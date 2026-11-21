"""Fold PEFT LoRA's ``* scaling`` and ``+ result`` into one GEMM.

PEFT's LoRA forward (``lora.Linear`` and ``lora.bnb.Linear4bit``, peft 0.21) does, per adapted
projection: ``result = base(x)``; ``result = result.clone()`` (4-bit only, "defensive");
``result = result + lora_B(lora_A(x)) * scaling``. That is a clone, a multiply and an add over the
full ``[tokens, out]`` output, twice under gradient checkpointing, plus ``grad * scaling`` over the
same size in backward. Here: ``result = base(x2d)``; ``result.addmm_(lora_A(x2d), B^T,
alpha=scaling)``: cuBLAS adds the scaled low-rank product into the base output in its epilogue;
backward touches only rank-sized tensors besides the GEMMs.

Falls back to PEFT's own forward whenever the fast path's assumptions don't hold: adapters not in
the base output's dtype (no silent precision change), dropout active in training, several active
adapters, LoRA variants (DoRA, ...), lora bias, merged/disabled adapters, extra forward arguments,
a base output that is a view, or autocast on.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG = logging.getLogger("micro_engram.axolotl_port")
_STATS = {"fused": 0, "fallback": 0, "logged": False}


def _fast_ok(self, x, args, kwargs):
    if args or kwargs or self.disable_adapters or self.merged or torch.is_autocast_enabled():
        return None
    active = self.active_adapters
    if len(active) != 1:
        return None
    a = active[0]
    if a not in self.lora_A or a in getattr(self, "lora_variant", {}):
        return None
    if getattr(self, "lora_bias", {}).get(a, False):
        return None
    drop = self.lora_dropout[a]
    if not isinstance(drop, nn.Identity) and self.training and getattr(drop, "p", 0) > 0:
        return None
    return a


def _make_forward(orig):
    def forward(self, x, *args, **kwargs):
        a = _fast_ok(self, x, args, kwargs)
        if a is None:
            return orig(self, x, *args, **kwargs)
        A = self.lora_A[a].weight
        B = self.lora_B[a].weight
        if x.dtype != A.dtype or B.dtype != A.dtype:  # e.g. fp32 adapters on a bf16 base: PEFT's path
            _count(False)
            return orig(self, x, *args, **kwargs)
        s = float(self.scaling[a])
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        result = self.base_layer(x2)
        h = F.linear(x2, A)
        if result.dim() == 2 and result._base is None and result.is_contiguous() and result.dtype == A.dtype:
            result.addmm_(h, B.t(), alpha=s)
            _count(True)
        else:  # unusual base output: PEFT's math on the result we already have (no second base call)
            result = result + (F.linear(h, B) * s).to(result.dtype)
            _count(False)
        return result.reshape(*shape[:-1], result.shape[-1])

    forward._me_fused_lora = True
    forward._me_original = orig
    return forward


def _count(fused):
    _STATS["fused" if fused else "fallback"] += 1
    if not _STATS["logged"] and _STATS["fused"] + _STATS["fallback"] >= 200:
        _STATS["logged"] = True
        LOG.warning("[qlora_port] fused LoRA: %d of the first %d LoRA calls on the addmm path "
                    "(the rest fell back to PEFT's forward: dtype/layout)", _STATS["fused"],
                    _STATS["fused"] + _STATS["fallback"])


def patch_fused_lora():
    """Patch PEFT's ``lora.Linear`` and ``lora.bnb.Linear4bit`` forwards (idempotent). Returns the
    names of the classes patched."""
    done = []
    targets = []
    try:
        from peft.tuners.lora.layer import Linear

        targets.append(("lora.Linear", Linear))
    except Exception:  # noqa: BLE001
        pass
    try:
        from peft.tuners.lora.bnb import Linear4bit

        targets.append(("lora.bnb.Linear4bit", Linear4bit))
    except Exception:  # noqa: BLE001
        pass
    for name, cls in targets:
        if not getattr(cls.forward, "_me_fused_lora", False):
            cls.forward = _make_forward(cls.forward)
        done.append(name)
    return done
