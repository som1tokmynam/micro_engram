"""Training patches for the Ricursive architecture (remote code, ``model_type: ricursive``).

The checkpoint's modeling_ricursive.py was written for inference. Its attention always calls
``F.scaled_dot_product_attention`` and builds masks only from ``attention_mask``: ``position_ids``
feed RoPE (sliding layers only; the full-attention layers have no RoPE) and the sliding window is
built from ``cache_position`` over the whole row. So with sample packing every document attends to
the previous ones, silently, and the 64 sliding layers compute full S x S attention (an explicit
mask, so no causal skip) instead of a 1024 window.

``patch_ricursive_varlen`` fixes both with FlashAttention 2 varlen:
- ``RicursiveModel.forward``: a forward without a KV cache, with explicit ``position_ids``
  (each row starting at 0) and no left padding derives the document boundaries ``cu_seqlens``
  from ``position_ids`` once (one host sync per forward), drops ``attention_mask`` (no S x S
  mask) and hands the boundaries to every layer through the layers' ``**kwargs`` (they also pass
  through the model's gradient checkpointing). Anything else (generation, a cache, left padding)
  runs the original code unchanged.
- ``RicursiveAttention.forward`` with those boundaries: the model's own projections, per-head
  Q/K norms, RoPE (sliding layers, from position_ids), value residual, output gate and o_proj,
  with ``flash_attn_varlen_func`` (causal; ``window_size=(sliding_window - 1, 0)`` on sliding
  layers, matching the model's ``key >= query - (window - 1)``; GQA native; default 1/sqrt(d)
  scale, as SDPA) instead of SDPA. Without flash-attn or off CUDA, an exact per-document SDPA
  reference runs instead (slow; for tests).

Nothing here imports Axolotl. The classes are found on the loaded model (remote code lives in a
dynamic ``transformers_modules`` package), so call it after the model is built.
"""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

LOG = logging.getLogger("micro_engram.axolotl_port")

VARLEN_KW = "ricursive_varlen"  # kwarg carrying (cu_seqlens int32 [n_docs+1], max_seqlen int)
_WARNED = {"reference": False}


# ----------------------------------------------------------------------------- boundaries
def varlen_from_position_ids(position_ids, attention_mask=None):
    """(cu_seqlens, max_seqlen) for the flattened [B*S] tokens, or None when this batch must take
    the original path: rows not starting at position 0, or left padding (a 2D mask with 0 in
    column 0). Right padding is fine: pads come last, so no real token attends to them (with
    Axolotl's padding they form their own segment, with continued positions they extend the last
    document)."""
    if position_ids is None or position_ids.dim() != 2:
        return None
    if attention_mask is not None:
        if attention_mask.dim() != 2 or bool((attention_mask[:, 0] == 0).any()):
            return None
    if bool((position_ids[:, 0] != 0).any()):
        return None
    flat = position_ids.reshape(-1)
    starts = torch.nonzero(flat == 0, as_tuple=False).flatten()
    cu = torch.cat([starts, starts.new_tensor([flat.numel()])]).to(torch.int32)
    max_len = int((cu[1:] - cu[:-1]).max())
    return cu, max_len


# ----------------------------------------------------------------------------- attention kernel
def _flash_varlen():
    try:
        from flash_attn import flash_attn_varlen_func
    except Exception:  # noqa: BLE001
        return None
    return flash_attn_varlen_func


def varlen_attention(q, k, v, cu, max_len, window):
    """Causal attention within each document. q [N, Hq, D], k/v [N, Hkv, D] (N = B*S flattened),
    ``window`` = the model's sliding_window or None. Returns [N, Hq, D]."""
    fn = _flash_varlen()
    if fn is not None and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
        ws = (window - 1, 0) if window else (-1, -1)
        return fn(q, k, v, cu, cu, max_len, max_len, dropout_p=0.0, causal=True, window_size=ws)
    if q.is_cuda and not _WARNED["reference"]:
        _WARNED["reference"] = True
        LOG.warning("[qlora_port] ricursive: flash-attn not available, packed attention uses the slow "
                    "per-document reference")
    return _reference_varlen(q, k, v, cu, window)


def _reference_varlen(q, k, v, cu, window):
    out = torch.empty_like(q)
    groups = q.shape[1] // k.shape[1]
    bounds = cu.tolist()
    for s, e in zip(bounds[:-1], bounds[1:]):
        n = e - s
        qd = q[s:e].transpose(0, 1)
        kd = k[s:e].repeat_interleave(groups, dim=1).transpose(0, 1)
        vd = v[s:e].repeat_interleave(groups, dim=1).transpose(0, 1)
        i = torch.arange(n, device=q.device)
        allowed = i[None, :] <= i[:, None]
        if window:
            allowed &= i[None, :] >= i[:, None] - (window - 1)
        out[s:e] = F.scaled_dot_product_attention(qd, kd, vd, attn_mask=allowed).transpose(0, 1)
    return out


# ----------------------------------------------------------------------------- patched forwards
def _attention_forward(orig, apply_rotary):
    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                output_attentions=False, use_cache=False, cache_position=None, value_residual=None, **kwargs):
        varlen = kwargs.get(VARLEN_KW)
        if varlen is None or past_key_value is not None or output_attentions:
            kwargs.pop(VARLEN_KW, None)
            return orig(self, hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                        past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache,
                        cache_position=cache_position, value_residual=value_residual, **kwargs)
        cu, max_len = varlen
        bsz, q_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        value_states = hidden_states
        if value_residual is not None:
            blend = torch.sigmoid(self.value_res_lambda[0]).to(dtype=hidden_states.dtype)
            value_states = torch.lerp(value_residual.to(dtype=hidden_states.dtype), hidden_states, blend)
        v = self.v_proj(value_states)
        q = self.q_norm(q.view(bsz, q_len, -1, self.head_dim))
        k = self.k_norm(k.view(bsz, q_len, -1, self.head_dim))
        v = v.view(bsz, q_len, -1, self.head_dim)
        window = None
        if self.layer_type == "sliding_attention":
            cos, sin = self.rotary_emb(v, position_ids)
            qt, kt = apply_rotary(q.transpose(1, 2), k.transpose(1, 2), cos, sin)
            q, k = qt.transpose(1, 2), kt.transpose(1, 2)
            window = self.config.sliding_window
        n = bsz * q_len
        out = varlen_attention(q.reshape(n, q.shape[2], self.head_dim).contiguous(),
                               k.reshape(n, k.shape[2], self.head_dim).contiguous(),
                               v.reshape(n, v.shape[2], self.head_dim).contiguous(), cu, max_len, window)
        out = out.reshape(bsz, q_len, -1)
        if self.config.use_gated_attention:
            gate = self.g_proj(hidden_states)
            if self.config.gate_full_precision:
                gate = gate.float()
            out = out * torch.sigmoid(gate).to(out.dtype)
        return self.o_proj(out), None, None

    forward._me_varlen = True
    forward._me_original = orig
    return forward


def _model_forward(orig):
    def forward(self, *args, **kwargs):
        if args or kwargs.get("past_key_values") is not None or kwargs.get(VARLEN_KW) is not None:
            return orig(self, *args, **kwargs)
        hs = kwargs.get("inputs_embeds")
        if hs is None and kwargs.get("input_ids") is not None:
            hs = kwargs["input_ids"]
        pos = kwargs.get("position_ids")
        if pos is None and self.training and not _WARNED.get("no_pos"):
            _WARNED["no_pos"] = True
            LOG.warning("[qlora_port] ricursive: a training batch has no position_ids, so the original "
                        "attention runs; if rows are packed, documents attend to each other")
        if hs is None or pos is None or hs.shape[1] < 2:
            return orig(self, *args, **kwargs)
        varlen = varlen_from_position_ids(pos, kwargs.get("attention_mask"))
        if varlen is None:
            return orig(self, *args, **kwargs)
        cu, max_len = varlen
        kwargs = dict(kwargs, attention_mask=None, use_cache=False)
        kwargs[VARLEN_KW] = (cu.to(pos.device), max_len)
        return orig(self, **kwargs)

    forward._me_varlen = True
    forward._me_original = orig
    return forward


def patch_ricursive_varlen(model):
    """Patch the Ricursive classes found on ``model`` (idempotent). Returns the number of attention
    modules now on the varlen path (0: not a Ricursive model)."""
    attn_cls = model_cls = None
    n = 0
    for mod in model.modules():
        name = type(mod).__name__
        if name == "RicursiveAttention":
            attn_cls = type(mod)
            n += 1
        elif name == "RicursiveModel":
            model_cls = type(mod)
    if attn_cls is None or model_cls is None:
        return 0
    if not getattr(attn_cls.forward, "_me_varlen", False):
        import sys

        apply_rotary = sys.modules[attn_cls.__module__].apply_rotary_pos_emb
        attn_cls.forward = _attention_forward(attn_cls.forward, apply_rotary)
    if not getattr(model_cls.forward, "_me_varlen", False):
        model_cls.forward = _model_forward(model_cls.forward)
    return n
