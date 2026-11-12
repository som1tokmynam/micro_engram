"""The Axolotl plugin: applies ``patches`` at Axolotl's hooks. Axolotl objects are looked up by
module/attribute name, so this also works on a patched Axolotl build with the same names."""
from __future__ import annotations

import importlib
import logging

from axolotl.integrations.base import BasePlugin

from . import patches

LOG = logging.getLogger("micro_engram.axolotl_port")

_GEMMA4_MODULES = (
    "transformers.models.gemma4.modeling_gemma4",
    "transformers.models.gemma4_unified.modeling_gemma4_unified",
)


def _on(cfg, key):
    v = cfg.get(key) if hasattr(cfg, "get") else getattr(cfg, key, None)
    return True if v is None else bool(v)  # unset -> default on


def _import(name):
    try:
        return importlib.import_module(name)
    except Exception as e:  # noqa: BLE001
        LOG.warning("[qlora_port] cannot import %s: %s", name, e)
        return None


class QLoRAPortPlugin(BasePlugin):
    def get_input_args(self):
        return "micro_engram.axolotl_port.args.QLoRAPortArgs"

    def register(self, cfg):
        if _on(cfg, "qlora_port_fail_fast") and patches.install_fail_fast():
            LOG.warning("[qlora_port] fail fast: a rank's uncaught error prints and exits (no NCCL hang)")

    def post_model_build(self, cfg, model):
        done = []
        pad, mask = _on(cfg, "qlora_port_continue_pad_positions"), _on(cfg, "qlora_port_mask_cross_doc_targets")
        if pad or mask:
            batching = _import("axolotl.utils.collators.batching")
            if batching is not None and patches.patch_collator(batching.DataCollatorForSeq2Seq, pad, mask):
                done.append("collator: " + ", ".join(
                    s for s, on in (("pads continue position_ids", pad),
                                    ("no cross-document targets", mask)) if on))
        if not cfg.get("gemma4_hybrid_attn_impl"):
            if _on(cfg, "qlora_port_dynamic_packing") or _on(cfg, "qlora_port_unmask_global_layers"):
                LOG.warning("[qlora_port] gemma4_hybrid_attn_impl is off: the global-layer fixes (items 1-2) "
                            "need it and are skipped")
        else:
            large_head = _import("axolotl.monkeypatch.attention.large_head")
            if large_head is not None and _on(cfg, "qlora_port_dynamic_packing"):
                if cfg.get("torch_compile"):
                    LOG.warning("[qlora_port] torch_compile is on: the per-call packing probe is a graph "
                                "break; set qlora_port_dynamic_packing: false if compile matters more")
                patches.dynamic_packing_probe(large_head)
                done.append("global layers probe packing per call")
            if _on(cfg, "qlora_port_unmask_global_layers"):
                n = sum(bool(m is not None and patches.unmask_global_layers(m))
                        for m in map(_import_quiet, _GEMMA4_MODULES))
                if n:
                    done.append(f"global layers get no model mask ({n} namespace(s))")
                policy = (large_head.get_large_head_policy() if large_head is not None else "?")
                if policy == "sdpa":
                    LOG.warning("[qlora_port] large_head_attention is 'sdpa': packed rows on the global "
                                "layers use the block mask, not the Triton kernel. Set "
                                "large_head_attention: auto for item 2.")
            flash = _import_quiet("axolotl.monkeypatch.attention.flash_attn_d512")
            if flash is not None and patches.warn_on_kernel_failure(flash):
                done.append("Triton kernel failures are logged")
        if done:
            LOG.warning("[qlora_port] %s", "; ".join(done))

    def post_trainer_create(self, cfg, trainer):
        if not _on(cfg, "qlora_port_token_loss"):
            return
        info = patches.token_normalized_loss(trainer)
        LOG.warning("[qlora_port] loss normalization: %s", info)
        if not info["after"]:
            LOG.warning("[qlora_port] model_accepts_loss_kwargs stays False: the loss is a mean of "
                        "per-micro-batch means (biased for assistant-only chat SFT)")
        elif "liger" in info["forward_from"].lower():
            LOG.warning("[qlora_port] forward comes from %s: check it divides by num_items_in_batch",
                        info["forward_from"])

    def add_callbacks_post_trainer(self, cfg, trainer):
        step = cfg.get("qlora_port_profile_step")
        if not step:
            return []
        from transformers import TrainerCallback

        return [patches.profiler_callback(int(step), trainer.args.output_dir, TrainerCallback)]


def _import_quiet(name):
    try:
        return importlib.import_module(name)
    except Exception:  # noqa: BLE001
        return None
