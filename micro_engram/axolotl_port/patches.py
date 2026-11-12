"""micro_engram.train's speed / correctness / tooling fixes, as patches for an Axolotl Gemma 4
(Q)LoRA run. See docs/NEXT_TASK_qlora_port.md for why each one exists and what it gained.

Nothing here imports Axolotl: every function takes the module, class or trainer it patches, so
the plugin (plugin.py) can hand in Axolotl's objects (looked up by name, which also works on a
patched Axolotl build) and the tests can hand in micro_engram's vendored copies of the same
Axolotl 0.19.0 modules. All patches are idempotent.
"""
from __future__ import annotations

import logging
import os
import sys

import numpy as np

LOG = logging.getLogger("micro_engram.axolotl_port")


# ----------------------------------------------------------------------------- item 1: collator
def patch_collator(collator_cls, pad_positions=True, mask_cross_doc_targets=True):
    """Wrap ``collator_cls.__call__`` (Axolotl's ``DataCollatorForSeq2Seq``; every packing collator
    ends in it through ``super().__call__``) so that:

    - (``pad_positions``) right padding continues the last document's ``position_ids`` (``p[-1]+1, p[-1]+2, ...``)
      instead of starting a new one at 0. The pads sit at the end, have no labels, and causal
      attention never lets a real token see them, so they need no isolation; a one-document row with
      padding then counts as ONE document, so the 512-dim global layers take SDPA's causal path
      (no S x S mask) and FA2 varlen on the sliding layers gets one sequence less.
    - (``mask_cross_doc_targets``) the first token of each packed document after the first is not a
      label: HF's loss shifts labels, so that label would be the target of the PREVIOUS document's
      last token. For chat SFT the first token is <bos> (already -100), so this is usually a no-op.

    The original then finds ``position_ids`` already at full length and pads only the rest.
    Left padding (not used with packing) is left alone."""
    cur = collator_cls.__call__
    if getattr(cur, "_me_pad_positions", False):
        return False

    def __call__(self, features, return_tensors=None):
        if features and isinstance(features[0], dict) and "position_ids" in features[0]:
            _prepad_positions(self, features, pad_positions, mask_cross_doc_targets)
        return cur(self, features, return_tensors=return_tensors)

    __call__._me_pad_positions = True
    __call__._me_original = cur
    collator_cls.__call__ = __call__
    return True


def _prepad_positions(collator, features, pad_positions, mask_cross_doc_targets):
    if mask_cross_doc_targets and "labels" in features[0]:
        for f in features:
            pos = np.asarray(f["position_ids"])
            starts = np.flatnonzero(pos == 0)
            starts = starts[starts > 0]  # index 0 is never a target (labels shift left by one)
            if len(starts):
                lab = np.array(f["labels"], dtype=np.int64)  # copy: features may share arrays
                lab[starts] = collator.label_pad_token_id
                f["labels"] = lab.tolist() if isinstance(f["labels"], list) else lab
    if not pad_positions or collator.tokenizer.padding_side != "right":
        return
    longest = max(len(f["position_ids"]) for f in features)
    m = collator.pad_to_multiple_of
    if m:
        longest = (longest + m - 1) // m * m
    for f in features:
        pos = f["position_ids"]
        n = longest - len(pos)
        if n <= 0:
            continue
        start = int(pos[-1]) + 1 if len(pos) else 0
        tail = np.arange(start, start + n, dtype=np.int64)
        if isinstance(pos, list):
            f["position_ids"] = pos + tail.tolist()
        else:
            f["position_ids"] = np.concatenate([np.asarray(pos, dtype=np.int64), tail])


# ----------------------------------------------------------------------------- item 1: packing probe
def dynamic_packing_probe(large_head):
    """``large_head.set_large_head_packed(None)``: decide per call (from position_ids) whether a row
    holds several documents, instead of Axolotl's ``bool(sample_packing)``, which declared every row
    packed and forced an S x S block mask (no causal skip, ~2x the work) on one-document rows too.
    Costs one tiny GPU sync per global layer per forward; incompatible with torch.compile's
    fullgraph (a data-dependent branch), which Axolotl only wanted to avoid for that.

    Returns False when this Axolotl has no such setter (before v0.18.0)."""
    setter = getattr(large_head, "set_large_head_packed", None)
    if setter is None:
        return False
    setter(None)
    return True


# Axolotl's packing-aware attention for the Gemma 4 global layers (v0.18.0+). Before it (the hybrid
# work started at b8358aa, v0.17.0) those layers ran plain "sdpa" and relied on the model's mask.
GLOBAL_PACKED_SDPA = "sdpa_global_packed"


def global_attention_impls(model):
    """(number of attention modules on GLOBAL_PACKED_SDPA, number on anything else), read from each
    ``self_attn.config._attn_implementation``. Item 2 is only safe when the first is > 0: dropping the
    model mask from plain-SDPA global layers would let them attend across packed documents."""
    packed = other = 0
    if model is None:
        return packed, other
    for name, mod in model.named_modules():
        if not name.endswith("self_attn"):
            continue
        impl = getattr(getattr(mod, "config", None), "_attn_implementation", None)
        if impl == GLOBAL_PACKED_SDPA:
            packed += 1
        else:
            other += 1
    return packed, other


# ----------------------------------------------------------------------------- item 2: unmasked globals
def unmask_global_layers(modeling_module):
    """Wrap ``modeling_module.create_causal_mask`` (the Gemma 4 namespace's binding, which Axolotl's
    hybrid patch already wrapped) so the full-attention (global-layer) mask is ``None`` in training.

    With sample packing Axolotl removes ``attention_mask`` and transformers builds a block-diagonal
    4D mask from position_ids for the global layers; Axolotl's ``sdpa_global_packed`` only routes to
    the Triton head_dim-512 kernel when the mask is None, so packed rows never reached it. Without
    a mask that function decides per call: one document -> SDPA causal; several -> Triton kernel
    (``large_head_attention: auto``), or the block mask it builds itself from position_ids.

    Passed through unchanged:
    - calls with ``and_mask_function`` / ``or_mask_function``: the sliding mask that
      ``create_masks_for_vision_model`` (every Gemma 4 batch under Axolotl: it passes
      ``mm_token_type_ids``) builds through this function; the FA2 sliding layers crash on a 4D one;
    - calls with a KV cache (generation);
    - batches with LEFT padding (a 2D attention_mask with a 0 in column 0): causal-only attention
      would let real tokens see the pads. Right padding is harmless (pads come last).

    Only correct when the global layers use Axolotl's ``sdpa_global_packed`` attention
    (``gemma4_hybrid_attn_impl: true``): plain SDPA layers would attend across documents."""
    cur = getattr(modeling_module, "create_causal_mask", None)
    if cur is None:
        return False
    if getattr(cur, "_me_unmasked", False):
        return True

    def create_causal_mask(*args, **kwargs):
        if kwargs.get("and_mask_function") is not None or kwargs.get("or_mask_function") is not None:
            return cur(*args, **kwargs)
        if kwargs.get("past_key_values") is not None:
            return cur(*args, **kwargs)
        am = kwargs.get("attention_mask")
        if am is not None and hasattr(am, "dim") and am.dim() == 2 and bool((am[:, 0] == 0).any()):
            return cur(*args, **kwargs)
        return None

    create_causal_mask._me_unmasked = True
    create_causal_mask._me_original = cur
    modeling_module.create_causal_mask = create_causal_mask
    return True


def warn_on_kernel_failure(flash_module):
    """Axolotl's ``flash_d512_route`` swallows any exception from the Triton kernel and silently runs
    the slow masked path. Wrap ``flash_module.flash_d512`` (looked up at call time by the router) so
    the first failure is logged, then re-raised for the router's fallback."""
    cur = getattr(flash_module, "flash_d512", None)
    if cur is None or getattr(cur, "_me_warn", False):
        return False
    state = {"warned": False}

    def flash_d512(*args, **kwargs):
        try:
            return cur(*args, **kwargs)
        except Exception as e:
            if not state["warned"]:
                state["warned"] = True
                LOG.warning("[qlora_port] Triton head_dim>256 flash kernel failed, packed rows fall back "
                            "to masked SDPA: %s: %s", type(e).__name__, e)
            raise

    flash_d512._me_warn = True
    flash_d512._me_original = cur
    flash_module.flash_d512 = flash_d512
    return True


# ----------------------------------------------------------------------------- item 3: loss normalization
def _unwrap(trainer):
    m = trainer.model
    try:
        m = trainer.accelerator.unwrap_model(m)
    except Exception:  # noqa: BLE001
        pass
    if hasattr(m, "get_base_model"):  # PEFT
        m = m.get_base_model()
    return m


def _loss_takes_num_items(model):
    import inspect

    fn = getattr(model, "loss_function", None)
    try:
        return fn is not None and "num_items_in_batch" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def token_normalized_loss(trainer):
    """Undo ``AxolotlTrainer.__init__``'s ``model_accepts_loss_kwargs = False`` when the model class
    explicitly declares ``accepts_loss_kwargs = True`` (Axolotl's own gemma4_loss_kwargs patch sets
    it on Gemma 4). Axolotl checks ``Gemma4Model.forward`` (the inner model, no
    ``num_items_in_batch`` parameter) and turns it off, which makes the loss a mean of per-micro-
    batch means / accumulation steps: bins with few trained tokens get more weight per token. With
    it on, the Trainer counts trained tokens over the whole step (all-gathered across ranks when
    ``average_tokens_across_devices``), passes ``num_items_in_batch`` to the model (stock
    ForCausalLMLoss, and CCE's apply_lce, sum then divide), and skips the / accumulation steps.

    Returns a dict describing what it saw / did (logged by the plugin)."""
    m = _unwrap(trainer)
    declared = getattr(type(m), "accepts_loss_kwargs", None)
    fwd = getattr(type(m), "forward", None)
    info = {
        "model_class": type(m).__name__,
        "forward_from": getattr(fwd, "__module__", "?"),
        "class_accepts_loss_kwargs": declared,
        "before": bool(trainer.model_accepts_loss_kwargs),
        "average_tokens_across_devices": bool(getattr(trainer.args, "average_tokens_across_devices", False)),
        "compute_loss_func": trainer.compute_loss_func is not None,
    }
    # Before Axolotl v0.18.0 (no gemma4_loss_kwargs patch) Gemma 4 still declares False, inherited
    # from PaliGemma; its loss is the stock ForCausalLMLoss, so decide from the loss signature.
    gemma4 = getattr(getattr(m, "config", None), "model_type", None) in ("gemma4", "gemma4_unified", "gemma4_text")
    info["loss_takes_num_items"] = _loss_takes_num_items(m)
    ok = declared is True or (gemma4 and info["loss_takes_num_items"])
    if ok and not trainer.model_accepts_loss_kwargs and trainer.compute_loss_func is None:
        trainer.model_accepts_loss_kwargs = True
    info["after"] = bool(trainer.model_accepts_loss_kwargs)
    return info


# ----------------------------------------------------------------------------- item 5: profiler
def profiler_callback(step, output_dir, callback_base):
    """A ``TrainerCallback`` (``callback_base`` is transformers.TrainerCallback) that records
    optimizer step ``step`` (1-based; 3 is a good choice, the optimizer state exists by then) with
    torch.profiler on every rank, then rank 0 writes output_dir/profile/rank0_step<N>.txt (top 40
    ops by CUDA time) and .json (timeline for https://ui.perfetto.dev)."""

    class StepProfiler(callback_base):
        def __init__(self):
            self.prof = None

        def on_step_begin(self, args, state, control, **kw):
            if self.prof is None and state.global_step + 1 == step:
                import torch
                from torch.profiler import ProfilerActivity, profile

                acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
                self.prof = profile(activities=acts, record_shapes=False)
                self.prof.__enter__()

        def on_step_end(self, args, state, control, **kw):
            if not self.prof:  # None: not started yet; False: already written
                return
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            prof, self.prof = self.prof, False  # False: done, never start again
            prof.__exit__(None, None, None)
            if int(os.environ.get("RANK", "0")) != 0:
                return
            folder = os.path.join(output_dir, "profile")
            base = os.path.join(folder, f"rank0_step{step}")
            try:
                os.makedirs(folder, exist_ok=True)
                prof.export_chrome_trace(base + ".json")
                key = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
                with open(base + ".txt", "w") as f:
                    f.write(prof.key_averages().table(sort_by=key, row_limit=40))
                LOG.warning("[qlora_port] profile written: %s.txt (top ops), %s.json (timeline)", base, base)
            except Exception as e:  # noqa: BLE001
                LOG.warning("[qlora_port] writing the profile failed (training continues): %s", e)

    return StepProfiler()


# ----------------------------------------------------------------------------- item 6: fail fast
def install_fail_fast():
    """On an uncaught exception, print the traceback and this GPU's memory NOW and ``os._exit(1)``,
    so torchrun stops the other ranks. Otherwise interpreter shutdown (process-group teardown) waits
    for ranks stuck in a collective this rank never reaches: the traceback never shows and the GPUs
    spin at 100% "util". Only in multi-process runs."""
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return False
    prev = sys.excepthook
    if getattr(prev, "_me_fail_fast", False):
        return True

    def hook(etype, value, tb):
        import traceback

        rank = os.environ.get("RANK", "?")
        try:
            print(f"\n[qlora_port] rank {rank} FAILED:", file=sys.stderr, flush=True)
            traceback.print_exception(etype, value, tb)
            sys.stderr.flush()
            import torch

            if torch.cuda.is_available():
                d = torch.cuda.current_device()
                print(f"[qlora_port] rank {rank}: {torch.cuda.memory_allocated(d) / 2**30:.2f} GiB allocated, "
                      f"{torch.cuda.memory_reserved(d) / 2**30:.2f} GiB reserved", file=sys.stderr, flush=True)
        finally:
            os._exit(1)

    hook._me_fail_fast = True
    sys.excepthook = hook
    return True
