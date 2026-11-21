# Axolotl plugin: micro_engram.train's fixes for a Gemma 4 (Q)LoRA run

`micro_engram.axolotl_port` is an Axolotl plugin. It carries the training fixes that
`micro_engram.train` proved on 5x RTX 3090 over to a normal Axolotl (Q)LoRA SFT, items 1, 2, 3, 5
and 6 of [NEXT_TASK_qlora_port.md](NEXT_TASK_qlora_port.md). Item 4 (vocab-parallel
embedding / lm_head) is not done yet.

It patches Axolotl objects that it finds by module and attribute name, at Axolotl's plugin hooks, so
it also works on a patched Axolotl build as long as those names exist. The logic lives in
`micro_engram/axolotl_port/patches.py` and doesn't import Axolotl. `plugin.py` is the Axolotl glue.

## Axolotl versions

Axolotl's Gemma 4 hybrid-attention work started at upstream commit `b8358aa` (2026-04-12, PR #3598,
first released in v0.17.0). What each item needs, per upstream history up to v0.21.1 / `75fe8a1c`:

| piece the plugin uses | upstream since | if missing |
|---|---|---|
| `sdpa_global_packed` on the global layers | v0.18.0 (before it, they ran plain `sdpa` and needed the model's mask) | item 2 is **skipped** (the plugin counts the layers' attention impl), because dropping the mask would let attention cross packed documents |
| `large_head.set_large_head_packed` | v0.18.0 (`c35810da`) | item 1's per-call probe is skipped; continued pad positions still apply |
| `large_head` + `flash_attn_d512` (Triton kernel) | v0.18.0 | no kernel: packed rows use the block mask |
| pass-through of sliding-mask calls in `gemma4_hybrid_mask` | v0.19.0 (`6e63e060`) | v0.18.x has Axolotl's own FA2-sliding-mask crash on the vision-mask path; the plugin doesn't change it |
| `gemma4_loss_kwargs` (Gemma 4 declares `accepts_loss_kwargs`) | v0.18.0 | item 3 decides from the Gemma 4 loss signature instead |
| Trainer no longer forces `model_accepts_loss_kwargs = False` | **v0.20.0** (`e6e8db31`, same diagnosis as item 3) | from v0.20.0, item 3 finds `before: True` and does nothing |

In every version, the plugin hooks run after Axolotl's own setters. Both
`set_large_head_packed(bool(sample_packing))` calls (pre-model-load and hybrid setup) run before
`post_model_build`. From v0.21.0 the collator precomputes FA varlen `cu_seqlens` from the final
`position_ids`. The plugin pads before that, so `cu_seqlens` already sees the continued padding.

### The owner's server: PR #4180 + the micro_engram engram plugin

PR #4180 only adds CoreWeave `host_paths` mounts. Its base is upstream `cfeff472` (2026-10-18,
between v0.21.0 and v0.21.1). There, the attention, collator and loss-kwargs modules are identical to
the latest upstream (`75fe8a1c`), and:

- **Items 1, 2, 5, 6 apply.** Axolotl's two `set_large_head_packed(bool(sample_packing))` calls run
  before the plugin's `post_model_build`. `compute_loss` still deletes `attention_mask` for packed
  Gemma 4 batches, so the global layers still get transformers' block mask without item 2. The
  collator still pads `position_ids` with `range(n)`.
- **Item 3 is already fixed** (`model_loss_accepts_num_items_in_batch`). Expect
  `loss normalization: {... 'before': True, 'after': True}`. The plugin changes nothing there.
- **The engram plugin** (`micro_engram/plugin.py`, tag `axolotl-plugin-final`) patches the optimizer,
  checkpoint save/resume and the LR scheduler. It doesn't touch attention, masks, the collator or
  loss kwargs, so there's no overlap. The engram reads `position_ids` only to keep n-grams inside a
  document (`pos >= i`). With continued padding, pad tokens' n-grams read the last real tokens.
  That doesn't affect the loss, because pads have no labels and causal attention never shows them
  to a real token. `micro_engram.train` pads the same way.
- **Install by copying, not `pip install -e` of this branch.** Current master no longer has
  `micro_engram/plugin.py`, so installing it over the `axolotl-plugin-final` checkout would remove
  the engram plugin. `micro_engram/axolotl_port/` only imports numpy, pydantic, Axolotl and its own
  `patches`, so copy that folder into the server's checkout (editable install, or reinstall). Then
  list both plugins:

  ```yaml
  plugins:
    - micro_engram.MicroEngramPlugin          # as before
    - micro_engram.axolotl_port.plugin.QLoRAPortPlugin
  ```
- **Check that it loaded.** This Axolotl skips a plugin that fails to import, with only a warning
  (`Could not load plugin ...; skipping it`), and trains on without it. Look for
  `Plugin loaded successfully: micro_engram.axolotl_port.plugin.QLoRAPortPlugin` and the
  `[qlora_port] ...` lines at startup.

## Ricursive (remote code, `model_type: ricursive`)

The checkpoint's `modeling_ricursive.py` was written for inference and has three problems for
packed training:
- Its attention always calls SDPA and builds masks only from `attention_mask`. With sample packing,
  every document attends to the previous ones, silently. Axolotl only warns, and its warning
  assumes SDPA keeps packed samples apart.
- The 64 sliding layers compute full S×S attention instead of a 1024 window.
- `RIPPLE_MLP_CHUNK` row chunking dequantizes the 4-bit MLP weights again for every chunk.

`micro_engram/axolotl_port/ricursive.py` (`qlora_port_ricursive_varlen`, default on, applied by the
same plugin) fixes the first two:
- Training forwards with `position_ids` run FlashAttention 2 varlen: one segment per document,
  causal, window 1023 on the sliding layers, GQA native.
- The model's own projections, Q/K norms, RoPE, value residual and output gate are kept.
- Generation, cached decoding, left padding and batches without `position_ids` run the original
  code. A training batch without `position_ids` logs a warning once.
- The plugin sets `RIPPLE_MLP_CHUNK=0` unless you set it.

```yaml
base_model: /path/to/ricursive
trust_remote_code: true
attn_implementation: sdpa   # transformers refuses flash_attention_2 for this remote class; the patch calls flash-attn itself
sample_packing: true
plugins:
  - micro_engram.axolotl_port.plugin.QLoRAPortPlugin
```

Startup log: `[qlora_port] ricursive: 80 attention modules on FlashAttention-2 varlen for packed rows`.
`flash-attn` must be installed, or packed rows fall back to a slow per-document path (with a
warning).

Test: `RICURSIVE_MODEL_DIR=<folder with modeling_ricursive.py + configuration_ricursive.py> python test_ricursive_port.py`.
On CPU it checks a tiny model:
- packed rows give each document the hidden states, loss and gradients it gets alone (sliding
  window binding, gradient checkpointing);
- the original model leaks (control);
- the fallbacks are unchanged.

On a GPU with flash-attn it also checks the FA2 kernel against the reference (bf16, forward and
backward, with and without the window).

**Chunked loss** (`qlora_port_ricursive_loss_chunk`, default 1024; `0` turns it off): the model's
loss builds the full `[S, 100352]` logits, an fp32 copy and an fp32 gradient, about 8 GB at 8k
tokens. In training with labels, the patched forward computes the loss 1024 tokens at a time
instead, about 1 GB at most.
- Each chunk is checkpointed, so only one chunk's logits exist, in forward and in backward.
- Every chunk goes through `self.lm_head(...)`, so PEFT/bnb/FSDP hooks still apply.
- The math is the same as the model's: bf16 `lm_head`, then fp32, then `cross_entropy`, with
  `sum / num_items_in_batch` or `mean`.
- It returns `logits=None`, which the Trainer doesn't need for the training loss.
- Eval, generation and calls asking for logits use the original forward.
- If `lm_head` is its own FSDP unit, the plugin falls back to the full loss with a warning: a
  different chunk count per rank could otherwise hang NCCL. In Axolotl, `lm_head` sits in the root
  unit.

The CPU test checks the loss value and every gradient (lm_head included) against the full-logits
loss, and that `lm_head` never sees more than one chunk.

**Fused MLP activation** (`qlora_port_ricursive_fused_mlp`, default on): the model computes
`silu(gate.clamp(max=T)) * up.clamp(-T, T)` as separate bf16 ops on `[tokens, 40960]` tensors. That
elementwise work was ~30% of compute in the step profile.
- One Triton kernel now does the forward in fp32 and writes once; one kernel does the backward.
- Only `gate` and `up` are saved for backward.
- The clamp masks match torch's clamp backward, which gives no gradient exactly at ±T.
- The projections stay module calls (LoRA / bnb / FSDP).
- Without CUDA or Triton, the original ops run.

`python test_ricursive_mlp.py`:
- **CPU:** runs the kernels through Triton's interpreter. Forward and both gradients match the
  model's ops (fp32, clamps binding, no-clamp case). With `RICURSIVE_MODEL_DIR` it also checks
  that loss and every gradient of a tiny model are unchanged (with gradient checkpointing).
- **GPU:** checks bf16 at `[4096, 40960]` (fused error at most 1.5× torch's own bf16 error,
  against fp32) and prints fwd+bwd timing. The interpreter truncates on bf16 stores, so the CPU
  bf16 bound is looser.

## Use

Install this repo in the Axolotl venv (`pip install -e /path/to/micro_engram`), then add to the
Axolotl config:

```yaml
plugins:
  - micro_engram.axolotl_port.plugin.QLoRAPortPlugin

gemma4_hybrid_attn_impl: true     # items 1-2 need it
large_head_attention: auto        # item 2: packed rows on the 512-dim global layers -> Triton kernel
qlora_port_profile_step: 3        # item 5: only for a measuring run
```

Every fix is on by default. Each key below turns one off:

| key | default | item | what it does |
|---|---|---|---|
| `qlora_port_profile_step` | unset | 5 | Profile this optimizer step (1-based). Rank 0 writes `output_dir/profile/rank0_stepN.txt` (top 40 ops by CUDA time) and `.json` (open in ui.perfetto.dev). |
| `qlora_port_dynamic_packing` | true | 1 | `set_large_head_packed(None)`: the global layers check each call whether the row really holds several documents. Axolotl declared every row packed under `sample_packing`. Costs one tiny sync per global layer; the plugin warns if `torch_compile` is on. |
| `qlora_port_continue_pad_positions` | true | 1 | Right padding continues the last document's `position_ids` instead of restarting at 0. A padded one-document row is then one document, which gets SDPA causal on the global layers and one FA2 varlen sequence less on the sliding ones. |
| `qlora_port_mask_cross_doc_targets` | true | 3 (minor) | In a packed row, the first token of each later document is no longer a target of the previous document's last token. In chat SFT that token is `<bos>` and already masked, so this is usually a no-op. |
| `qlora_port_unmask_global_layers` | true | 2 | The model builds no full-attention mask in training. Axolotl's `sdpa_global_packed` then sends packed rows to the Triton kernel, or builds the block mask itself when the kernel is off or fails; one-document rows take SDPA causal. Kernel failures are logged once (Axolotl falls back silently). Sliding masks, generation and left-padded batches are passed through. |
| `qlora_port_token_loss` | true | 3 | Undoes `AxolotlTrainer.__init__`'s `model_accepts_loss_kwargs = False` when the model class declares `accepts_loss_kwargs = True` (Axolotl's own Gemma 4 patch does). The loss becomes the per-token mean over the whole step (count all-gathered across ranks), not a mean of per-micro-batch means. Stock loss and CCE (`apply_lce`: sum, then divide by `num_items_in_batch`) both honor it. |
| `qlora_port_fail_fast` | true | 6 | In multi-process runs, an uncaught error prints the traceback and VRAM, then calls `os._exit(1)`, so torchrun stops the other ranks. Without it, the run hangs in NCCL. |

At startup, rank 0 logs one `[qlora_port] ...` line listing what was applied, then one
`loss normalization: {...}` line. In that line, `before` / `after` show
`trainer.model_accepts_loss_kwargs`, `forward_from` shows which module's forward computes the loss
(transformers, CCE, Liger), and it also shows `average_tokens_across_devices`.

## What changes in the numbers

- **Step-1 loss:** items 1 and 2 must not change it beyond noise. Item 3 changes it on purpose,
  because the loss and gradient become the per-token mean over the step. To compare with an older
  run, set `qlora_port_token_loss: false` for the A/B on items 1-2.
- **Logged loss:** with item 3 it's the true per-token mean. Before, it was a mean of
  per-micro-batch means.

## Verifying on the server (plan from the handoff)

1. Baseline: current config with only `plugins:` + `qlora_port_profile_step: 3` and all other
   `qlora_port_*` keys `false`. Record the step-1 loss, peak memory per GPU, s/step, and the profile.
2. Turn on `qlora_port_dynamic_packing` + `qlora_port_continue_pad_positions`. The `fmha_cutlassF/B`
   rows should shrink, and the step-1 loss should match.
3. Turn on `qlora_port_unmask_global_layers` with `large_head_attention: auto`. Triton `_fwd` /
   `_bwd_dkdv` / `_bwd_dq` should replace most `fmha_cutlass*` time, the step-1 loss should match,
   and no `kernel failed` warning should appear.
4. Turn on `qlora_port_token_loss`. Check that the log shows `after: True`. The loss value moves.
5. Read samples on known scenes. Loss alone doesn't tell the story.

## Tests

`python test_axolotl_port.py` (CPU, about 30 s) checks:

- the collator: it loads Axolotl 0.19.0's real `batching.py` from `AXOLOTL_SRC`, default
  `/root/micro_engram_work/ax/src`, and skips this part if the file isn't there
- the global-layer path on a tiny Gemma 4 against plain SDPA with transformers' packed mask: equal
  loss on packed and padded rows, no full mask built, block mask only for the packed row
- a real HF `Trainer` step with gradient accumulation: the gradient matches the per-token reference
  with the port, and differs from it under Axolotl's override
- the profiler output
- fail-fast in a subprocess

The Triton kernel path itself needs a GPU (step 3 above).
