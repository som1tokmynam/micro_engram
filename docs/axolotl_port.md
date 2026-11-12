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
