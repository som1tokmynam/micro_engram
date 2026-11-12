# Next task: bring micro_engram.train's optimizations to the Axolotl QLoRA run

Written 2026-11-12. The goal is to make the next Gemma 4 31B (Q)LoRA SFT in Axolotl use the speed and
memory fixes that `micro_engram.train` proved on 5x RTX 3090.

Line numbers for Axolotl refer to **0.19.0** (`/root/micro_engram_work/ax/src/axolotl` on the dev box).
The owner's server Axolotl is a **patched build**, so find the same functions by name there.

## What was measured (engram stage 1, same model, same 5 GPUs)

| | Axolotl + engram plugin | micro_engram.train |
|---|---|---|
| s/step | ~100-110 | ~70 |
| peak allocated / GPU | ~20.7 GiB | 16.9 GiB |
| global-attention GPU time / step | 27.2 s | 14.4 s |

Not every gain carries over: part of it came from engram-only changes (row-sharded tables, sparse row
gradients, no LoRA). The table below sorts them.

## What carries over to QLoRA, by value

| # | change | type | value for QLoRA SFT | work |
|---|---|---|---|---|
| 1 | Causal fast path for one-document rows on the 512-dim global layers + padding that continues positions | speed | high | small (2 edits) |
| 2 | Triton head_dim-512 kernel for **packed** rows (global layers get no model mask) | speed | **very high for chat SFT** (almost every row is packed) | small-medium |
| 3 | Loss normalization across gradient accumulation | correctness | real for chat SFT (trained tokens per bin vary a lot) | check + small |
| 4 | Vocab-parallel embedding / lm_head + chunked loss | memory (~2.2 GiB/GPU) | high if memory-bound | medium-large |
| 5 | One-step profiler | tooling | needed to verify 1-4 | tiny |
| 6 | Fail fast on a rank error (no NCCL hang) | tooling | quality of life | tiny |
| - | no-grad prefix, sparse row gradients, row-sharded tables, 4-bit without adapter | engram-only | does not apply (LoRA trains every layer; no tables) | - |

Recommended order: **5 → 1 → 3 → 2 → 4**. Profile first, then take the cheap wins, then the big one.

### Status (2026-11-12)

Items 1, 2, 3, 5 and 6 are implemented as an Axolotl plugin, `micro_engram.axolotl_port`. See
[axolotl_port.md](axolotl_port.md) for config keys and the GPU verification steps. They are
CPU-tested (`test_axolotl_port.py`) but **not yet run on GPUs or on the owner's patched build**.
Item 4 is still open.

Found while porting:

- Stock 0.19.0 pads `position_ids` with `range(n_pad)`, not all zeros. The padding is one extra
  document, not one per pad token. The effect described in item 1 still holds: a padded
  one-document row counts as packed.
- `AxolotlTrainer.compute_loss` deletes `attention_mask` for packed Gemma 4 batches. transformers
  then builds the block mask from `position_ids`, which is the 4D mask that kept packed rows away
  from the Triton kernel (item 2).
- Axolotl's CCE fork (`ml-cross-entropy@4dfa522`, `apply_lce`) sums the loss and divides by
  `num_items_in_batch` when it gets that value, so item 3 also works with CCE.

---

## 1. Causal path for one-document rows (global layers)

**Problem.** Gemma 4's 10 global layers (head_dim 512) can't use FA2, so Axolotl's
`gemma4_hybrid_attn_impl` runs them on SDPA with a block-diagonal S×S mask. Two things make
one-document rows take that slow masked path too:

- `patch_manager._apply_gemma_hybrid_attention` (0.19.0 `loaders/patch_manager.py` ~line 330) calls
  `set_large_head_packed(bool(self.cfg.sample_packing))`. That declares every row packed, so
  `monkeypatch/attention/large_head.py:_multidoc_position_ids` always returns the position ids and
  `sdpa_global_packed_forward` always builds the mask.
- The collator pads `position_ids` with **0** (`utils/collators/batching.py`, `position_pad_token_id: int = 0`).
  Every pad token then looks like a new one-token document. A one-document row with padding is "packed",
  and FA2 varlen on the sliding layers also gets one sequence per pad token.

With a mask, SDPA can't skip the causal upper triangle, so it does ~2x the work.

**Fix (as in micro_engram):**
- `set_large_head_packed(None)`: probe each call. That costs one tiny sync per global layer per forward,
  and Axolotl only wanted to avoid it for `torch.compile`. See `micro_engram/train/model.py`,
  `apply_kernel_patches`.
- Pad `position_ids` as a continuation of the last document, `arange(last_len, last_len + n_pad)`, not
  zeros. See `micro_engram/train/data.py`, `collate`. Pads sit at the end, have no labels, and causal
  attention never lets a real token see them, so they need no isolation. Axolotl pads in
  `DataCollatorForSeq2Seq.__call__`. Override there for `position_ids`, for example with a subclass used
  by `V2BatchSamplerDataCollatorForSeq2Seq`.

**Measured:** global-attention time dropped from 27.2 s to 14.4 s per step on stage 1 (mostly
one-document rows). For chat SFT the gain from this item alone is smaller, because most rows really are
packed. Item 2 covers those.

**Check:** with `profile_step`-style profiling (item 5), the `fmha_cutlassF/B` (SDPA efficient) rows
should shrink, and losses should be unchanged within noise.

## 2. Triton kernel for packed rows (global layers)

**Problem.** Axolotl vendors a Triton flash kernel for head_dim 512 with varlen packing
(`monkeypatch/attention/flash_attn_d512.py`, routed by `large_head.py:flash_d512_route`, policy
`large_head_attention: auto`). `sdpa_global_packed_forward` only routes to it when
`attention_mask is None`. But Gemma 4's text model builds a 4D block mask for packed rows through the
patched `create_causal_mask` (`monkeypatch/gemma4_hybrid_mask.py`, which forces SDPA format). So packed
rows arrive with a mask and **never reach the kernel**. In chat SFT, nearly every row is packed, so the
global layers stay on the slowest path.

**Fix (as in micro_engram):** wrap `transformers.models.gemma4.modeling_gemma4.create_causal_mask` so that
during training it returns `None` for the full-attention mask. Calls with `and_mask_function` (the
sliding mask built through it) and calls with a KV cache (generation) pass through unchanged. See
`micro_engram/train/model.py`, `_global_layers_unmasked`. The global-layer function then decides per
call:
- one document: SDPA causal (item 1)
- several documents: the Triton kernel (with `large_head_attention: auto`)
- kernel off or failed: it builds the block mask itself (`_packing_block_causal_mask`)

micro_engram also logs one warning when the kernel fails (vendored `large_head.py`). Axolotl's version
falls back silently.

**Axolotl-specific:** Axolotl passes `mm_token_type_ids` for Gemma 4, which takes
`create_masks_for_vision_model`. That path builds the sliding mask through `create_causal_mask` with
`and_mask_function`. Keep the pass-through for those calls exactly as written, or the FA2 sliding layers
get a 4D mask and crash (see the long docstring in `gemma4_hybrid_mask.py`).

**Correctness test (CPU, exists):** `test_train.py` checks that hybrid mode on packed multi-document rows
gives the same loss as plain SDPA and the HF reference. The Triton path itself was **not yet GPU-tested**
in micro_engram. Check it with a profile (Triton `_fwd` / `_bwd_dkdv` / `_bwd_dq` kernels instead of
`fmha_cutlass*`) and a step-1 loss match, with and without `large_head_attention: auto`, same seed.

## 3. Loss normalization across gradient accumulation

**Finding (0.19.0, read, not run):** `monkeypatch/gemma4_loss_kwargs.py` sets
`Gemma4ForConditionalGeneration.accepts_loss_kwargs = True`, so the Trainer passes `num_items_in_batch`
and normalizes per token. `AxolotlTrainer.__init__` (`core/trainers/base.py` ~114-131) then unwraps PEFT
(`base_model`) and `.model`, checks that object's `forward` signature for `num_items_in_batch`
(`Gemma4Model.forward` has none), and **sets `model_accepts_loss_kwargs = False` again**. Its comment
says only the logged loss is affected. In fact the loss becomes a mean of per-micro-batch means, divided
by the accumulation steps.

- **Completion/CPT** (bins full, trained tokens per bin ~equal): little effect.
- **Chat SFT with assistant-only labels:** trained tokens per bin vary a lot, so bins with few trained
  tokens get more weight per token than bins with many. That's a real bias, not just a logging issue.

**Do:** on the patched build, log `trainer.model_accepts_loss_kwargs` at train start. If it's `False` for
Gemma 4, skip that override for Gemma 4, so the stock `ForCausalLMLoss` gets `num_items_in_batch`. Under
FSDP, also check that the Trainer all-reduces the count across ranks (HF does when
`average_tokens_across_devices` is on).

micro_engram's reference: sum the token losses of every micro-batch, all-reduce the trained-token count,
divide once. See `micro_engram/train/loop.py`, `train_step`.

**Minor, optional:** with packing, HF's shifted labels make the first token of the next document a
target for the previous document's last token. micro_engram masks it (`data.py`, `collate`). It's about
one token per document, and only worth doing alongside a collator change for item 1.

## 4. Vocab-parallel embedding / lm_head + chunked loss

**Problem.** Gemma 4's tied embedding/lm_head (262,144 x 5,376, ~2.7 GiB bf16) sits in FSDP's root unit,
**whole on every GPU** for the whole forward and backward. CCE avoids building full logits, but not the
unsharded weight.

**micro_engram's version** (`micro_engram/train/head.py`, ~180 lines, CPU-tested vs a full-logit
reference on 1 and 3 processes, GPU-verified in training):
- Each GPU keeps rows `row_range(V, rank, world)` of the weight, outside FSDP (~0.5 GiB at 5 GPUs).
- **Embedding:** an all_to_all lookup of the token ids' rows (`rows._RowLookup`, the same exchange as the
  engram tables), times Gemma's `sqrt(hidden)` scale. The text model then gets `inputs_embeds`.
- **Loss:** chunks of tokens. Every GPU all-gathers the chunk's hidden states, computes logits against
  its own rows with Gemma's final softcap, and max / sum-exp / target logit are combined with small
  all_reduces. The backward recomputes the chunk and reduce-scatters dH. Chunk size comes from the
  rounded-up row count, **identical on every rank**: an uneven vocab split once deadlocked NCCL.

**What porting to Axolotl takes:**
- Keep `embed_tokens` / `lm_head` out of FSDP: `ignored_params` on the root `fully_shard`, in Axolotl's
  `monkeypatch/accelerate/fsdp2.py:fsdp2_prepare_model`. Build the shard after loading.
- Replace the input embedding: patch `Gemma4TextModel.embed_tokens` with a module that calls the shard's
  lookup. Gemma 4 calls `self.embed_tokens(input_ids)`, so a module with that interface works. With PLE
  (not in 31B), keep `embed_tokens_per_layer`.
- Replace the loss: patch `Gemma4ForConditionalGeneration.forward` to stop after the text model's last
  hidden state and return `VocabShard.loss_sum(h, shifted_labels)`, divided by the step's token count
  (item 3). CCE/Liger FLCE must be off for this model.
- **LoRA consequence:** if the run trains embedding rows or lm_head (for example the reserved tokens
  repurposed for tribe names: 22 used, 68 left), the shard needs gradients for its own weight.
  - The embedding side already supports it: `_RowLookup.backward` returns row gradients.
  - The loss side needs dW = (p - onehot)ᵀ · H per chunk, added to `_VocabParallelCE.backward`.
  - Then an optimizer for the shard, like the engram's `RowTables`. That's new work.
  - With embeddings frozen, as in plain LoRA, no gradient is needed.

**Value:** ~2.2 GiB per GPU at 5 GPUs, plus whatever the loss's transient buffers were. It's the largest
memory item, and it's what let the engram run drop `offload_params`.

## 5. One-step profiler (do this first)

`micro_engram.train`'s `profile_step: N` wraps one optimizer step in `torch.profiler`, then writes
`profile/rank0_stepN.txt` (top 40 kernels by CUDA time) and `.json` (timeline for ui.perfetto.dev). See
`micro_engram/train/loop.py`, `_profiled_step`.

As an Axolotl plugin or `TrainerCallback`: start the profiler in `on_step_begin` at step N, stop it in
`on_step_end`, synchronize, and write on rank 0. About 30 lines.

Use step 3, after the optimizer state exists. Compare the `fmha_cutlass*`, NCCL `AllGather`, `aten::mm`
and loss rows before and after each change.

## 6. Fail fast on a rank error

When one rank raised, micro_engram's old `finally: destroy_process_group()` waited for the other ranks,
which were stuck in a collective. The traceback never printed and the GPUs sat at 100% "util" with
~130 W: an NCCL spin. Now the failing rank prints its traceback and VRAM and calls `os._exit(1)`, and
torchrun stops the rest. See `micro_engram/train/loop.py`, `main`.

In Axolotl, the same thing helps when a run "hangs": wrap the train entry point, or set
`TORCH_NCCL_ASYNC_ERROR_HANDLING=1` plus a shorter NCCL timeout so the watchdog kills it sooner.

**Diagnosing a hang:** `py-spy dump --pid <rank pid>` on every rank (sudo with the full path to py-spy).
Ranks in different functions point at the mismatched collective.

## Verification plan

1. Baseline: current Axolotl QLoRA config, `profile_step`-style profile of step 3, plus step-1 loss,
   peak memory per GPU, and s/step.
2. After each item, the same three numbers. Step-1 loss must match within noise for 1, 2 and 4. Item 3
   changes the loss value on purpose.
3. Read samples on known scenes (owner's judgment), since loss alone doesn't tell the story.

## Pointers

- micro_engram repo: `micro_engram/train/{head,model,data,loop}.py`, `micro_engram/rows.py`,
  `micro_engram/train/vendor/` (Axolotl 0.19.0 kernels, Apache-2.0).
- Tests: `test_train.py` (2-process CPU: loss vs HF, hybrid attention, resume...), `test_head.py`
  (vocab-parallel loss and embedding vs a full-logit reference, 1 and 3 processes, uneven split).
- Axolotl 0.19.0 source on the dev box: `/root/micro_engram_work/ax/src/axolotl`.
