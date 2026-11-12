# micro-engram

A small **hashed n-gram memory** (Engram-style) added to the residual stream of a frozen decoder model, with its own trainer. Built for and tested on Gemma 4 (31B).

At each position `t`, the module looks up the 2-, 3- (and optionally 4-) grams that **end** at `t` (so it stays causal) in hashed embedding tables. A learned gate decides how much of that memory goes into the hidden state after one decoder layer.

- **No data pipeline changes.** Hashes are computed directly from the token ids.
- **Sample packing works.** `position_ids` stop n-grams from crossing document boundaries.
- **It starts as an exact no-op.** The output projection is zero-initialised, so step 0 behaves exactly like the base model.
- **Token merging is optional.** A "canon" map can merge token variants (e.g. `▁The` / `▁the`) so they share table rows.
- **Serving:** vLLM plugin, an HF server, and a llama.cpp patch (see below).

---

## Installation

```bash
git clone https://github.com/som1tokmynam/micro_engram.git
cd micro_engram
pip install -r requirements.txt     # training; skip what your environment already has
pip install -e . --no-deps          # registers micro_engram (+ the vLLM plugin entry point)
python -c "import micro_engram.train.loop; print('ok')"
```

`requirements.txt` lists what training needs: torch, transformers with Gemma 4, bitsandbytes (4-bit base), flash-attn (FA2 sliding layers), liger-kernel (optional kernels), datasets (`axolotl_prepared_dataset`), wandb, pydantic, pyyaml, tqdm, safetensors. An existing Axolotl venv already has all of it. Serving with vLLM needs vLLM (pinned commit, see *Serving with vLLM*), not these.

---

## Training: `micro_engram.train`

Plain PyTorch + FSDP2 (no HF Trainer, Accelerate or PEFT). One YAML, launched with torchrun:

```bash
torchrun --nproc_per_node 5 -m micro_engram.train stage1.yaml
```

The key names follow the Axolotl configs this project used before, so an old config works after deleting `adapter`, `lora_*`, `unfrozen_parameters` (now errors: a 4-bit base loads directly and only the engram trains) and `plugins`. Axolotl-only keys (`strict`, `bf16`, `save_safetensors`, ...) are listed in one warning and ignored; any other unknown key is an error.

### Example (Gemma 4 31B, 5x RTX 3090)

```yaml
base_model: /path/to/G4-31B
load_in_4bit: true                       # NF4 base, no adapter
sequence_len: 8192

datasets:
  - path: /path/to/pretraining.jsonl
    type: completion                     # or chat_template (see Data)
dataset_prepared_path: /path/to/prepared
# axolotl_prepared_dataset: /path/to/axolotl/prepared/<hash>   # reuse Axolotl's tokenized data as-is

output_dir: /path/to/output
auto_resume_from_checkpoints: true
save_steps: 36
save_total_limit: 20

# --- engram ---
micro_engram_layer: 0
micro_engram_extra:
  - layer: 29
micro_engram_table_size: 2097152
micro_engram_table_sizes: {2: 1572864, 3: 4194304, 4: 4194304}
micro_engram_dim: 64
micro_engram_heads: 8
micro_engram_gate: scalar_sqrt
micro_engram_dtype: bf16                 # + stochastic rounding (default on)
micro_engram_optimizer: rowwise_adam
micro_engram_canon_path: /path/to/canon_paper.pt
micro_engram_stats_steps: 20

# --- schedule ---
micro_batch_size: 1
gradient_accumulation_steps: 3
num_epochs: 5
learning_rate: 5e-3                      # the tables
micro_engram_dense_lr: 1e-3              # key/value, gates, norm scales (default: learning_rate)
micro_engram_lr_epoch_decay: 0.5         # warmup, then x0.5 every epoch (or lr_scheduler: ...)
warmup_ratio: 0.05
max_grad_norm: 2.0
weight_decay: 0.0

# --- speed / memory ---
gradient_checkpointing: true
gradient_checkpointing_kwargs: {use_reentrant: false}
activation_offloading: true
gemma4_hybrid_attn_impl: true            # FA2 on the sliding layers, see Attention
attn_implementation: flash_attention_2
large_head_attention: auto               # Triton flash for packed rows on the 512-dim global layers
liger_rms_norm: true
liger_glu_activation: true
tf32: true
fsdp_config:
  offload_params: false
  reshard_after_forward: true

wandb_project: my-project
logging_steps: 1
seed: 42
```

Measured on 5x RTX 3090 with the layout above (stage 1, both engrams training): ~70 s/step and 16.9 GiB peak, against ~110 s/step and ~20.7 GiB with the old Axolotl plugin.

### What runs where (per GPU)

| Part | How |
|---|---|
| base model | frozen; FSDP2 per decoder layer, bf16 or bitsandbytes NF4 (`load_in_4bit`); `fsdp_config.offload_params` moves it to CPU; `cpu_ram_efficient_loading` loads it on rank 0 only (bf16) |
| embedding / output head (tied, ~2.7 GiB) | rows split over the GPUs, outside FSDP. The loss is vocab-parallel and chunked (Gemma's final softcap included): full logits never exist. `loss_chunk_tokens` caps the chunk |
| engram tables | row-sharded: each GPU keeps its share of the rows and fetches the rows a batch uses from their owners (all_to_all). Own optimizer and clipping |
| table gradients | with `lazy_adam` / `rowwise_adam`, only the touched rows (fp32, coalesced) instead of a full-size gradient per table (`sparse_row_grads: true`, the default) |
| engram key/value/gates/scales | replicated on every GPU, outside FSDP, gradients all-reduced (they are small) |
| layers before the first trained engram | forward only, no autograd (`no_grad_prefix: true`) |
| loss over gradient accumulation | per-token mean over the whole step: all micro-batches, all GPUs |
| activation checkpointing | per-layer non-reentrant, as HF's `gradient_checkpointing` (FSDP's own `activation_checkpointing` is refused) |
| activation offloading | the checkpointed layers' inputs go to a reused pinned-memory pool, copied on a side stream |

### Attention (Gemma 4)

Gemma 4 has 512-dim "global" layers that FA2 can't run, and SDPA has no packing support (it needs an S×S block mask, and then computes the whole square). With `gemma4_hybrid_attn_impl`, the sliding layers run FA2 (varlen, from `position_ids`), and the global layers decide per call:

| row | global layers |
|---|---|
| one document (padding continues its positions) | SDPA causal, no mask |
| several packed documents | `large_head_attention: auto`: the Triton head_dim-512 flash kernel (varlen); unset: the block-diagonal mask |

Use `auto` for chat SFT, where nearly every row holds several conversations. If the Triton kernel fails, one warning is logged and those rows use the mask. The kernels are vendored from Axolotl 0.19.0 (Apache-2.0) in `micro_engram/train/vendor/`: fused q/k/v-norm + RoPE (`fused_attn_kernel`, on by default), the hybrid mask, the head_dim-512 flash kernel, and the FSDP2 patch for bitsandbytes 4-bit parameters.

### Data

- `type: completion`: Axolotl's rules: the text is cut at `sequence_len*64` tokens, EOS appended when shorter, split into `sequence_len` chunks; every token is trained. Gemma 4's tokenizer adds no `<bos>`, so put a literal `<bos>` at the start of each line.
- `type: chat_template`: rendered with `chat_template_jinja` + `chat_template_kwargs`; trains each assistant turn's content plus its `eot_tokens`, not the role header nor what follows the end-of-turn token. `field_messages`, `message_property_mappings` and `roles` work as in Axolotl. Conversations longer than `sequence_len` are dropped.
- Tokenized data is cached in `dataset_prepared_path/micro_engram-<hash>/` (rank 0 tokenizes, `dataset_num_proc` processes).
- `axolotl_prepared_dataset: <folder>` reads an Axolotl prepared dataset instead. `compare_axolotl_data.py cfg.yaml <folder>` compares our tokens and masks with it.
- Packing: every epoch the samples are reshuffled (seed + epoch) and packed into `sequence_len` bins; `position_ids` restart per document.

### Settings

| Key | Default | Meaning |
|---|---|---|
| `micro_engram_layer` | `0` | Inject the memory into the output of this decoder layer (0-indexed: `0` = after the first block, the Engram paper's best single placement) |
| `micro_engram_table_size` | `65536` | Rows per hash table |
| `micro_engram_table_sizes` | unset | Rows per table **for each n-gram order**, overriding `micro_engram_table_size`, e.g. `{2: 1048576, 3: 3145728}`. `0`/`null` drops an order; an order not listed keeps `micro_engram_table_size` (orders 2 and 3), so `{4: N}` **adds 4-grams** |
| `micro_engram_dim` | `128` | Width of each table row |
| `micro_engram_heads` | `4` | Hash tables per n-gram order |
| `micro_engram_canon_path` | `null` | Token-merge map from `build_canon.py`. `null` turns merging off |
| `micro_engram_gate` | `scalar` | `scalar` (the paper), `scalar_sqrt` (DeepSeek's code, recommended) or `ple` (Gemma 3n style). See *Gate modes* |
| `micro_engram_dtype` | unset | Engram weights: unset = its layer's dtype (bf16); `fp32` doubles the engram's memory (weights, gradients, momentum) |
| `micro_engram_stochastic_rounding` | `true` | bf16 weights and optimizer states written back with stochastic rounding, so updates smaller than half a bf16 step add up on average (`test_optim.py`) |
| `micro_engram_optimizer` | unset | Tables: unset = AdamW (every row every step, dense gradients); `lazy_adam` = a row moves only when its n-gram appears (like SparseAdam); `rowwise_adam` = lazy + one fp32 second moment per row (half the optimizer state; FBGEMM/TorchRec style). See *Row-wise Adam* |
| `learning_rate` | `1e-3` | The tables' LR (and everything's, without `micro_engram_dense_lr`). Hashed rows only learn when their n-gram shows up, so they want a high LR |
| `micro_engram_dense_lr` | `learning_rate` | LR of key/value, gates and norm scales, which learn on every token. Same schedule as the tables. Logged as `dense_lr` |
| `micro_engram_train_scales` | `true` | `false`: the `scalar_sqrt` gate's two norm scales stay where they are (1.0 for a new engram, the trained value with `init_from`), so the gate can't learn its own sharpness. The early bf16 runs had them stuck at 1.0 by accident |
| `micro_engram_lr_epoch_decay` | `null` | Warmup, then LR x this factor at every epoch start (replaces `lr_scheduler`). Otherwise `lr_scheduler`: `constant`, `constant_with_warmup`, `cosine` (+`cosine_min_lr_ratio`), `linear`, `warmup_stable_decay` (+`lr_scheduler_kwargs`) |
| `micro_engram_stats_steps` | `10` | Every N steps, engram stats in the log line (see *Watching training*) |
| `micro_engram_init_from` / `micro_engram_frozen` / `micro_engram_seed` / `micro_engram_extra` | | Several engrams, starting from trained ones (see below) |
| `no_grad_prefix` | `true` | Layers before the first trained engram run without autograd |
| `sparse_row_grads` | `true` | Sparse table gradients with lazy optimizers |
| `loss_chunk_tokens` | `2048` | Cap on tokens per loss chunk (auto-shrunk to ~256 MB of logits) |
| `profile_step` | unset | Profile that step: `output_dir/profile/rank0_step<N>.txt` (top kernels) and `.json` (timeline for https://ui.perfetto.dev) |

The engram has `heads × dim ×` (sum of the per-order rows) table parameters plus two `(orders·heads·dim) × hidden_size` projections.

#### Table sizes per n-gram order

There are many more distinct 3-grams than 2-grams in most data, so one size for every table leaves the 2-gram tables roomy and the 3-gram ones crowded. `engram_data_stats.py` measures it on your prepared dataset (`--orders 2,3,4 --canon <canon file> --sizes ...` prints how many n-grams worth memorizing share each row). Then size each order:

```yaml
micro_engram_table_size: 2097152   # default for the orders not listed
micro_engram_table_sizes:
  2: 1048576    # 2-grams: half the rows
  3: 3145728    # 3-grams: 1.5x the rows (same memory as 2M + 2M)
  4: 0          # 0, null or not listed: no 4-grams
```

- Every order keeps `micro_engram_heads` tables, each of its own size. 4-grams look up the 3 tokens before each position: training, vLLM and the llama.cpp patch all handle it.
- Extra engrams take `table_sizes` too; without it they copy the main engram's per-order sizes, and with only their own `table_size` they use one size for all.
- The config records the sizes per order, and every loader rebuilds the same tables. An engram trained with one size per order still loads as before. `init_from` always uses the trained engram's sizes (the YAML can't change them), and resuming a run with different sizes is refused.
- Memory grows with the total rows: with row sharding in fp32 + `rowwise_adam`, every 64-wide row costs ~768 bytes in training (weights, gradients, momentum), split over the GPUs.

#### Gate modes

All modes use the same tables and start as an exact no-op. `scalar` and `ple` have the same parameter count; `scalar_sqrt` adds two vectors of `hidden_size`.

- **`scalar`** (default, as in the Engram paper): one gate value per token, `α = sigmoid(rms(h)·rms(W_K·e)/√d)`, then `h += α · W_V·e`. The memory is let in or held back as a whole.
- **`scalar_sqrt`** (as in DeepSeek's released Engram code): like `scalar`, but the score goes through a signed square root, `α = sigmoid(sign(s)·√|s|)`, and both RMSNorms get learnable scales. The score is about `cos·√d`; with Gemma's `d = 5376` (`√d ≈ 73`) the plain sigmoid is already at 0.999 for a cosine of 0.1, where it barely learns. The square root keeps the gate responsive (cosine 0.1 gives 0.94), and the scales let it learn its own sharpness. Recommended for new runs; `scalar` checkpoints keep loading as `scalar`. Inside the module the two scales are stored as their difference from 1 (bf16 cannot move a value sitting at 1.0 by an Adam step: its spacing there is 1/128); saved files still hold the scales themselves, so old checkpoints load unchanged.
- **`ple`** (Gemma 3n per-layer-embedding style): a gate per dimension, computed from the hidden state: `g = gelu(W_down·rms(h))`, then `h += W_up·(g ⊙ e)`. The context chooses *which features* of the memory to use, and each hash head's slice gets its own slice of the gate.

The gate mode is saved in `micro_engram_config.json`, and `infer_engram.py` picks it up automatically. Checkpoints from before this option are `scalar`.

### Several engrams, and starting from a trained one

Each engram sits in its own decoder layer. `micro_engram_extra` adds more. Any engram can start from trained weights (`init_from`) and can be kept frozen (`frozen`):

| Setting (main engram / in `micro_engram_extra`) | Meaning |
|---|---|
| `micro_engram_init_from` / `init_from` | Start from a trained engram: an output folder (its newest complete checkpoint), a `checkpoint-N` folder, a `micro_engram_config.json`, or the weights themselves (DCP folder, `.pt`, `.safetensors`). Old Axolotl checkpoints (`pytorch_model_fsdp_0`) work too. With a config, the engram is rebuilt exactly as trained (sizes, gate, **seed**); the YAML can't change those |
| `micro_engram_frozen` / `frozen` | Load it but don't train it (needs `init_from`). A frozen engram costs only its weights |
| `micro_engram_seed` / `seed` | Hash seed. Default `10007 × layer` (DeepSeek's scheme); layer 0 keeps seed 0 |
| `layer`, `table_size`, `table_sizes`, `dim`, `heads`, `gate` (extra only) | `layer` is required (one engram per layer); the others default to the main engram's |

Stage 2 (SFT) from stage 1, both engrams training: `micro_engram_init_from: /path/to/stage1_output` with the same `micro_engram_extra`, and `init_from` on the extra too. A frozen layer-0 engram with only a deeper one training also makes the no-grad prefix skip everything up to that layer.

Every checkpoint's `micro_engram_config.json` lists all engrams under `"engrams"`, each with its own `weights` (a frozen one points to the files it was loaded from). The first engram is also written at the top level for older loaders.

#### Row-wise Adam (`rowwise_adam`)

For the tables only: lazy row updates, and **one second moment per row** instead of one per value: the running mean of the row's g². This is "partial row-wise Adam" as FBGEMM/TorchRec use for big embedding tables. The momentum stays per value. The tables' optimizer state drops from 2× to ~1× their size.

- The second moment is always fp32, also with bf16 weights. In bf16, `v = 0.999·v + 0.001·g²` changes by less than bf16's precision, so `v` can never decay (shown in `test_optim.py`); with the noisy, sparse gradients engram rows get, steps come out ~0.6–0.9× of the correct size.
- It equals AdamW whenever a row's gradient values have the same size, and approximates it otherwise.
- Resume works across `lazy_adam` ↔ `rowwise_adam`.

### Checkpoints, resume, output

- `checkpoint-N/micro_engram_rows/`: the complete engram (tables, key/value, scales, buffers) as a DCP folder; `micro_engram_rows_optim/`: the tables' Adam state; `micro_engram_dense_optim.pt`: the dense optimizer + scheduler; `micro_engram_config.json`; `trainer_state.json`, written last (it marks the checkpoint complete).
- `output_dir/micro_engram_config.json` always points to the newest checkpoint, so `output_dir` can be served directly. At the end, the engram is also written to `output_dir/micro_engram_rows/`.
- `auto_resume_from_checkpoints` continues from the newest complete checkpoint: same data order, optimizer state and LR schedule. The YAML's LRs win over the checkpoint's, so they can be changed when resuming. The GPU count may change (rows are resharded).
- vLLM (`--additional-config`), llama.cpp (`gguf_add_engram.py`), `assemble_engram_model.py`, `engram_stats.py` and `init_from` all read these checkpoints.
- If a GPU fails, it prints its traceback and VRAM and exits; torchrun stops the others (no silent hang).

```bash
python assemble_engram_model.py --base /path/to/base_model \
    --engram /path/to/output/checkpoint-400/micro_engram_rows --out /path/to/base_model-engram
```

### Logs

stdout and `output_dir/train_log.jsonl` (and wandb with `wandb_project`), one JSON line per logged step: `loss`, `ppl`, `grad_norm` (dense part), `table_grad_norm`, `learning_rate` (tables), `dense_lr`, peak VRAM per GPU, `s_per_step`, trained tokens/s, and every `micro_engram_stats_steps` the engram stats. A progress bar on rank 0 shows loss and ppl.

### Not supported

LoRA/adapters (the base is frozen), an eval split, `fsdp_config.activation_checkpointing`, models whose output head is not tied to the embedding.

The Axolotl plugin this project started as was removed after the new trainer matched it on GPUs; `git checkout axolotl-plugin-final` has its last version.

For ordinary (Q)LoRA SFT in Axolotl, `micro_engram.axolotl_port` is a separate, small plugin that carries this trainer's attention, loss-normalization, profiling and fail-fast fixes over: see [docs/axolotl_port.md](docs/axolotl_port.md).

---

## Helper scripts

| Script | What it does |
|---|---|
| `zero_ngram_order.py SRC [--order 4] [--scale 0.5] [--layer-scale 29=0.5] --out DIR` | Copy an engram (config, checkpoint or output_dir) with some n-gram orders' tables zeroed and/or its injection scaled down (all engrams, or one layer): quick tests of "is it the 4-grams?" and "is it just too strong?" without retraining. Writes `consolidated.pth` + `micro_engram_config.json` |
| `build_canon.py` | Builds the token-merge map. `python build_canon.py --tokenizer /path/to/model --out canon.pt --paper`. By default it merges only case/NFKC variants (`▁The` = `▁the`). Each option merges more: `--merge-space` (`▁the` = `the`), `--strip-accents` (`é` = `e`, Latin/Greek/Cyrillic diacritics only) and `--merge-whitespace` (all whitespace-only pieces share one id). `--paper` turns on all three, like the paper's tokenizer compression. `--protect names.txt` (one name per line, `#` comments, repeatable) keeps names that are also ordinary words apart from the word, e.g. `Peril` the dragon vs `peril`: a single-token name keeps its own id while the word forms still merge with each other. Names split into several tokens are listed and left alone. Special, added and byte tokens are never merged. It prints the class count for each step so you can compare. |
| `engram_data_stats.py` | Estimates how many n-grams your data trains and how crowded each table size would be. `python engram_data_stats.py /path/to/last_run_prepared/<hash> [--max-tokens N] [--orders 2,3,4] [--canon canon.pt] [--sizes 1048576,2097152,...]`. `--canon` counts the n-grams the engram really sees (after token merging); 4-grams are counted through a 64-bit hash. Use it to pick `micro_engram_table_size` / `micro_engram_table_sizes`. |
| `assemble_engram_model.py` | Builds a servable model folder from the base weights (symlinked, not copied) plus an engram from a DCP checkpoint, a safetensors folder or a `.pt`. See *Outputs*. |
| `extract_engram.py` | Pulls the engram tensors out of merged safetensors into a standalone `.pt` file. |
| `infer_engram.py` | Test generation with the engram attached. See below. |
| `engram_stats.py` | Engram weight stats from checkpoints on disk, **while training keeps running** (CPU only, reads only the engram tensors). `python engram_stats.py /path/to/output_dir` (newest checkpoint), `--all` (every checkpoint on disk), `--watch 120` (print each new checkpoint as it appears). Skips checkpoints that are still being written. |
| `eval_logprob.py` | Measures how well a served model predicts held-out text, through vLLM's prompt log-probabilities, and compares two configs (e.g. with and without the engram): loss, perplexity, per-chunk win rate and a 95% interval. `run --url ... --data heldout/ --out a.json` once per config, then `compare a.json b.json`. |
| `render_chat.py` | Renders N random conversations from a chat `.jsonl` (`messages`, or ShareGPT `conversations`) with the model's chat template (`--template` = training's `chat_template_jinja`, `--kwargs` = its `chat_template_kwargs`) into `{"text": ...}` lines for `eval_logprob.py`, e.g. to check that an engram doesn't hurt chat. |
| `engram_token_diff.py` | Shows **where** the engram pushes within one reply: scores every token with the engram on and off (vLLM prompt log-probabilities, same tokens both times), lists the biggest pushes both ways, and writes an HTML heat map (green: the engram made the token likelier, red: less likely; hover for both probabilities). `run --url ... --messages chat.json --reply reply.txt --out on.json` once per config, then `render on.json off.json --out diff.html`. |

### Inference

```bash
# merged model (copy micro_engram_config.json into the model folder first)
python infer_engram.py --model /path/to/merged_model --prompt "Hello there"

# compare against the same model with the engram turned off
python infer_engram.py --model /path/to/merged_model --prompt "Hello there" --no-engram

# base model + separate engram file
python infer_engram.py --model /path/to/base_model --engram-pt micro_engram.pt \
    --engram-config micro_engram_config.json --prompt "Hello there"
```

Engram settings are resolved in this order: explicit flag (`--layer`, `--table-size`, `--dim`, `--heads`), then `micro_engram_config.json`, then the shapes of the saved tensors, then the defaults.

> The engram is not part of the Hugging Face model class. Plain `transformers` and llama.cpp loading will **ignore** it. Use `infer_engram.py` / `serve_engram.py`, or vLLM with the plugin below.

### Serving with vLLM

`micro_engram_vllm` is a vLLM plugin (no vLLM fork). Installing this package registers it automatically; it replaces vLLM's Gemma 4 classes (text-only and multimodal) with ones that run the engram at its layer(s). Models without a `micro_engram_config.json` keep running as plain Gemma 4.

**Tables in host memory** (`--additional-config '{"micro_engram": "...", "micro_engram_offload": true}'`, or `MICRO_ENGRAM_OFFLOAD=1`): each rank keeps its share of the tables in pinned CPU RAM, and the GPU reads only the rows a step looks up, over PCIe, through a UVA view. That's the same mechanism vLLM uses for its own Engram/n-gram tables. The tables then take no GPU memory (all of it goes to the KV cache), at the cost of pinned host RAM (the whole engram once, split over the ranks) and a little latency per step (an 8k-token prefill reads a few tens of MB). CUDA graphs keep working. It needs a platform with UVA (NVIDIA).

With tensor parallelism the engram tables are **split by rows across the GPUs**, like vLLM's own vocabulary embedding: each GPU holds 1/TP of every table, and a lookup is a masked local read plus one all-reduce (exact: every row comes from one GPU). Two 2M-row engrams take ~1.1 GB per GPU at TP 4 instead of 4.3 GB on every GPU. The weight files are unchanged; `test_vllm_tp.py` checks it against the full tables.

Install the package **in the vLLM environment** (Axolotl is not needed there):

```bash
pip install -e /path/to/micro_engram
```

Serve the **merged** model folder, with `micro_engram_config.json` copied into it:

```bash
vllm serve /path/to/merged_model --tensor-parallel-size 4 \
    --reasoning-parser gemma4        # optional: thinking in reasoning_content
```

- **Images, video and audio:** the multimodal model is served as usual. Media placeholder tokens get no engram (training never saw them), and the text right after an image sees no earlier tokens through the engram, like the start of a sequence. Text-only serving with `--hf-overrides '{"architectures": ["Gemma4ForCausalLM"]}'` also works. (Multimodal serving is tested on CPU only, not yet on GPUs.)
- **Model Runner V2 is required.** It is vLLM's default; the plugin stops with an error if something forces the old runner.
- **A/B test:** start with `MICRO_ENGRAM_OFF=1` to load the same weights with the engram switched off.
- **vLLM version:** the plugin uses internal vLLM APIs and was written against commit `64175d3827`. Other versions may need changes.
- The startup log shows `[micro_engram] attached at layer ...` when the engram is active.

#### Choosing the engram at launch

The engram doesn't have to live in the model folder. Serve the **plain base model** and pick an engram at launch, like a LoRA adapter:

```bash
vllm serve /path/to/base_model --tensor-parallel-size 4 \
    --additional-config '{"micro_engram": "/path/to/run/checkpoint-400/micro_engram_config.json"}'
```

- `micro_engram` is a `micro_engram_config.json`, or the folder that holds it. Its `"weights"` field says where the engram is, relative to the config file: a DCP checkpoint folder (only the engram is read from it, even from a full checkpoint), a `.safetensors` file or folder, or a `.pt`.
- Training writes `"weights"` for you: FSDP checkpoints point to their `pytorch_model_fsdp_0`, a non-FSDP run's final config points to `micro_engram.pt`. For other files, add it by hand, e.g. `"weights": "my_engram.pt"`.
- Without `--additional-config` (or the `MICRO_ENGRAM_CONFIG` environment variable) the plugin uses a `micro_engram_config.json` next to the model weights (an assembled folder), and otherwise serves plain Gemma 4. A chosen engram replaces any engram already in the model folder.
- The engram must have been trained on this exact base model. Training records a fingerprint of the base weights in the config (`base_fingerprint`: a hash of the embeddings and of every projection in the first, middle and last layer, independent of the folder name). vLLM, `assemble_engram_model.py` and the HF scripts **warn** when the model they load differs, but load it anyway.
- `infer_engram.py` / `serve_engram.py` do the same with `--engram-config path/to/micro_engram_config.json`. Their `--engram-pt` accepts every format above.

---

### Serving with llama.cpp (Gemma 4)

llama.cpp has no plugin mechanism for models, so support comes as a small patch set in `llama_cpp/`: one new file (`src/llama-micro-engram.{h,cpp}`) plus 5 small edits to existing files (`micro-engram.patch`, ~25 changed lines). The engram goes **inside the GGUF**, so `llama-server` and the other tools need no new flags.

1. **Patch and build llama.cpp** (tested on llama.cpp `d38bea3`):
   ```bash
   ./llama_cpp/apply.sh /path/to/llama.cpp
   cd /path/to/llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j --target llama-server
   ```
   Only `libllama` changes (no CUDA code). To ship a prebuilt `libllama.so`, it must come from the same llama.cpp commit as the `llama-server` and ggml libraries it is used with.
2. **Add the engram to a GGUF** (any quantization; the engram is stored as BF16):
   ```bash
   pip install gguf    # or PYTHONPATH=/path/to/llama.cpp/gguf-py
   python llama_cpp/gguf_add_engram.py --gguf gemma-4-31B-it-Q4_K_M.gguf \
       --engram /path/to/output/checkpoint-400 --out gemma-4-31B-it-Q4_K_M-engram.gguf
   ```
   `--engram` takes a checkpoint folder, an assembled folder or a `micro_engram_config.json`; every engram it lists is added (all must share one token-merge map). Split GGUFs must be merged first.
3. **Run it as usual:** `llama-server -m gemma-4-31B-it-Q4_K_M-engram.gguf`. The load log shows `micro_engram: N engram(s)`.

How it works: on every batch, each token's previous two tokens come from llama.cpp's KV cache, which already stores token ids per sequence (added upstream for DeepSeek-V4.1's Engram). That covers chunked prompts, token-by-token generation, parallel slots and prompt-cache reuse. The patch also keeps that history in saved/restored slot states. The hash runs on the CPU (same formula as training), and the tables are looked up on the GPU.

Notes:
- A patched llama.cpp still loads every other GGUF normally. An **unpatched** llama.cpp refuses a GGUF that contains an engram (unknown tensors).
- Use the same tokenizer as training: the token-merge map is per token id.
- Image tokens (multimodal input) get no engram, and text right after an image sees no earlier tokens through the engram (same as vLLM).

Tested on CPU against the PyTorch engram (`llama_cpp/test/compare.py`, a tiny random Gemma 4 with two engrams per model, all three gates, f32 and bf16 storage):
- the logits match within the drift that llama.cpp shows even without an engram (~2e-2 on a tiny model)
- the same check also passes for a chunked prompt, token-by-token decoding, two sequences in one batch, and a state save/restore
- a negative control (hashing without token merging) misses by ~100x more, so a hashing bug cannot pass
- without the patch's cache fix, decoding after a state restore misses by 1.5, so the history check catches that too

Not yet tested on a GPU or with the real 31B model.

## Watching training

With `micro_engram_stats_steps` (default every 10 steps) the training log line gets `engram/...` entries (`engram_L<layer>/...` with several engrams), which also reach wandb:

| Entry | Meaning | What to look for |
|---|---|---|
| `engram/out_abs_mean` | mean \|weight\| of the output layer (`value`, or `up` for `ple`) | starts at 0 and should grow; 0 means it is not training |
| `engram/table_rms_2g`, `_3g` | RMS of the 2-gram / 3-gram tables (start at 0.02) | slowly rising |
| `engram/rows_learned_2g`, `_3g` | share of rows whose RMS grew past 1.5× the start | how much of each table is in use; 3-grams fill more slowly |
| `engram/gate_mean` | mean gate value over the tokens since the last log (scalar gates) | not stuck at 0 or 1 |
| `engram/gate_saturated` | share of tokens with gate > 0.99 | high values mean an on/off gate that learns slowly |
| `engram/q_scale_mean`, `k_scale_mean` | learnable norm scales (`scalar_sqrt`) | start at 1; drifting shows the gate tuning its sharpness |

The table numbers come from rank 0's share of the rows, a random sample, so no extra communication is needed. `rows_learned` uses a rule-of-thumb threshold (1.5x the starting RMS), so it stays near 0 early in training even while rows are moving.

For a run that started without this option (or to look at past checkpoints), `engram_stats.py` reads the same weight stats from the checkpoints on disk without stopping training.

---

## Tests (CPU only)

Run them from the repo root:

```bash
python test_train.py                          # micro_engram.train (2 procs): data, config, loss vs HF, no-grad prefix, resume, sparse grads, dense LR, hybrid attention, meta loading
python test_gate.py                           # all gate modes: no-op start, gradients, bf16, checkpoint names; weight stats
python test_canon.py                          # token-merge hashing and state-dict compatibility
python test_optim.py                          # lazy_adam / rowwise_adam: vs AdamW, unused rows untouched, bf16 second-moment freeze, FSDP2 + state save/load
python test_rows.py                           # micro_engram_sharding: rows, 3 processes: == full tables (forward, gradients, training, clipping), save/resume/reshard
python test_token_diff.py                     # engram_token_diff.py against a fake vLLM server
python test_render_chat.py                    # render_chat.py: formats, template override, fixed sampling
python test_vllm_ids.py                       # vLLM path: previous tokens per batch type, same output as training
python test_vllm_tp.py                        # vLLM tables split over TP GPUs: == full tables, loading (file and checkpoint), with vLLM stand-ins
python test_eval_spans.py                     # eval_logprob.py assistant-only scoring (render_chat spans) vs a fake vLLM server
python test_head.py                           # vocab-parallel embedding + chunked softcapped loss vs a full-logit reference (1 and 3 procs)
python test_zero_order.py                     # zero_ngram_order.py: zeroed orders, rest unchanged, output == engram without those lookups
python test_multi.py                          # several engrams, init_from + frozen, config, inference == training, assemble, stats (needs transformers)
python llama_cpp/test/compare.py --llama-cpp <patched llama.cpp> --driver <engram_logits>   # llama.cpp patch vs PyTorch (needs a build; see the file)
```

---

## Notes / limitations

- It is tested on Gemma 4 only. Other decoders should work as long as `find_text_layers` finds the text decoder `ModuleList` (it prefers a `*.language_model.*layers` path and skips vision/audio towers).
- N-gram orders: 2 and 3 by default; 4-grams via `micro_engram_table_sizes`.
- Hash multipliers are rebuilt from a fixed seed, so they are identical in every run and are not loaded from checkpoints.
