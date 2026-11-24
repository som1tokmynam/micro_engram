"""CPU tests for micro_engram.axolotl_port (the Axolotl plugin's patches). Run: python test_axolotl_port.py

Axolotl itself isn't needed: its collator file (numpy + transformers only) is loaded by path from
AXOLOTL_SRC (default /root/micro_engram_work/ax/src; that check is skipped if absent), and its
Gemma 4 hybrid-attention modules are micro_engram's vendored copies (same code, other imports).
  1. collator: right pads continue position_ids, a packed document's first token is no target,
     inputs / other labels unchanged
  2. global layers: with the model mask dropped and the per-call probe, packed rows and padded
     one-document rows give the reference loss (plain SDPA + transformers' packed mask); the full
     mask is None, one-document rows build no block mask, packed rows do; sliding / generation /
     left-padded calls pass through
  3. loss normalization: a real HF Trainer step with gradient accumulation gives the per-token
     gradient over the whole step once the port re-enables loss kwargs (and not with Axolotl's
     override)
  4. profiler callback writes the table and the timeline
  5. fail fast: an uncaught error in a multi-process run prints and exits 1
"""
import copy
import importlib.util
import os
import subprocess
import sys
import tempfile
import types

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from micro_engram.axolotl_port import patches  # noqa: E402

AXOLOTL_SRC = os.environ.get("AXOLOTL_SRC", "/root/micro_engram_work/ax/src")
FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail and not ok else ""), flush=True)
    if not ok:
        FAIL.append(name)


def tokenizer(V=64):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {f"t{i}": i for i in range(V)}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="t0"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="t0", eos_token="t1")


def tiny_gemma4(V=64):
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig

    cfg = Gemma4TextConfig(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                           num_attention_heads=4, num_key_value_heads=2, head_dim=16, global_head_dim=32,
                           layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
                           sliding_window=6, hidden_size_per_layer_input=0, vocab_size_per_layer_input=V,
                           max_position_embeddings=256, final_logit_softcapping=30.0, initializer_range=0.08,
                           pad_token_id=0, eos_token_id=1, bos_token_id=2)
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    m = Gemma4ForCausalLM(cfg).float().eval()
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() == 1:
                p.copy_(1 + 0.2 * torch.randn_like(p))
    return m


# ----------------------------------------------------------------------------- 1. collator
def test_collator():
    path = os.path.join(AXOLOTL_SRC, "axolotl/utils/collators/batching.py")
    if not os.path.exists(path):
        print(f"SKIP collator (no {path})")
        return
    spec = importlib.util.spec_from_file_location("ax_batching", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tok = tokenizer()

    def bins():  # one packed row (docs of 3 and 4) and one one-document row (5), as multipack hands them
        def doc(start, n):
            ids = list(range(start, start + n))
            return {"input_ids": ids, "attention_mask": [1] * n, "labels": [-100] + ids[1:],
                    "position_ids": list(range(n))}
        return [[doc(10, 3), doc(20, 4)], [doc(30, 5)]]

    kw = dict(pad_to_multiple_of=8, return_tensors="pt", padding=True)
    ref = mod.V2BatchSamplerDataCollatorForSeq2Seq(tok, **kw)(bins())
    patches.patch_collator(mod.DataCollatorForSeq2Seq)
    check("collator patch is idempotent", not patches.patch_collator(mod.DataCollatorForSeq2Seq))
    out = mod.V2BatchSamplerDataCollatorForSeq2Seq(tok, **kw)(bins())
    check("collator: inputs unchanged", torch.equal(out["input_ids"], ref["input_ids"]))
    check("collator: stock pads restart positions at 0 (what the port fixes)",
          ref["position_ids"][1].tolist() == [0, 1, 2, 3, 4, 0, 1, 2], ref["position_ids"][1].tolist())
    check("collator: pads continue the last document",
          out["position_ids"].tolist() == [[0, 1, 2, 0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5, 6, 7]],
          out["position_ids"].tolist())
    exp = ref["labels"].clone()
    exp[0, 3] = -100  # second document's first token
    check("collator: a packed document's first token is no target, other labels unchanged",
          torch.equal(out["labels"], exp), f"{out['labels'].tolist()} vs {exp.tolist()}")


# ----------------------------------------------------------------------------- 2. global layers
def test_global_layers():
    from transformers.models.gemma4 import modeling_gemma4 as mg

    from micro_engram.train.vendor import gemma4_hybrid_mask as hm
    from micro_engram.train.vendor import large_head as lh

    m = tiny_gemma4()
    g = torch.Generator().manual_seed(3)
    S = 24
    ids = torch.randint(3, 64, (2, S), generator=g)
    # row 0: documents of 7 and 9 tokens + 8 pads; row 1: one document of 13 tokens + 11 pads
    pos_stock = torch.tensor([list(range(7)) + list(range(9)) + list(range(8)),
                              list(range(13)) + list(range(11))])
    pos_port = torch.tensor([list(range(7)) + list(range(9)) + list(range(9, 17)),
                             list(range(13)) + list(range(13, 24))])
    labels = ids.clone()
    labels[0, 16:] = -100
    labels[1, 13:] = -100
    labels[0, 7] = -100  # cross-document target (masked in both runs)

    def loss(pos):
        with torch.no_grad():
            return [float(m(input_ids=ids[r:r + 1], position_ids=pos[r:r + 1], labels=labels[r:r + 1],
                            attention_mask=None, use_cache=False).loss) for r in range(2)]

    ref = loss(pos_stock)  # plain SDPA everywhere; transformers builds the packed mask from position_ids

    check("impl count: plain-SDPA global layers (Axolotl < v0.18.0) are not packing-aware -> item 2 skipped",
          patches.global_attention_impls(m) == (0, 4), patches.global_attention_impls(m))
    check("dynamic probe: no setter (Axolotl < v0.18.0) -> skipped", not patches.dynamic_packing_probe(types.ModuleType("x")))
    # Axolotl's hybrid setup (gemma4_hybrid_attn_impl), then the port on top
    hm.patch_gemma4_hybrid_mask()
    for layer, t in zip(m.model.layers, m.config.layer_types):
        if t == "full_attention":
            c = copy.copy(layer.self_attn.config)
            c._attn_implementation = hm.GLOBAL_PACKED_SDPA
            layer.self_attn.config = c
    check("impl count: the 2 global layers on sdpa_global_packed", patches.global_attention_impls(m) == (2, 2),
          patches.global_attention_impls(m))
    lh.set_large_head_packed(True)  # what Axolotl declares under sample_packing
    patches.dynamic_packing_probe(lh)
    check("dynamic probe: packing no longer declared", lh._PACKED is None)
    check("unmask: patched", patches.unmask_global_layers(mg))
    check("unmask: idempotent", patches.unmask_global_layers(mg) and
          not getattr(mg.create_causal_mask._me_original, "_me_unmasked", False))

    seen = {"full_masks": [], "block": 0}
    wrapped = mg.create_causal_mask

    def spy_mask(*a, **k):
        out = wrapped(*a, **k)
        if k.get("and_mask_function") is None:
            seen["full_masks"].append(out)
        return out

    real_block = hm._packing_block_causal_mask

    def spy_block(*a, **k):
        seen["block"] += 1
        return real_block(*a, **k)

    mg.create_causal_mask, hm._packing_block_causal_mask = spy_mask, spy_block
    try:
        seen["block"] = 0
        port0 = loss(pos_port)
        blocks = seen["block"]
        with torch.no_grad():
            seen["block"] = 0
            m(input_ids=ids[1:], position_ids=pos_port[1:], attention_mask=None, use_cache=False)
            single_blocks = seen["block"]
            seen["block"] = 0
            m(input_ids=ids[1:], position_ids=pos_stock[1:], attention_mask=None, use_cache=False)
            stock_single_blocks = seen["block"]
    finally:
        mg.create_causal_mask, hm._packing_block_causal_mask = wrapped, real_block
    n_global = m.config.layer_types.count("full_attention")
    check("global layers: loss on a packed row == reference", abs(port0[0] - ref[0]) < 1e-4, f"{port0} vs {ref}")
    check("global layers: loss on a padded one-document row == reference", abs(port0[1] - ref[1]) < 1e-4,
          f"{port0} vs {ref}")
    check("global layers: the model builds no full-attention mask",
          seen["full_masks"] and all(x is None for x in seen["full_masks"]))
    check("global layers: only the packed row builds a block mask (once per global layer)",
          blocks == n_global, f"{blocks} vs {n_global}")
    check("global layers: a one-document row with continued padding takes the causal path", single_blocks == 0,
          single_blocks)
    check("global layers: with stock padding the same row would build a block mask",
          stock_single_blocks == n_global, stock_single_blocks)

    # pass-through rules, against a stand-in for the wrapped function
    calls = []
    fake = types.SimpleNamespace(create_causal_mask=lambda *a, **k: calls.append(k) or "MASK")
    patches.unmask_global_layers(fake)
    f = fake.create_causal_mask
    check("unmask: sliding mask (and_mask_function) passes through", f(and_mask_function=len) == "MASK")
    check("unmask: or_mask_function passes through", f(or_mask_function=len) == "MASK")
    check("unmask: generation (KV cache) passes through", f(past_key_values=object()) == "MASK")
    check("unmask: left padding passes through", f(attention_mask=torch.tensor([[0, 1, 1], [1, 1, 1]])) == "MASK")
    check("unmask: right padding -> None", f(attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]])) is None)
    check("unmask: no mask -> None", f(attention_mask=None) is None)


# ----------------------------------------------------------------------------- 3. loss normalization
def test_loss_normalization():
    from transformers import Gemma4ForCausalLM, Trainer, TrainerCallback, TrainingArguments

    class Model(Gemma4ForCausalLM):
        accepts_loss_kwargs = True  # what Axolotl's gemma4_loss_kwargs patch declares

    base = tiny_gemma4()
    g = torch.Generator().manual_seed(5)
    S, trained = 16, [1, 2, 14, 15]  # very uneven trained tokens per micro-batch (assistant-only SFT)
    rows = []
    for n in trained:
        ids = torch.randint(3, 64, (S,), generator=g)
        lab = torch.full((S,), -100)
        lab[S - n:] = ids[S - n:]
        rows.append({"input_ids": ids, "labels": lab})

    def ref_grad():
        m = Model(base.config)
        m.load_state_dict(base.state_dict())
        tot, cnt = 0.0, 0
        for r in rows:
            logits = m(input_ids=r["input_ids"][None]).logits[0, :-1].float()
            t = r["labels"][1:]
            tot = tot + torch.nn.functional.cross_entropy(logits, t, ignore_index=-100, reduction="sum")
            cnt += int((t != -100).sum())
        (tot / cnt).backward()
        return m.lm_head.weight.grad.clone()

    def trainer_grad(port):
        m = Model(base.config)
        m.load_state_dict(base.state_dict())
        m.train()
        got = {}

        class Grab(TrainerCallback):
            def on_pre_optimizer_step(self, args, state, control, model=None, **kw):
                got["g"] = model.lm_head.weight.grad.clone()

        with tempfile.TemporaryDirectory() as d:
            args = TrainingArguments(output_dir=d, per_device_train_batch_size=1, gradient_accumulation_steps=4,
                                     max_steps=1, learning_rate=0.0, max_grad_norm=0.0, report_to=[], use_cpu=True,
                                     remove_unused_columns=False, logging_strategy="no", save_strategy="no",
                                     dataloader_num_workers=0)
            t = Trainer(model=m, args=args, train_dataset=rows, callbacks=[Grab()])
            t.model_accepts_loss_kwargs = False  # AxolotlTrainer.__init__'s override for Gemma 4
            info = patches.token_normalized_loss(t) if port else None
            t.train()
        return got["g"], info

    ref = ref_grad()
    g_ax, _ = trainer_grad(False)
    g_port, info = trainer_grad(True)

    def rel(a):
        return float((a - ref).norm() / ref.norm())

    check("loss kwargs: the port turns them back on", info["before"] is False and info["after"] is True, info)
    check("loss kwargs: step gradient == per-token mean over the whole step", rel(g_port) < 1e-4, rel(g_port))
    check("loss kwargs: Axolotl's override gives a different (per-micro-batch mean) gradient", rel(g_ax) > 0.05,
          rel(g_ax))

    def fake_trainer(model):
        return types.SimpleNamespace(model=model, accelerator=types.SimpleNamespace(unwrap_model=lambda x: x),
                                     model_accepts_loss_kwargs=False, compute_loss_func=None,
                                     args=types.SimpleNamespace(average_tokens_across_devices=True))

    class Gemma4PreV018(Gemma4ForCausalLM):
        accepts_loss_kwargs = False  # PaliGemma's flag, before Axolotl v0.18.0's gemma4_loss_kwargs patch

    check("loss kwargs: Gemma 4 without Axolotl's declaration (< v0.18.0) is turned on from its loss signature",
          patches.token_normalized_loss(fake_trainer(Gemma4PreV018(base.config)))["after"])

    class Other(torch.nn.Module):
        accepts_loss_kwargs = False
        config = types.SimpleNamespace(model_type="paligemma")

        def loss_function(self, logits, labels, num_items_in_batch=None):
            pass

    check("loss kwargs: left off for a non-Gemma 4 model that declares False",
          not patches.token_normalized_loss(fake_trainer(Other()))["after"])


# ----------------------------------------------------------------------------- 4. profiler
def test_profiler():
    from transformers import TrainerCallback

    with tempfile.TemporaryDirectory() as d:
        cb = patches.profiler_callback(3, d, TrainerCallback)
        for step in range(5):
            st = types.SimpleNamespace(global_step=step)
            cb.on_step_begin(None, st, None)
            torch.randn(64, 64) @ torch.randn(64, 64)
            cb.on_step_end(None, st, None)
        files = sorted(os.listdir(os.path.join(d, "profile")))
        check("profiler: writes step 3 only", files == ["rank0_step3.json", "rank0_step3.txt"], files)
        txt = open(os.path.join(d, "profile", "rank0_step3.txt")).read()
        check("profiler: table lists the matmul", "aten::mm" in txt)


# ----------------------------------------------------------------------------- 5. fail fast
def test_fail_fast():
    code = ("import sys; sys.path.insert(0, %r); from micro_engram.axolotl_port import patches; "
            "assert patches.install_fail_fast(); raise RuntimeError('boom')") % ROOT
    env = {**os.environ, "WORLD_SIZE": "2", "RANK": "1"}
    p = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    check("fail fast: exit code 1 with the traceback and rank", p.returncode == 1 and "rank 1 FAILED" in p.stderr
          and "RuntimeError: boom" in p.stderr, (p.returncode, p.stderr[-500:]))
    check("fail fast: off in a single-process run", not patches.install_fail_fast())


if __name__ == "__main__":
    test_collator()
    test_global_layers()
    test_loss_normalization()
    test_profiler()
    test_fail_fast()
    print(f"\n{'ALL PASSED' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}'}")
    sys.exit(1 if FAIL else 0)
