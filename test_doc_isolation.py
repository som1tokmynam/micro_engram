"""CPU test: micro_engram.train keeps packed documents isolated. Run: python test_doc_isolation.py

For each trainer configuration (gemma4_hybrid_attn_impl on/off, sliding layers on sdpa/eager), the
trainer's own setup must give each document of a packed row the same hidden states as that
document run alone, and the engram's n-gram ids of a document must not include the previous
document's tokens. The setup under test: apply_kernel_patches (with hybrid, the global layers get
no model mask), hybrid_attention (every global layer on sdpa_global_packed), an engram attached
through get_ids, the text model on inputs_embeds with attention_mask=None, and pads continuing the
positions as collate() does.

A control (mask dropped while the global layers are plain SDPA, the situation of Axolotl before
v0.18.0) must show the leak, so the check is known to catch one. Each case runs in its own
process, since the patches are global. The FA2 sliding path needs a GPU and isn't covered here.
"""
import os
import subprocess
import sys
import types

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def worker(mode, impl):
    import torch
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig

    from micro_engram.module import MicroEngram, attach_engram
    from micro_engram.train import model as M

    cfg = types.SimpleNamespace(fused_attn_kernel=False, gemma4_hybrid_attn_impl=(mode != "plain"),
                                large_head_policy="auto", liger_rms_norm=False, liger_glu_activation=False,
                                gradient_checkpointing=False, activation_offloading=False, attn_implementation=impl)
    M.apply_kernel_patches(cfg, on_cuda=False)
    V = 64
    conf = Gemma4TextConfig(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=16, global_head_dim=32,
                            layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
                            sliding_window=6, hidden_size_per_layer_input=0, vocab_size_per_layer_input=V,
                            max_position_embeddings=256, final_logit_softcapping=30.0, pad_token_id=0)
    conf._attn_implementation = impl
    torch.manual_seed(0)
    model = Gemma4ForCausalLM(conf).float().eval()
    text = model.model
    if mode == "hybrid":
        M.hybrid_attention(text, cfg)  # as Trainer.build does right after loading
    ctx = {}
    eng = MicroEngram(64, table_size=97, engram_dim=16, num_heads=2, orders=(2, 3), vocab_size=V, seed=1)
    attach_engram(model, eng, 1, dtype=torch.float32, get_ids=lambda: (ctx["ids"], ctx["pos"]))
    with torch.no_grad():
        for p in eng.parameters():
            if p.is_floating_point():
                p.copy_(torch.randn_like(p) * 0.5)  # make the engram (gate included) contribute

    def run(ids, pos):
        ctx["ids"], ctx["pos"] = ids, pos
        with torch.no_grad():
            return text(inputs_embeds=text.embed_tokens(ids), position_ids=pos, attention_mask=None,
                        use_cache=False).last_hidden_state

    g = torch.Generator().manual_seed(2)
    la, lb, npad = 7, 9, 8
    a = torch.randint(3, V, (la,), generator=g)
    b = torch.randint(3, V, (lb,), generator=g)
    ids = torch.cat([a, b, torch.zeros(npad, dtype=torch.long)])[None]
    pos = torch.cat([torch.arange(la), torch.arange(lb), torch.arange(lb, lb + npad)])[None]
    packed = run(ids, pos)
    da = float((packed[0, :la] - run(a[None], torch.arange(la)[None])[0]).abs().max())
    db = float((packed[0, la:la + lb] - run(b[None], torch.arange(lb)[None])[0]).abs().max())
    eng_ok = all(torch.equal(x[0, la:la + lb], y[0]) for x, y in
                 zip(eng.hash_ids(ids, pos), eng.hash_ids(b[None], torch.arange(lb)[None])))
    impls = sorted({getattr(layer.self_attn.config, "_attn_implementation", "?") for layer in text.layers})
    ok = max(da, db) < 1e-4 and eng_ok
    print(f"{mode} {impl} layers={impls} doc1 {da:.2e} doc2 {db:.2e} engram_isolated={eng_ok} -> "
          f"{'ISOLATED' if ok else 'LEAK'}")


def main():
    fail = []
    for impl in ("sdpa", "eager"):
        for mode in ("hybrid", "plain", "leak_control"):
            p = subprocess.run([sys.executable, os.path.abspath(__file__), mode, impl], capture_output=True,
                               text=True, timeout=600)
            line = next((ln for ln in p.stdout.splitlines() if "->" in ln), p.stderr[-400:]).strip()
            isolated = line.endswith("ISOLATED")
            ok = (not isolated and line.endswith("LEAK")) if mode == "leak_control" else isolated
            name = f"{mode} / {impl}: " + ("leak detected (control)" if mode == "leak_control"
                                           else "packed documents isolated (attention + engram)")
            print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  ({line})"), flush=True)
            if not ok:
                fail.append(name)
    print(f"\n{'ALL PASSED' if not fail else f'{len(fail)} FAILED: {fail}'}")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        worker(sys.argv[1], sys.argv[2])
    else:
        main()
