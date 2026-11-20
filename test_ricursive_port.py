"""CPU tests for micro_engram.axolotl_port.ricursive (FlashAttention-2 varlen packing for the
Ricursive remote-code model). Run: RICURSIVE_MODEL_DIR=/path/with/modeling_ricursive.py python test_ricursive_port.py

RICURSIVE_MODEL_DIR is a folder holding the checkpoint's modeling_ricursive.py and
configuration_ricursive.py (e.g. the HF snapshot); the test is skipped without it. On CPU the
patch runs its exact per-document reference instead of flash-attn; the FA2 call itself needs a GPU.
  1. control: the original model lets a packed document see the previous one
  2. packed rows (continued or Axolotl-style padding, segment-id attention_mask, 2 rows) give
     each document the hidden states it gets alone, sliding window binding
  3. loss (sum / num_items_in_batch) and gradients == the documents trained alone
  4. gradient checkpointing passes the boundaries through
  5. generation / cached forwards / left padding / no position_ids keep the original path
"""
import importlib
import os
import sys
import types

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from micro_engram.axolotl_port import ricursive as R  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail and not ok else ""), flush=True)
    if not ok:
        FAIL.append(name)


def load_remote(folder):
    pkg = types.ModuleType("ricursive_remote")
    pkg.__path__ = [folder]
    sys.modules["ricursive_remote"] = pkg
    return (importlib.import_module("ricursive_remote.configuration_ricursive"),
            importlib.import_module("ricursive_remote.modeling_ricursive"))


def tiny(cm, mm, W=4):
    cfg = cm.RicursiveConfig(vocab_size=50, hidden_size=64, intermediate_size=96, num_hidden_layers=5,
                             num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                             layer_types=["sliding_attention"] * 4 + ["full_attention"], sliding_window=W,
                             rope_theta=10000.0, rms_norm_eps=1e-6, use_gated_attention=True,
                             gate_full_precision=True, use_valres=True, ffn_clamp_threshold=20.0, embed_scale=8.0,
                             tie_word_embeddings=False, use_cache=False)
    torch.manual_seed(0)
    m = mm.RicursiveForCausalLM(cfg).float()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "value_res_lambda" in n:
                p.fill_(0.3)
            elif p.dim() == 1:
                p.copy_(1 + 0.2 * torch.randn_like(p))
            else:
                p.normal_(0, 0.08)
    return m


def hidden(m, ids, pos=None, mask=None, **kw):
    with torch.no_grad():
        return m.model(input_ids=ids, position_ids=pos, attention_mask=mask, use_cache=False, **kw).last_hidden_state


def test_fa2_kernel():
    """GPU only: FlashAttention-2 varlen == the per-document reference (bf16, fwd + bwd)."""
    if not torch.cuda.is_available() or R._flash_varlen() is None:
        print("SKIP FA2 kernel check (needs CUDA + flash-attn)")
        return
    torch.manual_seed(0)
    lens = [700, 1500, 33, 1200]  # one longer than the 1024 window
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device="cuda")
    n = int(cu[-1])
    for window in (1024, None):
        q = torch.randn(n, 8, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(n, 2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(n, 2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        go = torch.randn(n, 8, 128, device="cuda", dtype=torch.bfloat16)
        out = R.varlen_attention(q, k, v, cu, max(lens), window)
        grads = torch.autograd.grad(out, (q, k, v), go)
        qf, kf, vf = (t.detach().float().requires_grad_() for t in (q, k, v))
        ref = R._reference_varlen(qf, kf, vf, cu, window)
        rgrads = torch.autograd.grad(ref, (qf, kf, vf), go.float())
        rel = [float((a.float() - b).norm() / b.norm()) for a, b in zip((out,) + grads, (ref,) + rgrads)]
        check(f"FA2 varlen == reference (window {window}): out/dq/dk/dv rel err {['%.1e' % r for r in rel]}",
              max(rel) < 2e-2, rel)


def main():
    test_fa2_kernel()
    folder = os.environ.get("RICURSIVE_MODEL_DIR")
    if not folder or not os.path.exists(os.path.join(folder, "modeling_ricursive.py")):
        print("SKIP: set RICURSIVE_MODEL_DIR to a folder with modeling_ricursive.py + configuration_ricursive.py")
        return
    os.environ["RIPPLE_MLP_CHUNK"] = "0"
    cm, mm = load_remote(folder)
    m = tiny(cm, mm).eval()
    g = torch.Generator().manual_seed(1)
    la, lb, npad = 7, 9, 3  # window 4: binds inside both documents
    a = torch.randint(3, 50, (la,), generator=g)
    b = torch.randint(3, 50, (lb,), generator=g)
    ids = torch.cat([a, b, torch.zeros(npad, dtype=torch.long)])[None]
    pos_cont = torch.cat([torch.arange(la), torch.arange(lb), torch.arange(lb, lb + npad)])[None]
    pos_ax = torch.cat([torch.arange(la), torch.arange(lb), torch.arange(npad)])[None]
    seg_mask = torch.cat([torch.full((la,), 1), torch.full((lb,), 2), torch.zeros(npad, dtype=torch.long)])[None]
    alone_a = hidden(m, a[None])
    alone_b = hidden(m, b[None])

    def diff(h, off=0):
        return max(float((h[0, off:off + la] - alone_a[0]).abs().max()),
                   float((h[0, off + la:off + la + lb] - alone_b[0]).abs().max()))

    # untouched model, for the fallbacks later
    ref_plain = hidden(m, ids)
    left = torch.cat([torch.zeros(1, 3, dtype=torch.long), a[None]], 1)
    left_mask = torch.cat([torch.zeros(1, 3, dtype=torch.long), torch.ones(1, la, dtype=torch.long)], 1)
    ref_left = hidden(m, left, mask=left_mask)
    check("control: the original model leaks across packed documents", diff(hidden(m, ids, pos_cont)) > 1e-2,
          diff(hidden(m, ids, pos_cont)))

    n = R.patch_ricursive_varlen(m)
    check("patch: all 5 attention modules", n == 5, n)
    R.patch_ricursive_varlen(m)
    check("patch: idempotent", not getattr(type(m.model).forward._me_original, "_me_varlen", False))

    check("packed row, continued padding: each document == alone", diff(hidden(m, ids, pos_cont)) < 1e-5,
          diff(hidden(m, ids, pos_cont)))
    check("packed row, Axolotl padding + segment-id mask: each document == alone",
          diff(hidden(m, ids, pos_ax, seg_mask)) < 1e-5, diff(hidden(m, ids, pos_ax, seg_mask)))
    two = hidden(m, torch.cat([ids, ids.flip(1)]), torch.cat([pos_cont, pos_cont]))
    check("two packed rows: row 0 unaffected by row 1", diff(two[0:1]) < 1e-5, diff(two[0:1]))

    # 3. loss and gradients
    labels = ids.clone()
    labels[0, la + lb:] = -100
    labels[0, la] = -100  # next document's first token is no target

    def doc_loss_sum(model, tok):
        out = model(input_ids=tok[None], labels=tok[None], num_items_in_batch=1, use_cache=False)
        return out.loss

    m.train()
    m.zero_grad()
    count = (la - 1) + (lb - 1)
    loss_p = m(input_ids=ids, position_ids=pos_cont, labels=labels, num_items_in_batch=count, use_cache=False).loss
    loss_p.backward()
    g_p = {k: p.grad.clone() for k, p in m.named_parameters() if p.grad is not None}
    m.zero_grad()
    # reference: each document alone (original path: no position_ids -> unpatched forward), summed
    loss_r = (doc_loss_sum(m, a) + doc_loss_sum(m, b)) / count
    loss_r.backward()
    g_r = {k: p.grad.clone() for k, p in m.named_parameters() if p.grad is not None}
    gd = max(float((g_p[k] - g_r[k]).abs().max() / (g_r[k].abs().max() + 1e-12)) for k in g_r)
    check("loss (sum / num_items_in_batch) == documents alone", abs(float(loss_p) - float(loss_r)) < 1e-5,
          f"{float(loss_p)} vs {float(loss_r)}")
    check("gradients == documents alone (every parameter)", set(g_p) == set(g_r) and gd < 1e-4, gd)

    # 4. gradient checkpointing
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.zero_grad()
    loss_c = m(input_ids=ids, position_ids=pos_cont, labels=labels, num_items_in_batch=count, use_cache=False).loss
    loss_c.backward()
    gc = max(float((p.grad - g_p[k]).abs().max() / (g_p[k].abs().max() + 1e-12))
             for k, p in m.named_parameters() if p.grad is not None)
    check("gradient checkpointing: same loss and gradients", abs(float(loss_c) - float(loss_p)) < 1e-6 and gc < 1e-5,
          f"{float(loss_c)} vs {float(loss_p)}, {gc}")
    m.gradient_checkpointing_disable()
    m.eval()

    # 5. original path kept
    check("no position_ids: original path, unchanged output", float((hidden(m, ids) - ref_plain).abs().max()) == 0.0)
    check("left padding: original path, unchanged output",
          float((hidden(m, left, mask=left_mask) - ref_left).abs().max()) == 0.0)
    check("boundaries: left padding / rows not starting at 0 -> original path",
          R.varlen_from_position_ids(pos_cont, left_mask) is None and R.varlen_from_position_ids(pos_cont + 1) is None)
    with torch.no_grad():
        out = m(input_ids=a[None, :4], use_cache=True)
        cache = out.past_key_values
        nxt = m(input_ids=a[None, 4:5], past_key_values=cache, use_cache=True,
                position_ids=torch.tensor([[4]]), cache_position=torch.tensor([4]))
        full = m(input_ids=a[None, :5], use_cache=False)
    check("cached decoding: original path, matches a full forward",
          float((nxt.logits[0, -1] - full.logits[0, -1]).abs().max()) < 1e-4)

    print(f"\n{'ALL PASSED' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}'}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
