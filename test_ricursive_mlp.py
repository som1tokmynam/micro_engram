"""Tests for the fused Ricursive MLP activation (micro_engram.axolotl_port.ricursive_kernels).
Run: python test_ricursive_mlp.py   (optionally RICURSIVE_MODEL_DIR=... for the model-level check)

Without CUDA the Triton kernels run in Triton's interpreter (TRITON_INTERPRET=1, set below).
  1. fused == the model's ops (clamp, clamp, silu, mul): forward and both gradients, fp32,
     clamps binding (values beyond and exactly at +-T), and without a clamp threshold
  2. bf16: the fused result is at least as close to the fp32 truth as torch's own bf16 ops
  3. (RICURSIVE_MODEL_DIR) a tiny Ricursive model: loss and every gradient unchanged by the patch,
     with gradient checkpointing
  4. (CUDA) real size [4096, 40960] bf16: accuracy vs fp32 truth, and fwd+bwd time fused vs torch
"""
import os
import sys

import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")  # must be set before triton is imported

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from micro_engram.axolotl_port import ricursive as R  # noqa: E402
from micro_engram.axolotl_port import ricursive_kernels as K  # noqa: E402

FAIL = []
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail and not ok else ""), flush=True)
    if not ok:
        FAIL.append(name)


def run(fn, g, u, T, dh):
    g = g.detach().clone().requires_grad_()
    u = u.detach().clone().requires_grad_()
    h = fn(g, u, T)
    h.backward(dh)
    return h.detach(), g.grad, u.grad


def rel(a, b):
    return float((a.float() - b.float()).norm() / (b.float().norm() + 1e-12))


def test_function():
    check("triton kernels available", K._HAVE_TRITON and K.fused_available(torch.zeros(4, device=DEV)))
    torch.manual_seed(0)
    n, d, T = 37, 300, 2.0  # odd sizes: the last block is partial
    g = torch.randn(n, d, device=DEV) * 3
    u = torch.randn(n, d, device=DEV) * 3
    g[0, :5] = T  # exactly at the threshold: torch's clamp backward gives no gradient there
    u[0, 5:10] = T
    u[0, 10:15] = -T
    dh = torch.randn(n, d, device=DEV)
    hf, gf, uf = run(K.clamped_silu_mul, g, u, T, dh)
    hr, gr, ur = run(K.reference, g, u, T, dh)
    binding = float(((g > T) | (u.abs() > T)).float().mean())
    check(f"fp32, clamps binding on {binding:.0%} of entries: forward == model ops", rel(hf, hr) < 1e-6, rel(hf, hr))
    check("fp32: dgate == model ops (incl. clamp masks)", rel(gf, gr) < 1e-6, rel(gf, gr))
    check("fp32: dup == model ops (incl. clamp masks)", rel(uf, ur) < 1e-6, rel(uf, ur))
    for thr in (0.0, None):
        hf, gf, uf = run(K.clamped_silu_mul, g, u, thr, dh)
        hr, gr, ur = run(K.reference, g, u, thr, dh)
        check(f"fp32, no clamp (threshold {thr!r}): forward + grads == model ops",
              max(rel(hf, hr), rel(gf, gr), rel(uf, ur)) < 1e-6)

    # bf16: compare both against the fp32 truth. Triton's CPU interpreter TRUNCATES on the fp32 ->
    # bf16 store (GPUs round to nearest), which alone doubles the rounding error, so the bound is
    # looser there; the GPU check below holds the fused kernel to x1.5 of torch's own bf16 error.
    slack = 3.0 if os.environ.get("TRITON_INTERPRET") == "1" else 1.5
    gb, ub, dhb = g.bfloat16(), u.bfloat16(), dh.bfloat16()
    truth = run(K.reference, gb.float(), ub.float(), T, dhb.float())
    fused = run(K.clamped_silu_mul, gb, ub, T, dhb)
    torch_bf16 = run(K.reference, gb, ub, T, dhb)
    for i, name in enumerate(("forward", "dgate", "dup")):
        ef, et = rel(fused[i], truth[i]), rel(torch_bf16[i], truth[i])
        check(f"bf16 {name}: fused error {ef:.1e} <= torch bf16 error {et:.1e} (x{slack:g})",
              ef <= slack * et + 1e-6, (ef, et))


def run_norm(fn, x, w, dy):
    x = x.detach().clone().requires_grad_()
    y = fn(x, w, 1e-6)
    y.backward(dy)
    return y.detach(), x.grad


def test_rmsnorm():
    torch.manual_seed(2)
    for shape in ((7, 5, 300), (3, 4, 8, 128)):  # a layer norm (non-power-of-2 D) and a per-head norm
        D = shape[-1]
        x = torch.randn(*shape, device=DEV) * 3 + 0.5
        w = (1 + 0.3 * torch.randn(D, device=DEV))
        dy = torch.randn(*shape, device=DEV)
        yf, gf = run_norm(K.rms_norm, x, w, dy)
        yr, gr = run_norm(K.rms_norm_reference, x, w, dy)
        check(f"RMSNorm fp32 {shape}: forward == model", rel(yf, yr) < 1e-6, rel(yf, yr))
        check(f"RMSNorm fp32 {shape}: input gradient == model", rel(gf, gr) < 1e-5, rel(gf, gr))
        slack = 3.0 if os.environ.get("TRITON_INTERPRET") == "1" else 1.5
        xb, wb, dyb = x.bfloat16(), w.bfloat16(), dy.bfloat16()
        truth = run_norm(K.rms_norm_reference, xb.double(), wb.double(), dyb.double())
        fused = run_norm(K.rms_norm, xb, wb, dyb)
        ref = run_norm(K.rms_norm_reference, xb, wb, dyb)
        for i, name in enumerate(("forward", "input gradient")):
            ef, et = rel(fused[i], truth[i]), rel(ref[i], truth[i])
            check(f"RMSNorm bf16 {shape} {name}: fused error {ef:.1e} <= model bf16 error {et:.1e} (x{slack:g})",
                  ef <= slack * et + 1e-6, (ef, et))
    # an fp32 weight on a bf16 input promotes the output to fp32, as in the model
    x = torch.randn(4, 64, device=DEV).bfloat16()
    w = torch.ones(64, device=DEV)
    check("RMSNorm: fp32 weight x bf16 input -> fp32 output, as the model",
          K.rms_norm(x, w, 1e-6).dtype == K.rms_norm_reference(x, w, 1e-6).dtype == torch.float32)
    wt = torch.ones(64, device=DEV, requires_grad=True)
    check("RMSNorm: a trainable weight falls back to the model's code (weight gets a gradient)",
          K.rms_norm(x, wt, 1e-6).sum().backward() is None and wt.grad is not None)


def test_model():
    folder = os.environ.get("RICURSIVE_MODEL_DIR")
    if not folder or not os.path.exists(os.path.join(folder, "modeling_ricursive.py")):
        print("SKIP model-level check (set RICURSIVE_MODEL_DIR)")
        return
    os.environ["RIPPLE_MLP_CHUNK"] = "0"
    from test_ricursive_port import load_remote, tiny

    cm, mm = load_remote(folder)
    m = tiny(cm, mm).train()
    for mod in m.modules():
        if type(mod).__name__ == "RicursiveMLP":
            mod.clamp_threshold = 0.3  # make the clamps bind on this tiny model
        if type(mod).__name__ == "RicursiveRMSNorm":
            mod.weight.requires_grad_(False)  # frozen, as under LoRA: the fused norm path is taken
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    g = torch.Generator().manual_seed(4)
    ids = torch.randint(3, 50, (2, 19), generator=g)

    def loss_and_grads():
        m.zero_grad()
        loss = m(input_ids=ids, labels=ids, use_cache=False).loss
        loss.backward()
        return float(loss.detach()), {k: p.grad.clone() for k, p in m.named_parameters() if p.grad is not None}

    l0, g0 = loss_and_grads()
    n = R.patch_ricursive_fused_mlp(m)
    check("model: all 5 MLPs patched (idempotent)", n == 5 and R.patch_ricursive_fused_mlp(m) == 5, n)
    nn_ = R.patch_ricursive_fused_rmsnorm(m)
    # 5 layers x (4 layer norms + q_norm + k_norm) + final norm
    check("model: all 31 RMSNorms patched (idempotent)", nn_ == 31 and R.patch_ricursive_fused_rmsnorm(m) == 31, nn_)
    l1, g1 = loss_and_grads()
    gd = max(rel(g1[k], g0[k]) for k in g0)
    check("model: loss unchanged by the fused MLP + RMSNorm", abs(l1 - l0) < 1e-5, (l1, l0))
    check("model: every gradient unchanged (gradient checkpointing on)", set(g1) == set(g0) and gd < 1e-4, gd)


def test_gpu_size():
    if not torch.cuda.is_available():
        print("SKIP real-size GPU check (needs CUDA)")
        return
    torch.manual_seed(0)
    n, d, T = 4096, 40960, 20.0
    g = (torch.randn(n, d, device="cuda") * 8).bfloat16()
    u = (torch.randn(n, d, device="cuda") * 8).bfloat16()
    dh = torch.randn(n, d, device="cuda").bfloat16()
    truth = run(K.reference, g.float(), u.float(), T, dh.float())
    fused = run(K.clamped_silu_mul, g, u, T, dh)
    torch_bf16 = run(K.reference, g, u, T, dh)
    for i, name in enumerate(("forward", "dgate", "dup")):
        ef, et = rel(fused[i], truth[i]), rel(torch_bf16[i], truth[i])
        check(f"GPU [4096, 40960] bf16 {name}: fused error {ef:.1e} <= torch bf16 error {et:.1e} (x1.5)",
              ef <= 1.5 * et + 1e-6, (ef, et))
    del truth, fused, torch_bf16  # free before timing

    def timed(fn):
        for _ in range(2):
            run(fn, g, u, T, dh)
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(10):
            run(fn, g, u, T, dh)
        b.record()
        torch.cuda.synchronize()
        return a.elapsed_time(b) / 10

    tf, tr = timed(K.clamped_silu_mul), timed(K.reference)
    print(f"INFO MLP fwd+bwd per call at [4096, 40960] bf16: fused {tf:.1f} ms, torch {tr:.1f} ms ({tr / tf:.1f}x)")
    del g, u, dh

    x = (torch.randn(8192, 5120, device="cuda") * 3).bfloat16()
    w = (1 + 0.3 * torch.randn(5120, device="cuda")).bfloat16()
    dy = torch.randn(8192, 5120, device="cuda").bfloat16()
    truth = run_norm(K.rms_norm_reference, x.double(), w.double(), dy.double())
    fused = run_norm(K.rms_norm, x, w, dy)
    ref = run_norm(K.rms_norm_reference, x, w, dy)
    for i, name in enumerate(("forward", "input gradient")):
        ef, et = rel(fused[i], truth[i]), rel(ref[i], truth[i])
        check(f"GPU RMSNorm [8192, 5120] bf16 {name}: fused error {ef:.1e} <= model error {et:.1e} (x1.5)",
              ef <= 1.5 * et + 1e-6, (ef, et))
    del truth

    def timed_norm(fn):
        for _ in range(2):
            run_norm(fn, x, w, dy)
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(10):
            run_norm(fn, x, w, dy)
        b.record()
        torch.cuda.synchronize()
        return a.elapsed_time(b) / 10

    tf, tr = timed_norm(K.rms_norm), timed_norm(K.rms_norm_reference)
    print(f"INFO RMSNorm fwd+bwd per call at [8192, 5120] bf16: fused {tf:.2f} ms, model {tr:.2f} ms ({tr / tf:.1f}x)")


if __name__ == "__main__":
    test_function()
    test_rmsnorm()
    test_model()
    test_gpu_size()
    print(f"\n{'ALL PASSED' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}'}")
    sys.exit(1 if FAIL else 0)
