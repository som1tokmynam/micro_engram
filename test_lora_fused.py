"""Tests for micro_engram.axolotl_port.lora_fused (LoRA scale+add folded into addmm).
Run: python test_lora_fused.py

PEFT LoRA (rsLoRA, non-zero lora_B) on a small Linear stack, patched vs PEFT's own forward:
  1. fp32: outputs, input gradients and every LoRA gradient equal, with and without gradient
     checkpointing; the addmm path is the one taken
  2. bf16: error against a float64 truth no worse than PEFT's own bf16 path (x1.5)
  3. fallbacks give exactly PEFT's result: dropout active in training, disabled / merged
     adapters, fp32 adapters on a bf16 input
  (lora.bnb.Linear4bit is patched the same way; it needs bitsandbytes + CUDA and is exercised by
  the training run, whose log reports how many LoRA calls took the addmm path.)
"""
import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from micro_engram.axolotl_port import lora_fused as LF  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail and not ok else ""), flush=True)
    if not ok:
        FAIL.append(name)


class Block(nn.Module):
    def __init__(self, d=48, f=96):
        super().__init__()
        self.q = nn.Linear(d, d, bias=False)
        self.up = nn.Linear(d, f, bias=False)
        self.down = nn.Linear(f, d, bias=False)

    def forward(self, x):
        return x + self.down(torch.relu(self.up(self.q(x))))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(2)])

    def forward(self, x, ckpt=False):
        from torch.utils.checkpoint import checkpoint

        for b in self.blocks:
            x = checkpoint(b, x, use_reentrant=False) if ckpt else b(x)
        return x


def peft_net(dropout=0.0, seed=0):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)
    net = Net()
    cfg = LoraConfig(r=8, lora_alpha=8, use_rslora=True, lora_dropout=dropout,
                     target_modules=["q", "up", "down"])
    m = get_peft_model(net, cfg)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.1)  # PEFT inits B to zero: make the adapters contribute
    return m


def lora_grads(m):
    return {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}


def fwd_bwd(m, x, ckpt=False):
    x = x.detach().clone().requires_grad_()
    m.zero_grad()
    y = m.base_model.model(x, ckpt=ckpt)
    (y.float() ** 2).mean().backward()
    return y.detach(), x.grad, lora_grads(m)


def rel(a, b):
    return float((a.double() - b.double()).norm() / (b.double().norm() + 1e-30))


def main():
    torch.manual_seed(1)
    x = torch.randn(3, 11, 48)
    ref_m = peft_net()
    ref = {c: fwd_bwd(ref_m, x, c) for c in (False, True)}
    ref_bf16 = None
    m16 = peft_net().to(torch.bfloat16)
    ref_bf16 = fwd_bwd(m16, x.bfloat16())
    truth = fwd_bwd(peft_net().double(), x.double())
    # fallback references (unpatched)
    torch.manual_seed(5)
    drop_ref = fwd_bwd(peft_net(dropout=0.3).train(), x)
    m32_on_16 = peft_net()
    for mod in m32_on_16.modules():  # bf16 base weights, fp32 adapters
        if hasattr(mod, "base_layer"):
            mod.base_layer.to(torch.bfloat16)
    mixed_ref = fwd_bwd(m32_on_16, x.bfloat16())
    md = peft_net()
    with md.disable_adapter(), torch.no_grad():
        dis_ref = md.base_model.model(x)

    patched = LF.patch_fused_lora()
    check("patched PEFT lora.Linear (+ bnb.Linear4bit when importable)", "lora.Linear" in patched, patched)
    LF.patch_fused_lora()

    m = peft_net()
    for ckpt in (False, True):
        LF._STATS.update(fused=0, fallback=0)
        y, gx, gl = fwd_bwd(m, x, ckpt)
        ry, rgx, rgl = ref[ckpt]
        tag = "with checkpointing" if ckpt else "plain"
        check(f"fp32 {tag}: output == PEFT", rel(y, ry) < 1e-6, rel(y, ry))
        check(f"fp32 {tag}: input gradient == PEFT", rel(gx, rgx) < 1e-6, rel(gx, rgx))
        check(f"fp32 {tag}: every LoRA gradient == PEFT",
              set(gl) == set(rgl) and max(rel(gl[k], rgl[k]) for k in rgl) < 1e-5)
        check(f"fp32 {tag}: all LoRA calls took the addmm path", LF._STATS["fused"] > 0 and LF._STATS["fallback"] == 0,
              dict(LF._STATS))

    f16 = fwd_bwd(peft_net().to(torch.bfloat16), x.bfloat16())
    for i, name in enumerate(("output", "input gradient")):
        ef, ep = rel(f16[i], truth[i]), rel(ref_bf16[i], truth[i])
        check(f"bf16 {name}: fused error {ef:.1e} <= PEFT bf16 error {ep:.1e} (x1.5)", ef <= 1.5 * ep + 1e-6, (ef, ep))
    ef = max(rel(f16[2][k], truth[2][k]) for k in truth[2])
    ep = max(rel(ref_bf16[2][k], truth[2][k]) for k in truth[2])
    check(f"bf16 LoRA gradients: fused worst error {ef:.1e} <= PEFT worst {ep:.1e} (x1.5)", ef <= 1.5 * ep + 1e-6)

    torch.manual_seed(5)
    d = fwd_bwd(peft_net(dropout=0.3).train(), x)
    check("fallback: dropout active in training -> exactly PEFT", torch.equal(d[0], drop_ref[0]))
    mm = peft_net()
    with mm.disable_adapter(), torch.no_grad():
        y = mm.base_model.model(x)
    check("fallback: disabled adapters -> exactly PEFT (base only)", torch.equal(y, dis_ref))
    mm.merge_adapter()
    check("fallback: merged adapters -> same output as unmerged",
          rel(mm.base_model.model(x).detach(), ref[False][0]) < 1e-5)
    mm.unmerge_adapter()
    mx = peft_net()
    for mod in mx.modules():
        if hasattr(mod, "base_layer"):
            mod.base_layer.to(torch.bfloat16)
    LF._STATS.update(fused=0, fallback=0)
    got = fwd_bwd(mx, x.bfloat16())
    check("fallback: fp32 adapters on bf16 input -> exactly PEFT", torch.equal(got[0], mixed_ref[0])
          and LF._STATS["fused"] == 0, dict(LF._STATS))

    print(f"\n{'ALL PASSED' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}'}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
