"""Fused clamped-SwiGLU activation for the Ricursive MLP: ``silu(min(g, T)) * clamp(u, -T, T)``.

The model computes it as separate bf16 ops (clamp, clamp, silu, mul) on [tokens, 40960]
tensors, and autograd adds as many again in backward; under gradient checkpointing the forward
runs twice. Here one Triton kernel does the forward (fp32 math, one bf16 write) and saves only
``g`` and ``u``; one kernel does the backward (recomputes from them, writes dg and du). Gradient
masks follow torch's clamp backward, which is strict (``g < T``; ``-T < u < T``: no gradient
exactly at the threshold). No clamp threshold -> T = inf.

``clamped_silu_mul`` falls back to the plain torch ops when Triton can't run (no CUDA and no
``TRITON_INTERPRET=1``) or the tensor is too large for int32 offsets.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

_BLOCK = 1024

try:  # pragma: no cover - import guarded for machines without Triton
    import triton
    import triton.language as tl

    @triton.jit
    def _fwd_kernel(g_ptr, u_ptr, h_ptr, n, T, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        g = tl.load(g_ptr + offs, mask=m, other=0.0).to(tl.float32)
        u = tl.load(u_ptr + offs, mask=m, other=0.0).to(tl.float32)
        gc = tl.minimum(g, T)
        uc = tl.minimum(tl.maximum(u, -T), T)
        s = 1.0 / (1.0 + tl.exp(-gc))
        h = gc * s * uc
        tl.store(h_ptr + offs, h.to(h_ptr.dtype.element_ty), mask=m)

    @triton.jit
    def _bwd_kernel(dh_ptr, g_ptr, u_ptr, dg_ptr, du_ptr, n, T, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        dh = tl.load(dh_ptr + offs, mask=m, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + offs, mask=m, other=0.0).to(tl.float32)
        u = tl.load(u_ptr + offs, mask=m, other=0.0).to(tl.float32)
        gc = tl.minimum(g, T)
        uc = tl.minimum(tl.maximum(u, -T), T)
        s = 1.0 / (1.0 + tl.exp(-gc))
        silu = gc * s
        dsilu = s * (1.0 + gc * (1.0 - s))
        # torch's clamp backward is strict: no gradient exactly at the threshold
        dg = tl.where(g < T, dh * uc * dsilu, 0.0)
        du = tl.where((u > -T) & (u < T), dh * silu, 0.0)
        tl.store(dg_ptr + offs, dg.to(dg_ptr.dtype.element_ty), mask=m)
        tl.store(du_ptr + offs, du.to(du_ptr.dtype.element_ty), mask=m)

    @triton.jit
    def _rms_fwd_kernel(x_ptr, w_ptr, y_ptr, rstd_ptr, D, eps, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        m = cols < D
        x = tl.load(x_ptr + row * D + cols, mask=m, other=0.0).to(tl.float32)
        rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / D + eps)
        # the model: weight * (x * rstd).to(input dtype)
        xn = (x * rstd).to(x_ptr.dtype.element_ty).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(y_ptr + row * D + cols, (w * xn).to(y_ptr.dtype.element_ty), mask=m)
        tl.store(rstd_ptr + row, rstd)

    @triton.jit
    def _rms_bwd_kernel(dy_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, D, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        m = cols < D
        dy = tl.load(dy_ptr + row * D + cols, mask=m, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + row * D + cols, mask=m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + row)
        xn = x * rstd
        g = dy * w  # gradient w.r.t. the normalized x (the dtype cast passes it straight through)
        dx = rstd * (g - xn * (tl.sum(g * xn, axis=0) / D))
        tl.store(dx_ptr + row * D + cols, dx.to(dx_ptr.dtype.element_ty), mask=m)

    _HAVE_TRITON = True
except Exception:  # noqa: BLE001
    _HAVE_TRITON = False


def _rms_warps(block):
    return 16 if block >= 8192 else 8 if block >= 2048 else 4


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, eps):
        shape = x.shape
        D = shape[-1]
        x2 = x.reshape(-1, D).contiguous()
        rows = x2.shape[0]
        y = torch.empty((rows, D), dtype=torch.promote_types(w.dtype, x.dtype), device=x.device)
        rstd = torch.empty(rows, dtype=torch.float32, device=x.device)
        block = triton.next_power_of_2(D)
        _rms_fwd_kernel[(rows,)](x2, w.contiguous(), y, rstd, D, float(eps), BLOCK=block,
                                 num_warps=_rms_warps(block))
        ctx.save_for_backward(x2, w, rstd)
        ctx.shape = shape
        return y.view(shape)

    @staticmethod
    def backward(ctx, dy):
        x2, w, rstd = ctx.saved_tensors
        D = x2.shape[-1]
        dy2 = dy.reshape(-1, D).contiguous()
        dx = torch.empty_like(x2)
        block = triton.next_power_of_2(D)
        _rms_bwd_kernel[(x2.shape[0],)](dy2, x2, w.contiguous(), rstd, dx, D, BLOCK=block,
                                        num_warps=_rms_warps(block))
        return dx.view(ctx.shape), None, None


def rms_norm_reference(x, weight, eps):
    """RicursiveRMSNorm.forward."""
    input_dtype = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * xf.to(input_dtype)


def rms_norm(x, weight, eps):
    """The model's RMSNorm, fused when possible (frozen weight, Triton available, D <= 65536)."""
    if (weight.requires_grad or not fused_available(x) or x.dim() == 0 or x.numel() == 0
            or x.shape[-1] > 65536 or weight.shape != x.shape[-1:]):
        return rms_norm_reference(x, weight, eps)
    return _RMSNorm.apply(x, weight, eps)


def _grid(n):
    return ((n + _BLOCK - 1) // _BLOCK,)


class _ClampedSiluMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g, u, T):
        g = g.contiguous()
        u = u.to(g.dtype).contiguous()
        h = torch.empty_like(g)
        n = g.numel()
        _fwd_kernel[_grid(n)](g, u, h, n, T, BLOCK=_BLOCK)
        ctx.save_for_backward(g, u)
        ctx.T = T
        return h

    @staticmethod
    def backward(ctx, dh):
        g, u = ctx.saved_tensors
        dh = dh.contiguous()
        dg = torch.empty_like(g)
        du = torch.empty_like(u)
        n = g.numel()
        _bwd_kernel[_grid(n)](dh, g, u, dg, du, n, ctx.T, BLOCK=_BLOCK)
        return dg, du, None


def reference(g, u, threshold):
    """The model's own ops (RicursiveMLP._forward_rows between the projections)."""
    if threshold:
        g = g.clamp(min=None, max=threshold)
        u = u.clamp(min=-threshold, max=threshold)
    return F.silu(g) * u


def fused_available(t):
    if not _HAVE_TRITON or t.numel() >= 2**31 - _BLOCK:
        return False
    return t.is_cuda or os.environ.get("TRITON_INTERPRET") == "1"


def clamped_silu_mul(g, u, threshold):
    """``silu(min(g, T)) * clamp(u, -T, T)`` (no clamp when threshold is falsy); fused when possible."""
    if not fused_available(g) or g.shape != u.shape:
        return reference(g, u, threshold)
    T = float(threshold) if threshold else math.inf
    return _ClampedSiluMul.apply(g, u, T)
