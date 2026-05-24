"""STAGE 8 §1 — CausalConv3d cache_x byte-diff vs PT.

Two-layer verification per user directive:

  Layer 1 — cache 없이 T=5 한 번에 forward.  Stateless.  PT vs MLX byte-diff.
            cache 없이 틀리면 cache 넣어도 틀려.
  Layer 2 — T=5를 1+4 chunk로 쪼개서 feat_cache propagate.  단순 출력 일치
            뿐 아니라 chunk 경계에서 cache 상태가 PT와 동일한지 확인.
            (cache 상태가 어긋나면 multi-chunk 발산 = 진짜 seam)

PT 정답지: vae2_2.py:33-58 (CausalConv3d.forward).
"""
from __future__ import annotations

import os
import sys
import importlib

sys.path.insert(0, os.path.abspath("refs/Lance"))

import numpy as np
import torch
import mlx.core as mx

from lance_mlx.vae_wan22 import CausalConv3d


def _set_pt_conv_from_mlx(pt_conv, mlx_conv: CausalConv3d):
    """Copy MLX (O, kT, kH, kW, I) weight + bias → PT (O, I, kT, kH, kW)."""
    w_mlx = np.asarray(mlx_conv.weight, dtype=np.float32)  # (O, kT, kH, kW, I)
    w_pt = np.transpose(w_mlx, (0, 4, 1, 2, 3))            # → (O, I, kT, kH, kW)
    pt_conv.weight.data = torch.from_numpy(w_pt.copy())
    if mlx_conv._has_bias:
        pt_conv.bias.data = torch.from_numpy(np.asarray(mlx_conv.bias, dtype=np.float32).copy())


def cos_pt_mlx(pt: torch.Tensor, mlx_: mx.array) -> float:
    a = pt.detach().to(torch.float32).cpu().numpy().flatten()
    b = np.asarray(mlx_, dtype=np.float32).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def max_abs_diff(pt: torch.Tensor, mlx_: mx.array) -> float:
    a = pt.detach().to(torch.float32).cpu().numpy()
    # PT is NCTHW, MLX is NTHWC.  Permute PT to NTHWC for comparison.
    a = np.transpose(a, (0, 2, 3, 4, 1))
    b = np.asarray(mlx_, dtype=np.float32)
    return float(np.abs(a - b).max())


def main():
    print("=" * 70)
    print("STAGE 8 §1 — CausalConv3d cache_x byte-diff")
    print("=" * 70)

    # ---- import PT side ----
    vae_mod = importlib.import_module("modeling.vae.wan.vae2_2")
    PtCausalConv3d = vae_mod.CausalConv3d

    # ---- build matched pair: one CausalConv3d, kT=3 kH=3 kW=3 padding=1, like inside Encoder3d's conv1 ----
    in_c, out_c = 8, 16
    mx_conv = CausalConv3d(in_c, out_c, kernel_size=3, padding=1)
    pt_conv = PtCausalConv3d(in_c, out_c, kernel_size=3, padding=1)
    pt_conv.eval()
    # Initialize MLX weights deterministically, then copy to PT
    rng = np.random.default_rng(42)
    mx_conv.weight = mx.array(rng.standard_normal(mx_conv.weight.shape).astype("float32") * 0.05)
    mx_conv.bias = mx.array(rng.standard_normal(mx_conv.bias.shape).astype("float32") * 0.01)
    _set_pt_conv_from_mlx(pt_conv, mx_conv)
    print(f"[setup] CausalConv3d({in_c}→{out_c}, kT=3 kH=3 kW=3 pad=1)  weights synced")

    # ---- common input: T=5, H=8, W=8 ----
    B, T, H, W = 1, 5, 8, 8
    x_np = rng.standard_normal((B, T, H, W, in_c)).astype("float32")
    x_mx = mx.array(x_np)
    x_pt = torch.from_numpy(np.transpose(x_np, (0, 4, 1, 2, 3)).copy())  # NCTHW
    print(f"[input] T={T} H={H} W={W} C={in_c}  ||x||={np.linalg.norm(x_np):.3f}")

    # ====================================================================
    # LAYER 1 — single full T=5 forward, no cache
    # ====================================================================
    print("\n" + "=" * 70)
    print("LAYER 1 — full T=5 forward, no cache (stateless)")
    print("=" * 70)
    y_mx = mx_conv(x_mx)                          # cache_x default=None
    mx.eval(y_mx)
    with torch.no_grad():
        y_pt = pt_conv(x_pt)
    print(f"  PT  output shape (NCTHW): {tuple(y_pt.shape)}")
    print(f"  MLX output shape (NTHWC): {tuple(y_mx.shape)}")
    c1 = cos_pt_mlx(y_pt.permute(0, 2, 3, 4, 1).contiguous(), y_mx)
    d1 = max_abs_diff(y_pt, y_mx)
    print(f"  cos(PT, MLX) = {c1:.8f}    max|Δ| = {d1:.3e}    "
          f"{'PASS' if c1 >= 0.999999 else 'FAIL'}")

    # ====================================================================
    # LAYER 2 — split T=5 into chunks (1 + 4) with feat_cache propagation
    # ====================================================================
    print("\n" + "=" * 70)
    print("LAYER 2 — chunked (1 + 4) with feat_cache, byte-diff vs full T=5")
    print("=" * 70)

    CACHE_T = 2  # matches refs/Lance/modeling/vae/wan/vae2_2.py:30

    # ---- MLX side: chunked forward ----
    # Chunk 1: frames [0]  — feat_cache empty (cache_x=None for this conv)
    x_mx_c1 = x_mx[:, 0:1, :, :, :]
    y_mx_c1 = mx_conv(x_mx_c1, cache_x=None)
    # Cache last CACHE_T frames of the input that fed this conv (PT pattern,
    # see vae2_2.py:543 — `cache_x = x[:, :, -CACHE_T:, ...]` BEFORE conv).
    # For chunk 1 input T=1 < CACHE_T, the cache is the entire chunk (T=1).
    cache_after_c1 = x_mx_c1[:, -CACHE_T:, :, :, :]
    print(f"  [chunk 1] in T=1 → out T={y_mx_c1.shape[1]}, cache T={cache_after_c1.shape[1]}")

    # Chunk 2: frames [1, 2, 3, 4] — feed cache_x from chunk 1
    x_mx_c2 = x_mx[:, 1:5, :, :, :]
    y_mx_c2 = mx_conv(x_mx_c2, cache_x=cache_after_c1)
    print(f"  [chunk 2] in T=4 → out T={y_mx_c2.shape[1]}, cache_x T={cache_after_c1.shape[1]}")

    # Concatenate chunked outputs
    y_mx_chunked = mx.concatenate([y_mx_c1, y_mx_c2], axis=1)
    mx.eval(y_mx_chunked)

    # ---- PT side: same chunked forward ----
    x_pt_c1 = x_pt[:, :, 0:1, :, :]
    with torch.no_grad():
        y_pt_c1 = pt_conv(x_pt_c1, cache_x=None)
    cache_pt_after_c1 = x_pt_c1[:, :, -CACHE_T:, :, :]
    x_pt_c2 = x_pt[:, :, 1:5, :, :]
    with torch.no_grad():
        y_pt_c2 = pt_conv(x_pt_c2, cache_x=cache_pt_after_c1)
    y_pt_chunked = torch.cat([y_pt_c1, y_pt_c2], dim=2)

    # ---- (2a) chunked output PT vs MLX ----
    c2a_full = cos_pt_mlx(y_pt_chunked.permute(0, 2, 3, 4, 1).contiguous(), y_mx_chunked)
    d2a_full = max_abs_diff(y_pt_chunked, y_mx_chunked)
    print(f"\n  (2a) chunked-output cos(PT, MLX)       = {c2a_full:.8f}  "
          f"max|Δ| = {d2a_full:.3e}    "
          f"{'PASS' if c2a_full >= 0.999999 else 'FAIL'}")

    # ---- (2b) chunked output PT vs MLX, per-chunk (catch boundary alone) ----
    c2b_c1 = cos_pt_mlx(y_pt_c1.permute(0, 2, 3, 4, 1).contiguous(), y_mx_c1)
    c2b_c2 = cos_pt_mlx(y_pt_c2.permute(0, 2, 3, 4, 1).contiguous(), y_mx_c2)
    print(f"  (2b) per-chunk cos:  chunk1={c2b_c1:.8f}   chunk2={c2b_c2:.8f}")

    # ---- (2c) cache STATE byte-diff (the real seam test) ----
    cache_mlx_np = np.asarray(cache_after_c1, dtype=np.float32)         # NTHWC
    cache_pt_np = np.transpose(cache_pt_after_c1.numpy(), (0, 2, 3, 4, 1))  # → NTHWC
    cache_eq = np.allclose(cache_mlx_np, cache_pt_np, atol=0.0)
    cache_max_diff = float(np.abs(cache_mlx_np - cache_pt_np).max())
    print(f"  (2c) feat_cache state byte-equal: {cache_eq}   max|Δ| = {cache_max_diff:.3e}    "
          f"{'PASS' if cache_eq else 'FAIL'}")

    # ---- (2d) chunked vs full T=5 (the *real* gate — does chunking lose info?) ----
    c2d = cos_pt_mlx(y_pt.permute(0, 2, 3, 4, 1).contiguous(), y_mx_chunked)
    d2d = max_abs_diff(y_pt, y_mx_chunked)
    c2d_pt_only = float(((y_pt - y_pt_chunked) ** 2).mean().item())
    print(f"\n  (2d) chunked MLX vs full-T=5 PT:  cos = {c2d:.8f}  max|Δ| = {d2d:.3e}")
    print(f"       PT internal: full-T=5 vs PT-chunked MSE = {c2d_pt_only:.3e}  "
          f"(should be ~0 if feat_cache works in PT)")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Layer 1 (stateless full T=5)            cos = {c1:.8f}   "
          f"{'PASS' if c1 >= 0.999999 else 'FAIL'}")
    print(f"  Layer 2a (chunked output PT vs MLX)     cos = {c2a_full:.8f}  "
          f"{'PASS' if c2a_full >= 0.999999 else 'FAIL'}")
    print(f"  Layer 2c (feat_cache state byte-equal)  {'PASS' if cache_eq else 'FAIL'}")
    print(f"  Layer 2d (chunked MLX vs full-T=5 PT)   cos = {c2d:.8f}  "
          f"{'PASS' if c2d >= 0.999999 else 'FAIL'}")


if __name__ == "__main__":
    main()
