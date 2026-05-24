"""STAGE 8 §1.2 — Resample upsample3d / downsample3d byte-diff vs PT.

Same two-layer pattern as CausalConv3d compare, plus an EXPLICIT T-axis
frame-ordering check (STAGE 5 教訓: reshape sequence can lose order
silently; cos won't catch it if values happen to be similar across the
permutation).

For upsample3d, PT does:
  time_conv → (B, 2C, T, H, W)
  reshape   → (B, 2, C, T, H, W)
  stack dim=3 → (B, C, T, 2, H, W)
  reshape   → (B, C, T*2, H, W)
  Frame index in output: t_out = t_in*2 + g
"""
from __future__ import annotations

import os
import sys
import importlib

sys.path.insert(0, os.path.abspath("refs/Lance"))

import numpy as np
import torch
import mlx.core as mx

from lance_mlx.vae_wan22 import Resample, CausalConv3d


def _sync_pt_resample_from_mlx(pt_resample, mx_resample):
    """Copy MLX Resample weights → PT Resample.

    PT layout:
      - mode==upsample2d/3d: `self.resample = nn.Sequential(Upsample, Conv2d)` → conv at index 1
      - mode==downsample2d/3d: `self.resample = nn.Sequential(ZeroPad2d, Conv2d)` → conv at index 1
      - additionally upsample3d/downsample3d: self.time_conv = CausalConv3d
    """
    # spatial_conv (MLX) ↔ resample[1] (PT)
    w_mlx = np.asarray(mx_resample.spatial_conv.weight, dtype=np.float32)  # (O, H, W, I)
    w_pt = np.transpose(w_mlx, (0, 3, 1, 2))                                # → (O, I, H, W)
    pt_resample.resample[1].weight.data = torch.from_numpy(w_pt.copy())
    if mx_resample.spatial_conv.bias is not None:
        b_mlx = np.asarray(mx_resample.spatial_conv.bias, dtype=np.float32)
        pt_resample.resample[1].bias.data = torch.from_numpy(b_mlx.copy())

    # time_conv (3D modes only)
    if hasattr(mx_resample, "time_conv"):
        w_mlx_t = np.asarray(mx_resample.time_conv.weight, dtype=np.float32)  # (O, kT, kH, kW, I)
        w_pt_t = np.transpose(w_mlx_t, (0, 4, 1, 2, 3))                        # → (O, I, kT, kH, kW)
        pt_resample.time_conv.weight.data = torch.from_numpy(w_pt_t.copy())
        if mx_resample.time_conv._has_bias:
            b_mlx_t = np.asarray(mx_resample.time_conv.bias, dtype=np.float32)
            pt_resample.time_conv.bias.data = torch.from_numpy(b_mlx_t.copy())


def cos_pt_mlx(pt: torch.Tensor, mlx_: mx.array) -> float:
    a = pt.detach().to(torch.float32).cpu().numpy().flatten()
    b = np.asarray(mlx_, dtype=np.float32).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def max_abs_diff_nthwc(pt_nctw: torch.Tensor, mlx_nthwc: mx.array) -> float:
    """PT is (B, C, T, H, W); MLX is (B, T, H, W, C).  Permute then diff."""
    a = pt_nctw.detach().to(torch.float32).cpu().numpy()
    a = np.transpose(a, (0, 2, 3, 4, 1))   # → NTHWC
    b = np.asarray(mlx_nthwc, dtype=np.float32)
    return float(np.abs(a - b).max())


def _seed_resample(mx_r: Resample, rng):
    """Deterministic init for both spatial_conv and time_conv (if present)."""
    w = mx_r.spatial_conv.weight
    mx_r.spatial_conv.weight = mx.array(
        rng.standard_normal(w.shape).astype("float32") * 0.05
    )
    if mx_r.spatial_conv.bias is not None:
        mx_r.spatial_conv.bias = mx.array(
            rng.standard_normal(mx_r.spatial_conv.bias.shape).astype("float32") * 0.01
        )
    if hasattr(mx_r, "time_conv"):
        wt = mx_r.time_conv.weight
        mx_r.time_conv.weight = mx.array(
            rng.standard_normal(wt.shape).astype("float32") * 0.05
        )
        if mx_r.time_conv._has_bias:
            mx_r.time_conv.bias = mx.array(
                rng.standard_normal(mx_r.time_conv.bias.shape).astype("float32") * 0.01
            )


# ============================================================================
# UPSAMPLE3D
# ============================================================================
def test_upsample3d():
    print("=" * 70)
    print("UPSAMPLE3D — T expansion path (used in decoder)")
    print("=" * 70)

    vae_mod = importlib.import_module("modeling.vae.wan.vae2_2")
    PtResample = vae_mod.Resample

    dim = 8
    mx_r = Resample(dim, "upsample3d")
    pt_r = PtResample(dim, "upsample3d")
    pt_r.eval()
    rng = np.random.default_rng(123)
    _seed_resample(mx_r, rng)
    _sync_pt_resample_from_mlx(pt_r, mx_r)
    print(f"[setup] Resample(dim={dim}, mode='upsample3d')  weights synced")

    # --- T-expansion axis-order explicit check first ---
    # Set time_conv weights to identity-like so we can predict the output frame
    # ordering directly: output channel c = group g * dim + sub_c.
    # Then the output T should interleave (group 0 frame, group 1 frame, ...).
    print("\n[axis-order check] use identity-like time_conv to verify frame interleave")
    # Re-init time_conv with a known pattern: weight[oc, kt, kh, kw, ic] = 1 if oc % dim == ic and kt == 0 (center past frame after causal pad)
    test_w = np.zeros((dim * 2, 3, 1, 1, dim), dtype=np.float32)
    for oc in range(dim * 2):
        ic = oc % dim
        test_w[oc, 2, 0, 0, ic] = 1.0   # last past frame (index 2 in pad_t=2 → centered after causal pad)
    mx_r.time_conv.weight = mx.array(test_w)
    mx_r.time_conv.bias = mx.zeros((dim * 2,))
    pt_test_w = np.transpose(test_w, (0, 4, 1, 2, 3))   # PT: (O, I, kT, kH, kW)
    pt_r.time_conv.weight.data = torch.from_numpy(pt_test_w.copy())
    pt_r.time_conv.bias.data = torch.zeros(dim * 2)

    # Single frame input (multi-call simulation needs at least 2 calls to see expansion)
    B, T_in, H, W = 1, 1, 4, 4
    x1_np = rng.standard_normal((B, T_in, H, W, dim)).astype("float32")
    x1_mx = mx.array(x1_np)
    x1_pt = torch.from_numpy(np.transpose(x1_np, (0, 4, 1, 2, 3)).copy())

    # Call 1: feat_cache=[None] → first call → cache becomes "Rep", time_conv SKIPPED
    feat_cache_mx, feat_idx_mx = [None], [0]
    out1_mx = mx_r(x1_mx, feat_cache=feat_cache_mx, feat_idx=feat_idx_mx)
    mx.eval(out1_mx)
    pt_cache, pt_idx = [None], [0]
    with torch.no_grad():
        out1_pt = pt_r(x1_pt, feat_cache=pt_cache, feat_idx=pt_idx)
    print(f"  Call 1: T_in={T_in} → MLX T_out={out1_mx.shape[1]}  PT T_out={out1_pt.shape[2]}")
    c1 = cos_pt_mlx(out1_pt.permute(0, 2, 3, 4, 1).contiguous(), out1_mx)
    d1 = max_abs_diff_nthwc(out1_pt, out1_mx)
    print(f"    cos = {c1:.8f}  max|Δ| = {d1:.3e}  "
          f"{'PASS' if c1 >= 0.999999 else 'FAIL'}")
    print(f"    MLX feat_cache[0] = {feat_cache_mx[0]!r}  feat_idx = {feat_idx_mx}")
    print(f"    PT  feat_cache[0] = {pt_cache[0]!r}      feat_idx = {pt_idx}")

    # Call 2: feat_cache[0]='Rep' → time_conv applied, T 1→2 via reshape+stack
    x2_np = rng.standard_normal((B, T_in, H, W, dim)).astype("float32")
    x2_mx = mx.array(x2_np)
    x2_pt = torch.from_numpy(np.transpose(x2_np, (0, 4, 1, 2, 3)).copy())
    # Reset feat_idx to 0 at start of each call (PT pattern: self._conv_idx=[0] per iter).
    # feat_cache is the state that persists; feat_idx is the walk counter for one pass.
    feat_idx_mx[0] = 0
    pt_idx[0] = 0
    out2_mx = mx_r(x2_mx, feat_cache=feat_cache_mx, feat_idx=feat_idx_mx)
    mx.eval(out2_mx)
    with torch.no_grad():
        out2_pt = pt_r(x2_pt, feat_cache=pt_cache, feat_idx=pt_idx)
    print(f"\n  Call 2: T_in={T_in} → MLX T_out={out2_mx.shape[1]}  PT T_out={out2_pt.shape[2]}  "
          f"(expect 2× expansion)")
    c2 = cos_pt_mlx(out2_pt.permute(0, 2, 3, 4, 1).contiguous(), out2_mx)
    d2 = max_abs_diff_nthwc(out2_pt, out2_mx)
    print(f"    cos = {c2:.8f}  max|Δ| = {d2:.3e}  "
          f"{'PASS' if c2 >= 0.999999 else 'FAIL'}")

    # T-axis frame ordering check on call 2 output (the 2× expansion happened here)
    # With identity-like time_conv: out frame 0 should match group 0, out frame 1 group 1 (or vice versa).
    # We compare PT and MLX frame-by-frame.
    print(f"\n  [frame-order check] per-frame cos (call 2 output T={out2_mx.shape[1]}):")
    pt_nthwc = out2_pt.permute(0, 2, 3, 4, 1).contiguous().numpy()
    mlx_arr = np.asarray(out2_mx)
    for t in range(out2_mx.shape[1]):
        a = pt_nthwc[0, t].flatten()
        b = mlx_arr[0, t].flatten()
        c_t = float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))
        d_t = float(np.abs(a - b).max())
        print(f"    frame {t}:  cos = {c_t:.8f}  max|Δ| = {d_t:.3e}")

    # Cache state byte-diff (the real seam)
    cache_mx_arr = np.asarray(feat_cache_mx[0])                # (B, T, H, W, C)
    cache_pt_arr = np.transpose(pt_cache[0].numpy(), (0, 2, 3, 4, 1))
    cache_eq = np.allclose(cache_mx_arr, cache_pt_arr, atol=0.0)
    cache_diff = float(np.abs(cache_mx_arr - cache_pt_arr).max())
    print(f"\n  [seam] feat_cache state after call 2: byte-equal={cache_eq}  max|Δ|={cache_diff:.3e}  "
          f"{'PASS' if cache_eq else 'FAIL'}")

    return c1 >= 0.999999 and c2 >= 0.999999 and cache_eq


# ============================================================================
# DOWNSAMPLE3D
# ============================================================================
def test_downsample3d():
    print("\n" + "=" * 70)
    print("DOWNSAMPLE3D — stride-2 T conv with cache_x concat")
    print("=" * 70)

    vae_mod = importlib.import_module("modeling.vae.wan.vae2_2")
    PtResample = vae_mod.Resample

    dim = 8
    mx_r = Resample(dim, "downsample3d")
    pt_r = PtResample(dim, "downsample3d")
    pt_r.eval()
    rng = np.random.default_rng(456)
    _seed_resample(mx_r, rng)
    _sync_pt_resample_from_mlx(pt_r, mx_r)
    print(f"[setup] Resample(dim={dim}, mode='downsample3d')  weights synced")

    # Multi-call simulating encoder chunking: 1 + 4 frames
    B, H, W = 1, 8, 8
    x_full_np = rng.standard_normal((B, 5, H, W, dim)).astype("float32")
    x_full_mx = mx.array(x_full_np)
    x_full_pt = torch.from_numpy(np.transpose(x_full_np, (0, 4, 1, 2, 3)).copy())

    # Call 1: T=1 (first chunk) → cache stores x, time_conv NOT applied
    x1_mx = x_full_mx[:, 0:1]
    x1_pt = x_full_pt[:, :, 0:1]
    feat_cache_mx, feat_idx_mx = [None], [0]
    out1_mx = mx_r(x1_mx, feat_cache=feat_cache_mx, feat_idx=feat_idx_mx)
    mx.eval(out1_mx)
    pt_cache, pt_idx = [None], [0]
    with torch.no_grad():
        out1_pt = pt_r(x1_pt, feat_cache=pt_cache, feat_idx=pt_idx)
    c1 = cos_pt_mlx(out1_pt.permute(0, 2, 3, 4, 1).contiguous(), out1_mx)
    d1 = max_abs_diff_nthwc(out1_pt, out1_mx)
    print(f"  Call 1: T_in=1 → MLX T_out={out1_mx.shape[1]}  PT T_out={out1_pt.shape[2]}")
    print(f"    cos = {c1:.8f}  max|Δ| = {d1:.3e}  "
          f"{'PASS' if c1 >= 0.999999 else 'FAIL'}")

    # Call 2: T=4 (second chunk) → time_conv with cache concat
    x2_mx = x_full_mx[:, 1:5]
    x2_pt = x_full_pt[:, :, 1:5]
    # Reset feat_idx (per-pass counter); feat_cache (state) persists.
    feat_idx_mx[0] = 0
    pt_idx[0] = 0
    out2_mx = mx_r(x2_mx, feat_cache=feat_cache_mx, feat_idx=feat_idx_mx)
    mx.eval(out2_mx)
    with torch.no_grad():
        out2_pt = pt_r(x2_pt, feat_cache=pt_cache, feat_idx=pt_idx)
    c2 = cos_pt_mlx(out2_pt.permute(0, 2, 3, 4, 1).contiguous(), out2_mx)
    d2 = max_abs_diff_nthwc(out2_pt, out2_mx)
    print(f"\n  Call 2: T_in=4 → MLX T_out={out2_mx.shape[1]}  PT T_out={out2_pt.shape[2]}")
    print(f"    cos = {c2:.8f}  max|Δ| = {d2:.3e}  "
          f"{'PASS' if c2 >= 0.999999 else 'FAIL'}")

    # Per-frame check on call 2 (catches frame ordering bugs that cos can hide)
    print(f"\n  [frame-order check] per-frame cos (call 2 output T={out2_mx.shape[1]}):")
    pt_nthwc = out2_pt.permute(0, 2, 3, 4, 1).contiguous().numpy()
    mlx_arr = np.asarray(out2_mx)
    for t in range(out2_mx.shape[1]):
        a = pt_nthwc[0, t].flatten()
        b = mlx_arr[0, t].flatten()
        c_t = float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))
        d_t = float(np.abs(a - b).max())
        print(f"    frame {t}:  cos = {c_t:.8f}  max|Δ| = {d_t:.3e}")

    # Cache state check
    cache_mx_arr = np.asarray(feat_cache_mx[0])
    cache_pt_arr = np.transpose(pt_cache[0].numpy(), (0, 2, 3, 4, 1))
    cache_eq = np.allclose(cache_mx_arr, cache_pt_arr, atol=0.0)
    cache_diff = float(np.abs(cache_mx_arr - cache_pt_arr).max())
    print(f"\n  [seam] feat_cache state after call 2: byte-equal={cache_eq}  max|Δ|={cache_diff:.3e}  "
          f"{'PASS' if cache_eq else 'FAIL'}")

    return c1 >= 0.999999 and c2 >= 0.999999 and cache_eq


def main():
    print("STAGE 8 §1.2 — Resample T>1 byte-diff (upsample3d + downsample3d)\n")
    ok_up = test_upsample3d()
    ok_down = test_downsample3d()
    print("\n" + "=" * 70)
    print(f"SUMMARY  upsample3d={'PASS' if ok_up else 'FAIL'}  "
          f"downsample3d={'PASS' if ok_down else 'FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
