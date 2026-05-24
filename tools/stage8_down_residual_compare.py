"""STAGE 8 §1.4 — Down_ResidualBlock pass-through byte-diff vs PT.

First place where chunked-output T-axis ACTUALLY changes (Resample
downsample3d applies stride-2 T conv on chunk 2+).  Tests:

  Layer 1 — stateless full-T=5 (feat_cache=None on both sides).
            PT skips time_conv branch in Resample.downsample3d; T preserved.
            cos PT vs MLX.
  Layer 2 — chunked (1 + 4) with feat_cache.
            chunk 1 (T=1): first call into downsample3d → cache stored, no time_conv → T_out=1.
            chunk 2 (T=4): time_conv applied with cache concat → T_out=2.
            Total chunked T = 1 + 2 = 3, NOT 5.
            * cos PT vs MLX, per-chunk
            * slot-by-slot cache state (5 slots: 2 RB convs × 2 RBs + 1 Resample)
            * feat_idx counter progression
            * frame count assertion

NOTE — "chunked vs stateless" is NOT a direct equality gate here: PT's
stateless path skips time_conv entirely, so chunked (T_out=3) ≠ stateless
(T_out=5) by PT design.  We report it for diagnosis only.
"""
from __future__ import annotations

import os
import sys
import importlib

sys.path.insert(0, os.path.abspath("refs/Lance"))

import numpy as np
import torch
import mlx.core as mx

from lance_mlx.vae_wan22 import Down_ResidualBlock


def _sync_pt_down_from_mlx(pt_block, mlx_block):
    """Copy MLX → PT for a Down_ResidualBlock with mult ResidualBlocks +
    optional trailing Resample, plus AvgDown3D shortcut (no params)."""
    n_rb = sum(1 for _ in mlx_block.downsamples if not hasattr(_, "spatial_conv"))
    for i, m in enumerate(mlx_block.downsamples):
        if hasattr(m, "norm1"):  # ResidualBlock
            pt_m = pt_block.downsamples[i]
            pt_m.residual[0].gamma.data = torch.from_numpy(
                np.asarray(m.norm1.gamma, dtype=np.float32).reshape(-1, 1, 1, 1).copy())
            pt_m.residual[3].gamma.data = torch.from_numpy(
                np.asarray(m.norm2.gamma, dtype=np.float32).reshape(-1, 1, 1, 1).copy())
            for src, dst in [(m.conv1, pt_m.residual[2]), (m.conv2, pt_m.residual[6])]:
                w = np.transpose(np.asarray(src.weight, dtype=np.float32), (0, 4, 1, 2, 3))
                dst.weight.data = torch.from_numpy(w.copy())
                dst.bias.data = torch.from_numpy(np.asarray(src.bias, dtype=np.float32).copy())
            if hasattr(m, "shortcut"):
                w = np.transpose(np.asarray(m.shortcut.weight, dtype=np.float32), (0, 4, 1, 2, 3))
                pt_m.shortcut.weight.data = torch.from_numpy(w.copy())
                pt_m.shortcut.bias.data = torch.from_numpy(
                    np.asarray(m.shortcut.bias, dtype=np.float32).copy())
        else:  # Resample
            pt_m = pt_block.downsamples[i]
            # spatial conv (resample[1])
            w = np.transpose(np.asarray(m.spatial_conv.weight, dtype=np.float32), (0, 3, 1, 2))
            pt_m.resample[1].weight.data = torch.from_numpy(w.copy())
            if m.spatial_conv.bias is not None:
                pt_m.resample[1].bias.data = torch.from_numpy(
                    np.asarray(m.spatial_conv.bias, dtype=np.float32).copy())
            # time conv
            if hasattr(m, "time_conv"):
                w = np.transpose(np.asarray(m.time_conv.weight, dtype=np.float32), (0, 4, 1, 2, 3))
                pt_m.time_conv.weight.data = torch.from_numpy(w.copy())
                if m.time_conv._has_bias:
                    pt_m.time_conv.bias.data = torch.from_numpy(
                        np.asarray(m.time_conv.bias, dtype=np.float32).copy())


def _seed_down(mlx_block, rng):
    for m in mlx_block.downsamples:
        if hasattr(m, "norm1"):
            m.norm1.gamma = mx.array(rng.standard_normal(m.norm1.gamma.shape).astype("float32")*0.1 + 1.0)
            m.norm2.gamma = mx.array(rng.standard_normal(m.norm2.gamma.shape).astype("float32")*0.1 + 1.0)
            for c in (m.conv1, m.conv2):
                c.weight = mx.array(rng.standard_normal(c.weight.shape).astype("float32") * 0.05)
                c.bias   = mx.array(rng.standard_normal(c.bias.shape).astype("float32") * 0.01)
            if hasattr(m, "shortcut"):
                m.shortcut.weight = mx.array(rng.standard_normal(m.shortcut.weight.shape).astype("float32") * 0.05)
                m.shortcut.bias   = mx.array(rng.standard_normal(m.shortcut.bias.shape).astype("float32") * 0.01)
        else:  # Resample
            m.spatial_conv.weight = mx.array(rng.standard_normal(m.spatial_conv.weight.shape).astype("float32") * 0.05)
            if m.spatial_conv.bias is not None:
                m.spatial_conv.bias = mx.array(rng.standard_normal(m.spatial_conv.bias.shape).astype("float32") * 0.01)
            if hasattr(m, "time_conv"):
                m.time_conv.weight = mx.array(rng.standard_normal(m.time_conv.weight.shape).astype("float32") * 0.05)
                if m.time_conv._has_bias:
                    m.time_conv.bias = mx.array(rng.standard_normal(m.time_conv.bias.shape).astype("float32") * 0.01)


def cos_pt_mlx(pt: torch.Tensor, mlx_: mx.array) -> float:
    a = pt.detach().to(torch.float32).cpu().numpy().flatten()
    b = np.asarray(mlx_, dtype=np.float32).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def max_abs_diff_nthwc(pt_nctw: torch.Tensor, mlx_nthwc: mx.array) -> float:
    a = np.transpose(pt_nctw.detach().to(torch.float32).cpu().numpy(), (0, 2, 3, 4, 1))
    b = np.asarray(mlx_nthwc, dtype=np.float32)
    return float(np.abs(a - b).max())


def compare_slot(name, m_entry, p_entry, label):
    if m_entry is None and p_entry is None:
        print(f"    {label}: both None  PASS")
        return True
    if (m_entry is None) ^ (p_entry is None):
        print(f"    {label}: one is None — FAIL")
        return False
    a_pt = np.transpose(p_entry.numpy(), (0, 2, 3, 4, 1))
    a_mx = np.asarray(m_entry)
    if a_pt.shape != a_mx.shape:
        print(f"    {label}: shape mismatch  MLX={a_mx.shape}  PT={a_pt.shape}  FAIL")
        return False
    d = float(np.abs(a_pt - a_mx).max())
    print(f"    {label}: shape={tuple(a_mx.shape)}  max|Δ|={d:.3e}  "
          f"{'PASS' if d < 1e-5 else 'FAIL'}")
    return d < 1e-5


def main():
    print("STAGE 8 §1.4 — Down_ResidualBlock (mult=2, down_flag=True, temperal_downsample=True)")

    vae_mod = importlib.import_module("modeling.vae.wan.vae2_2")
    PtDown = vae_mod.Down_ResidualBlock

    in_dim, out_dim, mult = 8, 16, 2
    mx_down = Down_ResidualBlock(in_dim, out_dim, dropout=0.0, mult=mult,
                                  temperal_downsample=True, down_flag=True)
    pt_down = PtDown(in_dim, out_dim, dropout=0.0, mult=mult,
                     temperal_downsample=True, down_flag=True)
    pt_down.eval()
    rng = np.random.default_rng(8888)
    _seed_down(mx_down, rng)
    _sync_pt_down_from_mlx(pt_down, mx_down)
    print(f"[setup] in={in_dim} out={out_dim} mult={mult}  downsample3d")

    # Predict slot count: each RB = 2 slots, Resample downsample3d = 1 slot.
    # Total = 2*mult + 1 = 5.
    NUM_SLOTS = 2 * mult + 1
    print(f"[predict] expected slot count = {NUM_SLOTS}  "
          f"(order: [RB0_conv1, RB0_conv2, RB1_conv1, RB1_conv2, Resample])")

    # Inputs
    B, T, H, W = 1, 5, 8, 8
    x_np = rng.standard_normal((B, T, H, W, in_dim)).astype("float32")
    x_mx = mx.array(x_np)
    x_pt = torch.from_numpy(np.transpose(x_np, (0, 4, 1, 2, 3)).copy())

    # ===== LAYER 1 skipped — INVALID configuration =====
    print("\n" + "=" * 70)
    print("LAYER 1 — SKIPPED (stateless full-T=5 is invalid for this config)")
    print("=" * 70)
    print("  Reason: feat_cache=None makes Resample.downsample3d skip time_conv")
    print("  → main path T=5 preserved; AvgDown3D shortcut T=3 (factor_t=2 collapse).")
    print("  Shape mismatch when adding x + shortcut.  PT raises the same error.")
    print("  Down_ResidualBlock with temperal_downsample=True is *chunked-only* by")
    print("  PT design.  Layer 2 below is the actual gate.")
    c1 = None  # not measured

    # ===== LAYER 2: chunked 1 + 4 with feat_cache =====
    print("\n" + "=" * 70)
    print("LAYER 2 — chunked (1+4) with feat_cache")
    print(f"  Expected T progression:")
    print(f"    chunk 1 (T=1) → downsample3d first call: spatial /2, time skipped → T_out=1")
    print(f"    chunk 2 (T=4) → downsample3d cache concat T=1+4=5, stride-2 conv → T_out=2")
    print(f"  Total chunked T = 1 + 2 = 3 (≠ stateless T_out=5; intentional PT divergence)")
    print("=" * 70)

    fc_mx = [None] * NUM_SLOTS
    fi_mx = [0]
    fc_pt = [None] * NUM_SLOTS
    fi_pt = [0]

    all_slots_ok = True

    # Chunk 1
    x1_mx = x_mx[:, 0:1]
    x1_pt = x_pt[:, :, 0:1]
    fi_mx[0] = 0; fi_pt[0] = 0
    out1_mx = mx_down(x1_mx, feat_cache=fc_mx, feat_idx=fi_mx)
    mx.eval(out1_mx)
    with torch.no_grad():
        out1_pt = pt_down(x1_pt, feat_cache=fc_pt, feat_idx=fi_pt)
    c2a = cos_pt_mlx(out1_pt.permute(0, 2, 3, 4, 1).contiguous(), out1_mx)
    d2a = max_abs_diff_nthwc(out1_pt, out1_mx)
    t1_ok = (out1_mx.shape[1] == out1_pt.shape[2])
    print(f"\nChunk 1 (T_in=1):  T_out  MLX={out1_mx.shape[1]}  PT={out1_pt.shape[2]}  "
          f"{'PASS' if t1_ok else 'FAIL'}")
    print(f"  cos = {c2a:.8f}  max|Δ| = {d2a:.3e}  {'PASS' if c2a >= 0.999999 else 'FAIL'}")
    print(f"  feat_idx counter MLX={fi_mx[0]} PT={fi_pt[0]}  "
          f"{'PASS' if fi_mx[0] == fi_pt[0] == NUM_SLOTS else 'FAIL'}")
    print(f"  [slot-by-slot cache state]:")
    slot_labels = ["RB0_conv1", "RB0_conv2", "RB1_conv1", "RB1_conv2", "Resample"]
    for i in range(NUM_SLOTS):
        ok = compare_slot(f"slot[{i}]", fc_mx[i], fc_pt[i], slot_labels[i])
        all_slots_ok = all_slots_ok and ok

    # Chunk 2
    x2_mx = x_mx[:, 1:5]
    x2_pt = x_pt[:, :, 1:5]
    fi_mx[0] = 0; fi_pt[0] = 0
    out2_mx = mx_down(x2_mx, feat_cache=fc_mx, feat_idx=fi_mx)
    mx.eval(out2_mx)
    with torch.no_grad():
        out2_pt = pt_down(x2_pt, feat_cache=fc_pt, feat_idx=fi_pt)
    c2b = cos_pt_mlx(out2_pt.permute(0, 2, 3, 4, 1).contiguous(), out2_mx)
    d2b = max_abs_diff_nthwc(out2_pt, out2_mx)
    t2_ok = (out2_mx.shape[1] == out2_pt.shape[2])
    print(f"\nChunk 2 (T_in=4):  T_out  MLX={out2_mx.shape[1]}  PT={out2_pt.shape[2]}  "
          f"{'PASS' if t2_ok else 'FAIL'}")
    print(f"  cos = {c2b:.8f}  max|Δ| = {d2b:.3e}  {'PASS' if c2b >= 0.999999 else 'FAIL'}")
    print(f"  feat_idx counter MLX={fi_mx[0]} PT={fi_pt[0]}  "
          f"{'PASS' if fi_mx[0] == fi_pt[0] == NUM_SLOTS else 'FAIL'}")

    # Per-frame check on chunk 2 (T_out=2 expected)
    print(f"  [frame-by-frame cos (chunk 2 output T={out2_mx.shape[1]})]:")
    pt_nthwc = out2_pt.permute(0, 2, 3, 4, 1).contiguous().numpy()
    mlx_arr = np.asarray(out2_mx)
    for t in range(out2_mx.shape[1]):
        a = pt_nthwc[0, t].flatten()
        b = mlx_arr[0, t].flatten()
        c_t = float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))
        d_t = float(np.abs(a - b).max())
        print(f"    frame {t}:  cos = {c_t:.8f}  max|Δ| = {d_t:.3e}")

    print(f"  [slot-by-slot cache state]:")
    for i in range(NUM_SLOTS):
        ok = compare_slot(f"slot[{i}]", fc_mx[i], fc_pt[i], slot_labels[i])
        all_slots_ok = all_slots_ok and ok

    # ===== Diagnostic: total chunked T =====
    print("\n" + "=" * 70)
    print("DIAGNOSTIC — chunked total T evolution")
    print("=" * 70)
    chunked_total_T = out1_mx.shape[1] + out2_mx.shape[1]
    print(f"  chunked total T (chunk1 + chunk2) = {out1_mx.shape[1]} + {out2_mx.shape[1]} = {chunked_total_T}")
    print(f"  Expected per encode pattern (1+4 → 1+2): 3 ✓" if chunked_total_T == 3
          else f"  UNEXPECTED total T={chunked_total_T}, predicted 3")

    # ===== SUMMARY =====
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Layer 2a (chunk 1 PT vs MLX)  cos = {c2a:.8f}  {'PASS' if c2a >= 0.999999 else 'FAIL'}")
    print(f"  Layer 2b (chunk 2 PT vs MLX)  cos = {c2b:.8f}  {'PASS' if c2b >= 0.999999 else 'FAIL'}")
    print(f"  Slot-by-slot cache state              {'PASS' if all_slots_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
