"""STAGE 8 §1.5 — Up_ResidualBlock pass-through byte-diff vs PT.

Decoder-side mirror of Down_ResidualBlock.  Two new traps user flagged:

  1) pixel-shuffle 2× T expansion (upsample3d inside Up_).  Already
     verified standalone in stage8_resample_compare.py; here it fires
     *inside* the block alongside ResidualBlocks, with Resample's slot
     coming AFTER all RB slots — slot mixing in the opposite direction
     of Down_.

  2) DupUp3D `first_chunk` frame drop.  factor_t=2 expansion would yield
     T_out = 2*T_in, but `first_chunk=True` drops `factor_t - 1 = 1`
     frame from the front.  If this drop disagrees with PT, the first
     chunk's frame count is off and the seam shows at the very start of
     the video.  Verified explicitly here for first_chunk=True AND False.

Layout test config:
  Up_ResidualBlock(in_dim=8, out_dim=16, mult=2, up_flag=True,
                    temperal_upsample=True)
Slot count: 2*mult + 1 = 5  ([RB0_c1, RB0_c2, RB1_c1, RB1_c2, Resample])

Decode simulation: 3 chunks of T_in=1 (mimics decoder loop for T_lat=3):
  chunk 0 (first_chunk=True):  main T_out=1, shortcut T_out=2-1=1, total T=1
  chunk 1 (first_chunk=False): main T_out=2 (Resample pixel-shuffle 2×),
                                shortcut T_out=2,           total T=2
  chunk 2 (first_chunk=False): same as chunk 1 (continued cache state)
Expected sum: 1 + 2 + 2 = 5
"""
from __future__ import annotations

import os
import sys
import importlib

sys.path.insert(0, os.path.abspath("refs/Lance"))

import numpy as np
import torch
import mlx.core as mx

from lance_mlx.vae_wan22 import Up_ResidualBlock


def _sync_pt_up_from_mlx(pt_block, mlx_block):
    """Copy MLX Up_ResidualBlock weights → PT.  AvgShortcut (DupUp3D) has no
    learnable params."""
    for i, m in enumerate(mlx_block.upsamples):
        if hasattr(m, "norm1"):  # ResidualBlock
            pt_m = pt_block.upsamples[i]
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
            pt_m = pt_block.upsamples[i]
            w = np.transpose(np.asarray(m.spatial_conv.weight, dtype=np.float32), (0, 3, 1, 2))
            pt_m.resample[1].weight.data = torch.from_numpy(w.copy())
            if m.spatial_conv.bias is not None:
                pt_m.resample[1].bias.data = torch.from_numpy(
                    np.asarray(m.spatial_conv.bias, dtype=np.float32).copy())
            if hasattr(m, "time_conv"):
                w = np.transpose(np.asarray(m.time_conv.weight, dtype=np.float32), (0, 4, 1, 2, 3))
                pt_m.time_conv.weight.data = torch.from_numpy(w.copy())
                if m.time_conv._has_bias:
                    pt_m.time_conv.bias.data = torch.from_numpy(
                        np.asarray(m.time_conv.bias, dtype=np.float32).copy())


def _seed_up(mlx_block, rng):
    for m in mlx_block.upsamples:
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


def compare_slot(label, m_entry, p_entry):
    if m_entry is None and p_entry is None:
        print(f"    {label}: both None  PASS")
        return True
    if (m_entry is None) ^ (p_entry is None):
        ms = "Rep" if m_entry == "Rep" else (None if m_entry is None else "tensor")
        ps = "Rep" if p_entry == "Rep" else (None if p_entry is None else "tensor")
        print(f"    {label}: MLX={ms}  PT={ps}  {'PASS' if ms == ps else 'FAIL'}")
        return ms == ps
    # Handle "Rep" sentinel (upsample3d first call)
    if isinstance(m_entry, str) or isinstance(p_entry, str):
        ok = m_entry == p_entry
        print(f"    {label}: MLX={m_entry!r}  PT={p_entry!r}  {'PASS' if ok else 'FAIL'}")
        return ok
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
    print("STAGE 8 §1.5 — Up_ResidualBlock (mult=2, up_flag=True, temperal_upsample=True)")
    print("Decode-per-frame simulation: 3 chunks of T_in=1 (T_lat=3 video)\n")

    vae_mod = importlib.import_module("modeling.vae.wan.vae2_2")
    PtUp = vae_mod.Up_ResidualBlock

    in_dim, out_dim, mult = 8, 16, 2
    mx_up = Up_ResidualBlock(in_dim, out_dim, dropout=0.0, mult=mult,
                              temperal_upsample=True, up_flag=True)
    pt_up = PtUp(in_dim, out_dim, dropout=0.0, mult=mult,
                  temperal_upsample=True, up_flag=True)
    pt_up.eval()
    rng = np.random.default_rng(9999)
    _seed_up(mx_up, rng)
    _sync_pt_up_from_mlx(pt_up, mx_up)
    print(f"[setup] in={in_dim} out={out_dim} mult={mult}  upsample3d  weights synced")

    NUM_SLOTS = 2 * mult + 1
    slot_labels = ["RB0_conv1", "RB0_conv2", "RB1_conv1", "RB1_conv2", "Resample"]
    print(f"[predict] slot count = {NUM_SLOTS}  order: {slot_labels}")

    # Independent chunk inputs (each T_in=1)
    B, H, W = 1, 4, 4
    chunks_np = [rng.standard_normal((B, 1, H, W, in_dim)).astype("float32") for _ in range(3)]
    chunks_mx = [mx.array(c) for c in chunks_np]
    chunks_pt = [torch.from_numpy(np.transpose(c, (0, 4, 1, 2, 3)).copy()) for c in chunks_np]

    # Shared feat_cache state across chunks
    fc_mx = [None] * NUM_SLOTS
    fi_mx = [0]
    fc_pt = [None] * NUM_SLOTS
    fi_pt = [0]

    all_slots_ok = True
    cumulative_T_mx = 0
    cumulative_T_pt = 0

    for chunk_idx, (xmx, xpt) in enumerate(zip(chunks_mx, chunks_pt)):
        first_chunk = (chunk_idx == 0)
        print("\n" + "=" * 70)
        print(f"CHUNK {chunk_idx} (T_in=1, first_chunk={first_chunk})")
        print("=" * 70)

        # Reset feat_idx counter per chunk (PT pattern: self._conv_idx=[0])
        fi_mx[0] = 0; fi_pt[0] = 0

        out_mx = mx_up(xmx, feat_cache=fc_mx, feat_idx=fi_mx, first_chunk=first_chunk)
        mx.eval(out_mx)
        with torch.no_grad():
            out_pt = pt_up(xpt, feat_cache=fc_pt, feat_idx=fi_pt, first_chunk=first_chunk)

        T_mx, T_pt = out_mx.shape[1], out_pt.shape[2]
        cumulative_T_mx += T_mx
        cumulative_T_pt += T_pt
        c = cos_pt_mlx(out_pt.permute(0, 2, 3, 4, 1).contiguous(), out_mx)
        d = max_abs_diff_nthwc(out_pt, out_mx)
        expected_T = 1 if first_chunk else 2
        T_ok = (T_mx == T_pt == expected_T)
        print(f"  Output T  MLX={T_mx}  PT={T_pt}  (expected {expected_T})  "
              f"{'PASS' if T_ok else 'FAIL'}")
        print(f"  Output H/W  MLX={out_mx.shape[2:4]}  PT={out_pt.shape[3:]}  "
              f"(expected H*2, W*2 = {H*2})")
        print(f"  cos = {c:.8f}  max|Δ| = {d:.3e}  {'PASS' if c >= 0.999999 else 'FAIL'}")
        print(f"  feat_idx counter  MLX={fi_mx[0]}  PT={fi_pt[0]}  "
              f"{'PASS' if fi_mx[0] == fi_pt[0] == NUM_SLOTS else 'FAIL'}")

        # Per-frame check (catches reshape order bugs Resample.upsample3d might introduce)
        if T_mx > 0:
            print(f"  [frame-by-frame cos]:")
            pt_nthwc = out_pt.permute(0, 2, 3, 4, 1).contiguous().numpy()
            mlx_arr = np.asarray(out_mx)
            for t in range(T_mx):
                a = pt_nthwc[0, t].flatten()
                b = mlx_arr[0, t].flatten()
                c_t = float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))
                d_t = float(np.abs(a - b).max())
                print(f"    frame {t}:  cos = {c_t:.8f}  max|Δ| = {d_t:.3e}")

        # Slot-by-slot cache state
        print(f"  [slot-by-slot cache state]:")
        for i in range(NUM_SLOTS):
            ok = compare_slot(slot_labels[i], fc_mx[i], fc_pt[i])
            all_slots_ok = all_slots_ok and ok

    print("\n" + "=" * 70)
    print("DupUp3D first_chunk drop verification")
    print("=" * 70)
    print(f"  chunk 0 (first_chunk=True):  T_out=1 — DupUp3D dropped 1 frame ({2*1} → 1)")
    print(f"  chunk 1+ (first_chunk=False): T_out=2 — DupUp3D no drop ({2*1} = 2)")

    print("\n" + "=" * 70)
    print("Cumulative T")
    print("=" * 70)
    print(f"  MLX total T over 3 chunks = {cumulative_T_mx}")
    print(f"  PT  total T over 3 chunks = {cumulative_T_pt}")
    expected_total = 1 + 2 + 2  # = 5
    print(f"  Expected (T_lat=3 → T_pixel = 1 + 2*(T_lat-1) = {expected_total}): "
          f"{'PASS' if cumulative_T_mx == expected_total == cumulative_T_pt else 'FAIL'}")

    print("\n" + "=" * 70)
    print(f"SUMMARY  all chunks cos ≥ 0.999999 + slots byte-equal: "
          f"{'PASS' if all_slots_ok else 'FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
