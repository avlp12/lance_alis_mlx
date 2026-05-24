"""STAGE 8 §1.3 — ResidualBlock pass-through byte-diff vs PT.

Bottom-up assembly test.  Adds the *cross-contamination* check user
flagged: forward the whole block, then inspect `feat_cache` list
slot-by-slot against PT — counter progress + per-slot values.

PT ResidualBlock (vae2_2.py:194-229):
  residual = Sequential(norm1, silu, conv1, norm2, silu, dropout, conv2)
  shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else Identity
  Each call claims 2 feat_cache slots (conv1, conv2) in order.
  Shortcut conv (when present) is stateless 1×1×1, no slot.
"""
from __future__ import annotations

import os
import sys
import importlib

sys.path.insert(0, os.path.abspath("refs/Lance"))

import numpy as np
import torch
import mlx.core as mx

from lance_mlx.vae_wan22 import ResidualBlock, CausalConv3d


def _sync_pt_block_from_mlx(pt_block, mlx_block):
    """Copy MLX ResidualBlock weights → PT ResidualBlock.

    MLX names: norm1, conv1, norm2, conv2, [shortcut]
    PT names:  residual.0 (norm1), residual.2 (conv1),
               residual.3 (norm2), residual.6 (conv2), [shortcut]
    """
    # norm1
    pt_block.residual[0].gamma.data = torch.from_numpy(
        np.asarray(mlx_block.norm1.gamma, dtype=np.float32).reshape(-1, 1, 1, 1).copy()
    )
    # conv1
    w_mlx = np.asarray(mlx_block.conv1.weight, dtype=np.float32)   # (O, kT, kH, kW, I)
    w_pt = np.transpose(w_mlx, (0, 4, 1, 2, 3))                     # → (O, I, kT, kH, kW)
    pt_block.residual[2].weight.data = torch.from_numpy(w_pt.copy())
    pt_block.residual[2].bias.data = torch.from_numpy(
        np.asarray(mlx_block.conv1.bias, dtype=np.float32).copy()
    )
    # norm2
    pt_block.residual[3].gamma.data = torch.from_numpy(
        np.asarray(mlx_block.norm2.gamma, dtype=np.float32).reshape(-1, 1, 1, 1).copy()
    )
    # conv2
    w_mlx2 = np.asarray(mlx_block.conv2.weight, dtype=np.float32)
    w_pt2 = np.transpose(w_mlx2, (0, 4, 1, 2, 3))
    pt_block.residual[6].weight.data = torch.from_numpy(w_pt2.copy())
    pt_block.residual[6].bias.data = torch.from_numpy(
        np.asarray(mlx_block.conv2.bias, dtype=np.float32).copy()
    )
    # shortcut (CausalConv3d 1×1×1) if present
    if hasattr(mlx_block, "shortcut"):
        w_mlx_s = np.asarray(mlx_block.shortcut.weight, dtype=np.float32)
        w_pt_s = np.transpose(w_mlx_s, (0, 4, 1, 2, 3))
        pt_block.shortcut.weight.data = torch.from_numpy(w_pt_s.copy())
        pt_block.shortcut.bias.data = torch.from_numpy(
            np.asarray(mlx_block.shortcut.bias, dtype=np.float32).copy()
        )


def _seed_block(mlx_block, rng):
    # norm1.gamma, conv1.weight/bias, norm2.gamma, conv2.weight/bias, [shortcut]
    mlx_block.norm1.gamma = mx.array(rng.standard_normal(mlx_block.norm1.gamma.shape).astype("float32") * 0.1 + 1.0)
    mlx_block.norm2.gamma = mx.array(rng.standard_normal(mlx_block.norm2.gamma.shape).astype("float32") * 0.1 + 1.0)
    for c in (mlx_block.conv1, mlx_block.conv2):
        c.weight = mx.array(rng.standard_normal(c.weight.shape).astype("float32") * 0.05)
        c.bias   = mx.array(rng.standard_normal(c.bias.shape).astype("float32") * 0.01)
    if hasattr(mlx_block, "shortcut"):
        mlx_block.shortcut.weight = mx.array(
            rng.standard_normal(mlx_block.shortcut.weight.shape).astype("float32") * 0.05
        )
        mlx_block.shortcut.bias = mx.array(
            rng.standard_normal(mlx_block.shortcut.bias.shape).astype("float32") * 0.01
        )


def cos_pt_mlx(pt: torch.Tensor, mlx_: mx.array) -> float:
    a = pt.detach().to(torch.float32).cpu().numpy().flatten()
    b = np.asarray(mlx_, dtype=np.float32).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def max_abs_diff_nthwc(pt_nctw: torch.Tensor, mlx_nthwc: mx.array) -> float:
    a = np.transpose(pt_nctw.detach().to(torch.float32).cpu().numpy(), (0, 2, 3, 4, 1))
    b = np.asarray(mlx_nthwc, dtype=np.float32)
    return float(np.abs(a - b).max())


def compare_feat_cache_slot(name, mlx_cache_entry, pt_cache_entry):
    """Compare one slot of feat_cache list, PT (NCTHW) vs MLX (NTHWC)."""
    if mlx_cache_entry is None and pt_cache_entry is None:
        print(f"    {name}: both None  PASS")
        return True
    if mlx_cache_entry is None or pt_cache_entry is None:
        print(f"    {name}: one is None, other not — MLX={mlx_cache_entry}  PT={pt_cache_entry}  FAIL")
        return False
    a_pt = np.transpose(pt_cache_entry.numpy(), (0, 2, 3, 4, 1))
    a_mx = np.asarray(mlx_cache_entry)
    if a_pt.shape != a_mx.shape:
        print(f"    {name}: shape mismatch  MLX={a_mx.shape}  PT={a_pt.shape}  FAIL")
        return False
    d = float(np.abs(a_pt - a_mx).max())
    c = float(np.dot(a_pt.flatten(), a_mx.flatten()) /
              (np.linalg.norm(a_pt)*np.linalg.norm(a_mx) + 1e-12))
    print(f"    {name}: shape={tuple(a_mx.shape)}  cos={c:.8f}  max|Δ|={d:.3e}  "
          f"{'PASS' if d < 1e-5 else 'FAIL'}")
    return d < 1e-5


def run_one_case(in_dim, out_dim, label):
    print("\n" + "=" * 70)
    print(f"{label}  (in_dim={in_dim}, out_dim={out_dim})")
    print("=" * 70)

    vae_mod = importlib.import_module("modeling.vae.wan.vae2_2")
    PtBlock = vae_mod.ResidualBlock

    mx_block = ResidualBlock(in_dim, out_dim, dropout=0.0)
    pt_block = PtBlock(in_dim, out_dim, dropout=0.0)
    pt_block.eval()
    rng = np.random.default_rng(7777 + in_dim * 100 + out_dim)
    _seed_block(mx_block, rng)
    _sync_pt_block_from_mlx(pt_block, mx_block)
    has_shortcut = hasattr(mx_block, "shortcut")
    print(f"[setup] ResidualBlock weights synced; shortcut={'CausalConv3d 1×1×1' if has_shortcut else 'Identity (passthrough)'}")

    # ===== LAYER 1: stateless full T=5 with feat_cache initialised but unused-per-conv-init =====
    B, T, H, W = 1, 5, 8, 8
    x_np = rng.standard_normal((B, T, H, W, in_dim)).astype("float32")
    x_mx = mx.array(x_np)
    x_pt = torch.from_numpy(np.transpose(x_np, (0, 4, 1, 2, 3)).copy())

    # Stateless: pass feat_cache=None on both sides (no streaming)
    out_mx_stateless = mx_block(x_mx)
    mx.eval(out_mx_stateless)
    with torch.no_grad():
        out_pt_stateless = pt_block(x_pt)
    c1 = cos_pt_mlx(out_pt_stateless.permute(0, 2, 3, 4, 1).contiguous(), out_mx_stateless)
    d1 = max_abs_diff_nthwc(out_pt_stateless, out_mx_stateless)
    print(f"\nLayer 1 (stateless full T=5):")
    print(f"  cos = {c1:.8f}  max|Δ| = {d1:.3e}  "
          f"{'PASS' if c1 >= 0.999999 else 'FAIL'}")

    # ===== LAYER 2: chunked (1 + 4) with feat_cache propagation =====
    print(f"\nLayer 2 (chunked 1+4 with feat_cache):")
    # Number of CausalConv3d slots claimed by this block: 2 (conv1, conv2).
    # Shortcut not in feat_cache.
    NUM_SLOTS = 2
    fc_mx = [None] * NUM_SLOTS
    fi_mx = [0]
    fc_pt = [None] * NUM_SLOTS
    fi_pt = [0]

    # Chunk 1: T=1
    x1_mx = x_mx[:, 0:1]
    x1_pt = x_pt[:, :, 0:1]
    fi_mx[0] = 0; fi_pt[0] = 0
    out1_mx = mx_block(x1_mx, feat_cache=fc_mx, feat_idx=fi_mx)
    mx.eval(out1_mx)
    with torch.no_grad():
        out1_pt = pt_block(x1_pt, feat_cache=fc_pt, feat_idx=fi_pt)
    c2a = cos_pt_mlx(out1_pt.permute(0, 2, 3, 4, 1).contiguous(), out1_mx)
    d2a = max_abs_diff_nthwc(out1_pt, out1_mx)
    print(f"  Chunk 1 (T=1):  cos = {c2a:.8f}  max|Δ| = {d2a:.3e}  "
          f"{'PASS' if c2a >= 0.999999 else 'FAIL'}")
    print(f"  feat_idx counter:  MLX={fi_mx[0]}  PT={fi_pt[0]}  "
          f"{'PASS' if fi_mx[0] == fi_pt[0] == NUM_SLOTS else 'FAIL'}")

    # Slot-by-slot cache state inspection (the cross-contamination check)
    print(f"  [slot-by-slot cache state after chunk 1]:")
    all_slots_ok = True
    for i in range(NUM_SLOTS):
        ok = compare_feat_cache_slot(f"slot[{i}]", fc_mx[i], fc_pt[i])
        all_slots_ok = all_slots_ok and ok

    # Chunk 2: T=4 — reset feat_idx counter, cache state persists
    x2_mx = x_mx[:, 1:5]
    x2_pt = x_pt[:, :, 1:5]
    fi_mx[0] = 0; fi_pt[0] = 0
    out2_mx = mx_block(x2_mx, feat_cache=fc_mx, feat_idx=fi_mx)
    mx.eval(out2_mx)
    with torch.no_grad():
        out2_pt = pt_block(x2_pt, feat_cache=fc_pt, feat_idx=fi_pt)
    c2b = cos_pt_mlx(out2_pt.permute(0, 2, 3, 4, 1).contiguous(), out2_mx)
    d2b = max_abs_diff_nthwc(out2_pt, out2_mx)
    print(f"  Chunk 2 (T=4):  cos = {c2b:.8f}  max|Δ| = {d2b:.3e}  "
          f"{'PASS' if c2b >= 0.999999 else 'FAIL'}")
    print(f"  feat_idx counter:  MLX={fi_mx[0]}  PT={fi_pt[0]}  "
          f"{'PASS' if fi_mx[0] == fi_pt[0] == NUM_SLOTS else 'FAIL'}")

    print(f"  [slot-by-slot cache state after chunk 2]:")
    for i in range(NUM_SLOTS):
        ok = compare_feat_cache_slot(f"slot[{i}]", fc_mx[i], fc_pt[i])
        all_slots_ok = all_slots_ok and ok

    # LAYER 2d: chunked output concatenated == stateless full-T=5 PT output
    out_chunked_mx = mx.concatenate([out1_mx, out2_mx], axis=1)
    mx.eval(out_chunked_mx)
    c2d = cos_pt_mlx(out_pt_stateless.permute(0, 2, 3, 4, 1).contiguous(), out_chunked_mx)
    d2d = max_abs_diff_nthwc(out_pt_stateless, out_chunked_mx)
    print(f"\nLayer 2d (chunked MLX cat vs stateless full-T=5 PT):")
    print(f"  cos = {c2d:.8f}  max|Δ| = {d2d:.3e}  "
          f"{'PASS' if c2d >= 0.999999 else 'FAIL'}")
    print(f"  [NOTE] PT internal: chunked-vs-stateless may differ — feat_cache adds "
          f"context the stateless path doesn't have.  Reporting for diagnosis only.")

    return c1 >= 0.999999 and c2a >= 0.999999 and c2b >= 0.999999 and all_slots_ok


def main():
    print("STAGE 8 §1.3 — ResidualBlock feat_cache pass-through\n")
    ok_same = run_one_case(in_dim=8,  out_dim=8,  label="CASE A — in_dim==out_dim (Identity shortcut)")
    ok_diff = run_one_case(in_dim=8,  out_dim=16, label="CASE B — in_dim!=out_dim (CausalConv3d 1×1×1 shortcut)")
    print("\n" + "=" * 70)
    print(f"SUMMARY  case A (Identity shortcut)={'PASS' if ok_same else 'FAIL'}  "
          f"case B (CausalConv3d shortcut)={'PASS' if ok_diff else 'FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
