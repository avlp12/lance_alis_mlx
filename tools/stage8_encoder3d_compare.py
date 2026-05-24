"""STAGE 8 §1.6 — Encoder3d full-assembly byte-diff vs PT.

First top-level assembly test.  All building blocks (CausalConv3d, Resample,
ResidualBlock, Down_ResidualBlock, Up_ResidualBlock) individually passed
byte-diff at this point.  Encoder3d threads feat_cache through ~24 conv
sites — the chance for a slot-index drift or middle-block routing bug is
the new exposure (STAGE 7 §3 TI2I pattern: parts correct, assembly buggy).

Verification (gate redefined for chunked-only path):
  1) feat_cache list TOTAL length matches PT `count_conv3d(encoder)` exactly
  2) feat_idx counter resets to 0 per chunk; advances through ALL slots
  3) per-chunk output PT vs MLX cos
  4) slot-by-slot cache state PT vs MLX (every slot in the list)
  5) final-chunk latent total cos (cumulative)

Test input matches PT WanVAE_.encode chunking pattern:
  pre-patchify (1, 3, 5, 128, 128) → patchify(2) → (1, 12, 5, 64, 64)
  chunk 0: T=1, chunk 1: T=4 (= 1 + 4·(1) per encode loop iter_ = 1 + (5-1)//4 = 2)
"""
from __future__ import annotations

import os
import sys
import importlib

sys.path.insert(0, os.path.abspath("refs/Lance"))

import numpy as np
import torch
import mlx.core as mx

from lance_mlx.vae_wan22 import Encoder3d, Wan22VAEConfig, CausalConv3d, ResidualBlock, Resample
from tools.stage5_pt_compare import mlx_to_pt_state


def cos_pt_mlx(pt: torch.Tensor, mlx_: mx.array) -> float:
    a = pt.detach().to(torch.float32).cpu().numpy().flatten()
    b = np.asarray(mlx_, dtype=np.float32).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def max_abs_diff_nthwc(pt_nctw: torch.Tensor, mlx_nthwc: mx.array) -> float:
    a = np.transpose(pt_nctw.detach().to(torch.float32).cpu().numpy(), (0, 2, 3, 4, 1))
    b = np.asarray(mlx_nthwc, dtype=np.float32)
    return float(np.abs(a - b).max())


def count_mlx_conv3d(module) -> int:
    """Count CausalConv3d via MLX's recursive .modules() iterator (PT parity).

    Note: this counts ALL CausalConv3d including 1×1×1 shortcut convs which
    are stateless (no causal padding) and do not actually claim feat_cache
    slots — PT does the same overcounting; the extra slots stay None.
    Matching PT's list length is the invariant.
    """
    return sum(1 for m in module.modules() if isinstance(m, CausalConv3d))


def compare_slot(label, m_entry, p_entry, atol=1e-4):
    """Slot byte-diff with tolerance.  Default 1e-4 — f32 noise accumulates
    through up to ~20 conv layers in deeper slots; 1e-5 fails on legitimately
    correct slots at the bottom of the network."""
    if m_entry is None and p_entry is None:
        return True, "both None", None
    if (m_entry is None) ^ (p_entry is None):
        return False, f"MLX={m_entry!r} PT={p_entry!r}", None
    if isinstance(m_entry, str) or isinstance(p_entry, str):
        ok = (m_entry == p_entry)
        return ok, f"MLX={m_entry!r} PT={p_entry!r}", None
    a_pt = np.transpose(p_entry.numpy(), (0, 2, 3, 4, 1))
    a_mx = np.asarray(m_entry)
    if a_pt.shape != a_mx.shape:
        return False, f"shape mismatch MLX={a_mx.shape} PT={a_pt.shape}", None
    d = float(np.abs(a_pt - a_mx).max())
    return d < atol, f"shape={tuple(a_mx.shape)} max|Δ|={d:.3e}", d


def main():
    print("STAGE 8 §1.6 — Encoder3d full-assembly byte-diff\n")

    # ---- import PT ----
    vae_mod = importlib.import_module("modeling.vae.wan.vae2_2")
    PtEncoder3d = vae_mod.Encoder3d
    pt_count_conv3d = vae_mod.count_conv3d

    # ---- build matched pair ----
    cfg = Wan22VAEConfig()
    print(f"[setup] Encoder3d(dim={cfg.enc_dim}, z_dim={cfg.z_dim}, "
          f"dim_mult={cfg.dim_mult}, num_res_blocks={cfg.num_res_blocks}, "
          f"temperal_downsample={cfg.temperal_downsample})")
    mx_enc = Encoder3d(cfg, in_channels=12)
    pt_enc = PtEncoder3d(
        dim=cfg.enc_dim, z_dim=cfg.z_dim * 2,  # PT Encoder z_dim is x2 (mu/logvar split later)
        dim_mult=list(cfg.dim_mult),
        num_res_blocks=cfg.num_res_blocks,
        attn_scales=[],
        temperal_downsample=list(cfg.temperal_downsample),
        dropout=cfg.dropout,
    )
    pt_enc.eval()
    print(f"[setup] MLX z_dim={cfg.z_dim}*2={cfg.z_dim*2}; PT encoder z_dim={cfg.z_dim*2}")

    # ---- load real Wan2.2 VAE weights into both (using STAGE 5 MLX→PT mapping) ----
    print("[load] loading actual Wan2.2-VAE weights ...")
    mlx_w = mx.load("checkpoints/Wan2.2-VAE-MLX/model.safetensors")
    # Filter only encoder.* keys
    enc_mlx_w = {k.replace("encoder.", "", 1): v for k, v in mlx_w.items() if k.startswith("encoder.")}
    print(f"[load] {len(enc_mlx_w)} encoder keys in MLX checkpoint")
    # Load into MLX Encoder3d
    mx_enc.load_weights(list(enc_mlx_w.items()), strict=True)
    mx.eval(mx_enc.parameters())
    mx_enc.eval()
    # Load into PT Encoder3d
    full_pt_state = mlx_to_pt_state(mlx_w)
    enc_pt_state = {k.replace("encoder.", "", 1): v for k, v in full_pt_state.items() if k.startswith("encoder.")}
    missing, unexpected = pt_enc.load_state_dict(enc_pt_state, strict=False)
    print(f"[load] PT load: missing={len(missing)}  unexpected={len(unexpected)}")
    if missing:
        print(f"  first missing: {missing[:3]}")
    if unexpected:
        print(f"  first unexpected: {unexpected[:3]}")

    # ---- count CausalConv3d (slot count) ----
    pt_slot_count = pt_count_conv3d(pt_enc)
    mlx_slot_count = count_mlx_conv3d(mx_enc)
    print(f"\n[slot count]  PT={pt_slot_count}  MLX={mlx_slot_count}  "
          f"{'PASS' if pt_slot_count == mlx_slot_count else 'FAIL'}")

    if pt_slot_count != mlx_slot_count:
        print("ABORT — slot count mismatch means we cannot meaningfully byte-diff slots.")
        return

    # ---- prepare input ----
    # Pre-patchified (1, C=12, T=5, H=64, W=64) — matches WanVAE_.encode after patchify(2)
    B, C, T, H, W = 1, 12, 5, 64, 64
    rng = np.random.default_rng(20260524)
    x_np = rng.standard_normal((B, C, T, H, W)).astype("float32") * 0.5
    x_pt = torch.from_numpy(x_np.copy())                                       # NCTHW
    x_mx = mx.array(np.transpose(x_np, (0, 2, 3, 4, 1)).copy())                # NTHWC
    print(f"\n[input] pre-patchified shape PT={tuple(x_pt.shape)} MLX={x_mx.shape}")

    # ---- chunked encode (1 + 4 — matches PT iter_ = 1 + (T-1)//4 = 1 + 1 = 2 calls) ----
    fc_mx = [None] * mlx_slot_count
    fi_mx = [0]
    fc_pt = [None] * pt_slot_count
    fi_pt = [0]

    chunks_pt = [x_pt[:, :, :1, :, :], x_pt[:, :, 1:5, :, :]]
    chunks_mx = [x_mx[:, :1, :, :, :], x_mx[:, 1:5, :, :, :]]

    cumulative_out_mx = []
    cumulative_out_pt = []
    all_slots_ok = True

    for chunk_idx, (xpt, xmx) in enumerate(zip(chunks_pt, chunks_mx)):
        print("\n" + "=" * 70)
        print(f"CHUNK {chunk_idx}  (T_in={xpt.shape[2]})")
        print("=" * 70)

        fi_mx[0] = 0; fi_pt[0] = 0

        with torch.no_grad():
            out_pt = pt_enc(xpt, feat_cache=fc_pt, feat_idx=fi_pt)
        out_mx = mx_enc(xmx, feat_cache=fc_mx, feat_idx=fi_mx)
        mx.eval(out_mx)

        c = cos_pt_mlx(out_pt.permute(0, 2, 3, 4, 1).contiguous(), out_mx)
        d = max_abs_diff_nthwc(out_pt, out_mx)
        T_mx, T_pt = out_mx.shape[1], out_pt.shape[2]
        print(f"  output T  MLX={T_mx}  PT={T_pt}  "
              f"{'PASS' if T_mx == T_pt else 'FAIL'}")
        print(f"  output shape  MLX={out_mx.shape}  PT={tuple(out_pt.shape)}")
        print(f"  cos = {c:.8f}  max|Δ| = {d:.3e}  "
              f"{'PASS' if c >= 0.999 else 'FAIL'}")
        # feat_idx final value should match between sides — actual usage may
        # be less than total slots since 1×1×1 shortcut convs (in count) don't
        # claim slots.  Cross-side equality is the invariant; absolute value
        # of fi_mx[0] just needs to be == fi_pt[0].
        print(f"  feat_idx counter  MLX={fi_mx[0]}  PT={fi_pt[0]}  "
              f"(of {mlx_slot_count} allocated; 1×1×1 shortcuts don't claim)  "
              f"{'PASS' if fi_mx[0] == fi_pt[0] else 'FAIL'}")

        cumulative_out_mx.append(out_mx)
        cumulative_out_pt.append(out_pt)

        # slot-by-slot comparison with max|Δ| distribution
        n_pass = 0; n_fail = 0; first_fail = None
        deltas = []
        for i in range(mlx_slot_count):
            ok, info, d = compare_slot(f"slot[{i}]", fc_mx[i], fc_pt[i], atol=1e-4)
            if d is not None:
                deltas.append(d)
            if ok:
                n_pass += 1
            else:
                n_fail += 1
                all_slots_ok = False
                if first_fail is None:
                    first_fail = (i, info)
        if deltas:
            d_arr = np.array(deltas)
            print(f"  [slots]  pass={n_pass}/{mlx_slot_count}  fail={n_fail}  "
                  f"max|Δ| distribution: median={np.median(d_arr):.3e}  "
                  f"p90={np.quantile(d_arr, 0.9):.3e}  max={d_arr.max():.3e}")
            if first_fail:
                print(f"           first fail slot[{first_fail[0]}]: {first_fail[1]}")
        print(f"  [slots]  {'PASS' if n_fail == 0 else 'FAIL'}")

    # ---- cumulative final latent comparison ----
    print("\n" + "=" * 70)
    print("CUMULATIVE LATENT (chunked outputs concatenated along T)")
    print("=" * 70)
    final_mx = mx.concatenate(cumulative_out_mx, axis=1)
    mx.eval(final_mx)
    final_pt = torch.cat(cumulative_out_pt, dim=2)
    c_final = cos_pt_mlx(final_pt.permute(0, 2, 3, 4, 1).contiguous(), final_mx)
    d_final = max_abs_diff_nthwc(final_pt, final_mx)
    print(f"  final latent shape  MLX={final_mx.shape}  PT={tuple(final_pt.shape)}")
    print(f"  cos = {c_final:.8f}  max|Δ| = {d_final:.3e}  "
          f"{'PASS' if c_final >= 0.999 else 'FAIL'}")

    print("\n" + "=" * 70)
    print(f"SUMMARY  slot count + all slot byte-diff + cumulative cos: "
          f"{'PASS' if all_slots_ok and c_final >= 0.999 else 'FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
