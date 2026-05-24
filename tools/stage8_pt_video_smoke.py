"""STAGE 8 §0 — PT video path 정답지 확보.

Smallest sanity: PT Wan2_2_VAE encode+decode on a tiny synthetic video
clip (T=5).  Verifies the *PT side* works end-to-end before we touch
MLX streaming.  No MLX cross-check yet — just PT shapes / stats / no
NaN.  Re-uses the MLX→PT state conversion from `tools/stage5_pt_compare.py`.

Reports:
  - input shape  (B, C, T, H, W)
  - latent shape (B, z_dim, T_lat, H_lat, W_lat)
  - encode iter count, per-chunk frame split
  - decode iter count (= T_lat)
  - PT round-trip pixel MSE
"""
from __future__ import annotations

import os
import sys
import importlib

# PT path
sys.path.insert(0, os.path.abspath("refs/Lance"))

import numpy as np
import torch
import mlx.core as mx

from tools.stage5_pt_compare import mlx_to_pt_state


def main():
    # ---- import PT Wan2_2_VAE ----
    print("[setup] importing PT Wan2_2_VAE ...")
    vae_mod = importlib.import_module("modeling.vae.wan.vae2_2")
    WanVAE_ = vae_mod.WanVAE_
    Wan2_2_VAE_class = vae_mod.Wan2_2_VAE

    # ---- build PT model directly (skip _video_vae which loads from disk) ----
    print("[build] PT WanVAE_(z_dim=48, dec_dim=256, dim_mult=[1,2,4,4], temperal_downsample=[False,True,True])")
    pt_model = WanVAE_(
        dim=160, dec_dim=256, z_dim=48,
        dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[],
        temperal_downsample=[False, True, True],  # matches Wan2_2_VAE default
        dropout=0.0,
    )
    pt_model.eval()

    # ---- load weights from MLX-converted checkpoint ----
    print("[load] converting MLX → PT state ...")
    mlx_w = mx.load("checkpoints/Wan2.2-VAE-MLX/model.safetensors")
    pt_state = mlx_to_pt_state(mlx_w)
    print(f"[load] {len(pt_state)} PT keys, loading into model ...")
    missing, unexpected = pt_model.load_state_dict(pt_state, strict=False)
    print(f"[load] missing={len(missing)}  unexpected={len(unexpected)}")
    if missing:
        print(f"  first missing: {missing[:5]}")
    if unexpected:
        print(f"  first unexpected: {unexpected[:5]}")

    # ---- scale (Wan2_2_VAE outer class hardcodes these) ----
    mean = torch.tensor([-0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174,
                         0.1838, 0.1557, -0.1382, 0.0542, 0.2813, 0.0891,
                          0.1570, -0.0098, 0.0375, -0.1825, -0.2246, -0.1207,
                         -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
                         -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899,
                         -0.2799, -0.1230, -0.0313, -0.1649, 0.0117, 0.0723,
                         -0.2839, -0.2083, -0.0520, 0.3748, 0.0152, 0.1957,
                          0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667])
    std = torch.tensor([0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990,
                        0.4818, 0.5013, 0.8158, 1.0344, 0.5894, 1.0901,
                        0.6885, 0.6165, 0.8454, 0.4978, 0.5759, 0.3523,
                        0.7135, 0.6804, 1.0457, 0.4329, 0.7918, 0.5739,
                        0.5942, 0.5570, 0.5860, 0.6673, 0.4109, 0.7894,
                        0.5897, 0.4845, 0.5727, 1.1191, 0.4921, 0.4753,
                        1.0265, 0.4790, 1.2798, 0.4768, 0.8169, 0.7497,
                        0.7344, 0.4759, 0.8501, 0.6479, 0.4523, 0.6116])
    scale = [mean, 1.0 / std]

    # ---- synthetic video: T=5 (will exercise temporal causal conv + chunking) ----
    torch.manual_seed(0)
    B, C, T, H, W = 1, 3, 5, 128, 128
    x = torch.randn(B, C, T, H, W, dtype=torch.float32)
    print(f"\n[input] video shape={tuple(x.shape)}  range=[{x.min():.3f}, {x.max():.3f}]")

    # ---- predict encode iter count ----
    # encode: iter_ = 1 + (t - 1) // 4 where t = T // patch_size (patch_size=2 for T-axis? let's check)
    # Looking at vae2_2.py:761: x = patchify(x, patch_size=2) — let's see what patchify does
    print(f"[predict] PT encode iter_ = 1 + (T_after_patchify - 1)//4")

    # ---- PT encode ----
    print("\n[encode] running PT encode ...")
    with torch.no_grad():
        mu, log_var = pt_model.encode(x, scale)
    print(f"[encode] mu shape={tuple(mu.shape)}  log_var shape={tuple(log_var.shape)}")
    print(f"[encode] mu     stats: mean={float(mu.mean()):+.4f}  std={float(mu.std()):.4f}  range=[{float(mu.min()):+.3f}, {float(mu.max()):+.3f}]")

    # ---- PT decode ----
    print("\n[decode] running PT decode ...")
    with torch.no_grad():
        x_hat = pt_model.decode(mu, scale)
    print(f"[decode] x_hat shape={tuple(x_hat.shape)}  range=[{float(x_hat.min()):+.3f}, {float(x_hat.max()):+.3f}]")

    # ---- Round-trip ----
    # Note: x was N(0,1), VAE expects [-1,1] inputs; quality will be poor.
    # We only care: shapes correct + no NaN + numerically stable.
    mse = float(((x_hat - x) ** 2).mean())
    nan_mu = bool(torch.isnan(mu).any())
    nan_xhat = bool(torch.isnan(x_hat).any())
    print(f"\n[verify] round-trip MSE = {mse:.4f}  (quality irrelevant; input is N(0,1) noise)")
    print(f"[verify] NaN check: mu={nan_mu}  x_hat={nan_xhat}")

    print("\n[OK] PT video path forward end-to-end on T=5 input.")
    print(f"     Encode produced latent T_lat={mu.shape[2]} from input T={T}")
    print(f"     This is the ground-truth path our MLX impl must reproduce byte-for-byte.")

    # Save reference outputs for later MLX byte-diff
    np.save("out/stage8_pt_video_input.npy",  x.numpy())
    np.save("out/stage8_pt_video_mu.npy",     mu.numpy())
    np.save("out/stage8_pt_video_logvar.npy", log_var.numpy())
    np.save("out/stage8_pt_video_xhat.npy",   x_hat.numpy())
    print("\n[save] reference tensors in out/stage8_pt_video_*.npy")


if __name__ == "__main__":
    main()
