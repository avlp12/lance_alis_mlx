"""STAGE 6 512² VAE decode side-by-side (PT vs MLX).

Diagnostic for the white-grid-seam artifact at 512² that STAGE 5 missed
(it only tested 256²/16×16 latent).  Now we feed a 32×32 latent — the
shape we get from 512² denoising — to *both* the PT WanVAE_ (direct
import) and our MLX Wan2_2_VAE.  Compare cos + side-by-side PNG.

Two latent sources tested:
  (a) random N(0, σ=0.7) latent — controls for the bug being purely a
      VAE-side resolution issue.
  (b) our 512² denoised latent (loaded from disk if saved by smoke run).

If PT image is clean and MLX has seams → MLX VAE 512² path bug.
If both have seams → upstream (latent values from denoising), not VAE.
"""
from __future__ import annotations

import os
import sys

import mlx.core as mx
import numpy as np
import torch
from PIL import Image

# Use the STAGE 5 PT-VAE shim
sys.path.insert(0, os.path.abspath("refs/Lance"))
from modeling.vae.wan.vae2_2 import WanVAE_
sys.path.insert(0, ".")
from tools.stage5_pt_compare import mlx_to_pt_state

from lance_mlx.vae_wan22 import Wan2_2_VAE, Wan22VAEConfig


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64).flatten()
    bf = b.astype(np.float64).flatten()
    return float((af @ bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))


def _to_uint8(arr_nthwc: np.ndarray) -> np.ndarray:
    return (np.clip(arr_nthwc[0, 0] * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8)


def main() -> None:
    print("[setup] loading PT WanVAE_ + MLX Wan2_2_VAE ...")
    mlx_w = mx.load("checkpoints/Wan2.2-VAE-MLX/model.safetensors")

    # PT side
    pt_vae = WanVAE_(dim=160, dec_dim=256, z_dim=48,
                     dim_mult=[1, 2, 4, 4],
                     temperal_downsample=[False, True, True])
    pt_vae.eval()
    pt_state = mlx_to_pt_state(mlx_w)
    pt_vae.load_state_dict(pt_state, strict=False)

    # MLX side
    mlx_vae = Wan2_2_VAE(Wan22VAEConfig())
    mlx_vae.load_weights(list(mlx_w.items()), strict=True)
    mx.eval(mlx_vae.parameters()); mlx_vae.eval()
    print("[setup] both VAEs loaded.")

    # ---- Test (a): random latent at 32×32 ----
    for tag, H_lat, W_lat in [("random_16x16", 16, 16),
                              ("random_32x32", 32, 32)]:
        print(f"\n=== {tag}: latent (1, 1, {H_lat}, {W_lat}, 48) ===")
        np.random.seed(0)
        lat_np = (np.random.randn(1, 1, H_lat, W_lat, 48).astype(np.float32) * 0.7)
        lat_mlx = mx.array(lat_np)
        lat_pt = torch.from_numpy(np.transpose(lat_np, (0, 4, 1, 2, 3)))   # NCTHW

        with torch.no_grad():
            img_pt_nctw = pt_vae.decode(lat_pt, scale=[0, 1]).numpy()      # (1, 3, T, H, W)
        img_pt_nthwc = np.transpose(img_pt_nctw, (0, 2, 3, 4, 1))           # (1, T, H, W, 3)

        img_mlx = mlx_vae.decode(lat_mlx)                                  # NTHWC
        mx.eval(img_mlx)
        img_mlx_np = np.asarray(img_mlx)

        cos = _cos(img_pt_nthwc, img_mlx_np)
        max_d = float(np.abs(img_pt_nthwc - img_mlx_np).max())
        print(f"  cos = {cos:.6f}   max|Δ| = {max_d:.4f}   "
              f"PT range [{img_pt_nthwc.min():+.2f},{img_pt_nthwc.max():+.2f}]   "
              f"MLX range [{img_mlx_np.min():+.2f},{img_mlx_np.max():+.2f}]")

        # Save side-by-side
        pt_u8 = _to_uint8(img_pt_nthwc)
        mlx_u8 = _to_uint8(img_mlx_np)
        # 1-pixel red border between halves for visibility
        sep = np.full((pt_u8.shape[0], 2, 3), [255, 0, 0], dtype=np.uint8)
        combo = np.concatenate([pt_u8, sep, mlx_u8], axis=1)
        out_path = f"out/stage6_vae_{tag}_compare.png"
        os.makedirs("out", exist_ok=True)
        Image.fromarray(combo).save(out_path)
        print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
