"""STAGE 9 §1 단계 6 closing — PT VAE decode → video pixel cos.

PT 최종 latent → PT Wan2_2_VAE decode → video pixel.  MLX video 와 cos.
교훈 12 (PT 와 같다 ≠ 작동): t2v latent 의 VAE end-to-end 는 *처음 도는 경로*,
명시적 게이트 1회.

PT VAE 의존성: refs/Lance/modeling/vae/wan/vae2_2.py:Wan2_2_VAE (STAGE 8 패턴).
Wan2.2 VAE 는 image/video 양쪽 동일 (byte-identical 확인됨, 단계 4-3 도중).

Gate: cos(PT_video_pixel, MLX_video_pixel) ≥ 0.999
"""
from __future__ import annotations

import os
import sys
import importlib
import time

sys.path.insert(0, ".")
sys.path.insert(0, os.path.abspath("refs/Lance"))

import numpy as np
import torch
import mlx.core as mx

# PT VAE import (STAGE 8 패턴)
vae_mod = importlib.import_module("modeling.vae.wan.vae2_2")
PtWanVAE_ = vae_mod.WanVAE_

# MLX → PT state 변환 (STAGE 5 stage5_pt_compare 패턴)
from tools.stage5_pt_compare import mlx_to_pt_state


def main():
    print("=" * 72)
    print("STAGE 9 §1 단계 6 closing — PT VAE decode → video pixel cos")
    print("=" * 72)

    # ---- load latents ----
    pt_latent_flat = np.load("out/stage9_pt_30step_final_x_t.npy")     # (128, 48)
    mlx_video = np.load("out/stage9_mlx_30step_video.npy")             # (1, 5, 128, 128, 3)
    print(f"[in] PT final latent: shape={pt_latent_flat.shape}")
    print(f"[in] MLX video:        shape={mlx_video.shape}  range=[{mlx_video.min():+.3f}, {mlx_video.max():+.3f}]")

    # ---- reshape PT latent: (n_video, 48) → (t, h, w, 48) → PT (B, C, T, H, W) ----
    t_lat, h_lat, w_lat = 2, 8, 8
    LATENT_CHANNEL = 48
    pt_latent_4d = pt_latent_flat.reshape(t_lat, h_lat, w_lat, LATENT_CHANNEL)        # (t, h, w, c)
    # PT Wan2_2_VAE.decode 가 받는 shape: (B, C, T, H, W)
    pt_latent_5d_pt = pt_latent_4d.transpose(3, 0, 1, 2)[None, ...]                  # (1, c, t, h, w)
    pt_latent_pt = torch.from_numpy(pt_latent_5d_pt.astype(np.float32))
    print(f"[reshape] PT latent shape for VAE decode: {tuple(pt_latent_pt.shape)}")

    # ---- build PT Wan2_2_VAE (STAGE 8 패턴) ----
    print("\n[build] PT WanVAE_ (z_dim=48, image/video 공용 Wan2.2 VAE) ...")
    t_load = time.time()
    pt_vae = PtWanVAE_(
        dim=160, dec_dim=256, z_dim=48,
        dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
    )
    pt_vae.eval()
    mlx_w = mx.load("checkpoints/Wan2.2-VAE-MLX/model.safetensors")
    pt_state = mlx_to_pt_state(mlx_w)
    missing, unexpected = pt_vae.load_state_dict(pt_state, strict=False)
    if missing:
        print(f"[load] missing keys: {len(missing)} (first 3: {missing[:3]})")
    if unexpected:
        print(f"[load] unexpected keys: {len(unexpected)}")
    print(f"[load] done in {time.time()-t_load:.1f}s")

    # ---- production scale (Wan2_2_VAE wrapper 의 mean/std) — STAGE 8 §0 코드 그대로 ----
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

    # ---- PT VAE decode ----
    print(f"\n[decode] PT VAE decode (n_video=128 → video 5×128×128×3) ...")
    t_dec = time.time()
    with torch.no_grad():
        pt_video_pt = pt_vae.decode(pt_latent_pt, scale)      # (B, C, T_pix, H, W) PyTorch layout
    print(f"[decode] done in {time.time()-t_dec:.1f}s")
    print(f"[decode] PT video shape: {tuple(pt_video_pt.shape)}")

    # Convert PT video to MLX layout (B, T, H, W, C)
    pt_video_np = pt_video_pt.cpu().numpy()                    # (1, 3, 5, 128, 128)
    pt_video_mlx_layout = pt_video_np.transpose(0, 2, 3, 4, 1)  # (1, 5, 128, 128, 3)
    print(f"[layout] PT video → MLX layout: {pt_video_mlx_layout.shape}")
    print(f"         range=[{pt_video_mlx_layout.min():+.3f}, {pt_video_mlx_layout.max():+.3f}]")

    # ---- cos comparison ----
    def cos(a, b):
        a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))

    c_total = cos(pt_video_mlx_layout, mlx_video)
    diff_abs = np.abs(pt_video_mlx_layout - mlx_video)
    print(f"\n[VIDEO PIXEL COS]")
    print(f"  cos(PT_video, MLX_video) = {c_total:.6f}   {'PASS' if c_total >= 0.999 else 'FAIL'}")
    print(f"  maxabs = {float(diff_abs.max()):.4e}")
    print(f"  mean abs diff = {float(diff_abs.mean()):.4e}")
    print(f"  p50 / p90 abs diff = {float(np.percentile(diff_abs, 50)):.4e} / {float(np.percentile(diff_abs, 90)):.4e}")
    print(f"  ||PT video||  = {float(np.linalg.norm(pt_video_mlx_layout)):.2f}")
    print(f"  ||MLX video|| = {float(np.linalg.norm(mlx_video)):.2f}")

    # Per-frame breakdown
    print(f"\n[per-frame cos]")
    for f in range(pt_video_mlx_layout.shape[1]):
        c = cos(pt_video_mlx_layout[0, f], mlx_video[0, f])
        d = float(np.abs(pt_video_mlx_layout[0, f] - mlx_video[0, f]).max())
        print(f"  frame {f}: cos={c:.6f}  maxabs={d:.4e}")

    # save
    np.save("out/stage9_pt_30step_video.npy", pt_video_mlx_layout)
    print(f"\n[save] out/stage9_pt_30step_video.npy  shape={pt_video_mlx_layout.shape}")

    print("\n" + "=" * 72)
    if c_total >= 0.999:
        print(f"STAGE 9 §1 단계 6 closing PASS — video pixel cos={c_total:.6f}")
        print("STAGE 9 §1 종료.  다음: STAGE 9 마무리 (LEARNING_LOG + reviewer + 회귀).")
    else:
        print(f"FAIL — video pixel cos={c_total:.6f} < 0.999")
    print("=" * 72)


if __name__ == "__main__":
    main()
