"""STAGE 11 video_edit — cond-VAE-encode SCALE de-blind.

The velocity gate (stage11_video_edit_verify) SHARES cond_flat to PT, so it cannot
see whether our cond-encode *scale* ((mu-mean)*inv_std) matches PT's Wan2_2_VAE
wrapper.  Here PT independently encodes the SAME preprocessed cond video and we
cos-assert against our scaled encode.  Encode only (no LLM forward) — light.

PT Wan VAE = stage8 pattern (modeling.vae.wan.vae2_2.WanVAE_ + mlx_to_pt_state),
scale=[mean, 1/std] (== t2v VAE_SCALE).  Same input fed both sides, so this checks
encode+scale, not preprocessing (which is shared, as in the velocity gate).
"""
from __future__ import annotations

import sys, importlib
sys.path.insert(0, ".")
sys.path.insert(0, "refs/Lance")

import numpy as np
import torch
import mlx.core as mx

from tools.stage5_pt_compare import mlx_to_pt_state
from lance_mlx.vae_wan22 import Wan2_2_VAE, Wan22VAEConfig
from lance_mlx.pipelines.t2v import VAE_SCALE_MEAN, VAE_SCALE_STD
from lance_mlx.pipelines.video_edit import _vae_preprocess_video

FRAMES = "out/stage11_assets/vqa01_frames.npy"
N_FRAMES = 8
H = W = 128
VAE_W = "checkpoints/Wan2.2-VAE-MLX/model.safetensors"


def _cos(a, b):
    a = np.asarray(a, np.float32).flatten(); b = np.asarray(b, np.float32).flatten()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    print("=" * 72)
    print("STAGE 11 video_edit — cond VAE encode SCALE de-blind (encode only)")
    print("=" * 72)
    clip = np.load(FRAMES)[:N_FRAMES]

    # shared preprocessed input (NTHWC [-1,1]) -> both sides
    vae_in = _vae_preprocess_video(clip, H=H, W=W)            # (1, N, H, W, 3) mx
    x_np = np.asarray(vae_in, dtype=np.float32)
    print(f"[in] cond video {x_np.shape} range=[{x_np.min():+.3f},{x_np.max():+.3f}]")

    scale_mx = (mx.array(VAE_SCALE_MEAN), mx.array(1.0 / VAE_SCALE_STD))
    scale_pt = [torch.tensor(VAE_SCALE_MEAN), 1.0 / torch.tensor(VAE_SCALE_STD)]

    # ---- ours (MLX) ----
    vae = Wan2_2_VAE(Wan22VAEConfig())
    vae.load_weights(list(mx.load(VAE_W).items()), strict=True)
    mx.eval(vae.parameters()); vae.eval()
    cond_ours = np.asarray(vae.encode(vae_in, scale=scale_mx), dtype=np.float32)   # (1,t,h,w,48)
    print(f"[ours] cond latent {cond_ours.shape} "
          f"mean={cond_ours.mean():+.4f} std={cond_ours.std():.4f}")

    # ---- PT (independent) ----
    vae_mod = importlib.import_module("modeling.vae.wan.vae2_2")
    pt_model = vae_mod.WanVAE_(
        z_dim=48, dec_dim=256, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[],
        temperal_downsample=[False, True, True], dropout=0.0)
    pt_model.eval()
    miss, unexp = pt_model.load_state_dict(mlx_to_pt_state(mx.load(VAE_W)), strict=False)
    print(f"[pt] load missing={len(miss)} unexpected={len(unexp)}")
    x_pt = torch.from_numpy(np.transpose(x_np, (0, 4, 1, 2, 3)).copy())            # NTHWC -> NCTHW
    with torch.no_grad():
        mu_pt, _ = pt_model.encode(x_pt, scale_pt)
    cond_pt = mu_pt.permute(0, 2, 3, 4, 1).to(torch.float32).cpu().numpy()         # -> NTHWC
    print(f"[pt]  cond latent {cond_pt.shape} "
          f"mean={cond_pt.mean():+.4f} std={cond_pt.std():.4f}")

    # ---- compare (scaled) + identity-scale control ----
    cos_scaled = _cos(cond_pt, cond_ours)
    cond_ours_noscale = np.asarray(vae.encode(vae_in), dtype=np.float32)
    cos_ctrl = _cos(cond_pt, cond_ours_noscale)                                    # PT(scaled) vs ours(no scale)
    print("\n" + "-" * 72)
    print(f"  cos(PT scaled, ours scaled)       = {cos_scaled:.6f}  "
          f"{'PASS' if cos_scaled >= 0.999 else 'FAIL'}")
    print(f"  cos(PT scaled, ours NO-scale ctrl) = {cos_ctrl:.6f}  "
          f"(should be < scaled if scale matters)")
    ok = cos_scaled >= 0.999 and cond_pt.shape == cond_ours.shape
    print("=" * 72)
    print(f"GATE cond-scale de-blind: {'PASS' if ok else 'FAIL'}  "
          f"(cond encode scale matches PT: {ok})")
    print("=" * 72)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
