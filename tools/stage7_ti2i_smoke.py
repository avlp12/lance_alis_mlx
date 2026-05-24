"""STAGE 7 §3 smoke — TI2I edit on the synthetic gradient image."""
from __future__ import annotations

import time

import mlx.core as mx
import numpy as np
from PIL import Image
from transformers import AutoTokenizer

from lance_mlx.backbone import LanceLLM, LanceTextConfig, load_full_lance
from lance_mlx.vit import LanceViT, load_lance_vit
from lance_mlx.vae_wan22 import Wan2_2_VAE, Wan22VAEConfig
from lance_mlx.pipelines.image_edit import image_edit


def main():
    print("[build] LanceLLM ...")
    model = LanceLLM(LanceTextConfig())
    load_full_lance(model, "checkpoints/Lance-3B-MLX/model.safetensors")
    model.eval()
    print("[build] LanceViT ...")
    vit = LanceViT()
    load_lance_vit(vit, "checkpoints/Lance-3B-MLX/vit.safetensors")
    vit.eval()
    print("[build] Wan2_2_VAE ...")
    vae = Wan2_2_VAE(Wan22VAEConfig())
    vae.load_weights(list(mx.load("checkpoints/Wan2.2-VAE-MLX/model.safetensors").items()),
                     strict=True)
    mx.eval(vae.parameters()); vae.eval()
    print("[build] OK\n")

    tok = AutoTokenizer.from_pretrained("checkpoints/Lance-3B-MLX", trust_remote_code=True)
    import os
    IMAGE = os.environ.get("LANCE_IMG", "out/test_synthetic.png")
    INSTRUCTION = os.environ.get("LANCE_INST", "Make it more vibrant and saturated.")
    OUTNAME = os.environ.get("LANCE_OUT", "out/stage7_ti2i.png")

    print(f"[ti2i] cond_image: {IMAGE}")
    print(f"[ti2i] instruction: {INSTRUCTION}")

    t0 = time.time()
    out = image_edit(
        model, vit, vae, tok, IMAGE, INSTRUCTION,
        size=512, num_steps=30, timestep_shift=3.5,
        cfg_text=4.0, cfg_vit=1.0, seed=0,
    )
    print(f"\n[ti2i] full pipeline {time.time()-t0:.1f}s")
    print(f"[ti2i] latent shape={out.latent.shape}  mean={out.latent.mean().item():+.3f}  "
          f"std={out.latent.std().item():.3f}")
    print(f"[ti2i] image  shape={out.image_recon.shape}  range="
          f"[{out.image_recon.min().item():+.3f}, {out.image_recon.max().item():+.3f}]")

    arr = np.asarray(out.image_recon[0, 0])
    arr = (np.clip(arr * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr).save(OUTNAME)
    print(f"[save] {OUTNAME}")


if __name__ == "__main__":
    main()
