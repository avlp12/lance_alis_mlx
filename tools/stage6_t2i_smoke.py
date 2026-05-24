"""STAGE 6 first-image smoke: 256² t2i, 8 steps, save PNG."""
from __future__ import annotations

import time

import mlx.core as mx
import numpy as np
from PIL import Image
from transformers import AutoTokenizer

from lance_mlx.backbone import LanceLLM, LanceTextConfig, load_full_lance
from lance_mlx.pipelines.t2i import t2i
from lance_mlx.vae_wan22 import Wan2_2_VAE, Wan22VAEConfig


def main() -> None:
    print("[build] LanceLLM ...")
    cfg = LanceTextConfig()
    model = LanceLLM(cfg)
    t0 = time.time()
    stats = load_full_lance(model, "checkpoints/Lance-3B-MLX/model.safetensors")
    model.eval()
    print(f"[load] {stats['loaded_keys']} keys in {time.time()-t0:.1f}s "
          f"(language_model={stats['language_model_keys']}, adapters={stats['adapter_keys']})")

    print("[build] Wan2_2_VAE ...")
    vae = Wan2_2_VAE(Wan22VAEConfig())
    vae.load_weights(list(mx.load("checkpoints/Wan2.2-VAE-MLX/model.safetensors").items()),
                     strict=True)
    mx.eval(vae.parameters()); vae.eval()
    print(f"[load] vae OK")

    tok = AutoTokenizer.from_pretrained("checkpoints/Lance-3B-MLX", trust_remote_code=True)

    prompt = "a photo of a sunset over mountains"
    print(f"\n[t2i] prompt: {prompt!r}\n")
    out = t2i(
        model, tok, prompt,
        height=512, width=512,
        num_steps=30,
        timestep_shift=3.5,
        cfg_scale=4.0,
        seed=0,
    )
    latent = out["latent"]                  # (1, 1, 16, 16, 48)
    print(f"\n[result] latent: shape={latent.shape}  "
          f"mean={latent.mean().item():+.4f}  std={latent.std().item():.4f}  "
          f"range=[{latent.min().item():+.3f}, {latent.max().item():+.3f}]")

    print("\n[vae] decoding ...")
    t0 = time.time()
    recon = vae.decode(latent)              # (1, 1, 256, 256, 3) in [-1, 1]
    mx.eval(recon)
    print(f"[vae] decoded in {time.time()-t0:.1f}s   shape={recon.shape}")
    print(f"[vae] range=[{recon.min().item():+.3f}, {recon.max().item():+.3f}]   "
          f"mean={recon.mean().item():+.4f}  std={recon.std().item():.4f}")

    # Convert (1, 1, H, W, 3) in [-1, 1] → uint8 PNG
    img = np.asarray(recon[0, 0])
    img = (np.clip(img, -1, 1) * 0.5 + 0.5) * 255
    img = img.astype(np.uint8)
    out_path = "out/stage6_first_512.png"
    import os; os.makedirs("out", exist_ok=True)
    Image.fromarray(img).save(out_path)
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
