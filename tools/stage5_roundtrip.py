"""STAGE 5 verification: Wan 2.2 VAE encode→decode round-trip MSE ≤ 1e-3."""
from __future__ import annotations

import time

import mlx.core as mx
import numpy as np

from lance_mlx.vae_wan22 import Wan2_2_VAE, Wan22VAEConfig


def _cosine(a: mx.array, b: mx.array) -> float:
    af = a.flatten().astype(mx.float32)
    bf = b.flatten().astype(mx.float32)
    return float((mx.sum(af * bf) / (mx.linalg.norm(af) * mx.linalg.norm(bf) + 1e-12)).item())


def main() -> None:
    print("[build] Wan2_2_VAE ...")
    t0 = time.time()
    m = Wan2_2_VAE(Wan22VAEConfig())
    w = mx.load("checkpoints/Wan2.2-VAE-MLX/model.safetensors")
    m.load_weights(list(w.items()), strict=True)
    mx.eval(m.parameters())
    m.eval()
    print(f"[load] {len(w)} keys strict-loaded in {time.time()-t0:.1f}s")

    rng = np.random.default_rng(0)
    for H in (64, 128, 256):
        W = H
        print(f"\n=== single image {H}×{W} ===")
        # Build a slow-frequency sinusoid (period > spatial_downsample_factor=16).
        # Higher freqs lose info under 16× downsampling — RockTalk reports
        # 37.99 dB on a similar slow pattern.
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        img = np.stack([
            np.sin(2 * np.pi * yy / 64),
            np.cos(2 * np.pi * xx / 64),
            np.sin(2 * np.pi * (yy + xx) / 48),
        ], axis=-1).astype(np.float32) * 0.7
        img = (img + rng.standard_normal((H, W, 3), dtype=np.float32) * 0.01).clip(-1, 1)
        x = mx.array(img[None, ...], dtype=mx.float32)               # (1, H, W, 3)

        t0 = time.time()
        z = m.encode(x)
        mx.eval(z)
        t_enc = time.time() - t0

        t0 = time.time()
        recon = m.decode(z)
        mx.eval(recon)
        t_dec = time.time() - t0

        recon_img = recon[:, 0]                                       # (1, H, W, 3)
        diff = float(mx.abs(recon_img - x).max().item())
        mse  = float(((recon_img - x) ** 2).mean().item())
        cos  = _cosine(recon_img, x)
        # Per-pixel range
        print(f"  z shape={tuple(z.shape)}  z mean={z.mean().item():+.3f} std={z.std().item():.3f}")
        print(f"  recon range=[{recon_img.min().item():+.3f}, {recon_img.max().item():+.3f}]")
        print(f"  MSE = {mse:.5f}   max|Δ| = {diff:.4f}   cos = {cos:.6f}")
        print(f"  encode {t_enc*1000:.0f}ms   decode {t_dec*1000:.0f}ms")
        # PSNR (input range 2.0 since [-1,1]) — same metric RockTalk reports.
        psnr = 10 * np.log10(4.0 / max(mse, 1e-12))
        print(f"  PSNR ≈ {psnr:.2f} dB")
        ok = mse <= 1e-3
        print(f"  {'PASS' if ok else 'FAIL'} (criterion MSE ≤ 1e-3)")


if __name__ == "__main__":
    main()
