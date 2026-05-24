"""Diagnostic: convert MLX VAE weights back to PT layout, run PT forward,
compare with our MLX forward.  Settles whether the bug is in our impl or
in the way we load the checkpoint."""
from __future__ import annotations

import sys
sys.path.insert(0, "refs/Lance")           # so we can `from modeling.vae.wan...`

import mlx.core as mx
import numpy as np
import torch

from modeling.vae.wan.vae2_2 import WanVAE_

from lance_mlx.vae_wan22 import Wan2_2_VAE, Wan22VAEConfig


def mlx_to_pt_state(mlx_w: dict) -> dict:
    """Map RockTalk MLX VAE keys + shapes to PT keys + shapes.

    Conv weight layout: MLX (O, [T,] H, W, I) → PT (O, I, [T,] H, W).
    Module rename: MLX named attributes (conv1, conv2, norm1, norm2) →
                   PT nn.Sequential indices (residual.0/.2/.3/.6).
    """
    out = {}
    for k, v in mlx_w.items():
        # Permute conv weights from MLX to PT layout.
        arr = np.asarray(v)
        if "weight" in k and ("conv" in k or "shortcut" in k or "to_qkv" in k or "proj" in k or "spatial_conv" in k or "time_conv" in k):
            if arr.ndim == 5:                              # 3D conv: O T H W I → O I T H W
                arr = np.transpose(arr, (0, 4, 1, 2, 3))
            elif arr.ndim == 4:                            # 2D conv: O H W I → O I H W
                arr = np.transpose(arr, (0, 3, 1, 2))
        # gamma → gamma but PT shape (C, 1, 1) for images=True or (C, 1, 1, 1) for images=False
        # For our purpose, PT does broadcast just fine if we pass (C,) too; PyTorch
        # nn.Parameter accepts any shape if we set .data shape-compatibly.  Easier:
        # reshape gamma based on PT's expected shape from the parent module.
        if "gamma" in k:
            # PT keys are like "residual.0.gamma" with shape (C, 1, 1, 1).
            # We'll always upcast to (C, 1, 1, 1) for ResidualBlock.RMS_norm(images=False)
            # and (C, 1, 1) for AttentionBlock.norm (images=True).  Detect via parent path.
            if "middle.1.norm" in k:                       # AttentionBlock: images=True
                arr = arr.reshape(-1, 1, 1)
            else:                                          # all other RMS_norm: images=False
                arr = arr.reshape(-1, 1, 1, 1)

        # PT module name mapping (RockTalk MLX renamed Sequential children).
        # ResidualBlock: norm1 / conv1 / norm2 / conv2  →  residual.0 / residual.2 / residual.3 / residual.6
        pt_k = (
            k
            .replace("downsamples.0.norm1", "downsamples.0.residual.0")
            .replace("downsamples.0.conv1", "downsamples.0.residual.2")
            .replace("downsamples.0.norm2", "downsamples.0.residual.3")
            .replace("downsamples.0.conv2", "downsamples.0.residual.6")
            .replace("downsamples.1.norm1", "downsamples.1.residual.0")
            .replace("downsamples.1.conv1", "downsamples.1.residual.2")
            .replace("downsamples.1.norm2", "downsamples.1.residual.3")
            .replace("downsamples.1.conv2", "downsamples.1.residual.6")
            .replace("downsamples.2.norm1", "downsamples.2.residual.0")
            .replace("downsamples.2.conv1", "downsamples.2.residual.2")
            .replace("downsamples.2.norm2", "downsamples.2.residual.3")
            .replace("downsamples.2.conv2", "downsamples.2.residual.6")
            .replace("middle.0.norm1", "middle.0.residual.0")
            .replace("middle.0.conv1", "middle.0.residual.2")
            .replace("middle.0.norm2", "middle.0.residual.3")
            .replace("middle.0.conv2", "middle.0.residual.6")
            .replace("middle.2.norm1", "middle.2.residual.0")
            .replace("middle.2.conv1", "middle.2.residual.2")
            .replace("middle.2.norm2", "middle.2.residual.3")
            .replace("middle.2.conv2", "middle.2.residual.6")
            # Up_ResidualBlock has up to 4 sub-items
            .replace("upsamples.0.norm1", "upsamples.0.residual.0")
            .replace("upsamples.0.conv1", "upsamples.0.residual.2")
            .replace("upsamples.0.norm2", "upsamples.0.residual.3")
            .replace("upsamples.0.conv2", "upsamples.0.residual.6")
            .replace("upsamples.1.norm1", "upsamples.1.residual.0")
            .replace("upsamples.1.conv1", "upsamples.1.residual.2")
            .replace("upsamples.1.norm2", "upsamples.1.residual.3")
            .replace("upsamples.1.conv2", "upsamples.1.residual.6")
            .replace("upsamples.2.norm1", "upsamples.2.residual.0")
            .replace("upsamples.2.conv1", "upsamples.2.residual.2")
            .replace("upsamples.2.norm2", "upsamples.2.residual.3")
            .replace("upsamples.2.conv2", "upsamples.2.residual.6")
            # Resample.spatial_conv is at .resample.1 in PT (after ZeroPad or Upsample at index 0).
            .replace("spatial_conv", "resample.1")
            # head_norm / head_conv → head.0 / head.2
            .replace("head_norm", "head.0")
            .replace("head_conv", "head.2")
        )
        out[pt_k] = torch.from_numpy(arr.copy())
    return out


def main():
    mlx_w = mx.load("checkpoints/Wan2.2-VAE-MLX/model.safetensors")
    print(f"MLX VAE: {len(mlx_w)} keys")

    pt_state = mlx_to_pt_state(mlx_w)
    print(f"after conversion: {len(pt_state)} PT keys")

    pt_model = WanVAE_(dim=160, dec_dim=256, z_dim=48,
                       dim_mult=[1, 2, 4, 4],
                       temperal_downsample=[False, True, True])
    pt_model.eval()
    missing, unexpected = pt_model.load_state_dict(pt_state, strict=False)
    print(f"PT load: missing={len(missing)}, unexpected={len(unexpected)}")
    if missing[:5]:
        print(f"  first missing: {missing[:5]}")
    if unexpected[:5]:
        print(f"  first unexpected: {unexpected[:5]}")

    # Same input on both sides
    np.random.seed(0)
    img_np = np.random.uniform(-0.5, 0.5, size=(1, 3, 1, 64, 64)).astype(np.float32)
    pt_img = torch.from_numpy(img_np)

    with torch.no_grad():
        pt_mu, pt_logvar = pt_model.encode(pt_img, scale=[0, 1])
    pt_mu_np = pt_mu.cpu().numpy()                # (1, 48, T', 4, 4)
    print(f"PT mu: shape={pt_mu_np.shape}  mean={pt_mu_np.mean():+.4f}  std={pt_mu_np.std():.4f}")

    # Reorder PT mu (1, 48, T', H, W) → MLX (1, T', H, W, 48) for comparison
    pt_mu_mlx = np.transpose(pt_mu_np, (0, 2, 3, 4, 1))

    # MLX side
    mlx_model = Wan2_2_VAE(Wan22VAEConfig())
    mlx_model.load_weights(list(mlx_w.items()), strict=True)
    mx.eval(mlx_model.parameters()); mlx_model.eval()

    # Convert PT input to MLX NTHWC
    img_mlx = mx.array(np.transpose(img_np, (0, 2, 3, 4, 1)))     # (1, 1, 64, 64, 3)
    mlx_mu = mlx_model.encode(img_mlx)
    mlx_mu_np = np.asarray(mlx_mu)
    print(f"MLX mu: shape={mlx_mu_np.shape}  mean={mlx_mu_np.mean():+.4f}  std={mlx_mu_np.std():.4f}")

    # Compare
    diff = np.abs(pt_mu_mlx - mlx_mu_np).max()
    af = pt_mu_mlx.flatten().astype(np.float64)
    bf = mlx_mu_np.flatten().astype(np.float64)
    cos = float((af @ bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))
    print(f"\nMLX mu vs PT mu: cos={cos:.6f}  max|Δ|={diff:.4e}")


if __name__ == "__main__":
    main()
