---
license: apache-2.0
language:
- en
tags:
- mlx
- vae
- video
- diffusion
- wan
- wan2.2
- apple-silicon
library_name: mlx
base_model: Wan-AI/Wan2.2-VAE
---

# Wan 2.2 VAE — MLX port

First MLX port of the Wan 2.2 video VAE. Runs natively on Apple Silicon via MLX.

This is the VAE component used by **bytedance-research/Lance** (their unified multimodal model) and originally trained by the Alibaba Wan team. Useful as a standalone for any Wan-family model or for building MLX-native video diffusion pipelines on Apple Silicon.

## What it is

A 3D causal video VAE:

| Property | Value |
|---|---|
| Latent channels (`z_dim`) | 48 |
| Spatial downsample | 16× |
| Temporal downsample | 4× |
| Input patchify | 2× |
| Encoder/decoder stages | 4 with `dim_mult=[1, 2, 4, 4]` |
| Layout | NTHWC (MLX native) |
| Size | 2.82 GB (float32, 196 tensors) |

Encoder: `(B, T, H, W, 3)` RGB in `[-1, 1]` → `(B, T', H', W', 48)` latent.
Decoder: latent → reconstructed RGB clamped to `[-1, 1]`.

## Validation

Single-image reconstruction PSNR vs the original PT checkpoint on a structured sinusoid test pattern: **37.99 dB** (target ≥ 25 dB for parity).

```
Latent: shape=(1, 1, 4, 4, 48)  mean=-0.070  std=0.699
Recon : shape=(1, 1, 64, 64, 3)  range=[-0.717, 0.735]
PSNR(input, recon) = 37.99 dB
```

## Status — v0.1.0

- **Image mode (T=1):** working, validated at 37.99 dB PSNR
- **Video streaming-cache mode (T>1):** ✅ **working** — chunked encode/decode with per-conv `feat_cache` matching the PyTorch reference. Verified at T=5 (36.17 dB overall) and T=9 (35.62 dB overall) on synthetic moving sinusoid video.

Encode pattern: first chunk is frame 0 (1 frame), then chunks of 4. T input frames → T_lat = 1 + (T-1)//4 latent frames.

Decode pattern: one latent frame at a time, expanded to 4 output frames per latent frame after the first (which is 1 frame). T_lat latent frames → T = (T_lat - 1) × 4 + 1 output frames.

## Usage

Requires `mlx >= 0.29`, `numpy`, and `einops` (used internally for tensor rearrangement).

```bash
pip install mlx numpy einops
```

Verified on M3 Ultra and M4 Studio — bit-identical reconstruction (zero diff) across both. Deterministic: same input → same output on repeat runs.

### Performance (M3 Ultra, steady-state)

Image mode (T=1):

| Output size | Encode | Decode | Peak mem |
|---|---|---|---|
| 256² | ~80 ms | ~200 ms | 7.6 GiB |
| 512² | ~260 ms | ~780 ms | 13.2 GiB |
| 768² | ~620 ms | ~2.2 s | 17.2 GiB |
| 1024² | ~1.0 s | ~3.8 s | 27.4 GiB |

Decode scales as ~pixels¹·² (near-linear).

Video mode (T>1, streaming cache, 64×64):

| T input frames | T_lat (encode) | Encode | Decode | Round-trip PSNR |
|---|---|---|---|---|
| 5 | 2 | 30 ms | 70 ms | 36.17 dB |
| 9 | 3 | ~60 ms | ~140 ms | 35.62 dB |

Streaming cache means memory stays bounded per frame regardless of T — only the prior frame's intermediate state is retained between chunks.

**Cold-start note:** the first call at each new spatial resolution pays a Metal-kernel JIT compile cost (a few seconds for 1024², trivial for ≤ 768²). Warm the pipeline once at your target size before timing or batching.

### Reconstruction quality (real photos)

| Content | Resolution | PSNR |
|---|---|---|
| Smooth/cartoon content | 512×288 | ~49 dB |
| iPhone photos (high-freq detail) | 384×512 | ~34 dB |
| Synthetic sinusoid (baseline) | 64×64 | 37.99 dB |

Round-trip stability: successive encode→decode cycles converge to a fixed point on the latent manifold (not divergent). Latent statistics across diverse inputs: mean ≈ 0, std ≈ 0.6–0.9, no collapse.

```python
import mlx.core as mx
from lance_mlx.vae_wan22 import Wan2_2_VAE  # from RockTalk/Lance-MLX (companion repo)

# Build
vae = Wan2_2_VAE(z_dim=48, c_dim=160,
                 dim_mult=(1, 2, 4, 4),
                 temperal_downsample=(False, True, True))

# Load
weights = mx.load("model.safetensors")
vae.model.load_weights(list(weights.items()), strict=True)

# Encode an image (T=1)
img = mx.array(image_array_in_minus1_to_plus1)[None, None, ...]  # (1, 1, H, W, 3)
mu, log_var = vae.encode(img)

# Decode
recon = vae.decode(mu)

# Encode a video clip (T>1) — uses streaming cache automatically
video = mx.array(video_array_in_minus1_to_plus1)[None, ...]      # (1, T, H, W, 3)
mu_v, log_var_v = vae.encode(video)                              # (1, T_lat, H/16, W/16, 48)

# Decode back to T frames (T = (T_lat - 1) * 4 + 1)
video_recon = vae.decode(mu_v)                                   # (1, T, H, W, 3)
```

## Conversion source

Converted from `bytedance-research/Lance/Wan2.2_VAE.pth` using the open-source conversion tool at https://github.com/RockTalk/Lance-MLX (`tools/convert_wan22_vae.py`).

Layout transforms applied:
- Conv weights: PT `(O, I, [T,] H, W)` → MLX `(O, [T,] H, W, I)`
- RMS_norm gamma: PT `(C, 1, 1, 1)` → MLX `(C,)`
- ResidualBlock: PT `Sequential.{0,2,3,6}` → MLX `norm1/conv1/norm2/conv2`
- Encoder/decoder head: PT `Sequential.{0,2}` → MLX `head_norm/head_conv`
- Resample 2D conv: PT `resample.1` → MLX `spatial_conv`

## License

Apache 2.0 — inherited from the upstream Wan 2.2 release.

## Acknowledgements

- Alibaba Wan team — original VAE training
- ByteDance Research — distribution as part of Lance
- This MLX port — RockTalk

## Citation

```
@misc{wan22vae_mlx,
  title  = {Wan 2.2 VAE — MLX port},
  author = {RockTalk},
  year   = {2026},
  url    = {https://huggingface.co/RockTalk/Wan2.2-VAE-MLX}
}
```
