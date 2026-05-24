"""Wan 2.2 VAE — MLX port (image path, T=1).

STAGE 5 scope: single-image encode/decode round-trip.  The 3D causal
machinery (CausalConv3d, T-axis Resample, feat_cache for streaming)
is built but the API entry points only exercise T=1 — STAGE 8 will
open the T>1 video path on top of the same module tree.

Layout (matches RockTalk standalone `Wan2.2-VAE-MLX`):

  Wan2_2_VAE
    encoder:   Encoder3d   (12 → 2·z_dim)
    conv1:     CausalConv3d(2·z_dim → 2·z_dim, k=1)   ← mu/log_var separator
    conv2:     CausalConv3d(z_dim → z_dim, k=1)        ← pre-decoder
    decoder:   Decoder3d   (z_dim → 12)

Pixel ↔ VAE-input patchify is 2× spatial: a (T,H,W,3) image becomes
(T,H/2,W/2,12) before the encoder sees it.  Together with the encoder's
8× spatial downsample this gives the advertised 16× factor.

Tensor layout is NTHWC throughout (MLX channels-last for 3D conv).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


# Streaming cache window — PT `refs/Lance/modeling/vae/wan/vae2_2.py:30`.
# Number of past frames each CausalConv3d (kT=3) needs to "see" across a
# chunk boundary.  Used at every conv site that participates in
# `feat_cache`: ResidualBlock's two residual convs, Encoder3d's conv1,
# Decoder3d's head_conv2, and Resample's time_conv siblings.
CACHE_T = 2


# ----------------------------------------------------------------------------
# Config — defaults match `RockTalk/Wan2.2-VAE-MLX/config.json`.
# ----------------------------------------------------------------------------
@dataclass
class Wan22VAEConfig:
    """Wan 2.2 VAE hyperparameters.

    Note the *asymmetric capacity* between encoder and decoder:
    `enc_dim=160, dec_dim=256`.  Do NOT unify these — the VAE was
    trained with these specific bases and the checkpoint shapes
    (e.g. `decoder.conv1.weight (1024, 3, 3, 3, 48)` = `dec_dim*4`)
    depend on the distinction.
    """
    z_dim: int = 48
    # PT WanVAE_(dim=160, dec_dim=256): encoder and decoder use *separate*
    # base channel counts.  Stage i has `dim * dim_mult[i]` channels.
    enc_dim: int = 160
    dec_dim: int = 256
    dim_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attn_scales: tuple[int, ...] = ()                 # PT default — no inline attn outside middle
    # `temperal_downsample[i]` controls whether the encoder stage-transition
    # i→i+1 also halves T.  Length = len(dim_mult) - 1 = 3.  Decoder uses
    # the reverse: `temperal_upsample = temperal_downsample[::-1]`.
    temperal_downsample: tuple[bool, ...] = (False, True, True)
    dropout: float = 0.0
    patch_size_input: int = 2                         # 2× spatial patchify before encoder
    spatial_downsample_factor: int = 16               # encoder 8× × patchify 2×
    temporal_downsample_factor: int = 4
    # `scale` (mu = (mu - scale[0]) * scale[1]) is passed at call time
    # by the diffusion pipeline, not stored here.


# ----------------------------------------------------------------------------
# CausalConv3d — 3D conv with asymmetric T-axis padding (past-only,
# never future).  For T=1 input this collapses to a regular 3D conv
# with padding only on H/W axes (T pad = 0 because asymmetric pad
# `2*pad_t` on the "before" side is meaningful only when there's
# input *after* the current frame — for a single-frame input the
# causal pad acts as just zero-padding T to length 1+2*pad_t, but
# the output T is preserved at 1 by the kernel sliding once).
#
# Implementation uses `mx.pad` on the input and a regular Conv3d
# with `padding=0` (we own all padding).  This matches PT exactly
# (`self.padding = (0,0,0); F.pad(x, self._padding)`).
# ----------------------------------------------------------------------------
class CausalConv3d(nn.Module):
    """Conv3d with asymmetric causal padding (past-only) on the T axis.

    Holds `weight (O, kT, kH, kW, I)` and `bias (O,)` directly so the
    parameter keys are flat (`conv.weight`, not `conv.conv.weight`) —
    matches RockTalk standalone Wan2.2-VAE-MLX layout.  Forward uses
    `mx.conv_general` on the manually-padded input.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size, stride=1, padding=1, bias: bool = True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        self.kernel_size = kernel_size
        self.stride = stride
        # Asymmetric causal pad on T (past-only), symmetric on H/W.
        self._pad_n = (0, 0)
        self._pad_t = (2 * padding[0], 0)
        self._pad_h = (padding[1], padding[1])
        self._pad_w = (padding[2], padding[2])
        self._pad_c = (0, 0)

        # MLX 3D conv weight layout: (O, kT, kH, kW, I).
        # Initialise small — actual values overwritten by load_weights.
        kT, kH, kW = kernel_size
        scale_init = (in_channels * kT * kH * kW) ** -0.5
        self.weight = mx.random.uniform(
            -scale_init, scale_init, (out_channels, kT, kH, kW, in_channels)
        )
        self._has_bias = bias
        if bias:
            self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array, cache_x: mx.array | None = None) -> mx.array:
        """x: (B, T, H, W, C_in) → (B, T_out, H_out, W_out, C_out).

        Streaming option for STAGE 8 video path: when `cache_x` is provided,
        prepend it on the T axis and reduce the causal "past" pad by its
        T-length.  Mirrors PT `vae2_2.py:50-58`:
          - Concat cache_x along T (PT dim=2, here axis=1 since NTHWC)
          - padding[4] (= our `_pad_t[0]`) -= cache_x.shape[T]
        For T=1 image path (STAGE 5), callers pass `cache_x=None` and
        we behave identically to before.
        """
        pad_t_before = self._pad_t[0]
        if cache_x is not None and pad_t_before > 0:
            x = mx.concatenate([cache_x, x], axis=1)
            pad_t_before = pad_t_before - cache_x.shape[1]
        x = mx.pad(x, [self._pad_n, (pad_t_before, self._pad_t[1]),
                       self._pad_h, self._pad_w, self._pad_c])
        y = mx.conv_general(x, self.weight, stride=list(self.stride))
        if self._has_bias:
            y = y + self.bias
        return y


# ----------------------------------------------------------------------------
# RMS_norm (Wan-specific, NOT the same as nn.RMSNorm).
#
# PT: `F.normalize(x, dim=axis) * scale * gamma + bias`
#   where `scale = dim**0.5`, `gamma` is learned (init 1), `bias` is
#   either learned or 0.  `F.normalize` is L2 normalization along
#   `axis` — *not* the standard RMS divide-by-rms-of-squares pattern.
#
# The MLX weight key is `gamma` (matches PT's `self.gamma`).
# Layout: NTHWC, so the normalize axis is -1 (last channel dim).
# ----------------------------------------------------------------------------
class WanRMSNorm(nn.Module):
    """L2 RMS-norm-ish: F.normalize(x, dim=-1) * sqrt(dim) * gamma."""
    def __init__(self, dim: int, has_bias: bool = False):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = mx.ones((dim,))
        if has_bias:
            self.bias = mx.zeros((dim,))
        self._has_bias = has_bias

    def __call__(self, x: mx.array) -> mx.array:
        # F.normalize equivalent: x / ||x||_2 along last axis, with eps.
        norm = mx.linalg.norm(x, axis=-1, keepdims=True)
        x = x / mx.maximum(norm, 1e-12)
        x = x * (self.scale * self.gamma)
        if self._has_bias:
            x = x + self.bias
        return x


# ----------------------------------------------------------------------------
# ResidualBlock — Wan VAE's two-conv residual with RMS norms.
#
# MLX module tree (matches RockTalk standalone weights):
#   norm1 / conv1 / norm2 / conv2  [+ optional shortcut conv if in_dim != out_dim]
# PT puts these inside an `nn.Sequential(norm1, silu, conv1, norm2, silu, dropout, conv2)`,
# saved as `residual.0/.2/.3/.6`.  RockTalk renamed to named attributes
# (cleaner).  We follow RockTalk naming.
# ----------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm1 = WanRMSNorm(in_dim)
        self.conv1 = CausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = WanRMSNorm(out_dim)
        self.conv2 = CausalConv3d(out_dim, out_dim, 3, padding=1)
        # Shortcut only when in/out dims differ (Wan VAE convention).
        if in_dim != out_dim:
            self.shortcut = CausalConv3d(in_dim, out_dim, 1, padding=0)
        # else: residual just adds x

    def __call__(self, x: mx.array,
                 feat_cache: list | None = None,
                 feat_idx: list | None = None) -> mx.array:
        """PT `vae2_2.py:213-229` line-by-line.

        - shortcut conv (when present) is 1×1×1 → causal pad 0, *stateless*.
          Does NOT participate in feat_cache.
        - The two internal CausalConv3d (conv1, conv2) each claim ONE
          feat_cache slot, in forward order.  Per-slot pattern:
            cache_x = current_conv_input[:, -CACHE_T:, ...]
            if cache_x.T < 2 and feat_cache[idx] is not None:
                cache_x = [feat_cache[idx][:, -1:, ...], cache_x] concat
            x = conv(x, cache_x=feat_cache[idx])   # use PREVIOUS slot
            feat_cache[idx] = cache_x              # store NEW slot for next call
            feat_idx[0] += 1
        """
        identity = self.shortcut(x) if hasattr(self, "shortcut") else x

        # ---- conv1 ----
        h = self.norm1(x)
        h = nn.silu(h)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = h[:, -CACHE_T:, :, :, :]
            if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
                cache_x = mx.concatenate(
                    [feat_cache[idx][:, -1:, :, :, :], cache_x], axis=1
                )
            h = self.conv1(h, cache_x=feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] = idx + 1
        else:
            h = self.conv1(h)

        # ---- conv2 ----
        h = self.norm2(h)
        h = nn.silu(h)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = h[:, -CACHE_T:, :, :, :]
            if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
                cache_x = mx.concatenate(
                    [feat_cache[idx][:, -1:, :, :, :], cache_x], axis=1
                )
            h = self.conv2(h, cache_x=feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] = idx + 1
        else:
            h = self.conv2(h)

        return identity + h


# ----------------------------------------------------------------------------
# AttentionBlock — spatial self-attention via 1×1 convs.
#
# PT keys (RockTalk MLX preserves the same names):
#   norm    : RMS_norm (channel-first in PT; channel-last in MLX NTHWC)
#   to_qkv  : 1×1 conv producing 3·C channels
#   proj    : 1×1 conv (output projection)
#
# PT reshapes (B*T, 3, C, H*W) and runs attention per-frame independently.
# For T=1 image path the per-frame loop collapses to one batch.
# ----------------------------------------------------------------------------
class AttentionBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.norm = WanRMSNorm(dim)
        # 1×1 *3D* conv (kernel=1) — MLX checkpoint has 4D weight shape
        # `(3·C, 1, 1, C)` which is the 2D conv layout `(O,kH,kW,I)`.
        # We use `nn.Conv2d` because the attention block operates per-frame
        # spatially; PT applies 1×1 Conv2d after reshape to (B*T, C, H, W).
        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, padding=0, bias=True)
        self.proj   = nn.Conv2d(dim, dim,     kernel_size=1, padding=0, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, T, H, W, C).  Returns same shape.  Per-frame spatial attn."""
        B, T, H, W, C = x.shape
        identity = x
        x = self.norm(x)                                 # (B, T, H, W, C)
        # Fold T into batch for per-frame 2D conv.
        x = x.reshape(B * T, H, W, C)                    # (B·T, H, W, C)
        qkv = self.to_qkv(x)                             # (B·T, H, W, 3·C)
        # Split q/k/v along last axis.
        q, k, v = mx.split(qkv, 3, axis=-1)              # each (B·T, H, W, C)
        # Attention: flatten H*W into sequence, single head.
        L = H * W
        q = q.reshape(B * T, L, C)
        k = k.reshape(B * T, L, C)
        v = v.reshape(B * T, L, C)
        # (B·T, 1, L, C) for sdpa with single head.
        q = q[:, None]; k = k[:, None]; v = v[:, None]
        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=C ** -0.5)
        attn = attn[:, 0].reshape(B * T, H, W, C)
        out = self.proj(attn).reshape(B, T, H, W, C)
        return identity + out


# ----------------------------------------------------------------------------
# Resample (T=1 image-mode subset) — handles 2D up/downsample.
#
# Modes:
#   "none"           — identity
#   "upsample2d"     — nearest-neighbor 2× spatial + 3×3 conv
#   "upsample3d"     — same plus a time_conv (CausalConv3d, kT=3 kH=1 kW=1, out=2·C)
#   "downsample2d"   — zero-pad (0,1,0,1) + stride-2 3×3 conv
#   "downsample3d"   — same plus a time_conv (stride=(2,1,1))
#
# For T=1 the 3D variants degenerate: time_conv on a 1-frame input keeps T=1
# (causal pad before, kernel slides once).  We still wire it because the
# checkpoint may carry time_conv weights even for T=1 mode (and STAGE 8 will
# need them).
#
# MLX checkpoint key naming:
#   spatial_conv  ← PT had `self.resample = nn.Sequential(ZeroPad, Conv2d)`
#                   (or Sequential(Upsample, Conv2d)).  RockTalk extracted
#                   the conv itself and named it `spatial_conv`.
#   time_conv     ← PT name preserved.
# ----------------------------------------------------------------------------
class Resample(nn.Module):
    def __init__(self, dim: int, mode: str):
        super().__init__()
        assert mode in ("none", "upsample2d", "upsample3d",
                        "downsample2d", "downsample3d")
        self.dim = dim
        self.mode = mode

        if mode in ("upsample2d", "upsample3d"):
            self.spatial_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)
        elif mode in ("downsample2d", "downsample3d"):
            # PT: ZeroPad2d((0,1,0,1)) then stride-2 3×3 conv with no padding.
            # We bake the (0,1,0,1) pad into the call rather than into the
            # module (Conv2d itself doesn't support asymmetric pad).
            self.spatial_conv = nn.Conv2d(dim, dim, kernel_size=3,
                                           stride=2, padding=0, bias=True)
        # else "none": no spatial_conv

        if mode == "upsample3d":
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample3d":
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1),
                                           stride=(2, 1, 1), padding=(0, 0, 0))

    def __call__(self, x: mx.array,
                 feat_cache: list | None = None,
                 feat_idx: list | None = None) -> mx.array:
        """x: (B, T, H, W, C) → resampled.

        STAGE 8 video streaming: when `feat_cache` is provided, the 3D modes
        apply `time_conv` with cache-based propagation across chunk boundaries
        (PT `vae2_2.py:121-170` line-by-line).

          - `feat_cache`: list of cached tensors / "Rep" sentinel strings /
            None, indexed by `feat_idx[0]` which counts conv layers as we
            forward through the network.  Mutated in-place by this call.
          - `feat_idx`: 1-element list holding the next-to-claim cache slot.

        For STAGE 5 image path (T=1, feat_cache=None) the 3D branches
        degenerate — spatial step only, T unchanged.  Image callers pass
        no cache and observe no behavioural change.
        """
        B, T, H, W, C = x.shape

        # ==== upsample3d time branch (PT line 123-153) ====
        if self.mode == "upsample3d" and feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                # First call into this conv ever — sentinel.  time_conv skipped.
                feat_cache[idx] = "Rep"
                feat_idx[0] = idx + 1
            else:
                # Build cache_x for the *next* call: last CACHE_T frames of
                # current input, optionally prepended with last frame of
                # previous cache (if current has <2 frames available).
                CACHE_T = 2
                cache_x = x[:, -CACHE_T:, :, :, :]
                prev = feat_cache[idx]
                if cache_x.shape[1] < 2 and prev is not None and prev != "Rep":
                    cache_x = mx.concatenate(
                        [prev[:, -1:, :, :, :], cache_x], axis=1
                    )
                if cache_x.shape[1] < 2 and prev == "Rep":
                    cache_x = mx.concatenate(
                        [mx.zeros_like(cache_x), cache_x], axis=1
                    )
                # Apply time_conv: "Rep" sentinel → no prepend, else feed prev as cache_x
                if prev == "Rep":
                    x = self.time_conv(x)
                else:
                    x = self.time_conv(x, cache_x=prev)
                feat_cache[idx] = cache_x
                feat_idx[0] = idx + 1
                # Pixel-shuffle 2× T expansion (PT line 151-153):
                #   PT: (B, 2C, T, H, W) → (B, 2, C, T, H, W) → stack dim=3
                #        → (B, C, T, 2, H, W) → reshape (B, C, T*2, H, W)
                #   MLX NTHWC equivalent:
                #     (B, T, H, W, 2C) → (B, T, H, W, 2, C)
                #                       → transpose (0,1,4,2,3,5): (B, T, 2, H, W, C)
                #                       → reshape (B, T*2, H, W, C)
                # Frame ordering: idx = t*2 + g — outer T, inner group.
                B2, T2, H2, W2, twoC = x.shape
                Cn = twoC // 2
                x = x.reshape(B2, T2, H2, W2, 2, Cn)
                x = x.transpose(0, 1, 4, 2, 3, 5)
                x = x.reshape(B2, T2 * 2, H2, W2, Cn)

        # Spatial step — fold T into batch, apply 2D conv per frame.
        if self.mode in ("upsample2d", "upsample3d"):
            B2, T2 = x.shape[0], x.shape[1]
            H_in, W_in, C_in = x.shape[2], x.shape[3], x.shape[4]
            x = mx.repeat(x, 2, axis=2)                 # H × 2
            x = mx.repeat(x, 2, axis=3)                 # W × 2
            x = x.reshape(B2 * T2, H_in * 2, W_in * 2, C_in)
            x = self.spatial_conv(x)
            x = x.reshape(B2, T2, H_in * 2, W_in * 2, C_in)
        elif self.mode in ("downsample2d", "downsample3d"):
            x = mx.pad(x, [(0, 0), (0, 0), (0, 1), (0, 1), (0, 0)])
            x = x.reshape(B * T, H + 1, W + 1, C)
            x = self.spatial_conv(x)                    # stride 2 → halves
            x = x.reshape(B, T, x.shape[1], x.shape[2], C)

        # ==== downsample3d time branch (PT line 159-170) ====
        if self.mode == "downsample3d" and feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                # First call — store input as cache, skip time_conv this chunk.
                feat_cache[idx] = x
                feat_idx[0] = idx + 1
            else:
                cache_x = x[:, -1:, :, :, :]
                x = self.time_conv(
                    mx.concatenate([feat_cache[idx][:, -1:, :, :, :], x], axis=1)
                )
                feat_cache[idx] = cache_x
                feat_idx[0] = idx + 1

        return x


# ----------------------------------------------------------------------------
# AvgDown3D / DupUp3D — average-pool / repeat-interleave shortcuts.
# For STAGE 5 (T=1) the temporal factor is 1 so these collapse to pure
# spatial pool/repeat.  Kept as classes mirroring the PT structure for
# STAGE 8 video extension.
# ----------------------------------------------------------------------------
class AvgDown3D(nn.Module):
    """PT AvgDown3D — temporal+spatial average pool with reshape-based
    channel mixing.  For T=1, factor_t collapses to 1; for image path it
    behaves as plain spatial avg pool.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 factor_t: int, factor_s: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, T, H, W, C).  Pad T to multiple of factor_t, then reshape
        to (B, T', H', W', C·factor) with PT's flatten order
        `(C, ft, fs_h, fs_w)` *C-slowest* (matches `permute(0,1,3,5,7,...)`
        in PT), and average over the `group_size` factor positions per
        output channel.

        Identity case (Wan 2.2 encoder stage 3): in_C == out_C, factor_s=1,
        factor_t=1 → factor=1, group_size = in_C / out_C = 1.  Each output
        channel pools 1 input position → identity passthrough.

        Inverted (C-fastest) flatten would silently merge input channels
        into groups instead of pooling factor positions of one channel —
        that was the STAGE 5 bug.
        """
        B, T, H, W, C = x.shape
        pad_t = (self.factor_t - T % self.factor_t) % self.factor_t
        if pad_t:
            x = mx.pad(x, [(0, 0), (pad_t, 0), (0, 0), (0, 0), (0, 0)])
            T = T + pad_t
        Tp = T // self.factor_t
        Hp = H // self.factor_s
        Wp = W // self.factor_s
        # axes: (B=0, Tp=1, ft=2, Hp=3, fs_h=4, Wp=5, fs_w=6, C=7)
        x = x.reshape(B, Tp, self.factor_t, Hp, self.factor_s, Wp, self.factor_s, C)
        # Target: (B, Tp, Hp, Wp, C, ft, fs_h, fs_w) — C slowest in the trailing flat.
        x = x.transpose(0, 1, 3, 5, 7, 2, 4, 6).reshape(B, Tp, Hp, Wp, C * self.factor)
        # Group-average pool: each out_channel pools `group_size` adjacent
        # entries (= factor positions of the same input channel).
        x = x.reshape(B, Tp, Hp, Wp, self.out_channels, self.group_size).mean(axis=-1)
        return x


class DupUp3D(nn.Module):
    """PT DupUp3D — repeat-interleave + reshape to expand a small latent
    tensor up to a larger one.  Used as an output shortcut in
    Up_ResidualBlock when avg_shortcut is wired.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 factor_t: int, factor_s: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        assert out_channels * self.factor % in_channels == 0
        self.repeats = out_channels * self.factor // in_channels

    def __call__(self, x: mx.array, first_chunk: bool = False) -> mx.array:
        # x: (B, T, H, W, C_in).  Expand channel by repeat, reshape to add
        # spatiotemporal factor, transpose, then merge into (B, T·ft, H·fs, W·fs, C_out).
        B, T, H, W, C = x.shape
        x = mx.repeat(x, self.repeats, axis=-1)            # (B, T, H, W, C·repeats)
        # Reshape: split last axis into (out_channels, factor_t, factor_s, factor_s)
        x = x.reshape(B, T, H, W,
                      self.out_channels, self.factor_t, self.factor_s, self.factor_s)
        # Move expansion axes outward of T/H/W: → (B, T, factor_t, H, factor_s, W, factor_s, out_channels)
        x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)
        x = x.reshape(B, T * self.factor_t,
                      H * self.factor_s, W * self.factor_s, self.out_channels)
        if first_chunk:
            # Drop the first (factor_t - 1) frames so the chunked encode
            # boundary aligns with PT — see PT `vae2_2.py:404`.
            x = x[:, self.factor_t - 1:]
        return x


# ----------------------------------------------------------------------------
# Down_ResidualBlock — one encoder "stage": N ResidualBlocks (possibly
# changing channel count) followed by an optional Resample.  An optional
# AvgDown3D "avg_shortcut" provides a residual bypass over the whole
# stage (Wan 2.2 only uses this when the stage actually changes shape).
#
# MLX key naming:  `downsamples` (list-attr) and optional `avg_shortcut`.
# ----------------------------------------------------------------------------
class Down_ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float,
                 mult: int, temperal_downsample: bool = False,
                 down_flag: bool = False):
        super().__init__()
        # PT *always* creates avg_shortcut, even when down_flag=False.
        # When down_flag=False the factor_s is 1 (identity spatial), and
        # the AvgDown3D collapses to a channel-only mean pool / passthrough.
        # AvgDown3D has no learnable parameters so strict-load doesn't care
        # — but the forward pass *does* need it (see PT line 439:
        # `return x + self.avg_shortcut(x_copy)`).
        self.avg_shortcut = AvgDown3D(
            in_dim, out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )

        downsamples: list[nn.Module] = []
        for i in range(mult):
            downsamples.append(
                ResidualBlock(in_dim if i == 0 else out_dim, out_dim, dropout)
            )
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            downsamples.append(Resample(out_dim, mode))
        self.downsamples = downsamples

    def __call__(self, x: mx.array,
                 feat_cache: list | None = None,
                 feat_idx: list | None = None) -> mx.array:
        """PT `vae2_2.py:434-439` — loop through downsamples passing
        feat_cache + feat_idx; avg_shortcut (AvgDown3D) is stateless.

        Slot order: each ResidualBlock claims 2 slots (conv1, conv2),
        then the trailing Resample (if down_flag) claims 1.
        """
        x_copy = x
        for m in self.downsamples:
            x = m(x, feat_cache=feat_cache, feat_idx=feat_idx)
        return x + self.avg_shortcut(x_copy)


# ----------------------------------------------------------------------------
# Up_ResidualBlock — mirror of Down_ResidualBlock.  Includes optional
# Resample at the *end* (upsample) and optional DupUp3D shortcut.
# ----------------------------------------------------------------------------
class Up_ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float,
                 mult: int, temperal_upsample: bool = False,
                 up_flag: bool = False):
        super().__init__()
        upsamples: list[nn.Module] = []
        # First mult ResidualBlocks (changes channel only on the first).
        for i in range(mult):
            upsamples.append(
                ResidualBlock(in_dim if i == 0 else out_dim, out_dim, dropout)
            )
        if up_flag:
            mode = "upsample3d" if temperal_upsample else "upsample2d"
            upsamples.append(Resample(out_dim, mode))
            self.avg_shortcut = DupUp3D(in_dim, out_dim,
                                        factor_t=2 if temperal_upsample else 1,
                                        factor_s=2)
        # else: no avg_shortcut for the last stage (no upsample needed)
        self.upsamples = upsamples

    def __call__(self, x: mx.array,
                 feat_cache: list | None = None,
                 feat_idx: list | None = None,
                 first_chunk: bool = True) -> mx.array:
        """PT `vae2_2.py:470-478`.

        - Loop through `self.upsamples` (ResidualBlocks + trailing
          Resample if up_flag) passing feat_cache + feat_idx.
        - `avg_shortcut` (DupUp3D) is stateless but takes `first_chunk`.

        `first_chunk=True` for the first (or only) chunk; STAGE 5 image
        path used the default.  STAGE 8 streaming decode passes
        `first_chunk=(i == 0)` per-latent-frame.
        """
        x_in = x
        for m in self.upsamples:
            x = m(x, feat_cache=feat_cache, feat_idx=feat_idx)
        if hasattr(self, "avg_shortcut") and self.avg_shortcut is not None:
            return x + self.avg_shortcut(x_in, first_chunk=first_chunk)
        return x


# ----------------------------------------------------------------------------
# Encoder3d / Decoder3d — the actual VAE backbones.
# ----------------------------------------------------------------------------
class Encoder3d(nn.Module):
    def __init__(self, cfg: Wan22VAEConfig, in_channels: int = 12):
        super().__init__()
        dims = [cfg.enc_dim * m for m in cfg.dim_mult]   # (160, 320, 640, 640)
        # First conv: patchified input (12 channels) → enc_dim
        self.conv1 = CausalConv3d(in_channels, dims[0], 3, padding=1)

        # Stacked Down_ResidualBlocks.  Stage i transitions to dims[i].
        downsamples: list[nn.Module] = []
        prev = dims[0]
        for i, mult_dim in enumerate(dims):
            down_flag = i != len(cfg.dim_mult) - 1
            t_down = cfg.temperal_downsample[i] if i < len(cfg.temperal_downsample) else False
            downsamples.append(
                Down_ResidualBlock(prev, mult_dim, cfg.dropout,
                                   mult=cfg.num_res_blocks,
                                   temperal_downsample=t_down,
                                   down_flag=down_flag)
            )
            prev = mult_dim
        self.downsamples = downsamples

        # Middle: Resid + Attn + Resid (all at final stage channels)
        c_mid = dims[-1]
        self.middle = [
            ResidualBlock(c_mid, c_mid, cfg.dropout),
            AttentionBlock(c_mid),
            ResidualBlock(c_mid, c_mid, cfg.dropout),
        ]

        # Head: norm + silu + conv → 2·z_dim
        self.head_norm = WanRMSNorm(c_mid)
        self.head_conv = CausalConv3d(c_mid, cfg.z_dim * 2, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv1(x)
        for stage in self.downsamples:
            x = stage(x)
        for m in self.middle:
            x = m(x)
        x = self.head_norm(x)
        x = nn.silu(x)
        x = self.head_conv(x)
        return x


class Decoder3d(nn.Module):
    def __init__(self, cfg: Wan22VAEConfig, out_channels: int = 12):
        super().__init__()
        # Decoder dim list is the reverse of encoder, but using dec_dim base.
        # PT: dims = [dec_dim * u for u in [dim_mult[-1], *dim_mult[::-1]]]
        # = [dec_dim*4, dec_dim*4, dec_dim*4, dec_dim*2, dec_dim*1]
        rev = list(cfg.dim_mult)[::-1]
        dims = [cfg.dec_dim * rev[0]] + [cfg.dec_dim * m for m in rev]
        # → [1024, 1024, 1024, 512, 256] for Wan 2.2 (dec_dim=256, dim_mult=(1,2,4,4))
        self.conv1 = CausalConv3d(cfg.z_dim, dims[0], 3, padding=1)
        c_mid = dims[0]
        self.middle = [
            ResidualBlock(c_mid, c_mid, cfg.dropout),
            AttentionBlock(c_mid),
            ResidualBlock(c_mid, c_mid, cfg.dropout),
        ]

        # Up_ResidualBlocks.  The decoder reverses temperal_downsample for upsampling.
        t_up = list(cfg.temperal_downsample)[::-1]
        upsamples: list[nn.Module] = []
        prev = dims[0]
        for i in range(len(cfg.dim_mult)):
            target = dims[i + 1]
            up_flag = i != len(cfg.dim_mult) - 1
            t_flag = t_up[i] if (i < len(t_up) and up_flag) else False
            # +1 to mult so the first stage has an extra ResidualBlock per PT.
            mult = cfg.num_res_blocks + 1
            upsamples.append(
                Up_ResidualBlock(prev, target, cfg.dropout,
                                 mult=mult,
                                 temperal_upsample=t_flag,
                                 up_flag=up_flag)
            )
            prev = target
        self.upsamples = upsamples

        c_final = dims[-1]
        self.head_norm = WanRMSNorm(c_final)
        self.head_conv = CausalConv3d(c_final, out_channels, 3, padding=1)

    def __call__(self, x: mx.array, first_chunk: bool = True) -> mx.array:
        x = self.conv1(x)
        for m in self.middle:
            x = m(x)
        for stage in self.upsamples:
            x = stage(x, first_chunk=first_chunk)
        x = self.head_norm(x)
        x = nn.silu(x)
        x = self.head_conv(x)
        return x


# ----------------------------------------------------------------------------
# Patchify / unpatchify — 2× spatial.  PT pattern is
#   patchify(x, p) : (B, T, H, W, 3) → (B, T, H/p, W/p, 3·p·p)
# implemented via reshape + transpose.
# ----------------------------------------------------------------------------
def patchify(x: mx.array, patch_size: int = 2) -> mx.array:
    """(B, T, H, W, C) → (B, T, H/p, W/p, C·p·p).

    Channel flatten order matches PT einops `b c f (h q) (w r) -> b (c r q) f h w`
    — *c slowest, then r (W-inner), then q (H-inner) fastest*.  Worked
    example for C=3, p=2 (flat length 12):

        flat_index = c * (p*p) + r * p + q

      k=0  : c=0, r=0, q=0   ← pixel (qOff=0, rOff=0) channel 0
      k=1  : c=0, r=0, q=1   ← pixel (qOff=1, rOff=0) channel 0
      k=2  : c=0, r=1, q=0   ← pixel (qOff=0, rOff=1) channel 0
      k=3  : c=0, r=1, q=1   ← pixel (qOff=1, rOff=1) channel 0
      k=4  : c=1, …          ← channel 1 starts here
      …

    The trained `encoder.conv1` weight expects this order; reversing the
    flatten direction (e.g. p_h slowest) silently breaks reconstruction
    while passing strict-load (shape matches but values are scrambled).
    """
    B, T, H, W, C = x.shape
    p = patch_size
    assert H % p == 0 and W % p == 0, f"H/W not divisible by p={p}: ({H},{W})"
    # axes after reshape: (B=0, T=1, H'=2, q=3, W'=4, r=5, C=6)
    x = x.reshape(B, T, H // p, p, W // p, p, C)
    # PT order at the flattened tail is (c, r, q) → axes (6, 5, 3).
    x = x.transpose(0, 1, 2, 4, 6, 5, 3)
    return x.reshape(B, T, H // p, W // p, C * p * p)


def unpatchify(x: mx.array, patch_size: int = 2) -> mx.array:
    """Inverse of `patchify`.  Last axis is C·p·p in (c, r, q) order."""
    B, T, Hp, Wp, Cpp = x.shape
    p = patch_size
    assert Cpp % (p * p) == 0
    C = Cpp // (p * p)
    # Reverse of the patchify transpose: last 3 axes were (C, r, q).
    x = x.reshape(B, T, Hp, Wp, C, p, p)                  # axes (0,1,2,3,4,5,6)
    # Map back to (B, T, H'=2, q=5? no... )
    # Forward transpose was (0,1,2,4,6,5,3) on axes (B,T,H',q,W',r,C).
    # Inverse permutation: axes go (0,1,2, _, 6, _, _) → need pre-image positions.
    # Easier: derive directly.  After reshape (B,T,H',W',C,r,q), put back as (B,T,H',q,W',r,C):
    #   target axes (B=0, T=1, H'=2, q=6, W'=3, r=5, C=4)
    x = x.transpose(0, 1, 2, 6, 3, 5, 4)
    return x.reshape(B, T, Hp * p, Wp * p, C)


# ----------------------------------------------------------------------------
# Wan2_2_VAE — top-level class matching RockTalk standalone weights.
#
# Module tree (from checkpoint inspection):
#   encoder : Encoder3d
#   conv1   : CausalConv3d(z_dim*2, z_dim*2, k=1)   ← mu/log_var separator
#   conv2   : CausalConv3d(z_dim,   z_dim,   k=1)   ← pre-decoder
#   decoder : Decoder3d
# ----------------------------------------------------------------------------
class Wan2_2_VAE(nn.Module):
    def __init__(self, cfg: Optional[Wan22VAEConfig] = None):
        super().__init__()
        cfg = cfg or Wan22VAEConfig()
        self.cfg = cfg
        self.encoder = Encoder3d(cfg, in_channels=12)
        self.conv1 = CausalConv3d(cfg.z_dim * 2, cfg.z_dim * 2, 1, padding=0)
        self.conv2 = CausalConv3d(cfg.z_dim,     cfg.z_dim,     1, padding=0)
        self.decoder = Decoder3d(cfg, out_channels=12)

    def encode(self, image: mx.array,
               scale: Optional[tuple[mx.array, mx.array]] = None) -> mx.array:
        """Encode a single image (or T=1 video clip) to latent.

        image: (B, H, W, 3) or (B, 1, H, W, 3), values in [-1, 1].
        Returns: (B, 1, H/16, W/16, z_dim).
        """
        if image.ndim == 4:
            image = image[:, None, :, :, :]
        x = patchify(image, patch_size=self.cfg.patch_size_input)
        enc_out = self.encoder(x)
        mu_logvar = self.conv1(enc_out)
        # Deterministic mean encode — we drop log_var.  Lance pipelines
        # use the mean only.  Stochastic VAE sampling would need
        # `log_var = mu_logvar[..., self.cfg.z_dim:]` and `eps * exp(0.5*log_var)`.
        mu = mu_logvar[..., : self.cfg.z_dim]
        if scale is not None:
            mu = (mu - scale[0]) * scale[1]
        return mu

    def decode(self, z: mx.array,
               scale: Optional[tuple[mx.array, mx.array]] = None) -> mx.array:
        """Decode latent to image.  z: (B, 1, H/16, W/16, z_dim)."""
        if z.ndim == 4:
            z = z[:, None, :, :, :]
        if scale is not None:
            z = z / scale[1] + scale[0]
        x = self.conv2(z)
        x = self.decoder(x, first_chunk=True)
        x = unpatchify(x, patch_size=self.cfg.patch_size_input)
        return x

