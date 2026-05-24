"""Flow matching scheduler + CFG denoising helpers for Lance.

Lance uses velocity-prediction flow matching (Rectified Flow / DiT-style).
Differences from DDPM:

  - Continuous t in [0, 1] (not discrete integer steps).
  - Prediction target is **velocity** `v_t = dx_t/dt = x_1 - x_0`,
    where x_0 = data, x_1 = noise.  Sampling integrates this ODE backward
    from x_1 to x_0 via Euler steps.
  - No prediction of noise or x_0 directly; the model outputs `v_t`.

PT reference: `refs/Lance/modeling/lance/lance.py:599-726`.

Step schedule (PT line 599-602):

    timesteps = linspace(1, 0, N+1)
    timesteps = shift * timesteps / (1 + (shift - 1) * timesteps)
    dts = timesteps[:-1] - timesteps[1:]

`shift` (default 3.5 per Lance config, was 4.0 in PT default kwarg) biases
the schedule to spend more steps near t=1 (noise), where the denoiser
needs more resolution to disambiguate signal from chaos.

CFG (PT line 707-724) is *velocity-CFG with global norm re-scale*:

    v_t_ = v_uncond + scale * (v_cond - v_uncond)
    scale = clamp(||v_cond|| / ||v_t_||, min=0, max=1)
    v_t   = v_t_ * scale

The re-scale is unusual — most diffusion CFG just uses v_t_.  Lance's
extra step preserves the conditional's velocity magnitude while keeping
its direction-blend.  Empirically improves quality at high CFG scales.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------
@dataclass
class FlowMatchingSchedule:
    """Pre-computed (timesteps, dts) arrays for an Euler sampler."""
    timesteps: mx.array   # (N,) — the t value used at each step
    dts: mx.array         # (N,) — Δt for each step (timesteps[i] - timesteps[i+1])

    @property
    def num_steps(self) -> int:
        return int(self.timesteps.shape[0])


def make_schedule(num_steps: int = 30, timestep_shift: float = 3.5) -> FlowMatchingSchedule:
    """Build a Lance-style flow-matching schedule.

    Matches PT `lance.py:599-602` byte-for-byte:
      timesteps = linspace(1, 0, num_steps + 1)
      timesteps = shift * timesteps / (1 + (shift - 1) * timesteps)
      dts       = timesteps[:-1] - timesteps[1:]
      timesteps = timesteps[:-1]

    For shift > 1, the transform concentrates more steps near t = 1
    (noise side).  shift = 1 reproduces uniform spacing.
    """
    t = mx.linspace(1.0, 0.0, num_steps + 1)
    t = timestep_shift * t / (1 + (timestep_shift - 1) * t)
    dts = t[:-1] - t[1:]
    timesteps = t[:-1]
    return FlowMatchingSchedule(timesteps=timesteps, dts=dts)


# ---------------------------------------------------------------------------
# CFG velocity blend with global norm re-scale (Lance-specific)
# ---------------------------------------------------------------------------
def cfg_velocity(
    v_cond: mx.array,
    v_uncond: mx.array,
    *,
    scale: float,
    renorm_type: str = "global",
    renorm_min: float = 0.0,
) -> mx.array:
    """Lance's velocity CFG.

    PT ref `lance.py:707-724`.  Computes `v_ = v_uncond + scale*(v_cond -
    v_uncond)` (classical CFG), then rescales to recover ||v_cond||
    direction-blended but magnitude-preserved.

    renorm_type:
      - "global":  use scalar norms of the entire tensors.  Default.
      - "channel": per-last-axis norm (channel-wise).
      - "none":    classical CFG, no rescale.
    """
    v_ = v_uncond + scale * (v_cond - v_uncond)
    # `scale == 1.0` short-circuit: PT lance.py:688 gates the entire renorm
    # block on `if cfg_text_scale_ > 1.0`, so at scale=1.0 PT skips renorm
    # and uses v_cond unchanged.  Our `v_ = v_uncond + 1.0*(v_cond - v_uncond)
    # = v_cond` exactly, so returning `v_` here is *mathematically identical*
    # to PT's "skip renorm" path.
    if scale == 1.0 or renorm_type.lower() in ("", "none", "null"):
        return v_
    if renorm_type == "global":
        n_v_cond = mx.linalg.norm(v_cond)
        n_v_     = mx.linalg.norm(v_)
        ratio = n_v_cond / (n_v_ + 1e-8)
        # PT uses .clamp(min=renorm_min, max=1.0)
        ratio = mx.clip(ratio, renorm_min, 1.0)
        return v_ * ratio
    if renorm_type == "channel":
        n_v_cond = mx.linalg.norm(v_cond, axis=-1, keepdims=True)
        n_v_     = mx.linalg.norm(v_, axis=-1, keepdims=True)
        ratio = mx.clip(n_v_cond / (n_v_ + 1e-8), renorm_min, 1.0)
        return v_ * ratio
    raise ValueError(f"unknown renorm_type: {renorm_type!r}")


# ---------------------------------------------------------------------------
# Single Euler step
# ---------------------------------------------------------------------------
def cfg_velocity_3comp(
    v_full: mx.array,
    v_t_uncond: mx.array,
    v_tv_uncond: mx.array,
    *,
    cfg_text: float,
    cfg_vit: float,
    renorm_type: str = "global",
    renorm_min: float = 0.0,
) -> mx.array:
    """Lance 3-component CFG for TI2I (image edit).

    Matches PT `lance.py:707` (`v_t_ = cfg_text_vit_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
    + cfg_vit_scale * (cfg_text_v_t - cfg_text_vit_v_t)`):

        v_blend = v_tv_uncond
                + cfg_text * (v_full - v_t_uncond)
                + cfg_vit  * (v_t_uncond - v_tv_uncond)

    Then the *same* Lance global-norm rescale as the 2-comp path:
        scale = clamp(||v_full|| / ||v_blend||, renorm_min, 1.0)
        v_final = v_blend * scale
    """
    v_blend = (v_tv_uncond
               + cfg_text * (v_full - v_t_uncond)
               + cfg_vit  * (v_t_uncond - v_tv_uncond))
    # PT `lance.py:688` outer gate: `if cfg_text_scale_ > 1.0:` skips the entire
    # renorm block.  When both scales are 1.0, v_blend = v_full exactly, so the
    # ratio would be 1.0 anyway — but make the short-circuit explicit for parity.
    if cfg_text == 1.0 and cfg_vit == 1.0:
        return v_blend
    if renorm_type.lower() in ("", "none", "null"):
        return v_blend
    if renorm_type == "global":
        n_full  = mx.linalg.norm(v_full)
        n_blend = mx.linalg.norm(v_blend)
        ratio = mx.clip(n_full / (n_blend + 1e-8), renorm_min, 1.0)
        return v_blend * ratio
    if renorm_type == "channel":
        n_full  = mx.linalg.norm(v_full,  axis=-1, keepdims=True)
        n_blend = mx.linalg.norm(v_blend, axis=-1, keepdims=True)
        ratio = mx.clip(n_full / (n_blend + 1e-8), renorm_min, 1.0)
        return v_blend * ratio
    raise ValueError(f"unknown renorm_type: {renorm_type!r}")


def euler_step(x_t: mx.array, v_t: mx.array, dt: mx.array | float) -> mx.array:
    """One Euler step of the reverse-time flow.

    Lance integrates from t=1 (noise) backward to t=0 (data).  At each
    step, `dt = t_curr - t_next > 0`, so we *subtract* `v_t * dt` to
    move toward data (PT `lance.py:726`: `x_t -= v_t * dts[i]`).
    """
    return x_t - v_t * dt


# ---------------------------------------------------------------------------
# Initial noise sampler
# ---------------------------------------------------------------------------
def sample_init_noise(shape: tuple[int, ...], *,
                       seed: int = 0, dtype=mx.float32) -> mx.array:
    """Gaussian noise for the t=1 starting point.

    Lance uses standard normal directly — no scaling.  Shape is
    `(T_lat, H_lat, W_lat, z_dim)` for image generation (z_dim=48,
    16× spatial downsample of the pixel image).

    Uses NumPy's Generator → mx.array (not `mx.random.normal`) so the
    sample bytes match what our PT side-by-side harness uses for cross-
    checking.  MLX's PRNG vs NumPy's PRNG produce *different* samples
    for the same nominal seed; standardising on NumPy gives us
    reproducibility across the (MLX, PT) pair.

    Note on `dtype`: NumPy's `standard_normal` is f64 by default; we
    cast to f32 before crossing into MLX so the byte pattern matches
    the cross-validation harness (which also goes f32).  If a caller
    passes `dtype=mx.bfloat16` (none today; STAGE 9 video may want it),
    this would silently downcast through the f32 buffer — at that
    point swap to numpy's `dtype="float32"` direct sample of the
    appropriate width, or accept the f32→bf16 narrow as acceptable.
    """
    import numpy as np
    assert dtype == mx.float32, (
        f"sample_init_noise pinned to f32 (cross-validation invariant); "
        f"got {dtype}.  See docstring before relaxing."
    )
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal(shape).astype("float32"), dtype=dtype)
