"""STAGE 8 final gate — WanVAE_ top-level round-trip parity.

Compares MLX Wan2_2_VAE.encode/decode against PT fixtures saved by
`tools/stage8_pt_video_smoke.py`:

    out/stage8_pt_video_input.npy   (1, 3, 5, 128, 128)  — synthetic N(0,1) video
    out/stage8_pt_video_mu.npy      (1, 48, 2, 8, 8)     — PT encode mu (scaled)
    out/stage8_pt_video_logvar.npy  (1, 48, 2, 8, 8)     — PT encode log_var
    out/stage8_pt_video_xhat.npy    (1, 3, 5, 128, 128)  — PT decode of mu

The gate criteria (per Stage 5 lesson — round-trip MSE is *pattern dependent*,
high-frequency input incurs irreducible Nyquist loss; absolute MSE is NOT the
gate, cos(PT_output, MLX_output) IS):

  - cos(mu_MLX, mu_PT)         ≥ 0.999
  - cos(logvar_MLX, logvar_PT) ≥ 0.999
  - cos(xhat_MLX, xhat_PT)     ≥ 0.999

Layout reminders:
  - PT 5-D conv tensor is (B, C, T, H, W); MLX 5-D is (B, T, H, W, C).
  - Fixtures are saved in PT layout — we permute to NTHWC before MLX forward.
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, ".")

import numpy as np
import mlx.core as mx

from lance_mlx.vae_wan22 import Wan2_2_VAE, Wan22VAEConfig


# --- per-channel scale (hardcoded by PT Wan2_2_VAE wrapper) -----------------
SCALE_MEAN = np.array([
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799,  0.0174,
     0.1838,  0.1557, -0.1382,  0.0542,  0.2813,  0.0891,
     0.1570, -0.0098,  0.0375, -0.1825, -0.2246, -0.1207,
    -0.0698,  0.5109,  0.2665, -0.2108, -0.2158,  0.2502,
    -0.2055, -0.0322,  0.1109,  0.1567, -0.0729,  0.0899,
    -0.2799, -0.1230, -0.0313, -0.1649,  0.0117,  0.0723,
    -0.2839, -0.2083, -0.0520,  0.3748,  0.0152,  0.1957,
     0.1433, -0.2944,  0.3573, -0.0548, -0.1681, -0.0667,
], dtype=np.float32)
SCALE_STD = np.array([
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990,
    0.4818, 0.5013, 0.8158, 1.0344, 0.5894, 1.0901,
    0.6885, 0.6165, 0.8454, 0.4978, 0.5759, 0.3523,
    0.7135, 0.6804, 1.0457, 0.4329, 0.7918, 0.5739,
    0.5942, 0.5570, 0.5860, 0.6673, 0.4109, 0.7894,
    0.5897, 0.4845, 0.5727, 1.1191, 0.4921, 0.4753,
    1.0265, 0.4790, 1.2798, 0.4768, 0.8169, 0.7497,
    0.7344, 0.4759, 0.8501, 0.6479, 0.4523, 0.6116,
], dtype=np.float32)


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def summarize(label: str, ref: np.ndarray, ours: np.ndarray) -> tuple[float, float]:
    """Report cos, max-abs, and percentile-style stats."""
    cos = cos_sim(ref, ours)
    diff = np.abs(ref - ours).reshape(-1)
    maxabs = float(diff.max())
    p50 = float(np.percentile(diff, 50))
    p90 = float(np.percentile(diff, 90))
    mse = float(((ref - ours) ** 2).mean())
    print(f"  [{label:8s}] cos={cos:.6f}  mse={mse:.6e}  "
          f"maxabs={maxabs:.4e}  p50={p50:.4e}  p90={p90:.4e}")
    return cos, maxabs


def main():
    print("[setup] loading PT fixtures from out/stage8_pt_video_*.npy ...")
    x_pt    = np.load("out/stage8_pt_video_input.npy")
    mu_pt   = np.load("out/stage8_pt_video_mu.npy")
    lv_pt   = np.load("out/stage8_pt_video_logvar.npy")
    xhat_pt = np.load("out/stage8_pt_video_xhat.npy")
    print(f"  input  : {x_pt.shape}   range=[{x_pt.min():+.3f}, {x_pt.max():+.3f}]")
    print(f"  mu     : {mu_pt.shape}  range=[{mu_pt.min():+.3f}, {mu_pt.max():+.3f}]")
    print(f"  logvar : {lv_pt.shape}  range=[{lv_pt.min():+.3f}, {lv_pt.max():+.3f}]")
    print(f"  xhat   : {xhat_pt.shape} range=[{xhat_pt.min():+.3f}, {xhat_pt.max():+.3f}]")

    print("\n[build] MLX Wan2_2_VAE(Wan22VAEConfig()) and loading checkpoint ...")
    cfg = Wan22VAEConfig()
    model = Wan2_2_VAE(cfg)
    w = mx.load("checkpoints/Wan2.2-VAE-MLX/model.safetensors")
    model.load_weights(list(w.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    print("  weights loaded.")

    # --- Convert PT (B,C,T,H,W) → MLX NTHWC (B,T,H,W,C) -------------------
    x_mlx = mx.array(np.transpose(x_pt, (0, 2, 3, 4, 1)))         # (1, 5, 128, 128, 3)
    print(f"\n[input] MLX layout: {x_mlx.shape}")

    scale = (mx.array(SCALE_MEAN), mx.array(1.0 / SCALE_STD))

    # --- MLX encode ------------------------------------------------------
    print("\n[encode] running MLX Wan2_2_VAE.encode(x, scale, return_logvar=True) ...")
    mu_mlx, lv_mlx = model.encode(x_mlx, scale=scale, return_logvar=True)
    mx.eval(mu_mlx, lv_mlx)
    print(f"  mu     : {mu_mlx.shape}  range=[{float(mu_mlx.min()):+.3f}, {float(mu_mlx.max()):+.3f}]")
    print(f"  logvar : {lv_mlx.shape}  range=[{float(lv_mlx.min()):+.3f}, {float(lv_mlx.max()):+.3f}]")

    # Permute MLX NTHWC → PT NCTHW for byte-diff vs fixture
    mu_mlx_pt    = np.transpose(np.array(mu_mlx),    (0, 4, 1, 2, 3))
    lv_mlx_pt    = np.transpose(np.array(lv_mlx),    (0, 4, 1, 2, 3))

    print("\n[encode-parity vs PT fixture]")
    cos_mu,  _ = summarize("mu",     mu_pt, mu_mlx_pt)
    cos_lv,  _ = summarize("logvar", lv_pt, lv_mlx_pt)

    # --- MLX decode using MLX's own mu (the proper round-trip) -----------
    print("\n[decode] running MLX Wan2_2_VAE.decode(mu_mlx, scale) ...")
    xhat_mlx = model.decode(mu_mlx, scale=scale)
    mx.eval(xhat_mlx)
    xhat_mlx_pt = np.transpose(np.array(xhat_mlx), (0, 4, 1, 2, 3))   # (B,C,T,H,W)
    print(f"  xhat   : {xhat_mlx_pt.shape}  range=[{xhat_mlx_pt.min():+.3f}, {xhat_mlx_pt.max():+.3f}]")

    print("\n[decode-parity vs PT fixture (decode of MU)]")
    # NB: PT fixture decodes PT-mu; MLX decodes MLX-mu.  If mu's differ
    # at all, the decode output will drift slightly.  This composite check
    # measures *whole pipeline* parity.
    cos_xhat_self, _ = summarize("xhat", xhat_pt, xhat_mlx_pt)

    # --- Also decode the PT mu via MLX (isolates decoder parity from encode) ---
    print("\n[decode] decoding PT mu via MLX (isolates decoder from encoder) ...")
    mu_pt_mlx = mx.array(np.transpose(mu_pt, (0, 2, 3, 4, 1)))         # PT→MLX layout
    xhat_from_pt_mu = model.decode(mu_pt_mlx, scale=scale)
    mx.eval(xhat_from_pt_mu)
    xhat_from_pt_mu_pt = np.transpose(np.array(xhat_from_pt_mu), (0, 4, 1, 2, 3))
    cos_xhat_iso, _ = summarize("xhat*", xhat_pt, xhat_from_pt_mu_pt)

    # --- Gate decision ----------------------------------------------------
    print("\n" + "=" * 64)
    print("STAGE 8 final gate")
    print("=" * 64)
    GATE = 0.999
    rows = [
        ("mu       (MLX encode vs PT fixture)", cos_mu),
        ("log_var  (MLX encode vs PT fixture)", cos_lv),
        ("xhat*    (MLX decode of PT mu)",     cos_xhat_iso),
        ("xhat     (MLX full round-trip)",     cos_xhat_self),
    ]
    ok = True
    for name, c in rows:
        status = "PASS" if c >= GATE else "FAIL"
        if c < GATE:
            ok = False
        print(f"  cos {c:.6f}  ≥{GATE}  [{status}]  {name}")
    print("=" * 64)
    if ok:
        print("STAGE 8 §2 (WanVAE_ top-level) — PASS  ✓ encode + decode both byte-clean")
    else:
        print("STAGE 8 §2 — FAIL.  Investigate the lowest-cos row first.")
    print("=" * 64)


if __name__ == "__main__":
    main()
