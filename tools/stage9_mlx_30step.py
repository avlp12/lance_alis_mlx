"""STAGE 9 §1 단계 6 (MLX side) — t2v 30-step + per-step latent dump.

PRNG: numpy seed=0 (Lesson 9), x_t_init = out/stage9_pt_video_x_t_init_prod.npy
(PT 와 byte-identical 확인됨).

Mirrors PT 30-step harness (cfg_text_scale=4.0, cfg_interval=[0.4,1.0],
global renorm, timestep_shift=3.5) — t2v.t2v() 내부 동일.  여기서는 t2v()
호출 대신 *step loop 안에 latent intercept* 추가해 dump.

Per-step x_t dump → out/stage9_mlx_30step_latent.npy (30, n_video, 48)
Final latent  → VAE decode → out/stage9_mlx_30step_video.npy

PT 비교: tools/stage9_per_step_cos.py 가 PT vs MLX cos 계산.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, ".")

import numpy as np
import mlx.core as mx
import mlx.utils as mu
from transformers import AutoTokenizer

from lance_mlx.backbone import LanceLLM, LanceTextConfig, PositionEmbedding3D
from lance_mlx.pipelines.t2v import build_t2v_layout, _forward_v, LATENT_CHANNEL
from lance_mlx.scheduler import make_schedule, cfg_velocity, euler_step
from lance_mlx.vae_wan22 import Wan2_2_VAE, Wan22VAEConfig


IMG_WEIGHTS = "checkpoints/Lance-3B-MLX/model.safetensors"
VID_SUP = "checkpoints/Lance-3B-Video-MLX/model_supplement.safetensors"
VAE_WEIGHTS = "checkpoints/Wan2.2-VAE-MLX/model.safetensors"   # STAGE 5/6/8 검증된 path (prefix 없는 standalone)

MAX_NUM_LATENT_FRAMES = 31
MAX_LATENT_SIZE = 64
USER_PROMPT = "A red panda riding a wave at sunset."
T_VIDEO, H_PIX, W_PIX = 5, 128, 128
NUM_STEPS = 30
CFG_TEXT_SCALE = 4.0
CFG_INTERVAL = (0.4, 1.0)
CFG_RENORM_MIN = 0.0
TIMESTEP_SHIFT = 3.5


def main():
    t_full = time.time()
    print("=" * 72)
    print("STAGE 9 §1 단계 6 (MLX) — t2v 30-step + per-step latent dump")
    print("=" * 72)

    # ---- build model + load video weights ----
    print("[build] LanceLLM + replace latent_pos_embed for video ...")
    cfg = LanceTextConfig()
    model = LanceLLM(cfg)
    model.latent_pos_embed = PositionEmbedding3D(
        max_num_latent_frames=MAX_NUM_LATENT_FRAMES,
        max_latent_size=MAX_LATENT_SIZE,
        hidden_size=cfg.hidden_size,
    )
    img_w = mx.load(IMG_WEIGHTS)
    sup_w = mx.load(VID_SUP)
    merged = dict(img_w)
    for k, v in sup_w.items():
        merged[k] = v
    ours = set(dict(mu.tree_flatten(model.parameters())).keys())
    to_load = {k: v for k, v in merged.items() if k in ours}
    model.load_weights(list(to_load.items()), strict=True)
    mx.eval(model.parameters())
    print(f"[load] keys loaded: {len(to_load)}")

    # ---- VAE (STAGE 8 byte-clean) ----
    print("[build] Wan2_2_VAE for decode ...")
    vae = Wan2_2_VAE(Wan22VAEConfig())
    vae_w = mx.load(VAE_WEIGHTS)
    vae.load_weights(list(vae_w.items()), strict=True)
    vae.eval()
    mx.eval(vae.parameters())

    # ---- tokenizer + layout ----
    tok = AutoTokenizer.from_pretrained("checkpoints/Lance-3B-MLX", use_fast=True)
    layout = build_t2v_layout(USER_PROMPT, tok,
                              num_frames=T_VIDEO, H=H_PIX, W=W_PIX)
    print(f"[layout] L={layout.L} uncond_L={layout.uncond_L} n_video={layout.n_video}")

    # ---- noise (numpy seed=0, byte-identical with PT v3 fixture) ----
    x_t_np = np.load("out/stage9_pt_video_x_t_init_prod.npy")   # = numpy seed=0
    x_t = mx.array(x_t_np)
    print(f"[x_t init] shape={x_t_np.shape}  std={x_t_np.std():.6f}")

    # ---- schedule ----
    sch = make_schedule(num_steps=NUM_STEPS, timestep_shift=TIMESTEP_SHIFT)
    timesteps = sch.timesteps
    dts = sch.dts
    print(f"[schedule] {NUM_STEPS} steps, shift={TIMESTEP_SHIFT}, "
          f"t[0]={float(timesteps[0]):.4f} t[-1]={float(timesteps[-1]):.4f}")
    print(f"[CFG] interval={CFG_INTERVAL}, scale={CFG_TEXT_SCALE}, global renorm min={CFG_RENORM_MIN}")

    # ---- step loop ----
    latents_per_step = np.zeros((NUM_STEPS, layout.n_video, LATENT_CHANNEL), dtype=np.float32)
    cfg_history = []
    t_loop = time.time()
    for i in range(NUM_STEPS):
        t_scalar = float(timesteps[i])
        dt = float(dts[i])
        cfg_on = (CFG_INTERVAL[0] < t_scalar <= CFG_INTERVAL[1]) and (CFG_TEXT_SCALE > 1.0)
        cfg_history.append(cfg_on)

        v_full = _forward_v(model, layout.input_ids, layout.pos_ids,
                            layout.attn_mask, layout.vae_token_indices,
                            x_t, t_scalar, layout.vae_pos_ids)
        if cfg_on:
            v_unc = _forward_v(model, layout.uncond_input_ids, layout.uncond_pos_ids,
                               layout.uncond_attn_mask, layout.uncond_vae_token_indices,
                               x_t, t_scalar, layout.vae_pos_ids)
            v_t = cfg_velocity(v_full, v_unc,
                               scale=CFG_TEXT_SCALE,
                               renorm_type="global",
                               renorm_min=CFG_RENORM_MIN)
        else:
            v_t = v_full

        x_t = euler_step(x_t, v_t, dt)
        mx.eval(x_t)
        latents_per_step[i] = np.asarray(x_t, dtype=np.float32)

        if (i + 1) % 5 == 0 or i == 0 or i == NUM_STEPS - 1:
            elapsed = time.time() - t_loop
            print(f"[step {i+1:3d}/{NUM_STEPS}]  t={t_scalar:.4f}  cfg={'ON ' if cfg_on else 'off'}  "
                  f"||v_t||={float(mx.linalg.norm(v_t)):.2f}  "
                  f"||x_t||={float(mx.linalg.norm(x_t)):.2f}  ({elapsed:.0f}s)")

    np.save("out/stage9_mlx_30step_latent.npy", latents_per_step)
    np.save("out/stage9_mlx_30step_final_x_t.npy", np.asarray(x_t, dtype=np.float32))

    # ---- VAE decode (STAGE 8) + production scale ----
    from lance_mlx.pipelines.t2v import VAE_SCALE_MEAN, VAE_SCALE_STD
    vae_scale = (mx.array(VAE_SCALE_MEAN), mx.array(1.0 / VAE_SCALE_STD))
    print(f"\n[VAE decode] reshape {layout.n_video}×48 → ({layout.t_lat}, {layout.h_lat}, {layout.w_lat}, 48) ...")
    latent_np = np.asarray(x_t, dtype=np.float32).reshape(layout.t_lat, layout.h_lat, layout.w_lat, LATENT_CHANNEL)
    latent_mx = mx.array(latent_np)[None, ...]
    video = vae.decode(latent_mx, scale=vae_scale)
    mx.eval(video)
    video_np = np.asarray(video, dtype=np.float32)
    np.save("out/stage9_mlx_30step_video.npy", video_np)
    print(f"[VAE] video shape={video_np.shape}  range=[{video_np.min():+.3f}, {video_np.max():+.3f}]")

    meta = {
        "num_steps": NUM_STEPS,
        "cfg_text_scale": CFG_TEXT_SCALE,
        "cfg_interval": list(CFG_INTERVAL),
        "cfg_renorm_min": CFG_RENORM_MIN,
        "cfg_renorm_type": "global",
        "timestep_shift": TIMESTEP_SHIFT,
        "cfg_on_per_step": cfg_history,
        "video_grid_thw": [layout.t_lat, layout.h_lat, layout.w_lat],
        "elapsed_total_seconds": time.time() - t_full,
    }
    with open("out/stage9_mlx_30step_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[OK] MLX 30-step + VAE decode 완료. 총 {(time.time()-t_full)/60:.1f} min")
    print(f"     saved: stage9_mlx_30step_{{latent,final_x_t,video}}.npy")
    print(f"     CFG on steps: {sum(cfg_history)}/{NUM_STEPS}")


if __name__ == "__main__":
    main()
