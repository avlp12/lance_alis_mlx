"""STAGE 9 §1 단계 6 (PT side) — 30-step end-to-end + per-step latent dump.

PT 측 step loop (validation_gen line 643-727):
  for i in range(num_steps):
      v_full = forward(x_t, t)
      if cfg_interval[0] < t <= cfg_interval[1]:
          v_unc = uncond_forward(x_t, t)
          v_blend = v_unc + cfg * (v_full - v_unc)
          scale = clamp(||v_full|| / ||v_blend||, min, 1.0)
          v_t = v_blend * scale
      else:
          v_t = v_full
      x_t -= v_t * dt

Per-step x_t intercept → out/stage9_pt_30step_latent.npy (30, n_video, 48).
또한 최종 latent → VAE decode → out/stage9_pt_30step_video.npy 도 함께.

PRNG: numpy seed=0 (Lesson 9), x_t_init = out/stage9_pt_video_x_t_init_prod.npy (v3 와 동일).

PT bf16 30-step CPU 시간 ~10 분 예상.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, os.path.abspath("refs/Lance"))
from tools._pt_smoke_common import install_pt_smoke_env, pt_layer_mask
install_pt_smoke_env()

import numpy as np
import torch
from transformers import AutoTokenizer

from lance_mlx.pipelines.t2v import (
    build_t2v_positions, vae_latent_position_indices,
    MAX_NUM_LATENT_FRAMES, MAX_LATENT_SIZE, LATENT_CHANNEL,
    START_OF_IMAGE, VIDEO_TOKEN_ID,
)
from lance_mlx.pipelines._t2v_seq import build_t2v_sequence_pt

# v3 PT smoke 의 PtLanceVideoT2V + _pt_forward_one 재사용
import importlib
v3_mod = importlib.import_module("tools.stage9_pt_video_dit_smoke_v3")
PtLanceVideoT2V = v3_mod.PtLanceVideoT2V
_pt_forward_one = v3_mod._pt_forward_one

from data.data_utils import create_sparse_mask


# Config (production)
USER_PROMPT = "A red panda riding a wave at sunset."
T_VIDEO, H_PIX, W_PIX = 5, 128, 128
NUM_STEPS = 30
CFG_TEXT_SCALE = 4.0
CFG_INTERVAL = (0.4, 1.0)
CFG_RENORM_MIN = 0.0
TIMESTEP_SHIFT = 3.5
NUMPY_SEED = 0


def _dense_mask(L_, sl_, am_):
    am_resolved = ["full" if m in ("full_noise", "full_noise_target") else m for m in am_]
    predicate = create_sparse_mask(document_lens=[L_], split_lens=sl_,
                                   attn_modes=am_resolved, device=torch.device("cpu"))
    q = torch.arange(L_)[:, None]; k = torch.arange(L_)[None, :]
    b = torch.tensor(0); h = torch.tensor(0)
    return predicate(b=b, h=h, q_idx=q, kv_idx=k).contiguous()


def main():
    t_full = time.time()
    os.makedirs("out", exist_ok=True)
    print("=" * 72)
    print("STAGE 9 §1 단계 6 (PT) — 30-step end-to-end + per-step latent dump")
    print("=" * 72)

    # ---- shapes + noise ----
    t_lat = (T_VIDEO - 1) // 4 + 1
    h_lat = H_PIX // 16
    w_lat = W_PIX // 16
    n_video = t_lat * h_lat * w_lat
    print(f"[shape] T={T_VIDEO} → t/h/w={t_lat}/{h_lat}/{w_lat}  n_video={n_video}")

    # x_t init from v3 fixture (numpy seed=0)
    x_t_np = np.load("out/stage9_pt_video_x_t_init_prod.npy")
    print(f"[x_t init] from v3 fixture: shape={x_t_np.shape}  std={x_t_np.std():.4f}")
    x_t = torch.from_numpy(x_t_np)

    # ---- sequence ----
    tok = AutoTokenizer.from_pretrained("checkpoints/Lance-3B-MLX", use_fast=True)
    seq = build_t2v_sequence_pt(USER_PROMPT, tok,
                                num_frames=T_VIDEO, H=H_PIX, W=W_PIX,
                                vae_down_t=4, vae_down_s=16)
    input_ids_np = seq["input_ids"].astype(np.int64)
    L = seq["L"]; modality = seq["sample_modality"]
    vae_token_indices = seq["packed_vae_token_indexes"].astype(np.int64)
    split_lens, attn_modes = seq["split_lens"], seq["attn_modes"]
    vs_idx = int(np.where(input_ids_np == START_OF_IMAGE)[0][0])

    # uncond
    uncond_mask = modality != 0
    uncond_input_ids_np = input_ids_np[uncond_mask]
    uncond_L = int(uncond_mask.sum())
    u_vs_idx = int(np.where(uncond_input_ids_np == START_OF_IMAGE)[0][0])
    uncond_split_lens, uncond_attn_modes = [], []
    cursor = 0
    for sl, am in zip(split_lens, attn_modes):
        sub_mod = modality[cursor:cursor + sl]
        keep = int((sub_mod != 0).sum())
        cursor += sl
        if keep == 0:
            continue
        uncond_split_lens.append(keep)
        uncond_attn_modes.append(am)
    uncond_vae_token_indices = np.where(uncond_input_ids_np == VIDEO_TOKEN_ID)[0].astype(np.int64)
    print(f"[seq] L={L}  uncond_L={uncond_L}")

    # ---- positions + masks (production: spt=1.0, tps=2) ----
    full_pos = build_t2v_positions(vs_idx, t_lat, h_lat, w_lat, L, second_per_grid_t=1.0)
    full_pos_pt = torch.from_numpy(full_pos.astype(np.int64))
    uncond_pos = build_t2v_positions(u_vs_idx, t_lat, h_lat, w_lat, uncond_L, second_per_grid_t=1.0)
    uncond_pos_pt = torch.from_numpy(uncond_pos.astype(np.int64))
    full_dense = _dense_mask(L, split_lens, attn_modes)
    uncond_dense = _dense_mask(uncond_L, uncond_split_lens, uncond_attn_modes)
    vae_pos_ids_np = vae_latent_position_indices(t_lat, h_lat, w_lat)
    vae_pos_ids_pt = torch.from_numpy(vae_pos_ids_np.astype(np.int64))
    print(f"[mask] full True={int(full_dense.sum())}  uncond True={int(uncond_dense.sum())}")

    # ---- build + load PT model ----
    print("\n[build+load] PtLanceVideoT2V (image + video supplement bf16) ...")
    t_load = time.time()
    pt = PtLanceVideoT2V()
    pt.load_pt()
    pt.to_bf16()
    print(f"[load] done in {time.time()-t_load:.1f}s")

    # ---- schedule (PT lance.py:599-602 byte-for-byte) ----
    t_arr = torch.linspace(1.0, 0.0, NUM_STEPS + 1)
    t_arr = TIMESTEP_SHIFT * t_arr / (1 + (TIMESTEP_SHIFT - 1) * t_arr)
    dts = (t_arr[:-1] - t_arr[1:]).tolist()
    timesteps = t_arr[:-1].tolist()
    print(f"[schedule] {NUM_STEPS} steps, shift={TIMESTEP_SHIFT}, "
          f"t[0]={timesteps[0]:.4f} t[-1]={timesteps[-1]:.4f}")
    print(f"[CFG] interval={CFG_INTERVAL}, scale={CFG_TEXT_SCALE}, global renorm min={CFG_RENORM_MIN}")

    # ---- step loop ----
    latents_per_step = np.zeros((NUM_STEPS, n_video, LATENT_CHANNEL), dtype=np.float32)
    cfg_history = []
    t_loop = time.time()
    with torch.no_grad():
        for i in range(NUM_STEPS):
            t_scalar = float(timesteps[i])
            dt = float(dts[i])
            cfg_on = (CFG_INTERVAL[0] < t_scalar <= CFG_INTERVAL[1]) and (CFG_TEXT_SCALE > 1.0)
            cfg_history.append(cfg_on)

            v_full = _pt_forward_one(
                pt, torch.from_numpy(input_ids_np), full_pos_pt, full_dense,
                vae_token_indices, x_t, t_scalar, vae_pos_ids_pt,
            )
            if cfg_on:
                v_unc = _pt_forward_one(
                    pt, torch.from_numpy(uncond_input_ids_np), uncond_pos_pt, uncond_dense,
                    uncond_vae_token_indices, x_t, t_scalar, vae_pos_ids_pt,
                )
                v_blend = v_unc + CFG_TEXT_SCALE * (v_full - v_unc)
                n_full = float(torch.linalg.norm(v_full))
                n_blend = float(torch.linalg.norm(v_blend))
                scale = min(n_full / (n_blend + 1e-8), 1.0)
                scale = max(scale, CFG_RENORM_MIN)
                v_t = v_blend * scale
            else:
                v_t = v_full

            x_t = x_t - v_t * dt
            latents_per_step[i] = x_t.cpu().numpy().astype(np.float32)

            if (i + 1) % 5 == 0 or i == 0 or i == NUM_STEPS - 1:
                elapsed = time.time() - t_loop
                print(f"[step {i+1:3d}/{NUM_STEPS}]  t={t_scalar:.4f}  cfg={'ON ' if cfg_on else 'off'}  "
                      f"||v_t||={float(torch.linalg.norm(v_t)):.2f}  "
                      f"||x_t||={float(torch.linalg.norm(x_t)):.2f}  ({elapsed:.0f}s)")

    np.save("out/stage9_pt_30step_latent.npy", latents_per_step)
    np.save("out/stage9_pt_30step_final_x_t.npy", x_t.cpu().numpy())
    meta = {
        "num_steps": NUM_STEPS,
        "cfg_text_scale": CFG_TEXT_SCALE,
        "cfg_interval": list(CFG_INTERVAL),
        "cfg_renorm_min": CFG_RENORM_MIN,
        "cfg_renorm_type": "global",
        "timestep_shift": TIMESTEP_SHIFT,
        "timesteps": timesteps,
        "dts": dts,
        "cfg_on_per_step": cfg_history,
        "prng_seed": NUMPY_SEED,
        "x_t_source": "out/stage9_pt_video_x_t_init_prod.npy (numpy seed=0)",
        "video_grid_thw": [t_lat, h_lat, w_lat],
        "elapsed_total_seconds": time.time() - t_full,
    }
    with open("out/stage9_pt_30step_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    elapsed_total = time.time() - t_full
    print(f"\n[OK] PT 30-step 완료. 총 {elapsed_total/60:.1f} min")
    print(f"     saved: out/stage9_pt_30step_latent.npy  shape={latents_per_step.shape}")
    print(f"            out/stage9_pt_30step_final_x_t.npy")
    print(f"     CFG on steps: {sum(cfg_history)}/{NUM_STEPS}")


if __name__ == "__main__":
    main()
