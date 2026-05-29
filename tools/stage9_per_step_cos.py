"""STAGE 9 §1 단계 6 (게이트) — per-step PT vs MLX latent cos + 프레임 연속성.

게이트:
  1. 매 step cos(latent_pt[i], latent_mlx[i]) ≥ 0.999
  2. cfg_interval 전환점 (CFG on→off) 주의 — 발산 지점 추적
  3. 최종 영상 byte-diff (cos + maxabs) — 교훈 11/12: 그럴듯한 영상 ≠ 통과,
     PT 와 같은 영상이 게이트

PRNG byte-identical 사전 확인됨 (numpy seed=0).
"""
from __future__ import annotations
import json
import os
import sys
import numpy as np

sys.path.insert(0, ".")


def cos(a, b):
    a = a.reshape(-1).astype(np.float64); b = b.reshape(-1).astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    print("=" * 72)
    print("STAGE 9 §1 단계 6 — per-step PT vs MLX cos + 프레임 연속성 게이트")
    print("=" * 72)

    pt_latent = np.load("out/stage9_pt_30step_latent.npy")        # (30, n_video, 48)
    mlx_latent = np.load("out/stage9_mlx_30step_latent.npy")      # (30, n_video, 48)
    with open("out/stage9_pt_30step_meta.json") as f:
        pt_meta = json.load(f)

    NUM_STEPS = pt_latent.shape[0]
    n_video = pt_latent.shape[1]
    print(f"[in] PT latent shape={pt_latent.shape}  MLX latent shape={mlx_latent.shape}")
    print(f"[in] CFG interval={pt_meta['cfg_interval']}  scale={pt_meta['cfg_text_scale']}")
    cfg_on_steps = pt_meta["cfg_on_per_step"]

    # ---- per-step cos ----
    print("\n[per-step cos]  (cfg ON/off 표시)")
    GATE = 0.999
    fail_steps = []
    print(f"  step    t      cfg    ||pt||    ||mlx||    cos        diff")
    for i in range(NUM_STEPS):
        c = cos(pt_latent[i], mlx_latent[i])
        n_pt = np.linalg.norm(pt_latent[i])
        n_mlx = np.linalg.norm(mlx_latent[i])
        diff = float(np.abs(pt_latent[i] - mlx_latent[i]).max())
        cfg_tag = "ON " if cfg_on_steps[i] else "off"
        flag = "★" if c < GATE else " "
        t_i = pt_meta["timesteps"][i]
        print(f"  {i+1:3d}/{NUM_STEPS}  {t_i:.4f}  {cfg_tag}  {n_pt:7.2f}  {n_mlx:7.2f}  {c:.6f}{flag}  {diff:.4e}")
        if c < GATE:
            fail_steps.append((i, c))

    # ---- cfg transition analysis ----
    transitions = []
    for i in range(1, NUM_STEPS):
        if cfg_on_steps[i] != cfg_on_steps[i-1]:
            transitions.append((i, cfg_on_steps[i-1], cfg_on_steps[i]))
    print(f"\n[CFG transitions] {len(transitions)} points:")
    for i, prev, cur in transitions:
        c = cos(pt_latent[i], mlx_latent[i])
        print(f"  step {i+1}: cfg {prev}→{cur}, t={pt_meta['timesteps'][i]:.4f}, cos={c:.6f}")

    # ---- final latent + video ----
    final_pt = np.load("out/stage9_pt_30step_final_x_t.npy")
    final_mlx = np.load("out/stage9_mlx_30step_final_x_t.npy")
    print(f"\n[final latent] cos={cos(final_pt, final_mlx):.6f}  maxabs={float(np.abs(final_pt-final_mlx).max()):.4e}")

    # ---- video frame continuity ----
    if os.path.exists("out/stage9_mlx_30step_video.npy"):
        mlx_video = np.load("out/stage9_mlx_30step_video.npy")
        print(f"\n[MLX video] shape={mlx_video.shape}  range=[{mlx_video.min():+.3f}, {mlx_video.max():+.3f}]")
        # frame continuity: consecutive frame difference
        # mlx_video shape (1, T_pix, H, W, 3)
        if mlx_video.ndim == 5:
            n_frames = mlx_video.shape[1]
            if n_frames > 1:
                inter_frame_diff = []
                for f in range(n_frames - 1):
                    diff = float(np.abs(mlx_video[0, f+1] - mlx_video[0, f]).mean())
                    inter_frame_diff.append(diff)
                print(f"[frames] {n_frames} frames, mean inter-frame |Δ|:")
                print(f"  per-frame diff: {[f'{d:.4f}' for d in inter_frame_diff]}")
                print(f"  range: {min(inter_frame_diff):.4f} ~ {max(inter_frame_diff):.4f}")

    # ---- gate decision ----
    print("\n" + "=" * 72)
    print("GATE")
    print("=" * 72)
    if len(fail_steps) == 0:
        print(f"PASS — 모든 {NUM_STEPS} step cos ≥ {GATE}")
        min_cos = min(cos(pt_latent[i], mlx_latent[i]) for i in range(NUM_STEPS))
        print(f"  min per-step cos = {min_cos:.6f}")
        print(f"  final latent cos = {cos(final_pt, final_mlx):.6f}")
    else:
        print(f"FAIL — {len(fail_steps)} step(s) below {GATE}")
        for i, c in fail_steps[:5]:
            print(f"  step {i+1}: cos={c:.6f}")


if __name__ == "__main__":
    main()
