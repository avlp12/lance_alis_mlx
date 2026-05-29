"""STAGE 9 §1 단계 5 — single-step PT byte-diff: v_full / v_unc / v_blend 각각 cos.

PT 정답지: v3 production smoke (out/stage9_pt_video_v_{full,unc,blend}_step0.npy)
MLX side: build_t2v_layout + _forward_v (production positions/mask byte-identical)

Gate (사용자 명시 — STAGE 7 §3 패턴 video 변형):
  cos(v_full_pt, v_full_mlx)   ≥ 0.999    ← forward (full context)
  cos(v_unc_pt,  v_unc_mlx)    ≥ 0.999    ← forward (uncond context, adapter 첫 발화)
  cos(v_blend_pt, v_blend_mlx) ≥ 0.999    ← cfg_velocity (renorm 포함)

진단:
  - v_full FAIL → MoE routing / mRoPE 적용 / full sequence forward
  - v_unc  FAIL → uncond context 의 adapter (vae2llm/time/latent_pos) 첫 발화
  - blend  FAIL → cfg_velocity 수식 (renorm/interval)
"""
import os
import sys

sys.path.insert(0, ".")

import numpy as np
import mlx.core as mx
import mlx.utils as mu
from transformers import AutoTokenizer

from lance_mlx.backbone import LanceLLM, LanceTextConfig, PositionEmbedding3D
from lance_mlx.pipelines.t2v import build_t2v_layout, _forward_v
from lance_mlx.scheduler import cfg_velocity

IMG_WEIGHTS = "checkpoints/Lance-3B-MLX/model.safetensors"
VID_SUP = "checkpoints/Lance-3B-Video-MLX/model_supplement.safetensors"

MAX_NUM_LATENT_FRAMES = 31
MAX_LATENT_SIZE = 64
USER_PROMPT = "A red panda riding a wave at sunset."


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    print("=" * 72)
    print("STAGE 9 §1 단계 5 — single-step PT byte-diff (production)")
    print("=" * 72)

    # ---- load PT fixtures (v3 production) ----
    v_full_pt = np.load("out/stage9_pt_video_v_full_step0.npy")
    v_unc_pt  = np.load("out/stage9_pt_video_v_unc_step0.npy")
    v_blend_pt = np.load("out/stage9_pt_video_v_blend_step0.npy")
    x_t_init = np.load("out/stage9_pt_video_x_t_init_prod.npy")
    print(f"[fx] v_full_pt  ||v||={np.linalg.norm(v_full_pt):.3f} std={v_full_pt.std():.4f}")
    print(f"[fx] v_unc_pt   ||v||={np.linalg.norm(v_unc_pt):.3f}  std={v_unc_pt.std():.4f}")
    print(f"[fx] v_blend_pt ||v||={np.linalg.norm(v_blend_pt):.3f} std={v_blend_pt.std():.4f}")
    print(f"[fx] x_t_init   shape={x_t_init.shape} std={x_t_init.std():.4f}")

    # ---- build MLX model + load video weights ----
    print("\n[build] LanceLLM + replace latent_pos_embed for video ...")
    cfg = LanceTextConfig()
    model = LanceLLM(cfg)
    model.latent_pos_embed = PositionEmbedding3D(
        max_num_latent_frames=MAX_NUM_LATENT_FRAMES,
        max_latent_size=MAX_LATENT_SIZE,
        hidden_size=cfg.hidden_size,
    )
    assert model.latent_pos_embed.pos_embed.shape == (
        MAX_NUM_LATENT_FRAMES * MAX_LATENT_SIZE * MAX_LATENT_SIZE, cfg.hidden_size
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
    print(f"[load] keys loaded: {len(to_load)} / {len(merged)} (vit_model.* dropped)")
    assert model.latent_pos_embed.pos_embed.shape == (
        MAX_NUM_LATENT_FRAMES * MAX_LATENT_SIZE * MAX_LATENT_SIZE, cfg.hidden_size
    )

    # ---- tokenizer + layout ----
    tok = AutoTokenizer.from_pretrained("checkpoints/Lance-3B-MLX", use_fast=True)
    layout = build_t2v_layout(USER_PROMPT, tok, num_frames=5, H=128, W=128)
    print(f"\n[layout] L={layout.L} uncond_L={layout.uncond_L} n_video={layout.n_video}")

    # ---- MLX forward: v_full ----
    x_t = mx.array(x_t_init)
    t_scalar = 1.0
    print(f"\n[forward FULL] MLX step 0 (t={t_scalar}) ...")
    v_full_mlx = _forward_v(model, layout.input_ids, layout.pos_ids,
                            layout.attn_mask, layout.vae_token_indices,
                            x_t, t_scalar, layout.vae_pos_ids)
    mx.eval(v_full_mlx)
    v_full_mlx_np = np.asarray(v_full_mlx, dtype=np.float32)
    print(f"  v_full_mlx: ||v||={np.linalg.norm(v_full_mlx_np):.3f} "
          f"mean={v_full_mlx_np.mean():+.4f} std={v_full_mlx_np.std():.4f}")

    # ---- MLX forward: v_unc ----
    print(f"\n[forward UNCOND] MLX step 0 (text drop, system+noise keep) ...")
    v_unc_mlx = _forward_v(model, layout.uncond_input_ids, layout.uncond_pos_ids,
                           layout.uncond_attn_mask, layout.uncond_vae_token_indices,
                           x_t, t_scalar, layout.vae_pos_ids)
    mx.eval(v_unc_mlx)
    v_unc_mlx_np = np.asarray(v_unc_mlx, dtype=np.float32)
    print(f"  v_unc_mlx: ||v||={np.linalg.norm(v_unc_mlx_np):.3f} "
          f"mean={v_unc_mlx_np.mean():+.4f} std={v_unc_mlx_np.std():.4f}")

    # ---- MLX blend: cfg_velocity (STAGE 6 검증된 함수) ----
    v_blend_mlx = cfg_velocity(v_full_mlx, v_unc_mlx,
                               scale=4.0, renorm_type="global", renorm_min=0.0)
    mx.eval(v_blend_mlx)
    v_blend_mlx_np = np.asarray(v_blend_mlx, dtype=np.float32)
    print(f"\n[blend] cfg_velocity (scale=4.0, global, min=0):")
    print(f"  v_blend_mlx: ||v||={np.linalg.norm(v_blend_mlx_np):.3f}")

    # ---- gate ----
    print("\n" + "=" * 72)
    print("GATE  (v_full / v_unc / v_blend 각각 cos ≥ 0.999)")
    print("=" * 72)
    GATE = 0.999
    results = [
        ("v_full   (forward, full context)", v_full_pt, v_full_mlx_np),
        ("v_unc    (forward, uncond — adapter 첫 발화)", v_unc_pt, v_unc_mlx_np),
        ("v_blend  (cfg_velocity, renorm global)", v_blend_pt, v_blend_mlx_np),
    ]
    all_pass = True
    for label, pt_arr, mlx_arr in results:
        c = cos(pt_arr, mlx_arr)
        maxabs = float(np.abs(pt_arr - mlx_arr).max())
        n_pt = float(np.linalg.norm(pt_arr))
        n_mlx = float(np.linalg.norm(mlx_arr))
        ratio = n_mlx / (n_pt + 1e-30)
        status = "PASS" if c >= GATE else "FAIL"
        if c < GATE:
            all_pass = False
        print(f"  [{status}] {label}")
        print(f"           cos={c:.6f}  maxabs={maxabs:.4e}  "
              f"||pt||={n_pt:.2f} ||mlx||={n_mlx:.2f} ratio={ratio:.4f}")
    print("=" * 72)
    if all_pass:
        print("STAGE 9 §1 단계 5 PASS — production single-step v_full/v_unc/v_blend 모두 통과")
        print("                       다음: 단계 6 (30-step + 첫 영상 + per-step PT cos)")
    else:
        print("FAIL — 진단 필요.")
        print("  v_full FAIL → MoE routing / mRoPE / full sequence forward")
        print("  v_unc FAIL  → uncond context 의 adapter 첫 발화 (vae2llm/time/latent_pos)")
        print("  blend FAIL  → cfg_velocity 수식")


if __name__ == "__main__":
    main()
