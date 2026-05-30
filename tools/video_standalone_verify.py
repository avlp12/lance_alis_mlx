"""GATE 4C — the standalone video weight reproduces STAGE 9 t2v end-to-end.

Loads the merged standalone Lance_3B_Video/model.safetensors (1411 keys) as a
SINGLE file, filters to the t2v model params (1021; vit_model.* is dropped —
t2v generation doesn't use the ViT), runs the production single-step forward,
and compares v_full / v_unc / v_blend against the STAGE 9 PT fixtures.

Reproducing cos >= 0.999 proves the standalone is end-to-end correct for t2v
(Lesson 23: verify at the output, not just the key-merge).  This mirrors
tools/stage9_single_step_compare.py exactly, except the weights come from the
one standalone file instead of an image-backbone + supplement merge.
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

import numpy as np
import mlx.core as mx
import mlx.utils as mu
from transformers import AutoTokenizer

from lance_mlx.backbone import LanceLLM, LanceTextConfig, PositionEmbedding3D
from lance_mlx.pipelines.t2v import build_t2v_layout, _forward_v
from lance_mlx.scheduler import cfg_velocity

STANDALONE = "out/lance_3b_video_mlx/model.safetensors"
MAX_NUM_LATENT_FRAMES = 31
MAX_LATENT_SIZE = 64
USER_PROMPT = "A red panda riding a wave at sunset."


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main() -> None:
    v_full_pt = np.load("out/stage9_pt_video_v_full_step0.npy")
    v_unc_pt = np.load("out/stage9_pt_video_v_unc_step0.npy")
    v_blend_pt = np.load("out/stage9_pt_video_v_blend_step0.npy")
    x_t_init = np.load("out/stage9_pt_video_x_t_init_prod.npy")

    cfg = LanceTextConfig()
    model = LanceLLM(cfg)
    model.latent_pos_embed = PositionEmbedding3D(
        max_num_latent_frames=MAX_NUM_LATENT_FRAMES,
        max_latent_size=MAX_LATENT_SIZE,
        hidden_size=cfg.hidden_size,
    )

    # ---- load the STANDALONE (single file) ----
    full = mx.load(STANDALONE)
    ours = set(dict(mu.tree_flatten(model.parameters())).keys())
    to_load = {k: v for k, v in full.items() if k in ours}
    model.load_weights(list(to_load.items()), strict=True)
    mx.eval(model.parameters())
    print(f"[load] standalone {len(full)} keys -> t2v loaded {len(to_load)} "
          f"(vit_model.* dropped: {len(full) - len(to_load)})")
    assert model.latent_pos_embed.pos_embed.shape == (
        MAX_NUM_LATENT_FRAMES * MAX_LATENT_SIZE * MAX_LATENT_SIZE, cfg.hidden_size
    )

    tok = AutoTokenizer.from_pretrained("checkpoints/Lance-3B-MLX", use_fast=True)
    layout = build_t2v_layout(USER_PROMPT, tok, num_frames=5, H=128, W=128)

    x_t = mx.array(x_t_init)
    t_scalar = 1.0
    v_full_mlx = _forward_v(model, layout.input_ids, layout.pos_ids,
                            layout.attn_mask, layout.vae_token_indices,
                            x_t, t_scalar, layout.vae_pos_ids)
    v_unc_mlx = _forward_v(model, layout.uncond_input_ids, layout.uncond_pos_ids,
                           layout.uncond_attn_mask, layout.uncond_vae_token_indices,
                           x_t, t_scalar, layout.vae_pos_ids)
    v_blend_mlx = cfg_velocity(v_full_mlx, v_unc_mlx, scale=4.0,
                               renorm_type="global", renorm_min=0.0)
    mx.eval(v_full_mlx, v_unc_mlx, v_blend_mlx)

    GATE = 0.999
    rows = [
        ("v_full ", v_full_pt, np.asarray(v_full_mlx, dtype=np.float32)),
        ("v_unc  ", v_unc_pt, np.asarray(v_unc_mlx, dtype=np.float32)),
        ("v_blend", v_blend_pt, np.asarray(v_blend_mlx, dtype=np.float32)),
    ]
    all_pass = True
    print("=" * 60)
    for label, pt, ml in rows:
        c = cos(pt, ml)
        maxabs = float(np.abs(pt - ml).max())
        ok = c >= GATE
        all_pass = all_pass and ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}  cos={c:.6f}  maxabs={maxabs:.4e}")
    print("=" * 60)
    print("GATE 4C:", "PASS — standalone reproduces STAGE 9 t2v end-to-end"
          if all_pass else "FAIL — standalone does not match STAGE 9")


if __name__ == "__main__":
    main()
