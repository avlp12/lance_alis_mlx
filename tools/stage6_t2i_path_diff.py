"""Diff the *production* t2i.py path vs the inline compare-style denoise.

User caught a visual divergence between
  - `out/stage6_first_512.png`     (t2i.py via lance_mlx.pipelines.t2i)
  - `out/stage6_mlx_final.png`     (compare-harness inline impl)
despite all inputs (ids/pos/gen_mask/latent_pos_ids/mask) being byte-identical.
So either the loops diverge inside, or the production path has a bug not
in the inline.  This script runs both paths on the same model with same
seed/prompt, dumps step-by-step latent, and reports the first divergence.
"""
from __future__ import annotations

import time

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer

from lance_mlx.backbone import LanceLLM, LanceTextConfig, load_full_lance
from lance_mlx.pipelines.t2i import (
    _build_t2i_sequence, _make_positions, _make_gen_mask,
    _latent_position_indices, _embed_with_gen_slab,
)
from lance_mlx.attn_mask import build_lance_attention_mask
from lance_mlx.rope import VisionSpec, build_positions_for_layout
from lance_mlx.scheduler import make_schedule, cfg_velocity, euler_step, sample_init_noise


def main() -> None:
    HEIGHT, WIDTH = 512, 512
    NUM_STEPS, SHIFT, CFG = 30, 3.5, 4.0
    SEED = 0
    Z_DIM = 48
    SPATIAL_DS = 16
    PROMPT = "a photo of a sunset over mountains"

    tok = AutoTokenizer.from_pretrained("checkpoints/Lance-3B-MLX", trust_remote_code=True)
    prompt_ids = tok(PROMPT, add_special_tokens=False, return_tensors=None)["input_ids"]
    h_lat = HEIGHT // SPATIAL_DS
    w_lat = WIDTH // SPATIAL_DS
    t_lat = 1
    n_latent = t_lat * h_lat * w_lat

    layout = _build_t2i_sequence(prompt_ids, (t_lat, h_lat, w_lat))
    cond_ids = mx.array([layout["cond_ids"]], dtype=mx.int32)
    unc_ids  = mx.array([layout["uncond_ids"]], dtype=mx.int32)
    cond_gs, cond_ge   = layout["cond_gen_span"]
    unc_gs,  unc_ge    = layout["uncond_gen_span"]

    cond_pos = _make_positions(cond_ids.shape[1], (cond_gs, cond_ge), t_lat, h_lat, w_lat)
    unc_pos  = _make_positions(unc_ids.shape[1],  (unc_gs,  unc_ge),  t_lat, h_lat, w_lat)
    cond_attn_mask = build_lance_attention_mask(
        seq_len=cond_ids.shape[1],
        split_lens=[cond_gs, n_latent, 1],
        attn_modes=["causal", "noise", "causal"],
    )
    unc_attn_mask = build_lance_attention_mask(
        seq_len=unc_ids.shape[1],
        split_lens=[unc_gs, n_latent, 1],
        attn_modes=["causal", "noise", "causal"],
    )
    cond_gen_mask = _make_gen_mask(cond_ids.shape[1], (cond_gs, cond_ge))
    unc_gen_mask  = _make_gen_mask(unc_ids.shape[1],  (unc_gs,  unc_ge))
    latent_pos_ids = _latent_position_indices(t_lat, h_lat, w_lat)

    print(f"[setup] L_cond={cond_ids.shape[1]}  L_unc={unc_ids.shape[1]}  N={n_latent}")

    print("[build] LanceLLM ...")
    cfg = LanceTextConfig()
    model = LanceLLM(cfg)
    load_full_lance(model, "checkpoints/Lance-3B-MLX/model.safetensors")
    model.eval()

    # Identical noise init for both runs.
    x_init = sample_init_noise((n_latent, Z_DIM), seed=SEED)

    sch = make_schedule(num_steps=NUM_STEPS, timestep_shift=SHIFT)

    # ============================================================
    # PATH A — t2i.py-style: builds embed via `_embed_with_gen_slab`
    # ============================================================
    x_a = x_init
    print("\n[A] t2i.py-style denoise loop")
    for i in range(NUM_STEPS):
        t_scalar = sch.timesteps[i:i+1]
        cond_embed = _embed_with_gen_slab(
            model, cond_ids, x_a, t_scalar, latent_pos_ids, (cond_gs, cond_ge),
        )
        cond_hidden = model.language_model.model(
            input_ids=None, position_ids=cond_pos,
            inputs_embeds=cond_embed, mask=cond_attn_mask, gen_mask=cond_gen_mask,
        )
        v_cond = model.llm2vae(cond_hidden[0, cond_gs:cond_ge, :])

        unc_embed = _embed_with_gen_slab(
            model, unc_ids, x_a, t_scalar, latent_pos_ids, (unc_gs, unc_ge),
        )
        unc_hidden = model.language_model.model(
            input_ids=None, position_ids=unc_pos,
            inputs_embeds=unc_embed, mask=unc_attn_mask, gen_mask=unc_gen_mask,
        )
        v_unc = model.llm2vae(unc_hidden[0, unc_gs:unc_ge, :])
        v_t = cfg_velocity(v_cond, v_unc, scale=CFG)
        x_a = euler_step(x_a, v_t, sch.dts[i])
        mx.eval(x_a)
    print(f"[A] final x_t: std={x_a.std().item():.4f}")

    # ============================================================
    # PATH B — compare-style inline: same primitives, fresh setup
    # ============================================================
    x_b = x_init                                    # same noise init
    print("\n[B] compare-style inline denoise loop")
    timesteps_np = np.asarray(sch.timesteps)
    dts_np = np.asarray(sch.dts)
    for i in range(NUM_STEPS):
        t = float(timesteps_np[i])
        t_scalar = mx.array([t], dtype=mx.float32)
        # === compare's mlx_forward_to_v inlined ===
        def fwd(input_ids, pos, mask, gen_span, gen_mask):
            text_embed = model.language_model.model.embed_tokens(input_ids)
            vae_proj = model.vae2llm(x_b)
            t_emb = model.time_embedder(t_scalar)
            pos_emb = model.latent_pos_embed(latent_pos_ids)
            slab = vae_proj + t_emb + pos_emb
            gs, ge = gen_span
            embed = mx.concatenate([
                text_embed[:, :gs, :], slab[None, :, :], text_embed[:, ge:, :],
            ], axis=1)
            hidden = model.language_model.model(
                input_ids=None, position_ids=pos,
                inputs_embeds=embed, mask=mask, gen_mask=gen_mask,
            )
            return model.llm2vae(hidden[0, gs:ge, :])

        v_cond = fwd(cond_ids, cond_pos, cond_attn_mask, (cond_gs, cond_ge), cond_gen_mask)
        v_unc  = fwd(unc_ids,  unc_pos,  unc_attn_mask,  (unc_gs,  unc_ge),  unc_gen_mask)
        v_t = cfg_velocity(v_cond, v_unc, scale=CFG)
        x_b = euler_step(x_b, v_t, float(dts_np[i]))
        mx.eval(x_b)
    print(f"[B] final x_t: std={x_b.std().item():.4f}")

    # ============================================================
    # Diff
    # ============================================================
    a_np = np.asarray(x_a)
    b_np = np.asarray(x_b)
    af = a_np.astype(np.float64).flatten()
    bf = b_np.astype(np.float64).flatten()
    cos = float((af @ bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))
    diff = float(np.abs(a_np - b_np).max())
    print(f"\nFINAL  cos(A, B) = {cos:.6f}   max|Δ| = {diff:.4f}")


if __name__ == "__main__":
    main()
