"""Text-to-image pipeline for Lance.

Sequence layout (matches RockTalk README and `sample_t2i`):

    <|im_start|>  [prompt_tokens]  <|im_end|>  <|vision_start|>
        [N = T_lat * H_lat * W_lat placeholder image tokens]
    <|vision_end|>

Uncond sequence (for CFG): same layout with the prompt body dropped
(only `<|im_start|> <|im_end|>` text envelope + latent slab).

Per-step (mirrors PT `lance.py:643-726`):

  1. Build the *VAE-embed* for the current x_t:
        embed = vae2llm(x_t) + time_embedder(t) + latent_pos_embed(pos)
  2. Inject `embed` into the GEN slab of the input embedding tensor.
  3. Forward conditional (full sequence + gen_mask) → hidden states.
  4. v_cond  = llm2vae(hidden[gen_slab])
  5. Forward uncond (uncond sequence) → hidden states.
  6. v_uncond = llm2vae(hidden_uncond[gen_slab])
  7. CFG blend + global-norm rescale (Lance-specific):
        v_t = cfg_velocity(v_cond, v_uncond, scale)
  8. Euler: x_t -= v_t * dts[i]

Returns the final x_t (latent), ready for VAE decode.
"""
from __future__ import annotations

import time
from typing import Optional

import mlx.core as mx

from ..backbone import LanceLLM, LanceTextConfig
from ..rope import VisionSpec, build_positions_for_layout
from ..scheduler import (
    FlowMatchingSchedule, cfg_velocity, euler_step, make_schedule, sample_init_noise,
)
from ..attn_mask import build_lance_attention_mask


# Lance token IDs — also exposed via tokenizer special tokens; pinned here so
# the pipeline never silently breaks if tokenizer config is missing.
# (Verified at STAGE 1 from `Lance_3B/llm_config.json`.)
IM_START   = 151644      # <|im_start|>  (Qwen "bos" alias used by Lance for sequence open)
IM_END     = 151645      # <|im_end|>
VIS_START  = 151652
VIS_END    = 151653
IMG_TOKEN  = 151655      # generic vision placeholder used for both UND ViT tokens
                         #   and GEN VAE-latent placeholders.


def _build_t2i_sequence(prompt_ids: list[int], latent_shape: tuple[int, int, int]) -> dict:
    """Return a dict describing the cond + uncond sequences.

    latent_shape: (T_lat, H_lat, W_lat).  For 512² image: (1, 32, 32) → 1024 tokens.
    """
    t_lat, h_lat, w_lat = latent_shape
    n_latent = t_lat * h_lat * w_lat

    # ---- conditional sequence ----
    cond_ids = ([IM_START] + list(prompt_ids) + [IM_END]
                + [VIS_START] + [IMG_TOKEN] * n_latent + [VIS_END])
    # ---- unconditional sequence (CFG): drop prompt body, keep envelope ----
    uncond_ids = ([IM_START, IM_END]
                  + [VIS_START] + [IMG_TOKEN] * n_latent + [VIS_END])

    # GEN slab location (the run of placeholder image tokens inside <vision_*>)
    cond_gen_start   = 1 + len(prompt_ids) + 1 + 1            # after <vis_start>
    cond_gen_end     = cond_gen_start + n_latent
    uncond_gen_start = 2 + 1                                   # after [IM_START, IM_END, <vis_start>]
    uncond_gen_end   = uncond_gen_start + n_latent

    return {
        "cond_ids": cond_ids,
        "uncond_ids": uncond_ids,
        "cond_gen_span": (cond_gen_start, cond_gen_end),
        "uncond_gen_span": (uncond_gen_start, uncond_gen_end),
        "t_lat": t_lat,
        "h_lat": h_lat,
        "w_lat": w_lat,
        "n_latent": n_latent,
    }


def _latent_position_indices(t_lat: int, h_lat: int, w_lat: int,
                             max_latent_size: int = 64) -> mx.array:
    """Flat indices into the (max_T * max_S², hidden) `latent_pos_embed` table.

    Each latent token at (t_i, h_i, w_i) maps to
        t_i * max_latent_size² + h_i * max_latent_size + w_i.
    Row-major (t, h, w) matches PT's `get_flattened_position_ids_extrapolate_video`.

    `max_latent_size=64` per `checkpoints/Lance-3B-MLX/config.json`
    (image-only variant: max_num_latent_frames=1, table size 1·64²=4096).
    *Not* the LanceConfig default of 32 — that's the video variant.
    """
    t_idx = mx.arange(t_lat).reshape(t_lat, 1, 1)
    h_idx = mx.arange(h_lat).reshape(1, h_lat, 1)
    w_idx = mx.arange(w_lat).reshape(1, 1, w_lat)
    flat = (t_idx * (max_latent_size ** 2)
            + h_idx * max_latent_size
            + w_idx)                                          # (t, h, w)
    return mx.broadcast_to(flat, (t_lat, h_lat, w_lat)).flatten()    # (N,)


def _embed_with_gen_slab(
    model: LanceLLM,
    input_ids: mx.array,                # (1, L) int32
    x_t: mx.array,                      # (N, z_dim=48)  flat over GEN slab
    t_scalar: mx.array,                 # (1,) float
    latent_pos_ids: mx.array,           # (N,) int32 — flat indices into pos_embed table
    gen_span: tuple[int, int],
) -> mx.array:
    """Build the (1, L, hidden) input embedding for the transformer.

    Text positions get `embed_tokens(input_ids)`.  GEN slab positions get
    `vae2llm(x_t) + time_embedder(t) + latent_pos_embed(pos)`.
    """
    text_embed = model.language_model.model.embed_tokens(input_ids)    # (1, L, D)
    # Build the GEN slab embed:
    vae_proj   = model.vae2llm(x_t)                                    # (N, D)
    t_emb      = model.time_embedder(t_scalar)                         # (1, D)
    pos_emb    = model.latent_pos_embed(latent_pos_ids)                # (N, D)
    slab_embed = vae_proj + t_emb + pos_emb                            # (N, D)
    # Splice: text_embed at [0..gen_start) and [gen_end..L), slab_embed in the middle.
    gs, ge = gen_span
    return mx.concatenate([
        text_embed[:, :gs, :],
        slab_embed[None, :, :],                                        # (1, N, D)
        text_embed[:, ge:, :],
    ], axis=1)


def _make_gen_mask(L: int, gen_span: tuple[int, int]) -> mx.array:
    gs, ge = gen_span
    cols = mx.arange(L)
    return ((cols >= gs) & (cols < ge))[None, :]


def _make_positions(L: int, gen_span: tuple[int, int], t_lat: int,
                    h_lat: int, w_lat: int) -> mx.array:
    """Build the (3, 1, L) mRoPE position_ids using STAGE 3 helpers.

    No `shift_position_ids` here.  PT's shift (lance.py:249, pos_shift=1000
    with pro_type=10) is a *no-op for T2I* because it only fires on
    sample_modality ∈ {2, 3, 4} which are TI2I/refedit-specific
    (validation_dataset.py:46-53: text=0, noise=1, ref_source=2,
    ref_image=3, ref_vit=4).  T2I sequence is purely {0, 1}.
    Empirically verified at `stage6_pt_denoise_compare.py`: with no shift
    here, step-by-step latent cos ≥ 0.99963 vs original PT denoise.
    """
    gs, ge = gen_span
    span = VisionSpec(start=gs - 1,         # <vision_start> sits at gs-1
                       length=ge - gs,
                       t=t_lat, h=h_lat, w=w_lat)
    return build_positions_for_layout(L, [span])


def t2i(
    model: LanceLLM,
    tokenizer,
    prompt: str,
    *,
    height: int = 256,
    width: int = 256,
    num_steps: int = 30,
    timestep_shift: float = 3.5,
    cfg_scale: float = 4.0,
    cfg_renorm_type: str = "global",
    cfg_renorm_min: float = 0.0,
    seed: int = 0,
    z_dim: int = 48,
    spatial_downsample: int = 16,
) -> dict:
    """Generate a single image latent.  Returns dict with x_t, schedule, etc."""

    # ---- shapes ----
    h_lat = height // spatial_downsample
    w_lat = width // spatial_downsample
    t_lat = 1
    print(f"[t2i] shape: {height}×{width} pixel → latent (t={t_lat}, h={h_lat}, w={w_lat}, c={z_dim})")

    # ---- tokenize ----
    prompt_ids = tokenizer(prompt, return_tensors=None, add_special_tokens=False)["input_ids"]
    layout = _build_t2i_sequence(prompt_ids, (t_lat, h_lat, w_lat))
    cond_ids   = mx.array([layout["cond_ids"]],   dtype=mx.int32)
    uncond_ids = mx.array([layout["uncond_ids"]], dtype=mx.int32)
    L_cond     = cond_ids.shape[1]
    L_uncond   = uncond_ids.shape[1]
    cond_gs, cond_ge     = layout["cond_gen_span"]
    uncond_gs, uncond_ge = layout["uncond_gen_span"]
    n_latent = layout["n_latent"]
    print(f"[t2i] seq lens: cond={L_cond}, uncond={L_uncond}  "
          f"prompt_tokens={len(prompt_ids)}  N_latent={n_latent}")

    # ---- position ids (both sequences) ----
    cond_pos   = _make_positions(L_cond,   (cond_gs, cond_ge),     t_lat, h_lat, w_lat)
    uncond_pos = _make_positions(L_uncond, (uncond_gs, uncond_ge), t_lat, h_lat, w_lat)

    # ---- attention masks (Lance bidirectional-within-noise-slab) ----
    # text prefix (causal) | latent slab (noise) | VIS_END (causal) — same split
    # convention as `stage6_pt_denoise_compare.py`, verified bit-equivalent
    # to PT `create_sparse_mask` on this layout.
    cond_prefix_len   = cond_gs                 # includes <vision_start>
    uncond_prefix_len = uncond_gs
    cond_attn_mask = build_lance_attention_mask(
        seq_len=L_cond,
        split_lens=[cond_prefix_len, n_latent, 1],
        attn_modes=["causal", "noise", "causal"],
    )
    uncond_attn_mask = build_lance_attention_mask(
        seq_len=L_uncond,
        split_lens=[uncond_prefix_len, n_latent, 1],
        attn_modes=["causal", "noise", "causal"],
    )

    # ---- gen masks for MoE-gen routing ----
    cond_gen_mask   = _make_gen_mask(L_cond,   (cond_gs, cond_ge))
    uncond_gen_mask = _make_gen_mask(L_uncond, (uncond_gs, uncond_ge))

    # ---- latent position indices into latent_pos_embed table ----
    latent_pos_ids = _latent_position_indices(t_lat, h_lat, w_lat)

    # ---- noise init (flow matching x_1) ----
    x_t = sample_init_noise((n_latent, z_dim), seed=seed)         # (N, z_dim)
    print(f"[t2i] x_t init: shape={x_t.shape}  mean={x_t.mean().item():+.4f}  std={x_t.std().item():.4f}")

    # ---- schedule ----
    sch = make_schedule(num_steps=num_steps, timestep_shift=timestep_shift)
    print(f"[t2i] {num_steps} steps, shift={timestep_shift}  "
          f"t[0]={sch.timesteps[0].item():.4f}  t[-1]={sch.timesteps[-1].item():.4f}")

    # ---- denoising loop ----
    t0 = time.time()
    for i in range(sch.num_steps):
        t_scalar = sch.timesteps[i:i+1]                              # (1,)

        # conditional forward
        cond_embed = _embed_with_gen_slab(
            model, cond_ids, x_t, t_scalar, latent_pos_ids, (cond_gs, cond_ge),
        )
        cond_hidden = model.language_model.model(
            input_ids=None, position_ids=cond_pos,
            inputs_embeds=cond_embed, mask=cond_attn_mask, gen_mask=cond_gen_mask,
        )
        v_cond = model.llm2vae(cond_hidden[0, cond_gs:cond_ge, :])    # (N, z_dim)

        # unconditional forward (text-dropped)
        uncond_embed = _embed_with_gen_slab(
            model, uncond_ids, x_t, t_scalar, latent_pos_ids, (uncond_gs, uncond_ge),
        )
        uncond_hidden = model.language_model.model(
            input_ids=None, position_ids=uncond_pos,
            inputs_embeds=uncond_embed, mask=uncond_attn_mask, gen_mask=uncond_gen_mask,
        )
        v_uncond = model.llm2vae(uncond_hidden[0, uncond_gs:uncond_ge, :])  # (N, z_dim)

        # CFG blend (+ Lance norm rescale)
        v_t = cfg_velocity(
            v_cond, v_uncond,
            scale=cfg_scale, renorm_type=cfg_renorm_type, renorm_min=cfg_renorm_min,
        )
        # Euler step
        x_t = euler_step(x_t, v_t, sch.dts[i])
        mx.eval(x_t)
        if (i + 1) % 5 == 0 or i == 0:
            print(f"[t2i] step {i+1:3d}/{sch.num_steps}  t={t_scalar.item():.4f}  "
                  f"||v_cond||={mx.linalg.norm(v_cond).item():.2f}  "
                  f"||x_t||={mx.linalg.norm(x_t).item():.2f}  "
                  f"({time.time()-t0:.1f}s)")

    print(f"[t2i] denoising done in {time.time()-t0:.1f}s")

    # Reshape flat (N, z_dim) → (1, t_lat, h_lat, w_lat, z_dim) for VAE decode.
    latent = x_t.reshape(1, t_lat, h_lat, w_lat, z_dim)
    return {
        "latent": latent,
        "schedule": sch,
        "x_t_flat": x_t,
        "n_latent": n_latent,
    }
