"""TI2I (text-conditioned image edit) pipeline.

Sequence layout (PT ref + RockTalk README):

    <|im_start|>system\n[EDIT_SYSTEM_PROMPT]<|im_end|>
    <|im_start|>user
      <|vision_start|>[N_vit ViT-placeholders]<|vision_end|>[instruction]
    <|im_end|>
    <|im_start|>assistant
    <|vision_start|>[N_cond VAE-encode-of-cond-image placeholders]<|vision_end|>
    <|vision_start|>[N_noise target-noise placeholders]<|vision_end|>

Routing: ViT tokens and VAE-cond tokens are UND; only the noise-target
slab is GEN (`moe_gen` weights).  Three-component CFG mixes:
  v_full       — all three conditions present (ViT + VAE-cond + text)
  v_t_uncond   — text dropped, ViT + VAE-cond present
  v_tv_uncond  — text + ViT dropped, VAE-cond present
  v_final = v_tv_uncond + cfg_text*(v_full - v_t_uncond) + cfg_vit*(v_t_uncond - v_tv_uncond)

Then Lance global-norm rescale (same as T2I CFG renorm).

Status: skeleton + smoke.  Full PT-direct cosine verification deferred
to its own harness (`tools/stage7_ti2i_compare.py` when needed).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from PIL import Image

from ..backbone import LanceLLM
from ..vit import LanceViT
from ..vae_wan22 import Wan2_2_VAE
from ..rope import VisionSpec, build_positions_for_layout
from ..attn_mask import build_lance_attention_mask
from ..scheduler import (
    make_schedule, cfg_velocity_3comp, euler_step, sample_init_noise,
)
from .x2t import (
    preprocess_image, IM_START_ID, IM_END_ID, VIS_START_ID, VIS_END_ID,
    IMG_TOKEN_ID, PATCH_SIZE, SPATIAL_MERGE_SIZE, TEMPORAL_PATCH_SIZE,
    QWEN_VL_IMAGE_MEAN, QWEN_VL_IMAGE_STD,
)
from .t2v import VAE_SCALE_MEAN, VAE_SCALE_STD   # Wan VAE per-channel decode scale


# Lance edit-mode system prompt (verbatim from `refs/Lance/data/common.py:35`,
# `system_prompt_type` containing "edit", `vision_type="image"`).
EDIT_SYSTEM_PROMPT = (
    "Describe the key features of the input image "
    "(color, shape, size, texture, objects, background), "
    "then explain how the user’s text instruction should alter or modify the image. "
    "Generate a new image that meets the user’s requirements "
    "while maintaining consistency with the original input where appropriate."
)


# z_dim and spatial downsample for Lance VAE
Z_DIM = 48
SPATIAL_DOWNSAMPLE = 16


def _vae_preprocess(image_path: str, size: int = 256) -> mx.array:
    """Load image, resize to square, normalize to [-1, 1] for VAE encode.

    Returns (1, 1, size, size, 3) NTHWC.
    """
    img = Image.open(image_path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = (np.asarray(img, dtype=np.float32) / 127.5) - 1.0     # [-1, 1]
    arr = arr[None, None, :, :, :]                               # (1, 1, H, W, 3)
    return mx.array(arr)


def _latent_position_indices(t_lat: int, h_lat: int, w_lat: int,
                             max_latent_size: int = 64) -> mx.array:
    """Flat lookup indices for `latent_pos_embed` (max_latent_size=64 image)."""
    t_idx = mx.arange(t_lat).reshape(t_lat, 1, 1)
    h_idx = mx.arange(h_lat).reshape(1, h_lat, 1)
    w_idx = mx.arange(w_lat).reshape(1, 1, w_lat)
    flat = (t_idx * (max_latent_size ** 2)
            + h_idx * max_latent_size
            + w_idx)
    return mx.broadcast_to(flat, (t_lat, h_lat, w_lat)).flatten()


@dataclass
class TI2IResult:
    latent: mx.array       # (1, T_lat, H_lat, W_lat, Z_DIM)
    image_recon: mx.array  # (1, 1, H_pix, W_pix, 3) in [-1, 1]


def image_edit(
    model: LanceLLM,
    vit: LanceViT,
    vae: Wan2_2_VAE,
    tokenizer,
    cond_image_path: str,
    instruction: str,
    *,
    size: int = 256,
    num_steps: int = 24,
    timestep_shift: float = 3.5,
    cfg_text: float = 3.0,
    cfg_vit: float = 1.0,
    cfg_renorm_type: str = "global",
    cfg_renorm_min: float = 0.0,
    seed: int = 0,
) -> TI2IResult:
    """TI2I single edit.  Returns latent + reconstructed image."""

    h_lat = size // SPATIAL_DOWNSAMPLE
    w_lat = size // SPATIAL_DOWNSAMPLE
    t_lat = 1

    # ---- preprocess cond image (twice: ViT path + VAE path) ----
    vit_patches, (T_g, H_g, W_g) = preprocess_image(cond_image_path)
    grid_thw = mx.array([[T_g, H_g, W_g]], dtype=mx.int32)
    visual_und = vit(vit_patches, grid_thw)                       # (N_vit, 2048)
    n_vit = int(visual_und.shape[0])

    vae_in = _vae_preprocess(cond_image_path, size=size)          # (1, 1, S, S, 3)
    cond_latent = vae.encode(vae_in)                              # (1, 1, h_lat, w_lat, 48)
    cond_flat = cond_latent.reshape(t_lat * h_lat * w_lat, Z_DIM) # (N_cond, 48)
    n_cond = int(cond_flat.shape[0])

    n_noise = t_lat * h_lat * w_lat                                # target slab
    print(f"[ti2i] image={size}² ViT tokens={n_vit}  VAE-cond tokens={n_cond}  "
          f"noise target={n_noise}  shape=({t_lat},{h_lat},{w_lat})")

    # ---- chat sequence ids ----
    sys_ids   = tokenizer(EDIT_SYSTEM_PROMPT, add_special_tokens=False)["input_ids"]
    inst_ids  = tokenizer(instruction,         add_special_tokens=False)["input_ids"]
    newline   = tokenizer("\n",                add_special_tokens=False)["input_ids"]
    sys_lbl   = tokenizer("system",            add_special_tokens=False)["input_ids"]
    usr_lbl   = tokenizer("user",              add_special_tokens=False)["input_ids"]
    asst_lbl  = tokenizer("assistant",         add_special_tokens=False)["input_ids"]

    def _build_seq(*, include_text: bool, include_vit: bool):
        """Three sequence variants for 3-component CFG."""
        sys_section  = ([IM_START_ID] + sys_lbl + newline
                        + (sys_ids if include_text else [])
                        + [IM_END_ID] + newline)
        # user header
        user_open = [IM_START_ID] + usr_lbl + newline
        # ViT slab (optional)
        vit_slab = ([VIS_START_ID] + [IMG_TOKEN_ID] * n_vit + [VIS_END_ID]
                    if include_vit else [])
        # instruction
        inst_section = (inst_ids if include_text else []) + [IM_END_ID] + newline
        # assistant header
        asst_open = [IM_START_ID] + asst_lbl + newline
        # VAE-cond slab + noise target slab
        vae_slab   = [VIS_START_ID] + [IMG_TOKEN_ID] * n_cond  + [VIS_END_ID]
        noise_slab = [VIS_START_ID] + [IMG_TOKEN_ID] * n_noise + [VIS_END_ID]
        full = (sys_section + user_open + vit_slab + inst_section
                + asst_open + vae_slab + noise_slab)
        # Compute slab positions (start indices)
        cursor = 0
        cursor += len(sys_section) + len(user_open)
        vit_start = (cursor + 1) if include_vit else -1                 # +1 past VIS_START
        cursor += len(vit_slab) + len(inst_section) + len(asst_open)
        vae_start   = cursor + 1                                         # +1 past VIS_START
        cursor += len(vae_slab)
        noise_start = cursor + 1
        return {
            "ids": full,
            "L": len(full),
            "vit_span":   (vit_start, vit_start + n_vit) if include_vit else None,
            "vae_span":   (vae_start, vae_start + n_cond),
            "noise_span": (noise_start, noise_start + n_noise),
        }

    full_layout       = _build_seq(include_text=True,  include_vit=True)
    t_uncond_layout   = _build_seq(include_text=False, include_vit=True)
    tv_uncond_layout  = _build_seq(include_text=False, include_vit=False)

    print(f"[ti2i] seq lens: full={full_layout['L']}  t_uncond={t_uncond_layout['L']}  "
          f"tv_uncond={tv_uncond_layout['L']}")

    # ---- scheduler + noise init ----
    sch = make_schedule(num_steps=num_steps, timestep_shift=timestep_shift)
    x_t = sample_init_noise((n_noise, Z_DIM), seed=seed)         # (N_noise, 48)

    # ---- per-step helper (mirror PT validation_gen step) ----
    def _forward_v(layout: dict, x_t_cur: mx.array, t_scalar: mx.array,
                   latent_pos_ids: mx.array) -> mx.array:
        """Inject ViT (if present) + VAE-cond + noise embeds into the sequence
        and forward through LanceLLM.  Returns v_t at the noise slab.

        Mirrors PT `lance.py:validation_gen` step body:
          - All latent tokens (cond + noise) get vae2llm + time_embedder
            + latent_pos_embed; cond timestep=0, noise timestep=current_t.
          - All latent tokens route through GEN (moe_gen) siblings —
            packed_gen_token_indexes = ALL VAE positions.
          - mRoPE: noise slab positions copied from cond slab
            (`shift_position_ids` pro_type=10 modality==2→1).
          - mRoPE: ViT slab base shifted to 1000 (pro_type=10 modality==4).
          - Attention mask: ViT 'full', VAE-cond 'full', noise 'noise',
            text/separators causal.
        """
        ids = mx.array([layout["ids"]], dtype=mx.int32)
        L = layout["L"]
        vae_s, vae_e = layout["vae_span"]
        noise_s, noise_e = layout["noise_span"]
        # Pre-condition for Fix C position copy below.  modality==1 ← modality==2
        # requires equal counts; future TI2I-V or refedit with mis-sized cond
        # would silently corrupt positions otherwise.
        assert (noise_e - noise_s) == (vae_e - vae_s), (
            f"cond span {vae_e - vae_s} != noise span {noise_e - noise_s}; "
            "Fix C position copy requires equal widths"
        )

        text_embed = model.language_model.model.embed_tokens(ids)        # (1, L, D)

        # ViT slab (if present): replace placeholders with `visual_und`.
        embed = text_embed
        if layout["vit_span"] is not None:
            vit_s, vit_e = layout["vit_span"]
            embed = mx.concatenate([
                embed[:, :vit_s, :], visual_und[None, :, :], embed[:, vit_e:, :],
            ], axis=1)

        # FIX A: combined latent embed (cond + noise) with per-token timestep.
        # PT lance.py:663-666 applies vae2llm + time_embedder + latent_pos_embed
        # to ALL latent tokens; cond timestep=0, noise timestep=current_t.
        # MLX `time_embedder(scalar)` broadcasts — call once per timestep group.
        t_zero = mx.zeros_like(t_scalar)
        vae_cond_embed = (model.vae2llm(cond_flat)
                          + model.time_embedder(t_zero)
                          + model.latent_pos_embed(latent_pos_ids))       # (N_cond, D)
        embed = mx.concatenate([
            embed[:, :vae_s, :], vae_cond_embed[None, :, :], embed[:, vae_e:, :],
        ], axis=1)

        noise_embed = (model.vae2llm(x_t_cur)
                       + model.time_embedder(t_scalar)
                       + model.latent_pos_embed(latent_pos_ids))           # (N_noise, D)
        embed = mx.concatenate([
            embed[:, :noise_s, :], noise_embed[None, :, :], embed[:, noise_e:, :],
        ], axis=1)

        # FIX B: gen_mask True for BOTH cond and noise slabs.  PT line 681:
        # packed_gen_token_indexes = current_vae_token_indexes_local (all VAE).
        cols = mx.arange(L)
        gen_mask = (((cols >= vae_s) & (cols < vae_e))
                    | ((cols >= noise_s) & (cols < noise_e)))[None, :]

        # positions: standard Qwen mRoPE through `build_positions_for_layout`.
        spans = []
        if layout["vit_span"] is not None:
            spans.append(VisionSpec(start=vit_s - 1, length=n_vit,
                                     t=T_g, h=H_g // SPATIAL_MERGE_SIZE,
                                     w=W_g // SPATIAL_MERGE_SIZE))
        spans.append(VisionSpec(start=vae_s - 1, length=n_cond,
                                 t=t_lat, h=h_lat, w=w_lat))
        spans.append(VisionSpec(start=noise_s - 1, length=n_noise,
                                 t=t_lat, h=h_lat, w=w_lat))
        pos = build_positions_for_layout(L, spans)

        # FIX C: apply pro_type=10 shifts.  PT common.py:60-67.
        #   - modality==4 (ViT): **T axis only** shifted by (1000 - T_first)
        #     (`position_ids[0, :, mask] += shift`)
        #   - modality==1 (noise) ← modality==2 (cond) positions (all 3 axes)
        # `pos` is (3, 1, L) MLX array.  Mutate via numpy then back.
        pos_np = np.asarray(pos)
        if layout["vit_span"] is not None:
            shift = 1000 - int(pos_np[0, 0, vit_s])
            pos_np[0, :, vit_s:vit_e] += shift
        # Copy cond positions → noise positions (all axes).  Pre-asserted equal widths.
        pos_np[:, :, noise_s:noise_e] = pos_np[:, :, vae_s:vae_e]
        pos = mx.array(pos_np)

        # FIX D: attention mask per PT validation_dataset.py:518-532.
        # Each vision slab includes vis_start + placeholders + vis_end as
        # ONE split (not three separately).
        split_lens, attn_modes = [], []
        if layout["vit_span"] is not None:
            sl_pre_vit = vit_s - 1                       # text before vis_start(vit)
            sl_vit     = (vit_e - vit_s) + 2             # vis_start + N + vis_end
            mid_start  = vit_e + 1
            split_lens += [sl_pre_vit, sl_vit]
            attn_modes += ["causal", "full"]
        else:
            mid_start = 0
        sl_mid   = (vae_s - 1) - mid_start               # text up to vis_start(vae)
        sl_vae   = (vae_e - vae_s) + 2                   # vae slab (full_noise→full)
        sl_noise = (noise_e - noise_s) + 2               # noise slab
        sl_tail  = L - (noise_e + 1)
        split_lens += [sl_mid, sl_vae, sl_noise]
        attn_modes += ["causal", "full_noise", "noise"]
        if sl_tail > 0:
            split_lens.append(sl_tail); attn_modes.append("causal")
        attn_mask = build_lance_attention_mask(seq_len=L, split_lens=split_lens,
                                                attn_modes=attn_modes)

        # Forward
        hidden = model.language_model.model(
            input_ids=None, position_ids=pos, inputs_embeds=embed,
            mask=attn_mask, gen_mask=gen_mask,
        )
        # v_t = llm2vae at the noise slab
        return model.llm2vae(hidden[0, noise_s:noise_e, :])                  # (N_noise, 48)

    # ---- shared latent_pos_ids ----
    latent_pos_ids = _latent_position_indices(t_lat, h_lat, w_lat)

    # ---- denoising loop ----
    t0 = time.time()
    for i in range(num_steps):
        t_scalar = sch.timesteps[i:i+1]
        v_full      = _forward_v(full_layout,      x_t, t_scalar, latent_pos_ids)
        v_t_uncond  = _forward_v(t_uncond_layout,  x_t, t_scalar, latent_pos_ids)
        v_tv_uncond = _forward_v(tv_uncond_layout, x_t, t_scalar, latent_pos_ids)
        v_final = cfg_velocity_3comp(
            v_full, v_t_uncond, v_tv_uncond,
            cfg_text=cfg_text, cfg_vit=cfg_vit,
            renorm_type=cfg_renorm_type, renorm_min=cfg_renorm_min,
        )
        x_t = euler_step(x_t, v_final, sch.dts[i])
        mx.eval(x_t)
        if (i + 1) % 5 == 0 or i == 0:
            print(f"[ti2i] step {i+1:3d}/{num_steps}  t={t_scalar.item():.4f}  "
                  f"||v_full||={mx.linalg.norm(v_full).item():.2f}  "
                  f"||x_t||={mx.linalg.norm(x_t).item():.2f}  "
                  f"({time.time()-t0:.1f}s)")

    latent = x_t.reshape(1, t_lat, h_lat, w_lat, Z_DIM)
    # Decode with the production Wan-VAE scale (un-normalize); without it the
    # dynamic range is off / oversaturated (same bug t2i had — PT's WanVAE.decode
    # always applies this scale).
    vae_scale = (mx.array(VAE_SCALE_MEAN), mx.array(1.0 / VAE_SCALE_STD))
    image = vae.decode(latent, scale=vae_scale)
    mx.eval(image)
    return TI2IResult(latent=latent, image_recon=image)
