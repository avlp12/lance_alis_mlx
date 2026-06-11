"""TIV2V (text-conditioned video edit) pipeline — STAGE 11, the 6th/last task.

PT `tiv2v_sample` is *image_edit's method on the video path* (validation_dataset.py
:923, `elif element_dtype in ["image", "video"]`).  Confirmed from PT source — the
xocialize "input video skips ViT" hint is WRONG: line 924 `if is_target == 0:` is
commented "condition 需要 vit 处理" and runs the cond media through BOTH the ViT
(`vit_video` stream) AND the VAE (`vae_video` stream) for image and video alike.

So video_edit = ASSEMBLY of already-verified pieces (no new algorithm).  The only
new risk is the wiring (3-slab layout + mRoPE position combination); that is what
the non-blind harness validates.

  layout (per PT, same as image_edit with T>1):
    <|im_start|>system\n[EDIT_SYSTEM_PROMPT_VIDEO]<|im_end|>
    <|im_start|>user
      <|vision_start|>[N_vit  video-ViT placeholders]<|vision_end|>[instruction]
    <|im_end|>
    <|im_start|>assistant
    <|vision_start|>[N_cond VAE-encode-of-cond-video placeholders]<|vision_end|>
    <|vision_start|>[N_noise target-noise placeholders]<|vision_end|>

  reused (verified) components:
    - ViT-cond            : x2t.preprocess_video + video ViT, VisionSpec temporal_scale=2   (STAGE 11)
    - VAE-cond / noise     : vae.encode T>1 (t_lat=(N-1)//4+1), scale=(mean,1/std)            (STAGE 8)
    - latent_pos_embed     : t2v.vae_latent_position_indices(t,h,w, max=64)                   (STAGE 9)
    - VAE/noise mRoPE      : VisionSpec temporal_scale=2 (== build_t2v_positions tps);
                             noise positions <- cond positions (pro_type=10 modality 1<-2)
    - 3-component CFG      : image_edit velocity machine + scheduler.cfg_velocity_3comp        (STAGE 11 re-verified)
    - decode               : Wan VAE T>1 with production scale (else dynamic range 1.5x off)   (STAGE 8/9)
    - model                : standalone video weight (Lance_3B_Video, latent_pos_embed 31x64^2)

Rule 3 (surgical): the verified functions are *imported*, never edited.  Only the
assembly (this file) is new.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import mlx.core as mx
from PIL import Image

from ..backbone import LanceLLM, LanceTextConfig, PositionEmbedding3D
from ..vit import LanceViT
from ..vae_wan22 import Wan2_2_VAE, Wan22VAEConfig
from ..rope import VisionSpec, build_positions_for_layout
from ..attn_mask import build_lance_attention_mask
from ..scheduler import make_schedule, cfg_velocity_3comp, euler_step, sample_init_noise
from .x2t import (
    preprocess_video, load_video_vit,
    VIDEO_TEMPORAL_SCALE, SPATIAL_MERGE_SIZE,
    IM_START_ID, IM_END_ID, VIS_START_ID, VIS_END_ID, IMG_TOKEN_ID, VIDEO_PAD_ID,
)
from .t2v import (
    vae_latent_position_indices,
    VAE_SCALE_MEAN, VAE_SCALE_STD,
    VAE_DOWN_TEMPORAL, VAE_DOWN_SPATIAL,
    MAX_NUM_LATENT_FRAMES, MAX_LATENT_SIZE,
)
from .image_edit import Z_DIM


# Edit system prompt, video variant — PT `generate_system_prompt(system_prompt_type
# "edit", vision_type="video")` (common.py:35).  Identical to image_edit's
# EDIT_SYSTEM_PROMPT with "image" -> "video".
EDIT_SYSTEM_PROMPT_VIDEO = (
    "Describe the key features of the input video "
    "(color, shape, size, texture, objects, background), "
    "then explain how the user’s text instruction should alter or modify the video. "
    "Generate a new video that meets the user’s requirements "
    "while maintaining consistency with the original input where appropriate."
)

VIDEO_WEIGHT_DEFAULT = "out/lance_3b_video_mlx/model.safetensors"
VAE_WEIGHT_DEFAULT   = "checkpoints/Wan2.2-VAE-MLX/model.safetensors"


# ---------------------------------------------------------------------------
# Model loading — video standalone weight (LLM + gen adapters + video ViT).
# ---------------------------------------------------------------------------
def load_video_edit_models(video_weight: str = VIDEO_WEIGHT_DEFAULT,
                           vae_weight: str = VAE_WEIGHT_DEFAULT):
    """Load the video LanceLLM (gen path, latent_pos_embed 31x64^2), the video
    ViT, and the Wan VAE.

    Mirrors the STAGE 9 verified loading (stage9_mlx_30step.py): build the image
    backbone, REPLACE latent_pos_embed with the video PositionEmbedding3D(31, 64),
    then filter-load the standalone video weight (load_full_lance can't be used —
    it strict-rejects the bundled `vit_model.*` keys).
    """
    from mlx.utils import tree_flatten

    cfg = LanceTextConfig()
    model = LanceLLM(cfg)
    # image latent_pos_embed (1*64^2=4096) -> video (31*64^2=126976)
    model.latent_pos_embed = PositionEmbedding3D(
        max_num_latent_frames=MAX_NUM_LATENT_FRAMES,   # 31
        max_latent_size=MAX_LATENT_SIZE,               # 64
        hidden_size=cfg.hidden_size,
    )
    full = mx.load(video_weight)
    ours = set(dict(tree_flatten(model.parameters())).keys())
    to_load = {k: v for k, v in full.items() if k in ours}
    model.load_weights(list(to_load.items()), strict=True)   # strict over the LLM+adapter set
    mx.eval(model.parameters())
    model.eval()

    vit = LanceViT()
    load_video_vit(vit, video_weight)                        # vit_model.* -> vision_tower.*
    vit.eval()

    vae = Wan2_2_VAE(Wan22VAEConfig())
    vae.load_weights(list(mx.load(vae_weight).items()), strict=True)
    mx.eval(vae.parameters()); vae.eval()
    return model, vit, vae


# ---------------------------------------------------------------------------
# Cond-video VAE preprocessing — frames -> (1, N, H, W, 3) in [-1, 1].
# ---------------------------------------------------------------------------
def _vae_preprocess_video(frames: np.ndarray, *, H: int, W: int) -> mx.array:
    """Resize sampled cond frames to (H, W) (multiples of VAE_DOWN_SPATIAL) and
    normalize to [-1, 1].  Returns NTHWC = (1, N, H, W, 3).

    N is the number of *sampled* cond frames (the VAE temporally compresses them
    to t_lat=(N-1)//4+1 inside encode); spatial gives h_lat=H//16, w_lat=W//16.
    """
    N = int(frames.shape[0])
    out = np.empty((N, H, W, 3), dtype=np.float32)
    for i in range(N):
        im = Image.fromarray(frames[i]).convert("RGB").resize((W, H), Image.BICUBIC)
        out[i] = (np.asarray(im, dtype=np.float32) / 127.5) - 1.0      # [-1, 1]
    return mx.array(out[None, ...])                                    # (1, N, H, W, 3)


@dataclass
class VideoEditResult:
    latent: mx.array        # (1, t_lat, h_lat, w_lat, Z_DIM)
    video_recon: mx.array   # (1, t_pix, h_pix, w_pix, 3) in [-1, 1]
    t_lat: int
    h_lat: int
    w_lat: int


# ---------------------------------------------------------------------------
# Layout + position builders — MODULE LEVEL so the non-blind harness exercises
# the *exact* pipeline code (not a re-implementation that could drift).
# ---------------------------------------------------------------------------
def build_video_edit_layouts(tokenizer, instruction: str,
                             n_vit: int, n_cond: int, n_noise: int) -> dict:
    """Build the 3 CFG-variant token layouts (full / t_uncond / tv_uncond).

    Identical structure to image_edit (system + user[ViT-cond] + instruction +
    assistant[VAE-cond + noise]); only the system prompt is the video variant and
    the counts are the video grids.  All vision placeholders use VIDEO_PAD_ID
    (cosmetic for our forward — embeds overwrite — but it is what PT's
    get_rope_index keys on as a *video* span, so the harness can reuse this).
    """
    sys_ids  = tokenizer(EDIT_SYSTEM_PROMPT_VIDEO, add_special_tokens=False)["input_ids"]
    inst_ids = tokenizer(instruction,              add_special_tokens=False)["input_ids"]
    newline  = tokenizer("\n",                     add_special_tokens=False)["input_ids"]
    sys_lbl  = tokenizer("system",                 add_special_tokens=False)["input_ids"]
    usr_lbl  = tokenizer("user",                   add_special_tokens=False)["input_ids"]
    asst_lbl = tokenizer("assistant",              add_special_tokens=False)["input_ids"]

    def _seq(*, include_text: bool, include_vit: bool):
        sys_section = ([IM_START_ID] + sys_lbl + newline
                       + (sys_ids if include_text else [])
                       + [IM_END_ID] + newline)
        user_open = [IM_START_ID] + usr_lbl + newline
        vit_slab = ([VIS_START_ID] + [VIDEO_PAD_ID] * n_vit + [VIS_END_ID]
                    if include_vit else [])
        inst_section = (inst_ids if include_text else []) + [IM_END_ID] + newline
        asst_open = [IM_START_ID] + asst_lbl + newline
        vae_slab   = [VIS_START_ID] + [VIDEO_PAD_ID] * n_cond  + [VIS_END_ID]
        noise_slab = [VIS_START_ID] + [VIDEO_PAD_ID] * n_noise + [VIS_END_ID]
        full = (sys_section + user_open + vit_slab + inst_section
                + asst_open + vae_slab + noise_slab)
        cursor = len(sys_section) + len(user_open)
        vit_start = (cursor + 1) if include_vit else -1
        cursor += len(vit_slab) + len(inst_section) + len(asst_open)
        vae_start   = cursor + 1
        cursor += len(vae_slab)
        noise_start = cursor + 1
        return {
            "ids": full, "L": len(full),
            "vit_span":   (vit_start, vit_start + n_vit) if include_vit else None,
            "vae_span":   (vae_start, vae_start + n_cond),
            "noise_span": (noise_start, noise_start + n_noise),
        }

    return {
        "v_full":      _seq(include_text=True,  include_vit=True),
        "v_t_uncond":  _seq(include_text=False, include_vit=True),
        "v_tv_uncond": _seq(include_text=False, include_vit=False),
    }


def build_video_edit_positions(layout: dict, *, T_g: int, H_g_m: int, W_g_m: int,
                               t_lat: int, h_lat: int, w_lat: int) -> np.ndarray:
    """mRoPE (3, 1, L) positions for one layout — the assembly under test.

    Equivalent to PT production: get_rope_index(video grids, sec=1 -> temporal x2
    for ViT-cond AND VAE-cond) then shift_position_ids(pro_type=10): ViT T-axis ->
    1000, noise <- cond.  Here: build_positions_for_layout with temporal_scale=2
    on every video slab + the manual pro_type=10 shifts.
    """
    L = layout["L"]
    vae_s, vae_e = layout["vae_span"]
    noise_s, noise_e = layout["noise_span"]
    spans = []
    if layout["vit_span"] is not None:
        vit_s, vit_e = layout["vit_span"]
        spans.append(VisionSpec(start=vit_s - 1, length=vit_e - vit_s,
                                t=T_g, h=H_g_m, w=W_g_m, temporal_scale=VIDEO_TEMPORAL_SCALE))
    spans.append(VisionSpec(start=vae_s - 1, length=vae_e - vae_s,
                            t=t_lat, h=h_lat, w=w_lat, temporal_scale=VIDEO_TEMPORAL_SCALE))
    spans.append(VisionSpec(start=noise_s - 1, length=noise_e - noise_s,
                            t=t_lat, h=h_lat, w=w_lat, temporal_scale=VIDEO_TEMPORAL_SCALE))
    pos_np = np.asarray(build_positions_for_layout(L, spans))
    # pro_type=10 shifts (PT common.py:60-67): ViT T-axis -> 1000; noise <- cond.
    if layout["vit_span"] is not None:
        vit_s, vit_e = layout["vit_span"]
        pos_np[0, :, vit_s:vit_e] += 1000 - int(pos_np[0, 0, vit_s])
    pos_np[:, :, noise_s:noise_e] = pos_np[:, :, vae_s:vae_e]
    return pos_np


# ---------------------------------------------------------------------------
# Main entry.
# ---------------------------------------------------------------------------
def video_edit(
    model: LanceLLM,
    vit: LanceViT,
    vae: Wan2_2_VAE,
    tokenizer,
    cond_frames: np.ndarray,        # (N, H, W, 3) uint8 — already sampled (video_io)
    instruction: str,
    *,
    vae_size: tuple[int, int] = (256, 256),   # (H, W) of the edited video (multiple of 16)
    vit_max_pixels: int = 14 * 14 * 12 * 12,
    num_steps: int = 24,
    timestep_shift: float = 3.5,
    cfg_text: float = 3.0,
    cfg_vit: float = 1.0,
    cfg_renorm_type: str = "global",
    cfg_renorm_min: float = 0.0,
    seed: int = 0,
) -> VideoEditResult:
    """TIV2V single video edit.  Returns latent + reconstructed video."""

    H_vae, W_vae = vae_size
    assert H_vae % VAE_DOWN_SPATIAL == 0 and W_vae % VAE_DOWN_SPATIAL == 0, \
        f"vae_size must be multiples of {VAE_DOWN_SPATIAL}"

    # ======================================================================
    # Slab 1 — ViT-cond  (x2t_video path: preprocess_video -> video ViT)
    # ======================================================================
    vit_patches, (T_g, H_g, W_g) = preprocess_video(cond_frames, max_pixels=vit_max_pixels)
    grid_thw = mx.array([[T_g, H_g, W_g]], dtype=mx.int32)
    visual_und = vit(vit_patches, grid_thw)                       # (N_vit, 2048)
    n_vit = int(visual_und.shape[0])
    H_g_m, W_g_m = H_g // SPATIAL_MERGE_SIZE, W_g // SPATIAL_MERGE_SIZE

    # ======================================================================
    # Slab 2 — VAE-cond  (Wan VAE T>1 encode with production scale)
    # ======================================================================
    vae_in = _vae_preprocess_video(cond_frames, H=H_vae, W=W_vae)  # (1, N, H, W, 3)
    vae_scale = (mx.array(VAE_SCALE_MEAN), mx.array(1.0 / VAE_SCALE_STD))
    cond_latent = vae.encode(vae_in, scale=vae_scale)             # (1, t_lat, h_lat, w_lat, 48)
    t_lat = int(cond_latent.shape[1]); h_lat = int(cond_latent.shape[2]); w_lat = int(cond_latent.shape[3])
    # sanity: PT temporal-compression formula
    assert t_lat == (int(cond_frames.shape[0]) - 1) // VAE_DOWN_TEMPORAL + 1, \
        f"t_lat {t_lat} != (N-1)//{VAE_DOWN_TEMPORAL}+1"
    n_cond = t_lat * h_lat * w_lat
    cond_flat = cond_latent.reshape(n_cond, Z_DIM)

    # ======================================================================
    # Slab 3 — noise target  (same latent shape as cond)
    # ======================================================================
    n_noise = n_cond
    x_t = sample_init_noise((n_noise, Z_DIM), seed=seed)

    # latent_pos_embed lookup (video, max=64) — shared by cond + noise slabs
    latent_pos_ids = mx.array(vae_latent_position_indices(t_lat, h_lat, w_lat,
                                                          max_latent_size=MAX_LATENT_SIZE))

    print(f"[video_edit] ViT grid=({T_g},{H_g},{W_g}) n_vit={n_vit} | "
          f"VAE lat=({t_lat},{h_lat},{w_lat}) n_cond={n_cond} | size={H_vae}x{W_vae}")

    # ======================================================================
    # Token layout — 3 CFG variants (full / t_uncond / tv_uncond)
    # ======================================================================
    layouts = build_video_edit_layouts(tokenizer, instruction, n_vit, n_cond, n_noise)
    full_layout      = layouts["v_full"]
    t_uncond_layout  = layouts["v_t_uncond"]
    tv_uncond_layout = layouts["v_tv_uncond"]
    print(f"[video_edit] seq L: full={full_layout['L']} t_uncond={t_uncond_layout['L']} "
          f"tv_uncond={tv_uncond_layout['L']}")

    # ======================================================================
    # Per-forward velocity (image_edit `_forward_v` structure + video temporal).
    # ======================================================================
    def _forward_v(layout: dict, x_t_cur: mx.array, t_scalar: mx.array) -> mx.array:
        ids = mx.array([layout["ids"]], dtype=mx.int32)
        L = layout["L"]
        vae_s, vae_e = layout["vae_span"]
        noise_s, noise_e = layout["noise_span"]
        assert (noise_e - noise_s) == (vae_e - vae_s), "cond/noise span widths must match (pos copy)"

        text_embed = model.language_model.model.embed_tokens(ids)

        # ViT-cond slab
        embed = text_embed
        if layout["vit_span"] is not None:
            vit_s, vit_e = layout["vit_span"]
            embed = mx.concatenate([embed[:, :vit_s, :], visual_und[None, :, :], embed[:, vit_e:, :]], axis=1)

        # latent embeds: cond timestep=0, noise timestep=current_t (PT lance.py:659)
        t_zero = mx.zeros_like(t_scalar)
        vae_cond_embed = (model.vae2llm(cond_flat)
                          + model.time_embedder(t_zero)
                          + model.latent_pos_embed(latent_pos_ids))
        embed = mx.concatenate([embed[:, :vae_s, :], vae_cond_embed[None, :, :], embed[:, vae_e:, :]], axis=1)
        noise_embed = (model.vae2llm(x_t_cur)
                       + model.time_embedder(t_scalar)
                       + model.latent_pos_embed(latent_pos_ids))
        embed = mx.concatenate([embed[:, :noise_s, :], noise_embed[None, :, :], embed[:, noise_e:, :]], axis=1)

        # gen_mask: cond + noise latent slabs route through moe_gen
        cols = mx.arange(L)
        gen_mask = (((cols >= vae_s) & (cols < vae_e)) | ((cols >= noise_s) & (cols < noise_e)))[None, :]

        # positions — assembly under test (module-level builder, shared verbatim
        # with the non-blind harness so the harness checks the *real* pipeline code).
        pos = mx.array(build_video_edit_positions(
            layout, T_g=T_g, H_g_m=H_g_m, W_g_m=W_g_m,
            t_lat=t_lat, h_lat=h_lat, w_lat=w_lat))

        # attention mask — same slab modes as image_edit (ViT 'full', VAE-cond
        # 'full_noise', noise 'noise', text causal).
        split_lens, attn_modes = [], []
        if layout["vit_span"] is not None:
            split_lens += [vit_s - 1, (vit_e - vit_s) + 2]
            attn_modes += ["causal", "full"]
            mid_start = vit_e + 1
        else:
            mid_start = 0
        split_lens += [(vae_s - 1) - mid_start, (vae_e - vae_s) + 2, (noise_e - noise_s) + 2]
        attn_modes += ["causal", "full_noise", "noise"]
        sl_tail = L - (noise_e + 1)
        if sl_tail > 0:
            split_lens.append(sl_tail); attn_modes.append("causal")
        attn_mask = build_lance_attention_mask(seq_len=L, split_lens=split_lens, attn_modes=attn_modes)

        hidden = model.language_model.model(
            input_ids=None, position_ids=pos, inputs_embeds=embed, mask=attn_mask, gen_mask=gen_mask,
        )
        return model.llm2vae(hidden[0, noise_s:noise_e, :])          # (n_noise, 48)

    # ======================================================================
    # Denoising loop — 3-component CFG (identical to image_edit)
    # ======================================================================
    sch = make_schedule(num_steps=num_steps, timestep_shift=timestep_shift)
    t0 = time.time()
    for i in range(num_steps):
        t_scalar = sch.timesteps[i:i + 1]
        v_full      = _forward_v(full_layout,      x_t, t_scalar)
        v_t_uncond  = _forward_v(t_uncond_layout,  x_t, t_scalar)
        v_tv_uncond = _forward_v(tv_uncond_layout, x_t, t_scalar)
        v_final = cfg_velocity_3comp(
            v_full, v_t_uncond, v_tv_uncond,
            cfg_text=cfg_text, cfg_vit=cfg_vit,
            renorm_type=cfg_renorm_type, renorm_min=cfg_renorm_min,
        )
        x_t = euler_step(x_t, v_final, sch.dts[i])
        mx.eval(x_t)
        if (i + 1) % 5 == 0 or i == 0:
            print(f"[video_edit] step {i+1:3d}/{num_steps}  t={t_scalar.item():.4f}  "
                  f"||v_full||={mx.linalg.norm(v_full).item():.2f}  "
                  f"||x_t||={mx.linalg.norm(x_t).item():.2f}  ({time.time()-t0:.1f}s)")

    latent = x_t.reshape(1, t_lat, h_lat, w_lat, Z_DIM)
    # decode with production scale (else dynamic range 1.5x off — STAGE 9 §1)
    video = vae.decode(latent, scale=vae_scale)
    mx.eval(video)
    return VideoEditResult(latent=latent, video_recon=video,
                           t_lat=t_lat, h_lat=h_lat, w_lat=w_lat)
