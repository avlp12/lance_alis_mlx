"""STAGE 9 §1 — t2v (text-to-video) MLX pipeline (production).

Production config (refs/Lance/inference_lance.sh):
  text_template=True + apply_qwen_2_5_vl_pos_emb=True
  num_frames=50, H=W=768 (production), 30 step, cfg_text_scale=4.0
  cfg_interval=[0.4, 1.0], cfg_renorm_type="global", cfg_renorm_min=0
  timestep_shift=3.5

검증된 컴포넌트 조립 (STAGE 9 §0 ~ §1 단계 4-3):
  - 시퀀스: PT ValidationDataset.t2v_sample (얇은 helper, 단계 4-3 byte-identical)
  - positions: `build_t2v_positions` (production: video case, second_per_grid_t=1, tps=2)
  - attn_mask: `build_lance_attention_mask` (단계 4-3 byte-identical)
  - CFG 2-comp: `scheduler.cfg_velocity` (STAGE 6 검증)
  - Schedule: `scheduler.make_schedule` (PT lance.py:599-602 byte-for-byte)
  - VAE decode: `Wan2_2_VAE.decode` (STAGE 8 cos=1.0)
  - PRNG: numpy seed only (Lesson 9)

Lesson E (Stage 7 §3 / Stage 9 §0) — 별도 PT smoke 에서 contract assertion.

t2v.py 본체는 *런타임 PT 의존성 0* (MLX only).  PT 의존성은 `_t2v_seq.py` 의
시퀀스 빌드 helper 안에만 격리.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import mlx.core as mx

from ..backbone import LanceLLM
from ..attn_mask import build_lance_attention_mask
from ..scheduler import make_schedule, cfg_velocity, euler_step
from ..vae_wan22 import Wan2_2_VAE
from ._t2v_seq import build_t2v_sequence_mlx


# Production constants (Lance-3B-Video-MLX/config.json + inference_lance.sh)
MAX_NUM_LATENT_FRAMES = 31           # = 121 // 4 + 1
MAX_LATENT_SIZE = 64
LATENT_CHANNEL = 48
LATENT_PATCH = (1, 1, 1)
VAE_DOWN_SPATIAL = 16
VAE_DOWN_TEMPORAL = 4
SPATIAL_MERGE_SIZE = 2
TOKENS_PER_SECOND = 2                # Lance-3B-Video vision_config

VIDEO_TOKEN_ID = 151656
START_OF_IMAGE = 151652
END_OF_IMAGE = 151653

# Wan2.2 VAE per-channel normalization scale.  PT validation_gen 의 t2v
# 호출이 `scale=[mean_t, 1.0/std_t]` 형태로 전달 (lance.py:787).  identity
# scale 로 decode 시 video 가 1.5× 큰 dynamic range 출력 (STAGE 9 §1 단계 6
# closing 에서 발견 — latent cos 0.999437 통과해도 video pixel cos 0.948 발산).
VAE_SCALE_MEAN = np.array([
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799,  0.0174,
     0.1838,  0.1557, -0.1382,  0.0542,  0.2813,  0.0891,
     0.1570, -0.0098,  0.0375, -0.1825, -0.2246, -0.1207,
    -0.0698,  0.5109,  0.2665, -0.2108, -0.2158,  0.2502,
    -0.2055, -0.0322,  0.1109,  0.1567, -0.0729,  0.0899,
    -0.2799, -0.1230, -0.0313, -0.1649,  0.0117,  0.0723,
    -0.2839, -0.2083, -0.0520,  0.3748,  0.0152,  0.1957,
     0.1433, -0.2944,  0.3573, -0.0548, -0.1681, -0.0667,
], dtype=np.float32)
VAE_SCALE_STD = np.array([
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990,
    0.4818, 0.5013, 0.8158, 1.0344, 0.5894, 1.0901,
    0.6885, 0.6165, 0.8454, 0.4978, 0.5759, 0.3523,
    0.7135, 0.6804, 1.0457, 0.4329, 0.7918, 0.5739,
    0.5942, 0.5570, 0.5860, 0.6673, 0.4109, 0.7894,
    0.5897, 0.4845, 0.5727, 1.1191, 0.4921, 0.4753,
    1.0265, 0.4790, 1.2798, 0.4768, 0.8169, 0.7497,
    0.7344, 0.4759, 0.8501, 0.6479, 0.4523, 0.6116,
], dtype=np.float32)


# ---------------------------------------------------------------------------
# Position builders — PT get_rope_index 분기 그대로 (단계 4-3 byte-identical).
# ---------------------------------------------------------------------------
def build_t2v_positions(text_split_len: int,
                        t_lat: int, h_lat: int, w_lat: int,
                        L: int,
                        *,
                        second_per_grid_t: float,   # ★ STAGE 9 reviewer A: required.
                        tokens_per_second: int = TOKENS_PER_SECOND) -> np.ndarray:
    """PT `get_rope_index` 의 t2v 분기 그대로.

    text + vis_start: 1D running [0..text_len]
    video span (IMG×N): t_index = repeat(arange(t_lat)*spt*tps, h*w)
                        h_index = tile(repeat(arange(h), w), t)
                        w_index = tile(arange(w), t*h)
                        모두 + (text_len + 1) offset
    post-video (vis_end): st_idx = max(prev) + 1

    `second_per_grid_t` *is keyword-only required* — no default — to prevent
    silent image/video case confusion (reviewer A, STAGE 9):
      - Production t2v (video_token_id): pass `1.0`.
      - text_template=False / image case (image_token_id): pass `0.0`.
    Same shape, very different numerics — silent if defaulted.
    """
    pos = np.zeros((3, 1, L), dtype=np.int32)
    n_video = t_lat * h_lat * w_lat
    text_len_inc_vs = text_split_len + 1
    v_start = text_len_inc_vs

    for axis in range(3):
        pos[axis, 0, :text_len_inc_vs] = np.arange(text_len_inc_vs)

    offset = text_len_inc_vs
    range_t = np.arange(t_lat, dtype=np.float64)
    time_long = (range_t * second_per_grid_t * tokens_per_second).astype(np.int32)
    t_index = np.repeat(time_long, h_lat * w_lat)
    h_index = np.tile(np.repeat(np.arange(h_lat, dtype=np.int32), w_lat), t_lat)
    w_index = np.tile(np.arange(w_lat, dtype=np.int32), t_lat * h_lat)

    pos[0, 0, v_start:v_start + n_video] = t_index + offset
    pos[1, 0, v_start:v_start + n_video] = h_index + offset
    pos[2, 0, v_start:v_start + n_video] = w_index + offset

    if v_start + n_video < L:
        max_t = int(t_index.max()) if t_index.size else 0
        max_pos = offset + max(max_t, h_lat - 1, w_lat - 1)
        st_idx_post = max_pos + 1
        post_len = L - (v_start + n_video)
        for axis in range(3):
            pos[axis, 0, v_start + n_video:] = np.arange(post_len) + st_idx_post
    return pos


def vae_latent_position_indices(t_lat: int, h_lat: int, w_lat: int,
                                max_latent_size: int = MAX_LATENT_SIZE) -> np.ndarray:
    """PT `get_flattened_position_ids_extrapolate_video`: flat = t·M² + h·M + w."""
    coords_t = np.arange(t_lat, dtype=np.int32)
    coords_h = np.arange(h_lat, dtype=np.int32)
    coords_w = np.arange(w_lat, dtype=np.int32)
    M = max_latent_size
    return (coords_t[:, None, None] * M * M
            + coords_h[None, :, None] * M
            + coords_w[None, None, :]).flatten()


# ---------------------------------------------------------------------------
# Single forward — returns v at vae_token positions (n_video, 48).
# ---------------------------------------------------------------------------
def _forward_v(model: LanceLLM,
               input_ids: mx.array,             # (1, L)
               pos_ids: mx.array,               # (3, 1, L)
               attn_mask: mx.array,             # (L, L) additive f32
               vae_token_indices: np.ndarray,   # (n_video,) — sequence positions of IMG tokens
               x_t: mx.array,                   # (n_video, 48)
               t_scalar: float,
               vae_pos_ids: mx.array            # (n_video,) — latent_pos_embed lookup
               ) -> mx.array:
    """Mirror PT `validation_gen` step 0 forward (lance.py:656-685).

    embed = text_embed(ids); embed[vae_indices] = vae2llm(x_t) + time_embed(t) + latent_pos(vae_pos)
    hidden = language_model(embed, pos_ids, mask, gen_mask=vae_indices_mask)
    v = llm2vae(hidden[vae_indices])
    """
    L = int(input_ids.shape[-1])
    n_video = int(vae_token_indices.shape[0])

    text_embed = model.language_model.model.embed_tokens(input_ids)   # (1, L, D)

    t_arr = mx.array([t_scalar] * n_video, dtype=mx.float32)
    vae_embed = (model.vae2llm(x_t)
                 + model.time_embedder(t_arr)
                 + model.latent_pos_embed(vae_pos_ids))                # (n_video, D)

    # Scatter vae_embed into text_embed at vae_token_indices.
    # MLX: split → concat 방식 (contiguous index 가정).  vae_indices 가 contiguous
    # span 인 경우만 검증됨 (production t2v 케이스).
    vi = vae_token_indices.astype(np.int64)
    assert np.array_equal(vi, np.arange(int(vi[0]), int(vi[0]) + n_video)), \
        "vae_token_indices must be contiguous (non-contiguous needs mx.scatter)"
    ns, ne = int(vi[0]), int(vi[0]) + n_video
    embed = mx.concatenate([
        text_embed[:, :ns, :],
        vae_embed[None, :, :],
        text_embed[:, ne:, :],
    ], axis=1)

    # gen_mask: True at vae_token positions (IMG only, not vis_start/vis_end)
    gm_np = np.zeros(L, dtype=bool)
    gm_np[vi] = True
    gen_mask = mx.array(gm_np)[None, :]                                 # (1, L)

    hidden = model.language_model.model(
        input_ids=None, position_ids=pos_ids,
        inputs_embeds=embed, mask=attn_mask, gen_mask=gen_mask,
    )
    return model.llm2vae(hidden[0, ns:ne, :])                           # (n_video, 48)


# ---------------------------------------------------------------------------
# t2v main entry.
# ---------------------------------------------------------------------------
@dataclass
class T2VLayout:
    """Pre-built layout for the denoising loop (full + uncond combined)."""
    # Full
    input_ids: mx.array              # (1, L)
    pos_ids: mx.array                # (3, 1, L)
    attn_mask: mx.array              # (L, L) additive f32
    vae_token_indices: np.ndarray    # (n_video,)
    vae_pos_ids: mx.array            # (n_video,) — latent_pos_embed lookup
    # Uncond
    uncond_input_ids: mx.array
    uncond_pos_ids: mx.array
    uncond_attn_mask: mx.array
    uncond_vae_token_indices: np.ndarray
    # Shapes
    L: int
    uncond_L: int
    t_lat: int
    h_lat: int
    w_lat: int
    n_video: int


def build_t2v_layout(prompt: str, tokenizer, *,
                     num_frames: int = 50, H: int = 768, W: int = 768
                     ) -> T2VLayout:
    """Build full + uncond layouts from a prompt.

    Uses PT `ValidationDataset.t2v_sample` (얇은 helper) for sequence build,
    then MLX-only logic for positions/mask/uncond filter (단계 4-3 검증).
    """
    seq = build_t2v_sequence_mlx(prompt, tokenizer,
                                 num_frames=num_frames, H=H, W=W,
                                 vae_down_t=VAE_DOWN_TEMPORAL,
                                 vae_down_s=VAE_DOWN_SPATIAL)
    input_ids_np = seq["input_ids"].astype(np.int32)
    L = seq["L"]
    modality = seq["sample_modality"]
    vae_token_indices = seq["packed_vae_token_indexes"].astype(np.int64)
    split_lens = seq["split_lens"]
    attn_modes = seq["attn_modes"]

    # Find vis_start → text_split_len
    vs_idx = int(np.where(input_ids_np == START_OF_IMAGE)[0][0])
    text_split_len = vs_idx

    # Latent shape
    t_lat = (num_frames - 1) // VAE_DOWN_TEMPORAL + 1
    h_lat = H // VAE_DOWN_SPATIAL
    w_lat = W // VAE_DOWN_SPATIAL
    n_video = t_lat * h_lat * w_lat

    # Full positions / mask
    pos_ids = build_t2v_positions(text_split_len, t_lat, h_lat, w_lat, L,
                                  second_per_grid_t=1.0)   # production video case
    attn_mask = build_lance_attention_mask(L, split_lens, attn_modes)

    # VAE latent position ids (flat 3D)
    vae_pos_ids = vae_latent_position_indices(t_lat, h_lat, w_lat)

    # Uncond filter (modality != 0 — PT uncond_split_pro_new line 628)
    uncond_mask = modality != 0
    uncond_input_ids = input_ids_np[uncond_mask]
    uncond_L = int(uncond_mask.sum())

    # Uncond split-level filter (PT line 778-792)
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

    u_vs_idx = int(np.where(uncond_input_ids == START_OF_IMAGE)[0][0])
    uncond_pos_ids = build_t2v_positions(u_vs_idx, t_lat, h_lat, w_lat, uncond_L,
                                         second_per_grid_t=1.0)   # production video case
    uncond_attn_mask = build_lance_attention_mask(uncond_L, uncond_split_lens, uncond_attn_modes)

    # Uncond vae_token_indices: same as where input == video_token_id
    uncond_vae_token_indices = np.where(uncond_input_ids == VIDEO_TOKEN_ID)[0].astype(np.int64)
    assert uncond_vae_token_indices.shape[0] == n_video, \
        f"uncond n_video mismatch: got {uncond_vae_token_indices.shape[0]}, expected {n_video}"

    return T2VLayout(
        input_ids=mx.array(input_ids_np)[None, :],
        pos_ids=mx.array(pos_ids),
        attn_mask=attn_mask,
        vae_token_indices=vae_token_indices,
        vae_pos_ids=mx.array(vae_pos_ids),
        uncond_input_ids=mx.array(uncond_input_ids.astype(np.int32))[None, :],
        uncond_pos_ids=mx.array(uncond_pos_ids),
        uncond_attn_mask=uncond_attn_mask,
        uncond_vae_token_indices=uncond_vae_token_indices,
        L=L, uncond_L=uncond_L,
        t_lat=t_lat, h_lat=h_lat, w_lat=w_lat, n_video=n_video,
    )


def t2v(prompt: str,
        model: LanceLLM,
        tokenizer,
        vae: Wan2_2_VAE,
        *,
        num_frames: int = 50,
        H: int = 768, W: int = 768,
        num_steps: int = 30,
        cfg_text_scale: float = 4.0,
        cfg_interval: tuple = (0.4, 1.0),
        cfg_renorm_type: str = "global",
        cfg_renorm_min: float = 0.0,
        timestep_shift: float = 3.5,
        seed: int = 0,
        log_every: int = 5,
        ) -> mx.array:
    """Lance t2v MLX pipeline.

    Args:
        prompt: user text.
        model: loaded LanceLLM with video weights (image backbone + video supplement).
        tokenizer: HF Qwen2 tokenizer (for sequence helper).
        vae: Wan2_2_VAE.
        num_frames / H / W: production defaults 50 / 768 / 768.

    Returns:
        video pixels (1, t_pix, h_pix, w_pix, 3) in [-1, 1].
    """
    import time

    layout = build_t2v_layout(prompt, tokenizer,
                              num_frames=num_frames, H=H, W=W)
    print(f"[t2v] layout: L={layout.L} uncond_L={layout.uncond_L} "
          f"n_video={layout.n_video} t/h/w={layout.t_lat}/{layout.h_lat}/{layout.w_lat}")

    # Initial noise — numpy seed (Lesson 9)
    rng = np.random.default_rng(seed)
    x_t_np = rng.standard_normal((layout.n_video, LATENT_CHANNEL), dtype=np.float32)
    x_t = mx.array(x_t_np)

    sch = make_schedule(num_steps=num_steps, timestep_shift=timestep_shift)
    timesteps = sch.timesteps                            # (num_steps,)
    dts = sch.dts                                        # (num_steps,)

    print(f"[t2v] {num_steps} steps, shift={timestep_shift}, "
          f"t[0]={float(timesteps[0]):.4f} t[-1]={float(timesteps[-1]):.4f}")
    print(f"[t2v] CFG: text_scale={cfg_text_scale}, interval={cfg_interval}, "
          f"renorm={cfg_renorm_type}@min={cfg_renorm_min}")

    t0 = time.time()
    for i in range(num_steps):
        t_scalar = float(timesteps[i])
        dt = float(dts[i])

        v_full = _forward_v(model, layout.input_ids, layout.pos_ids,
                            layout.attn_mask, layout.vae_token_indices,
                            x_t, t_scalar, layout.vae_pos_ids)

        if cfg_interval[0] < t_scalar <= cfg_interval[1] and cfg_text_scale > 1.0:
            v_unc = _forward_v(model, layout.uncond_input_ids, layout.uncond_pos_ids,
                               layout.uncond_attn_mask, layout.uncond_vae_token_indices,
                               x_t, t_scalar, layout.vae_pos_ids)
            v_t = cfg_velocity(v_full, v_unc,
                               scale=cfg_text_scale,
                               renorm_type=cfg_renorm_type,
                               renorm_min=cfg_renorm_min)
        else:
            v_t = v_full

        x_t = euler_step(x_t, v_t, dt)
        mx.eval(x_t)

        if (i + 1) % log_every == 0 or i == 0 or i == num_steps - 1:
            elapsed = time.time() - t0
            print(f"[t2v] step {i+1:3d}/{num_steps}  t={t_scalar:.4f}  "
                  f"||v||={float(mx.linalg.norm(v_t)):.2f}  "
                  f"||x_t||={float(mx.linalg.norm(x_t)):.2f}  ({elapsed:.1f}s)")

    # Unpatchify (pt=ph=pw=1 → reshape only)
    # PT: x_t (n_video, patch_latent_dim=48) → (t, h, w, 48) per
    #     "(t h w)(pt ph pw c) -> (t pt)(h ph)(w pw) c" with pt=ph=pw=1
    x_t_np = np.asarray(x_t, dtype=np.float32)
    latent_np = x_t_np.reshape(layout.t_lat, layout.h_lat, layout.w_lat, LATENT_CHANNEL)
    latent = mx.array(latent_np)[None, ...]              # (1, t, h, w, 48)

    # VAE decode (STAGE 8 byte-clean) — *production scale 필수*.  identity scale
    # 로 호출하면 video dynamic range 가 1.5× 발산 (STAGE 9 §1 단계 6 closing
    # 발견).  PT validation_gen 도 동일 scale 전달.
    vae_scale = (mx.array(VAE_SCALE_MEAN), mx.array(1.0 / VAE_SCALE_STD))
    print(f"[t2v] VAE decode latent shape={latent.shape}  scale=(mean, 1/std) ...")
    video = vae.decode(latent, scale=vae_scale)           # (1, t_pix, h_pix, w_pix, 3)
    print(f"[t2v] video shape={video.shape}  range=[{float(video.min()):+.3f}, {float(video.max()):+.3f}]")
    return video
