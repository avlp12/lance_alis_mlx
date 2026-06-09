"""X→T (image understanding) pipeline.

Sequence layout (PT ref `refs/Lance/modeling/lance/lance.py:879-1100`,
RockTalk README "X→T"):

    <|im_start|>system\n[system_prompt]<|im_end|>
    <|im_start|>user
      <|vision_start|>[N visual placeholders]<|vision_end|>[question]
    <|im_end|>
    <|im_start|>assistant
    [AR generated tokens...]

The visual placeholders are filled by `LanceViT(patches, grid_thw)` output
(2048-dim per token, already through the spatial-merge MLP / connector).
Backbone forward routes everything through the canonical (UND) path —
no GEN slab.  AR decoding samples one token at a time.

Minimal implementation (this STAGE): per-step *full-sequence* re-forward.
KV cache is wired through `LanceAttention` but not exercised here; STAGE 7
§2b will add it for the ~29 tok/s target.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from PIL import Image

from ..backbone import LanceLLM
from ..rope import VisionSpec, build_positions_for_layout
from ..attn_mask import build_lance_attention_mask
from ..vit import LanceViT


# Qwen2.5-VL / OpenAI CLIP normalization
QWEN_VL_IMAGE_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
QWEN_VL_IMAGE_STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

# Lance / Qwen2.5-VL ViT constants
PATCH_SIZE = 14
SPATIAL_MERGE_SIZE = 2
TEMPORAL_PATCH_SIZE = 2

# Temporal mRoPE scale for VIDEO understanding = second_per_grid_t(1.0) ·
# tokens_per_second(2).  PT `get_rope_index` scales the time axis by this on the
# video branch (qwen2_navit.py:1258); the generation path already does it
# (t2v.py:112).  x2t-image leaves it at 1 (T=1 → no-op).  [STAGE 11 fix]
VIDEO_TEMPORAL_SCALE = 2

# Special token ids (verified at STAGE 1, llm_config.json)
IM_START_ID  = 151644
IM_END_ID    = 151645
VIS_START_ID = 151652
VIS_END_ID   = 151653
IMG_TOKEN_ID = 151655
VIDEO_PAD_ID = 151656   # <|video_pad|> — placeholder for video visual tokens
EOS_ID       = 151645   # Qwen Lance uses <|im_end|> as EOS for chat


def preprocess_image(image_path: str, *, min_pixels: int = 28 * 28,
                     max_pixels: int = 14 * 14 * 16 * 16) -> tuple[mx.array, tuple[int, int, int]]:
    """Load image and patchify to ViT input format.

    Matches transformers' `Qwen2_5_VLImageProcessor` semantics:
      1. Open as RGB, scale to [0, 1].
      2. Resize so H, W are multiples of (patch_size · spatial_merge_size) = 28,
         honouring a target pixel budget (min_pixels ≤ H*W ≤ max_pixels).
      3. Normalize with OpenAI CLIP mean/std.
      4. Duplicate temporal axis to T=2 (since temporal_patch_size=2 and
         a single image is treated as a 2-frame "video" with both frames
         identical — gives a 1-token temporal grid after the temporal patch).
      5. Patchify via `_patchify_frames` — 2×2 spatial-merge-grouped token
         order (the layout PT + mlx-vlm ViT require; see _patchify_frames).

    Returns: (patches, (T_grid, H_grid, W_grid)).
    """
    img = Image.open(image_path).convert("RGB")
    W_orig, H_orig = img.size
    # Pixel-budget-aware resize: keep aspect; round H/W to multiples of 28.
    step = PATCH_SIZE * SPATIAL_MERGE_SIZE              # 28
    # Round-down to multiple of step, then enforce min/max pixel budget.
    H_target = max(step, (H_orig // step) * step)
    W_target = max(step, (W_orig // step) * step)
    if H_target * W_target > max_pixels:
        # Shrink proportionally to fit budget.
        scale = (max_pixels / (H_target * W_target)) ** 0.5
        H_target = max(step, int(H_target * scale) // step * step)
        W_target = max(step, int(W_target * scale) // step * step)
    img_resized = img.resize((W_target, H_target), Image.BICUBIC)
    arr = np.asarray(img_resized, dtype=np.float32) / 255.0       # (H, W, 3) in [0, 1]
    arr = (arr - QWEN_VL_IMAGE_MEAN) / QWEN_VL_IMAGE_STD          # normalize
    # T=2 by duplicating (single image): two identical frames → T_grid=1 after
    # the temporal patch.  Patchify is the shared 2×2-merge-grouped layout —
    # one code path with preprocess_video so they can never diverge again.
    arr_t2 = np.stack([arr, arr], axis=0)                          # (2, H, W, 3)
    return _patchify_frames(arr_t2)


def _patchify_frames(arr: np.ndarray) -> tuple[mx.array, tuple[int, int, int]]:
    """arr: (T, H, W, 3) normalized float32, T even.  Returns (patches (N, 1176),
    (T_grid, H_grid, W_grid)).

    ★ Token order is 2×2 SPATIAL-MERGE-GROUPED — the exact layout PT
    `data_utils.patchify_video_with_merge` emits and what *both* the PT ViT and
    mlx-vlm `VisionModel` expect (they read consecutive-4 patches as one 2×2
    spatial-merge block: `reshape(seq//merge_unit, merge_unit, -1)[window_index]`).

    History (STAGE 11 step 2B): this used to emit *plain raster* (t,h,w) order.
    That was wrong — vs the PT real pipeline our ViT output landed at cos≈0.29
    (image) / 0.36 (video).  STAGE 7's harness fed the *same* raster patches to
    both PT and MLX, so both mis-grouped identically and agreed at cos 1.0 —
    structurally blind to patch order.  Merge-grouped input gives cos 1.000000
    against PT's real `patchify_video_with_merge` → ViT pipeline.

    PT does this on a CTHW tensor with permute (0,3,6,4,7,2,1,5,8); we hold THWC,
    so the axes differ but the resulting (token, patch_dim) layout is identical:
      token order   = (T_grid, H_grid/2, W_grid/2, ms_h, ms_w)  ← merge-grouped
      patch_dim     = (C, t_p, p_h, p_w)                        ← channel-first
    Position assignment (rope._image_position_block) is unchanged: the ViT output
    (one token per 2×2 block, restored to llm-grid raster) already matches the
    raster (t, h/2, w/2) order rope.py emits.
    """
    ms = SPATIAL_MERGE_SIZE
    T, H, W = arr.shape[0], arr.shape[1], arr.shape[2]
    H_grid, W_grid = H // PATCH_SIZE, W // PATCH_SIZE
    T_grid = T // TEMPORAL_PATCH_SIZE
    # Expose every axis: (T_grid, t_p, gh/ms, ms_h, p_h, gw/ms, ms_w, p_w, C)
    a = arr.reshape(T_grid, TEMPORAL_PATCH_SIZE,
                    H_grid // ms, ms, PATCH_SIZE,
                    W_grid // ms, ms, PATCH_SIZE, 3)
    #            T_grid  gh/ms  gw/ms  ms_h  ms_w   C   t_p  p_h  p_w
    a = a.transpose(0,    2,     5,     3,    6,    8,   1,   4,   7)
    a = a.reshape(T_grid * H_grid * W_grid, 3 * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE)
    return mx.array(a.astype(np.float32)), (T_grid, H_grid, W_grid)


def preprocess_video(frames: np.ndarray, *,
                     max_pixels: int = 14 * 14 * 16 * 16) -> tuple[mx.array, tuple[int, int, int]]:
    """Video frames (N, H, W, 3) uint8 → ViT patch input.

    Mirrors preprocess_image but with N real frames instead of a duplicated
    single image.  Pads an odd N by repeating the last frame (PT
    get_video_tensor_online does the same — temporal_patch_size=2 needs even N).
    All frames are resized to one common multiple-of-28 size within the pixel
    budget.  Frame *sampling* (which N to take) is video_io.MultiClipsFrameSampler.
    """
    N = int(frames.shape[0])
    if N % 2 == 1:
        frames = np.concatenate([frames, frames[-1:]], axis=0)
        N += 1
    H0, W0 = int(frames.shape[1]), int(frames.shape[2])
    step = PATCH_SIZE * SPATIAL_MERGE_SIZE                      # 28
    H_t = max(step, (H0 // step) * step)
    W_t = max(step, (W0 // step) * step)
    if H_t * W_t > max_pixels:
        s = (max_pixels / (H_t * W_t)) ** 0.5
        H_t = max(step, int(H_t * s) // step * step)
        W_t = max(step, int(W_t * s) // step * step)
    out = np.empty((N, H_t, W_t, 3), dtype=np.float32)
    for i in range(N):
        im = Image.fromarray(frames[i]).resize((W_t, H_t), Image.BICUBIC)
        out[i] = (np.asarray(im, dtype=np.float32) / 255.0 - QWEN_VL_IMAGE_MEAN) / QWEN_VL_IMAGE_STD
    return _patchify_frames(out)


def load_video_vit(vit, video_weight_path: str) -> int:
    """Load the video ViT (`vit_model.*`) from the merged video weight into a
    LanceViT (which expects `vision_tower.*`).  Same architecture as the image
    ViT — only the key prefix differs (verified: shapes byte-identical)."""
    full = mx.load(video_weight_path)
    vit_w = {
        "vision_tower." + k[len("vit_model."):]: v
        for k, v in full.items() if k.startswith("vit_model.")
    }
    vit.load_weights(list(vit_w.items()), strict=True)
    mx.eval(vit.parameters())
    return len(vit_w)


@dataclass
class X2TResult:
    text: str
    tokens: list[int]
    n_visual_tokens: int


def x2t(
    model: LanceLLM,
    vit: LanceViT,
    tokenizer,
    image_path: str,
    question: str,
    *,
    system_prompt: str = "You are a helpful assistant.",
    max_new_tokens: int = 60,
    image_token_id: int = IMG_TOKEN_ID,
) -> X2TResult:
    """Single-image VQA / captioning."""

    # ---- ViT forward ----
    patches, (T_g, H_g, W_g) = preprocess_image(image_path)
    grid_thw = mx.array([[T_g, H_g, W_g]], dtype=mx.int32)
    visual = vit(patches, grid_thw)                          # (N_after_merge, 2048)
    n_vis = int(visual.shape[0])                              # = T_g · (H_g/2) · (W_g/2)
    h_lat = H_g // SPATIAL_MERGE_SIZE
    w_lat = W_g // SPATIAL_MERGE_SIZE
    print(f"[vit] image patches → {patches.shape}  grid=({T_g},{H_g},{W_g}) "
          f"→ visual tokens {visual.shape}  (LLM grid {T_g}×{h_lat}×{w_lat})")

    # ---- Build chat sequence ----
    # Manually construct chat tokens — Qwen tokenizer's `apply_chat_template`
    # would skip the vision placeholders we need to inject.
    sys_ids   = tokenizer(system_prompt, add_special_tokens=False)["input_ids"]
    q_ids     = tokenizer(question,      add_special_tokens=False)["input_ids"]
    # Chat scaffolding tokens
    newline_id = tokenizer("\n", add_special_tokens=False)["input_ids"]   # usually [198]
    system_lbl = tokenizer("system", add_special_tokens=False)["input_ids"]
    user_lbl   = tokenizer("user",   add_special_tokens=False)["input_ids"]
    assist_lbl = tokenizer("assistant", add_special_tokens=False)["input_ids"]

    seq = (
        [IM_START_ID] + system_lbl + newline_id + sys_ids + [IM_END_ID] + newline_id
        + [IM_START_ID] + user_lbl + newline_id
        + [VIS_START_ID] + [image_token_id] * n_vis + [VIS_END_ID]
        + q_ids + [IM_END_ID] + newline_id
        + [IM_START_ID] + assist_lbl + newline_id
    )
    L = len(seq)
    # Compute vis span (where we'll inject visual embeds)
    vis_start_idx = seq.index(VIS_START_ID) + 1                          # first placeholder
    vis_end_idx   = vis_start_idx + n_vis                                 # exclusive
    print(f"[seq] L={L}  vis=[{vis_start_idx},{vis_end_idx})  prompt tokens={L - n_vis}")

    # ---- Attention mask: pure causal (X→T is UND-only, no GEN slab) ----
    # Single split, causal — no bidirectional region.
    attn_mask = build_lance_attention_mask(
        seq_len=L, split_lens=[L], attn_modes=["causal"],
    )

    # ---- Position IDs: image grid mid-sequence (mRoPE) ----
    vis_span = VisionSpec(start=vis_start_idx - 1, length=n_vis,
                          t=T_g, h=h_lat, w=w_lat)
    pos = build_positions_for_layout(L, [vis_span])

    # ---- Build embedding sequence: text via embed_tokens, image placeholders via ViT ----
    ids = mx.array([seq], dtype=mx.int32)
    text_embed = model.language_model.model.embed_tokens(ids)            # (1, L, D)
    # Splice visual tokens into the placeholder positions
    embed = mx.concatenate([
        text_embed[:, :vis_start_idx, :],
        visual[None, :, :],                                              # (1, N_vis, D)
        text_embed[:, vis_end_idx:, :],
    ], axis=1)

    # ---- AR loop (no KV cache yet; STAGE 7 §2b will add) ----
    out_tokens: list[int] = []
    cur_embed = embed
    cur_pos = pos
    cur_mask = attn_mask
    cur_ids = ids
    for step in range(max_new_tokens):
        hidden = model.language_model.model(
            input_ids=None, position_ids=cur_pos, inputs_embeds=cur_embed,
            mask=cur_mask, gen_mask=None,
        )
        logits = model.language_model.lm_head(hidden[0, -1:, :])         # (1, V)
        next_id = int(mx.argmax(logits[0]).item())
        out_tokens.append(next_id)
        # IM_END_ID and EOS_ID are the same literal today; collapse to one check.
        # If they ever diverge upstream, extend STOP_IDS rather than re-grow this conditional.
        if next_id == IM_END_ID:
            break
        # Append + rebuild for next step.  Position is text-only beyond the
        # vision span — extends cur_pos by one text-position step.
        last_pos_vals = cur_pos[:, 0, -1]                                # (3,) — all equal for text
        new_pos_val = int(last_pos_vals[0].item()) + 1
        new_pos_col = mx.array([[[new_pos_val]],
                                 [[new_pos_val]],
                                 [[new_pos_val]]], dtype=mx.int32)        # (3, 1, 1)
        cur_pos = mx.concatenate([cur_pos, new_pos_col], axis=-1)
        # New token embed
        new_id_arr = mx.array([[next_id]], dtype=mx.int32)
        new_embed = model.language_model.model.embed_tokens(new_id_arr)
        cur_embed = mx.concatenate([cur_embed, new_embed], axis=1)
        # Extend causal mask by one row+col
        L_new = cur_embed.shape[1]
        cur_mask = build_lance_attention_mask(
            seq_len=L_new, split_lens=[L_new], attn_modes=["causal"],
        )

    text = tokenizer.decode(out_tokens, skip_special_tokens=True)
    return X2TResult(text=text, tokens=out_tokens, n_visual_tokens=n_vis)


def x2t_video(
    model: LanceLLM,
    vit: LanceViT,
    tokenizer,
    frames: np.ndarray,
    question: str,
    *,
    system_prompt: str = "You are a helpful assistant.",
    max_new_tokens: int = 60,
    max_pixels: int = 14 * 14 * 16 * 16,
) -> X2TResult:
    """Video VQA / captioning.  Same forward as x2t() but the visual stream is
    a multi-frame clip through the video ViT, placeheld with the *video* pad
    token.  `frames` are pre-sampled (video_io.read_video_frames / fixture);
    `vit` must already hold the video ViT (load_video_vit).

    NOTE: a "does it run" path — PT byte-diff (stage11 step 2B/C) proves it.
    Most of the body mirrors x2t() verbatim; kept separate so the verified
    image path stays untouched.
    """
    patches, (T_g, H_g, W_g) = preprocess_video(frames, max_pixels=max_pixels)
    grid_thw = mx.array([[T_g, H_g, W_g]], dtype=mx.int32)
    visual = vit(patches, grid_thw)
    n_vis = int(visual.shape[0])
    h_lat, w_lat = H_g // SPATIAL_MERGE_SIZE, W_g // SPATIAL_MERGE_SIZE

    sys_ids = tokenizer(system_prompt, add_special_tokens=False)["input_ids"]
    q_ids = tokenizer(question, add_special_tokens=False)["input_ids"]
    newline_id = tokenizer("\n", add_special_tokens=False)["input_ids"]
    system_lbl = tokenizer("system", add_special_tokens=False)["input_ids"]
    user_lbl = tokenizer("user", add_special_tokens=False)["input_ids"]
    assist_lbl = tokenizer("assistant", add_special_tokens=False)["input_ids"]

    seq = (
        [IM_START_ID] + system_lbl + newline_id + sys_ids + [IM_END_ID] + newline_id
        + [IM_START_ID] + user_lbl + newline_id
        + [VIS_START_ID] + [VIDEO_PAD_ID] * n_vis + [VIS_END_ID]
        + q_ids + [IM_END_ID] + newline_id
        + [IM_START_ID] + assist_lbl + newline_id
    )
    L = len(seq)
    vis_start_idx = seq.index(VIS_START_ID) + 1
    vis_end_idx = vis_start_idx + n_vis

    attn_mask = build_lance_attention_mask(seq_len=L, split_lens=[L], attn_modes=["causal"])
    vis_span = VisionSpec(start=vis_start_idx - 1, length=n_vis, t=T_g, h=h_lat, w=w_lat,
                          temporal_scale=VIDEO_TEMPORAL_SCALE)
    pos = build_positions_for_layout(L, [vis_span])

    ids = mx.array([seq], dtype=mx.int32)
    text_embed = model.language_model.model.embed_tokens(ids)
    embed = mx.concatenate([
        text_embed[:, :vis_start_idx, :],
        visual[None, :, :],
        text_embed[:, vis_end_idx:, :],
    ], axis=1)

    out_tokens: list[int] = []
    cur_embed, cur_pos, cur_mask = embed, pos, attn_mask
    for _ in range(max_new_tokens):
        hidden = model.language_model.model(
            input_ids=None, position_ids=cur_pos, inputs_embeds=cur_embed,
            mask=cur_mask, gen_mask=None,
        )
        logits = model.language_model.lm_head(hidden[0, -1:, :])
        next_id = int(mx.argmax(logits[0]).item())
        out_tokens.append(next_id)
        if next_id == IM_END_ID:
            break
        new_pos_val = int(cur_pos[:, 0, -1][0].item()) + 1
        new_pos_col = mx.array([[[new_pos_val]], [[new_pos_val]], [[new_pos_val]]], dtype=mx.int32)
        cur_pos = mx.concatenate([cur_pos, new_pos_col], axis=-1)
        new_embed = model.language_model.model.embed_tokens(mx.array([[next_id]], dtype=mx.int32))
        cur_embed = mx.concatenate([cur_embed, new_embed], axis=1)
        L_new = cur_embed.shape[1]
        cur_mask = build_lance_attention_mask(seq_len=L_new, split_lens=[L_new], attn_modes=["causal"])

    text = tokenizer.decode(out_tokens, skip_special_tokens=True)
    return X2TResult(text=text, tokens=out_tokens, n_visual_tokens=n_vis)
