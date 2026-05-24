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

# Special token ids (verified at STAGE 1, llm_config.json)
IM_START_ID  = 151644
IM_END_ID    = 151645
VIS_START_ID = 151652
VIS_END_ID   = 151653
IMG_TOKEN_ID = 151655
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
      5. Patchify: `(T=2, H, W, 3)` → `(N=T/2·H/14·W/14, C·t_p·p² = 1176)`.

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
    # T=2 by duplicating (single image)
    arr_t2 = np.stack([arr, arr], axis=0)                          # (2, H, W, 3)
    # Patchify: spatial 14×14 patches, temporal 2-patch (handles T=2 → T_grid=1).
    T = arr_t2.shape[0]
    H_grid = H_target // PATCH_SIZE
    W_grid = W_target // PATCH_SIZE
    T_grid = T // TEMPORAL_PATCH_SIZE
    # Reshape (T, H, W, 3) → (T_grid, t_p, H_grid, p, W_grid, p, 3)
    arr_t2 = arr_t2.reshape(T_grid, TEMPORAL_PATCH_SIZE,
                             H_grid, PATCH_SIZE,
                             W_grid, PATCH_SIZE, 3)
    # Move (t_p, p_h, p_w, c) to channel dim — PT order (c, t_p, p_h, p_w) flattened.
    # PT einops: 'b (g t_p) (h p_h) (w p_w) c -> (b g h w) (c t_p p_h p_w)'
    arr_t2 = arr_t2.transpose(0, 2, 4, 6, 1, 3, 5)                # (T_g, H_g, W_g, 3, t_p, p_h, p_w)
    arr_t2 = arr_t2.reshape(T_grid * H_grid * W_grid,
                             3 * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE)
    return mx.array(arr_t2.astype(np.float32)), (T_grid, H_grid, W_grid)


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
