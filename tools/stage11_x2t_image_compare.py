"""STAGE 11 (c)-image — NON-BLIND end-to-end x2t logit parity vs PT.

The STAGE-7 harness (stage7_x2t_compare) fed OUR patches to BOTH PT and MLX —
structurally blind to patch token-order (the raster bug found in step 2B).  This
re-verification fixes that: PT builds its OWN ViT input via PT's
`patchify_video_with_merge` (2×2 merge-grouped), MLX uses the fixed
`preprocess_image`.  Because the fix made our patchify byte-identical to PT's,
the two patch tensors match (asserted) — so the downstream LLM comparison is a
GENUINE PT-reference test, not a tautology.

What this seals beyond step 2B:
  - step 2B proved the ViT *output* matches PT (cos 1.0).  This runs the full
    LLM forward, so it also checks mRoPE position assignment is consistent with
    the corrected ViT output order — if it weren't, logits would diverge.
    (Lesson 23: prove it in the real forward, not by code-read alone.)

Reuses stage7_x2t_compare for the PT env + model builder + mrope helper; the
original stays untouched as the historical blind-harness artifact (audit trail).

Gate: first-token logits cos ≥ 0.999.  PT is bf16, MLX is f32 → expect ~0.9999
(crossing the precision boundary), NOT exactly 1.0 — that is the desirable sign.
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

import numpy as np
import torch
from PIL import Image

import mlx.core as mx
from transformers import AutoTokenizer

# --- transformers ≥5.9 compat (must precede any PT Lance modeling import) ------
# 5.9's is_flash_attn_*_available() does PACKAGE_DISTRIBUTION_MAPPING["flash_attn"]
# which KeyErrors against our fake flash_attn stub.  The modeling import chain
# (modeling_qwen2 → transformers.activations → integrations.flash_attention) calls
# it at import time.  stage7_x2t_compare's own setup predates this path, so we
# neutralise the probes here, BEFORE importing it (binds the patched fns into
# transformers.modeling_flash_attention_utils when that module first imports them).
import transformers.utils.import_utils as _iu
import transformers.utils as _tu
import transformers.modeling_flash_attention_utils as _mfa
def _false(*_a, **_k):  # noqa: D401
    return False
# `from transformers import AutoTokenizer` above already imported _mfa with the
# REAL is_flash_attn_* bound, so patching import_utils alone is too late — patch
# the module that flash_attention.py reads from (it binds these at its own import,
# which happens later in the chain).  flash_attn_supports_top_left_mask is the
# exact call at integrations/flash_attention.py:9.
for _m in (_iu, _tu, _mfa):
    for _fn in ("is_flash_attn_2_available", "is_flash_attn_3_available",
                "is_flash_attn_4_available", "flash_attn_supports_top_left_mask"):
        if hasattr(_m, _fn):
            setattr(_m, _fn, _false)

# Importing stage7_x2t_compare runs its module-level env setup (flash shim, refs/
# Lance path, qwen2_navit + flex_attention bind, PT class imports) and gives us
# the PT model builder + mrope helper + cosine.  main() is __name__-guarded.
import tools.stage7_x2t_compare as X
from data.data_utils import patchify_video_with_merge, create_sparse_mask
from torch.nn.attention.flex_attention import create_block_mask

from lance_mlx.backbone import LanceLLM, LanceTextConfig, load_full_lance
from lance_mlx.vit import LanceViT, load_lance_vit
from lance_mlx.pipelines.x2t import (
    preprocess_image, QWEN_VL_IMAGE_MEAN, QWEN_VL_IMAGE_STD,
    PATCH_SIZE, SPATIAL_MERGE_SIZE,
    IM_START_ID, IM_END_ID, VIS_START_ID, VIS_END_ID, IMG_TOKEN_ID,
)
from lance_mlx.rope import VisionSpec, build_positions_for_layout
from lance_mlx.attn_mask import build_lance_attention_mask

IMAGE = "refs/Lance/assets/image-understanding/cases/image-understanding-case-01.png"
QUESTION = "Describe this image briefly."

MLX_WEIGHTS = "checkpoints/Lance-3B-MLX/model.safetensors"
VIT_WEIGHTS = "checkpoints/Lance-3B-MLX/vit.safetensors"


def _norm_t2(path: str) -> np.ndarray:
    """Replicate preprocess_image's resize+normalize+T=2-dup → (2, H, W, 3).
    Used only to feed PT's OWN patchify with the same normalized frames."""
    img = Image.open(path).convert("RGB")
    W0, H0 = img.size
    step = PATCH_SIZE * SPATIAL_MERGE_SIZE
    mp = 14 * 14 * 16 * 16
    H = max(step, (H0 // step) * step)
    W = max(step, (W0 // step) * step)
    if H * W > mp:
        s = (mp / (H * W)) ** 0.5
        H = max(step, int(H * s) // step * step)
        W = max(step, int(W * s) // step * step)
    arr = (np.asarray(img.resize((W, H), Image.BICUBIC), np.float32) / 255.0
           - QWEN_VL_IMAGE_MEAN) / QWEN_VL_IMAGE_STD
    return np.stack([arr, arr], axis=0)


def main() -> None:
    tok = AutoTokenizer.from_pretrained("checkpoints/Lance-3B-MLX", trust_remote_code=True)

    # ---- patches: each side its OWN preprocessing (non-blind) ----
    patches_mlx, (T_g, H_g, W_g) = preprocess_image(IMAGE)          # fixed _patchify_frames
    patches_mlx_np = np.asarray(patches_mlx)
    norm = _norm_t2(IMAGE)
    patches_pt_np = patchify_video_with_merge(                       # PT's OWN patchify
        torch.from_numpy(np.transpose(norm, (3, 0, 1, 2))), PATCH_SIZE, 2).numpy()
    tie = np.array_equal(patches_mlx_np, patches_pt_np)
    print(f"[patches] MLX(preprocess_image) vs PT(patchify_video_with_merge): "
          f"byte-identical={tie}  grid=({T_g},{H_g},{W_g})  N_patch={patches_mlx_np.shape[0]}")
    if not tie:
        print("  ✗ patches differ — the two preprocessings diverge; STOP and localize.")
        sys.exit(2)

    grid_np = np.array([[T_g, H_g, W_g]], dtype=np.int64)
    grid_mlx = mx.array(grid_np.astype(np.int32))
    grid_pt = torch.from_numpy(grid_np)
    patches_pt = torch.from_numpy(patches_pt_np)

    # ---- MLX model + ViT ----
    print("[build] LanceLLM + LanceViT (MLX, f32) ...")
    mlx_model = LanceLLM(LanceTextConfig()); load_full_lance(mlx_model, MLX_WEIGHTS); mlx_model.eval()
    mlx_vit = LanceViT(); load_lance_vit(mlx_vit, VIT_WEIGHTS); mlx_vit.eval()
    visual_mlx = mlx_vit(patches_mlx, grid_mlx)
    n_vis = int(visual_mlx.shape[0])
    h_lat, w_lat = H_g // 2, W_g // 2

    # ---- PT model + ViT (bf16 LLM, fp32 ViT) ----
    print("[build] PT (full LLM + ViT) ...")
    pt_embed, pt_layers, pt_final_norm, pt_lm_head, pt_vit = X.build_pt_x2t_model()
    with torch.no_grad():
        visual_pt = pt_vit(hidden_states=patches_pt, grid_thw=grid_pt)
    visual_pt_bf16 = visual_pt.to(torch.bfloat16)
    print(f"[ViT] cos(PT, MLX) = {X._cosine(visual_pt.cpu().numpy(), np.asarray(visual_mlx)):.6f}")

    # ---- chat sequence (identical both sides) ----
    sys_ids = tok("You are a helpful assistant.", add_special_tokens=False)["input_ids"]
    q_ids = tok(QUESTION, add_special_tokens=False)["input_ids"]
    newline = tok("\n", add_special_tokens=False)["input_ids"]
    sys_lbl = tok("system", add_special_tokens=False)["input_ids"]
    usr_lbl = tok("user", add_special_tokens=False)["input_ids"]
    asst_lbl = tok("assistant", add_special_tokens=False)["input_ids"]
    seq = (
        [IM_START_ID] + sys_lbl + newline + sys_ids + [IM_END_ID] + newline
        + [IM_START_ID] + usr_lbl + newline
        + [VIS_START_ID] + [IMG_TOKEN_ID] * n_vis + [VIS_END_ID]
        + q_ids + [IM_END_ID] + newline
        + [IM_START_ID] + asst_lbl + newline
    )
    L = len(seq)
    vis_start = seq.index(VIS_START_ID) + 1
    vis_end = vis_start + n_vis
    print(f"[seq] L={L}  vis=[{vis_start},{vis_end})  N_vis={n_vis}")

    # ---- positions + mask ----
    pos_mlx = build_positions_for_layout(
        L, [VisionSpec(start=vis_start - 1, length=n_vis, t=T_g, h=h_lat, w=w_lat)])
    pos_pt = torch.from_numpy(np.asarray(pos_mlx).astype(np.int64))
    block_mask_pt = create_block_mask(
        create_sparse_mask([L], [L], ["causal"], torch.device("cpu")),
        B=1, H=16, Q_LEN=L, KV_LEN=L, device=torch.device("cpu"), BLOCK_SIZE=128, _compile=False)
    attn_mask_mlx = build_lance_attention_mask(seq_len=L, split_lens=[L], attn_modes=["causal"])

    # ---- MLX forward ----
    ids_mlx = mx.array([seq], dtype=mx.int32)
    text_embed_mlx = mlx_model.language_model.model.embed_tokens(ids_mlx)
    embed_mlx = mx.concatenate([text_embed_mlx[:, :vis_start, :], visual_mlx[None, :, :],
                                text_embed_mlx[:, vis_end:, :]], axis=1)
    hidden_mlx = mlx_model.language_model.model(
        input_ids=None, position_ids=pos_mlx, inputs_embeds=embed_mlx,
        mask=attn_mask_mlx, gen_mask=None)
    logits_mlx = mlx_model.language_model.lm_head(hidden_mlx[0, -1:, :])

    # ---- PT forward (packed) ----
    ids_pt = torch.tensor([seq], dtype=torch.long)
    text_embed_pt = pt_embed(ids_pt).to(torch.bfloat16)
    embed_pt = torch.cat([text_embed_pt[:, :vis_start, :], visual_pt_bf16.unsqueeze(0),
                          text_embed_pt[:, vis_end:, :]], dim=1)
    cos_pt, sin_pt = X.mrope_cos_sin(pos_pt, head_dim=128, base=1_000_000.0, mrope_section=[16, 24, 24])
    all_idx = torch.arange(L, dtype=torch.long)
    h = embed_pt[0]
    with torch.no_grad():
        for layer in pt_layers:
            h = layer(packed_sequence=h, sample_lens=[L], attention_mask=block_mask_pt,
                      packed_position_embeddings=(cos_pt, sin_pt),
                      packed_und_token_indexes=all_idx,
                      packed_gen_token_indexes=torch.empty(0, dtype=torch.long),
                      mode_forward="validation")
        h = pt_final_norm(h)
        logits_pt = pt_lm_head(h[-1:])

    # ---- compare ----
    pt_arr = logits_pt.to(torch.float32).cpu().numpy()
    mlx_arr = np.asarray(logits_mlx)
    cos = X._cosine(pt_arr, mlx_arr)
    pt_top, mlx_top = int(np.argmax(pt_arr[0])), int(np.argmax(mlx_arr[0]))
    print()
    print(f"first-token logits: cos = {cos:.6f}   max|Δ| = {float(np.abs(pt_arr - mlx_arr).max()):.3e}")
    print(f"PT  argmax = {pt_top:>6d}  ('{tok.decode([pt_top])}')")
    print(f"MLX argmax = {mlx_top:>6d}  ('{tok.decode([mlx_top])}')")
    print("=" * 64)
    ok = cos >= 0.999 and pt_top == mlx_top and tie
    print(f"GATE (c)-image: {'PASS' if ok else 'FAIL'}  (cos≥0.999 + top-1 agree + non-blind patches)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
