"""STAGE 11 (c) DISCRIMINATIVE POWER probe — does the (c) logit-cos metric
actually distinguish a CORRECT visual path from a BROKEN one?

The adversarial worry: first-token logits after "...assistant\n" might be
dominated by the text scaffolding, so cos≈0.9999 could appear REGARDLESS of
whether the visual tokens are right — which would make (c)'s PASS meaningless.

Test: run the SAME full LLM forward with visual tokens from
  - FIXED  patches (merge-grouped, current preprocess_image)   → expect cos≈truth
  - RASTER patches (the OLD pre-fix order, known wrong: ViT cos 0.29)
and compare BOTH against PT's real-pipeline truth.  If RASTER's logit-cos DROPS
clearly below FIXED's, the metric discriminates and (c) genuinely validates the
visual path.  If RASTER stays ≈ FIXED, (c) is blind to the visual path — STOP.

Single PT load + single MLX load; two cheap MLX LLM forwards.
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

import numpy as np
import torch
from PIL import Image

import mlx.core as mx
from transformers import AutoTokenizer

# transformers ≥5.9 flash-probe neutraliser (must precede PT modeling import).
import transformers.utils.import_utils as _iu
import transformers.utils as _tu
import transformers.modeling_flash_attention_utils as _mfa
def _false(*_a, **_k):
    return False
for _m in (_iu, _tu, _mfa):
    for _fn in ("is_flash_attn_2_available", "is_flash_attn_3_available",
                "is_flash_attn_4_available", "flash_attn_supports_top_left_mask"):
        if hasattr(_m, _fn):
            setattr(_m, _fn, _false)

import tools.stage7_x2t_compare as X
from data.data_utils import patchify_video_with_merge, create_sparse_mask
from torch.nn.attention.flex_attention import create_block_mask

from lance_mlx.backbone import LanceLLM, LanceTextConfig, load_full_lance
from lance_mlx.vit import LanceViT, load_lance_vit
from lance_mlx.pipelines.x2t import (
    preprocess_image, QWEN_VL_IMAGE_MEAN, QWEN_VL_IMAGE_STD,
    PATCH_SIZE, SPATIAL_MERGE_SIZE, TEMPORAL_PATCH_SIZE,
    IM_START_ID, IM_END_ID, VIS_START_ID, VIS_END_ID, IMG_TOKEN_ID,
)
from lance_mlx.rope import VisionSpec, build_positions_for_layout
from lance_mlx.attn_mask import build_lance_attention_mask

IMAGE = "refs/Lance/assets/image-understanding/cases/image-understanding-case-01.png"
QUESTION = "Describe this image briefly."
MLX_WEIGHTS = "checkpoints/Lance-3B-MLX/model.safetensors"
VIT_WEIGHTS = "checkpoints/Lance-3B-MLX/vit.safetensors"


def _norm_t2(path):
    img = Image.open(path).convert("RGB"); W0, H0 = img.size
    step = PATCH_SIZE * SPATIAL_MERGE_SIZE; mp = 14 * 14 * 16 * 16
    H = max(step, (H0 // step) * step); W = max(step, (W0 // step) * step)
    if H * W > mp:
        s = (mp / (H * W)) ** 0.5
        H = max(step, int(H * s) // step * step); W = max(step, int(W * s) // step * step)
    arr = (np.asarray(img.resize((W, H), Image.BICUBIC), np.float32) / 255.0
           - QWEN_VL_IMAGE_MEAN) / QWEN_VL_IMAGE_STD
    return np.stack([arr, arr], axis=0)


def _raster_patches(norm):
    """OLD pre-fix plain-raster patchify (the known-wrong order)."""
    p, tp = PATCH_SIZE, TEMPORAL_PATCH_SIZE
    T, H, W = norm.shape[:3]
    Tg, Hg, Wg = T // tp, H // p, W // p
    a = norm.reshape(Tg, tp, Hg, p, Wg, p, 3).transpose(0, 2, 4, 6, 1, 3, 5)
    return a.reshape(Tg * Hg * Wg, 3 * tp * p * p).astype(np.float32)


def _build_seq(tok, n_vis):
    sys_ids = tok("You are a helpful assistant.", add_special_tokens=False)["input_ids"]
    q_ids = tok(QUESTION, add_special_tokens=False)["input_ids"]
    nl = tok("\n", add_special_tokens=False)["input_ids"]
    s = tok("system", add_special_tokens=False)["input_ids"]
    u = tok("user", add_special_tokens=False)["input_ids"]
    a = tok("assistant", add_special_tokens=False)["input_ids"]
    seq = ([IM_START_ID] + s + nl + sys_ids + [IM_END_ID] + nl
           + [IM_START_ID] + u + nl
           + [VIS_START_ID] + [IMG_TOKEN_ID] * n_vis + [VIS_END_ID]
           + q_ids + [IM_END_ID] + nl + [IM_START_ID] + a + nl)
    return seq


def _mlx_logits(mlx_model, visual_mlx, seq, vis_start, vis_end, pos_mlx, attn_mask_mlx):
    ids = mx.array([seq], dtype=mx.int32)
    te = mlx_model.language_model.model.embed_tokens(ids)
    emb = mx.concatenate([te[:, :vis_start, :], visual_mlx[None, :, :], te[:, vis_end:, :]], axis=1)
    h = mlx_model.language_model.model(input_ids=None, position_ids=pos_mlx,
                                       inputs_embeds=emb, mask=attn_mask_mlx, gen_mask=None)
    return np.asarray(mlx_model.language_model.lm_head(h[0, -1:, :]))


def main():
    tok = AutoTokenizer.from_pretrained("checkpoints/Lance-3B-MLX", trust_remote_code=True)
    patches_fixed, (T_g, H_g, W_g) = preprocess_image(IMAGE)
    norm = _norm_t2(IMAGE)
    patches_raster = mx.array(_raster_patches(norm))
    grid_np = np.array([[T_g, H_g, W_g]], np.int64)
    grid_mlx = mx.array(grid_np.astype(np.int32))
    pt_merge = patchify_video_with_merge(torch.from_numpy(np.transpose(norm, (3, 0, 1, 2))), PATCH_SIZE, 2).numpy()

    print("[build] MLX LLM + ViT ...")
    mlx_model = LanceLLM(LanceTextConfig()); load_full_lance(mlx_model, MLX_WEIGHTS); mlx_model.eval()
    mlx_vit = LanceViT(); load_lance_vit(mlx_vit, VIT_WEIGHTS); mlx_vit.eval()
    visual_fixed = mlx_vit(patches_fixed, grid_mlx)
    visual_raster = mlx_vit(patches_raster, grid_mlx)
    n_vis = int(visual_fixed.shape[0]); h_lat, w_lat = H_g // 2, W_g // 2

    print("[build] PT LLM + ViT (truth) ...")
    pt_embed, pt_layers, pt_final_norm, pt_lm_head, pt_vit = X.build_pt_x2t_model()
    with torch.no_grad():
        visual_pt = pt_vit(hidden_states=torch.from_numpy(pt_merge), grid_thw=torch.from_numpy(grid_np))
    visual_pt_bf16 = visual_pt.to(torch.bfloat16)

    seq = _build_seq(tok, n_vis)
    L = len(seq); vis_start = seq.index(VIS_START_ID) + 1; vis_end = vis_start + n_vis
    pos_mlx = build_positions_for_layout(L, [VisionSpec(start=vis_start - 1, length=n_vis, t=T_g, h=h_lat, w=w_lat)])
    pos_pt = torch.from_numpy(np.asarray(pos_mlx).astype(np.int64))
    attn_mask_mlx = build_lance_attention_mask(seq_len=L, split_lens=[L], attn_modes=["causal"])
    block_mask_pt = create_block_mask(create_sparse_mask([L], [L], ["causal"], torch.device("cpu")),
                                      B=1, H=16, Q_LEN=L, KV_LEN=L, device=torch.device("cpu"),
                                      BLOCK_SIZE=128, _compile=False)

    # PT truth logits
    ids_pt = torch.tensor([seq], dtype=torch.long)
    te_pt = pt_embed(ids_pt).to(torch.bfloat16)
    emb_pt = torch.cat([te_pt[:, :vis_start, :], visual_pt_bf16.unsqueeze(0), te_pt[:, vis_end:, :]], dim=1)
    cos_pt, sin_pt = X.mrope_cos_sin(pos_pt, head_dim=128, base=1_000_000.0, mrope_section=[16, 24, 24])
    all_idx = torch.arange(L, dtype=torch.long)
    h = emb_pt[0]
    with torch.no_grad():
        for layer in pt_layers:
            h = layer(packed_sequence=h, sample_lens=[L], attention_mask=block_mask_pt,
                      packed_position_embeddings=(cos_pt, sin_pt), packed_und_token_indexes=all_idx,
                      packed_gen_token_indexes=torch.empty(0, dtype=torch.long), mode_forward="validation")
        truth = pt_lm_head(pt_final_norm(h)[-1:]).to(torch.float32).cpu().numpy()

    lg_fixed = _mlx_logits(mlx_model, visual_fixed, seq, vis_start, vis_end, pos_mlx, attn_mask_mlx)
    lg_raster = _mlx_logits(mlx_model, visual_raster, seq, vis_start, vis_end, pos_mlx, attn_mask_mlx)

    cos = X._cosine
    vit_fixed = cos(visual_pt.cpu().numpy(), np.asarray(visual_fixed))
    vit_raster = cos(visual_pt.cpu().numpy(), np.asarray(visual_raster))
    c_fixed = cos(truth, lg_fixed); c_raster = cos(truth, lg_raster)
    t_truth = int(np.argmax(truth[0])); t_fixed = int(np.argmax(lg_fixed[0])); t_raster = int(np.argmax(lg_raster[0]))
    print("=" * 70)
    print(f"{'path':8s} {'ViT cos':10s} {'logit cos':12s} top-1")
    print(f"{'truth':8s} {'—':10s} {'—':12s} {t_truth} ('{tok.decode([t_truth])}')")
    print(f"{'FIXED':8s} {vit_fixed:<10.6f} {c_fixed:<12.6f} {t_fixed} ('{tok.decode([t_fixed])}')")
    print(f"{'RASTER':8s} {vit_raster:<10.6f} {c_raster:<12.6f} {t_raster} ('{tok.decode([t_raster])}')")
    print("=" * 70)
    drop = c_fixed - c_raster
    discriminates = (c_fixed >= 0.999) and (c_raster < 0.99) and (t_fixed == t_truth)
    print(f"logit-cos drop (FIXED - RASTER) = {drop:.6f}")
    print(f"DISCRIMINATES: {'YES — (c) metric is sensitive to the visual path' if discriminates else 'NO — metric insensitive, (c) would be blind'}")
    print(f"  (raster top-1 {'DIFFERS from' if t_raster != t_truth else 'matches'} truth)")
    sys.exit(0 if discriminates else 1)


if __name__ == "__main__":
    main()
