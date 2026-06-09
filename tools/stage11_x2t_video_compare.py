"""STAGE 11 (c)-video — NON-BLIND end-to-end x2t_video logit parity vs PT.

Video analogue of stage11_x2t_image_compare.  Same Lance LLM backbone (the image
and video weights share it byte-identically), but the vision stream is the VIDEO
ViT (`vit_model`, 390 keys) over a T>1 clip.

What this adds beyond (b)-video (which proved the ViT output matches PT, cos 1.0)
and (c)-image (which sealed mRoPE consistency at T=1): the full LLM forward with a
**T>1** video grid — so the mRoPE position assignment for multi-frame visual
tokens is exercised end-to-end (Lesson 23: prove it in the real forward).

Non-blind: PT builds its OWN ViT input via patchify_video_with_merge; MLX uses
preprocess_video.  The patch-order fix makes them byte-identical (asserted).

Gate: first-token logits cos ≥ 0.999 (bf16 PT vs f32 MLX → ~0.9999).
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

import numpy as np
import torch
from PIL import Image

import mlx.core as mx
from transformers import AutoTokenizer

# transformers ≥5.9 flash-probe neutraliser — must precede PT modeling import.
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
import tools.stage7_vit_compare as S7
from data.data_utils import patchify_video_with_merge, create_sparse_mask
from torch.nn.attention.flex_attention import create_block_mask

from lance_mlx.backbone import LanceLLM, LanceTextConfig, load_full_lance
from lance_mlx.vit import LanceViT
from lance_mlx.pipelines.x2t import (
    preprocess_video, load_video_vit, QWEN_VL_IMAGE_MEAN, QWEN_VL_IMAGE_STD,
    PATCH_SIZE, SPATIAL_MERGE_SIZE, VIDEO_TEMPORAL_SCALE,
    IM_START_ID, IM_END_ID, VIS_START_ID, VIS_END_ID, VIDEO_PAD_ID,
)
from lance_mlx.rope import VisionSpec, build_positions_for_layout
from lance_mlx.attn_mask import build_lance_attention_mask

import importlib as _importlib
_qwen2_navit = _importlib.import_module("modeling.lance.qwen2_navit")  # X already put refs/Lance on path


class _MockVisionConfig(dict):
    pass


class _MockConfig:
    def __init__(self):
        self.image_token_id = 151655
        self.video_token_id = 151656
        self.vision_start_token_id = 151652
        self.vision_config = _MockVisionConfig(
            {"spatial_merge_size": 2, "tokens_per_second": 2, "temporal_patch_size": 2})


class _MockSelf:
    def __init__(self):
        self.config = _MockConfig()


def _pt_positions(seq, T_g, H_g, W_g):
    """PT Lance get_rope_index — PT's OWN mRoPE positions (NOT our pos_mlx).
    Closes the position-reuse blindness the adversarial review found."""
    grid = torch.tensor([[T_g, H_g, W_g]])
    pos, _ = _qwen2_navit.Qwen2ForCausalLM.get_rope_index(
        _MockSelf(), input_ids=torch.tensor([seq], dtype=torch.long),
        image_grid_thw=grid, video_grid_thw=grid,
        second_per_grid_ts=torch.tensor([1.0]), attention_mask=None)
    return pos  # (3, 1, L) int64

MLX_LLM   = "checkpoints/Lance-3B-MLX/model.safetensors"     # Lance LLM backbone (== video's)
VIDEO_W   = "out/lance_3b_video_mlx/model.safetensors"       # vit_model lives here
FRAMES    = "out/stage11_assets/vqa01_frames.npy"
QUESTION  = "What is happening in this video?"
MAX_PIX   = 14 * 14 * 12 * 12
N_FRAMES  = 8


def _norm_video(clip_uint8: np.ndarray, max_pixels: int) -> np.ndarray:
    """Replicate preprocess_video's resize+normalize → (N, H, W, 3).  Only used
    to feed PT's OWN patchify with the same normalized frames."""
    N, H0, W0 = clip_uint8.shape[:3]
    step = PATCH_SIZE * SPATIAL_MERGE_SIZE
    H = max(step, (H0 // step) * step)
    W = max(step, (W0 // step) * step)
    if H * W > max_pixels:
        s = (max_pixels / (H * W)) ** 0.5
        H = max(step, int(H * s) // step * step)
        W = max(step, int(W * s) // step * step)
    out = np.empty((N, H, W, 3), np.float32)
    for i in range(N):
        im = Image.fromarray(clip_uint8[i]).resize((W, H), Image.BICUBIC)
        out[i] = (np.asarray(im, np.float32) / 255.0 - QWEN_VL_IMAGE_MEAN) / QWEN_VL_IMAGE_STD
    return out


def _pt_video_vit():
    """PT Qwen2.5-VL ViT loaded with the video vit_model (from the MLX video
    weight, transposed back to PT layout)."""
    full = mx.load(VIDEO_W)
    vt = {"vision_tower." + k[len("vit_model."):]: v
          for k, v in full.items() if k.startswith("vit_model.")}
    pt_vit = S7.Qwen2_5_VisionTransformerPretrainedModel(S7.Qwen2_5_VLVisionConfig(
        depth=32, hidden_size=1280, intermediate_size=3420, num_heads=16, in_channels=3,
        patch_size=14, spatial_patch_size=14, spatial_merge_size=2, temporal_patch_size=2,
        window_size=112, layer_norm_eps=1e-6, tokens_per_second=2, out_hidden_size=2048,
        fullatt_block_indexes=[7, 15, 23, 31], hidden_act="silu"))
    pt_vit.load_state_dict(S7.mlx_to_pt_vit_state(vt), strict=False)
    pt_vit.eval()
    return pt_vit


def main() -> None:
    tok = AutoTokenizer.from_pretrained("checkpoints/Lance-3B-MLX", trust_remote_code=True)
    clip = np.load(FRAMES)[:N_FRAMES]

    # ---- patches: each side its OWN preprocessing (non-blind) ----
    patches_mlx, (T_g, H_g, W_g) = preprocess_video(clip, max_pixels=MAX_PIX)
    patches_mlx_np = np.asarray(patches_mlx)
    norm = _norm_video(clip, MAX_PIX)
    patches_pt_np = patchify_video_with_merge(
        torch.from_numpy(np.transpose(norm, (3, 0, 1, 2))), PATCH_SIZE, 2).numpy()
    tie = np.array_equal(patches_mlx_np, patches_pt_np)
    print(f"[patches] MLX(preprocess_video) vs PT(patchify_video_with_merge): "
          f"byte-identical={tie}  grid=({T_g},{H_g},{W_g})  N_patch={patches_mlx_np.shape[0]}")
    if not tie:
        print("  ✗ patches differ — STOP and localize."); sys.exit(2)

    grid_np = np.array([[T_g, H_g, W_g]], dtype=np.int64)
    grid_mlx = mx.array(grid_np.astype(np.int32))
    grid_pt = torch.from_numpy(grid_np)

    # ---- MLX: LLM backbone + video ViT ----
    print("[build] LanceLLM + video LanceViT (MLX, f32) ...")
    mlx_model = LanceLLM(LanceTextConfig()); load_full_lance(mlx_model, MLX_LLM); mlx_model.eval()
    mlx_vit = LanceViT(); load_video_vit(mlx_vit, VIDEO_W); mlx_vit.eval()
    visual_mlx = mlx_vit(patches_mlx, grid_mlx)
    n_vis = int(visual_mlx.shape[0])
    h_lat, w_lat = H_g // 2, W_g // 2

    # ---- PT: LLM backbone (reuse builder; ignore its image ViT) + video ViT ----
    print("[build] PT LLM + video ViT ...")
    pt_embed, pt_layers, pt_final_norm, pt_lm_head, _img_vit = X.build_pt_x2t_model()
    pt_vit = _pt_video_vit()
    with torch.no_grad():
        visual_pt = pt_vit(hidden_states=torch.from_numpy(patches_pt_np), grid_thw=grid_pt)
    visual_pt_bf16 = visual_pt.to(torch.bfloat16)
    print(f"[ViT] cos(PT, MLX) = {X._cosine(visual_pt.cpu().numpy(), np.asarray(visual_mlx)):.6f}")

    # ---- chat sequence (VIDEO_PAD_ID placeholders; overwritten by visual embeds) ----
    sys_ids = tok("You are a helpful assistant.", add_special_tokens=False)["input_ids"]
    q_ids = tok(QUESTION, add_special_tokens=False)["input_ids"]
    newline = tok("\n", add_special_tokens=False)["input_ids"]
    sys_lbl = tok("system", add_special_tokens=False)["input_ids"]
    usr_lbl = tok("user", add_special_tokens=False)["input_ids"]
    asst_lbl = tok("assistant", add_special_tokens=False)["input_ids"]
    seq = (
        [IM_START_ID] + sys_lbl + newline + sys_ids + [IM_END_ID] + newline
        + [IM_START_ID] + usr_lbl + newline
        + [VIS_START_ID] + [VIDEO_PAD_ID] * n_vis + [VIS_END_ID]
        + q_ids + [IM_END_ID] + newline
        + [IM_START_ID] + asst_lbl + newline
    )
    L = len(seq)
    vis_start = seq.index(VIS_START_ID) + 1
    vis_end = vis_start + n_vis
    print(f"[seq] L={L}  vis=[{vis_start},{vis_end})  N_vis={n_vis}  (T_g={T_g} → T>1 positions)")

    # OUR positions — x2t_video code path (temporal_scale=2 = STAGE 11 fix).
    pos_mlx = build_positions_for_layout(
        L, [VisionSpec(start=vis_start - 1, length=n_vis, t=T_g, h=h_lat, w=w_lat,
                       temporal_scale=VIDEO_TEMPORAL_SCALE)])
    # PT positions — INDEPENDENTLY from PT get_rope_index (no longer from_numpy(pos_mlx)).
    pos_pt = _pt_positions(seq, T_g, H_g, W_g)
    _pos_id = np.array_equal(np.asarray(pos_mlx).astype(np.int64), pos_pt.numpy())
    print(f"[positions] ours(temporal_scale={VIDEO_TEMPORAL_SCALE}) vs PT get_rope_index: "
          f"byte-identical={_pos_id}")
    if not _pos_id:
        print("  ✗ x2t-video positions DIVERGE from PT — STOP (the temporal-scale fix is wrong)."); sys.exit(2)
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
    ok = cos >= 0.999 and pt_top == mlx_top and tie and _pos_id
    print(f"GATE (c)-video: {'PASS' if ok else 'FAIL'}  "
          f"(cos≥0.999 + top-1 agree + non-blind patches + PT-independent positions byte-identical, T>1)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
