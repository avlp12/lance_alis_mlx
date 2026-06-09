"""STAGE 11 — canonical HARDENED x2t verification gate (image + video).

Supersedes the ad-hoc (c) scripts as the auditable gate.  Addresses every
weakness the adversarial review (workflow wzkjmih3i) raised about the thin
first-token check:

  * MULTI-TOKEN (K=8): greedy-decode K tokens with the production MLX path, then
    teacher-force seq+R through PT and MLX and compare PT-vs-MLX logits at EACH
    of the K generation steps (cos + top-1).  Exercises visually-conditioned
    tokens, not just the prior-dominated first token.  PT runs ONE forward per
    case (cheap) — greedy decode is MLX-only.
  * DISCRIMINATIVE CONTROL, GATED: the SAME K-step compare is run with RASTER
    (old-bug) visual tokens; the gate PASSES only if FIXED matches PT AND RASTER
    diverges (cos<0.99 or top-1 mismatch on some step) — so a green result can
    never be reported without proving the metric is sensitive to the visual path.
  * PRODUCTION PROMPT: instruction-as-system-prompt (the example's instruction),
    not the placeholder "You are a helpful assistant." (matches step 2B (a′)).
  * NON-BLIND: PT builds its own patches (patchify_video_with_merge) and its own
    mRoPE positions (get_rope_index, temporal step ×tokens_per_second); both are
    asserted byte-identical to ours before the logits are trusted.
  * RUN-RECORD: results persisted to out/stage11_x2t_verify.json.

SCOPE (honest): this verifies — GIVEN identical resized+normalized frames and
thus an identical grid — that our patchify→ViT→LLM matches PT's
patchify_video_with_merge→ViT→LLM to the bf16(PT)/f32(MLX) precision boundary.
It does NOT exercise PT's real `vit_transform` (bucket-mode NaResize → different
grid/pixels for raw media); the resize policy is a deliberate MLX-side choice,
verified out of band, not a PT match.
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, ".")

import numpy as np
import torch
from PIL import Image

import mlx.core as mx
from transformers import AutoTokenizer

# transformers >=5.9 flash-probe neutraliser (before any PT modeling import).
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

import importlib as _importlib
import tools.stage7_x2t_compare as X
import tools.stage7_vit_compare as S7
_qwen2_navit = _importlib.import_module("modeling.lance.qwen2_navit")
from data.data_utils import patchify_video_with_merge, create_sparse_mask
from torch.nn.attention.flex_attention import create_block_mask

from lance_mlx.backbone import LanceLLM, LanceTextConfig, load_full_lance
from lance_mlx.vit import LanceViT, load_lance_vit
from lance_mlx.pipelines.x2t import (
    preprocess_image, preprocess_video, load_video_vit,
    QWEN_VL_IMAGE_MEAN, QWEN_VL_IMAGE_STD,
    PATCH_SIZE, SPATIAL_MERGE_SIZE, TEMPORAL_PATCH_SIZE, VIDEO_TEMPORAL_SCALE,
    IM_START_ID, IM_END_ID, VIS_START_ID, VIS_END_ID, IMG_TOKEN_ID, VIDEO_PAD_ID,
)
from lance_mlx.rope import VisionSpec, build_positions_for_layout
from lance_mlx.attn_mask import build_lance_attention_mask

MLX_WEIGHTS = "checkpoints/Lance-3B-MLX/model.safetensors"
VIT_WEIGHTS = "checkpoints/Lance-3B-MLX/vit.safetensors"
VIDEO_W = "out/lance_3b_video_mlx/model.safetensors"
IMAGE = "refs/Lance/assets/image-understanding/cases/image-understanding-case-01.png"
FRAMES = "out/stage11_assets/vqa01_frames.npy"
K = 8
MAX_PIX_VIDEO = 14 * 14 * 12 * 12
_cos = X._cosine


# ---- PT get_rope_index (PT's own mRoPE positions) -----------------------------
class _MockVisionConfig(dict):
    pass
class _MockConfig:
    def __init__(self):
        self.image_token_id = 151655; self.video_token_id = 151656
        self.vision_start_token_id = 151652
        self.vision_config = _MockVisionConfig(
            {"spatial_merge_size": 2, "tokens_per_second": 2, "temporal_patch_size": 2})
class _MockSelf:
    def __init__(self):
        self.config = _MockConfig()
def _pt_positions(seq, T_g, H_g, W_g):
    grid = torch.tensor([[T_g, H_g, W_g]])
    pos, _ = _qwen2_navit.Qwen2ForCausalLM.get_rope_index(
        _MockSelf(), input_ids=torch.tensor([seq], dtype=torch.long),
        image_grid_thw=grid, video_grid_thw=grid,
        second_per_grid_ts=torch.tensor([1.0]), attention_mask=None)
    return pos.numpy()


def _raster_patches(norm):
    p, tp = PATCH_SIZE, TEMPORAL_PATCH_SIZE
    T, H, W = norm.shape[:3]
    Tg, Hg, Wg = T // tp, H // p, W // p
    a = norm.reshape(Tg, tp, Hg, p, Wg, p, 3).transpose(0, 2, 4, 6, 1, 3, 5)
    return mx.array(a.reshape(Tg * Hg * Wg, 3 * tp * p * p).astype(np.float32))


def _norm_image(path):
    img = Image.open(path).convert("RGB"); W0, H0 = img.size
    step = PATCH_SIZE * SPATIAL_MERGE_SIZE; mp = 14 * 14 * 16 * 16
    H = max(step, (H0 // step) * step); W = max(step, (W0 // step) * step)
    if H * W > mp:
        s = (mp / (H * W)) ** 0.5
        H = max(step, int(H * s) // step * step); W = max(step, int(W * s) // step * step)
    arr = (np.asarray(img.resize((W, H), Image.BICUBIC), np.float32) / 255.0 - QWEN_VL_IMAGE_MEAN) / QWEN_VL_IMAGE_STD
    return np.stack([arr, arr], 0)


def _norm_video(clip, mp):
    N, H0, W0 = clip.shape[:3]
    step = PATCH_SIZE * SPATIAL_MERGE_SIZE
    H = max(step, (H0 // step) * step); W = max(step, (W0 // step) * step)
    if H * W > mp:
        s = (mp / (H * W)) ** 0.5
        H = max(step, int(H * s) // step * step); W = max(step, int(W * s) // step * step)
    out = np.empty((N, H, W, 3), np.float32)
    for i in range(N):
        im = Image.fromarray(clip[i]).resize((W, H), Image.BICUBIC)
        out[i] = (np.asarray(im, np.float32) / 255.0 - QWEN_VL_IMAGE_MEAN) / QWEN_VL_IMAGE_STD
    return out


def _build_seq(tok, instruction, question, n_vis, placeholder_id):
    sys_ids = tok(instruction, add_special_tokens=False)["input_ids"]
    q_ids = tok(question, add_special_tokens=False)["input_ids"]
    nl = tok("\n", add_special_tokens=False)["input_ids"]
    s = tok("system", add_special_tokens=False)["input_ids"]
    u = tok("user", add_special_tokens=False)["input_ids"]
    a = tok("assistant", add_special_tokens=False)["input_ids"]
    seq = ([IM_START_ID] + s + nl + sys_ids + [IM_END_ID] + nl
           + [IM_START_ID] + u + nl
           + [VIS_START_ID] + [placeholder_id] * n_vis + [VIS_END_ID]
           + q_ids + [IM_END_ID] + nl + [IM_START_ID] + a + nl)
    vis_start = seq.index(VIS_START_ID) + 1
    return seq, vis_start, vis_start + n_vis


def _mlx_logits_all(mlx_model, seq_tokens, visual, vis_start, vis_end, pos, mask):
    ids = mx.array([seq_tokens], dtype=mx.int32)
    te = mlx_model.language_model.model.embed_tokens(ids)
    emb = mx.concatenate([te[:, :vis_start, :], visual[None, :, :], te[:, vis_end:, :]], axis=1)
    h = mlx_model.language_model.model(input_ids=None, position_ids=pos,
                                       inputs_embeds=emb, mask=mask, gen_mask=None)
    return mlx_model.language_model.lm_head(h[0])           # (L, V)


def _pt_logits_all(pt, seq_tokens, visual_pt_bf16, vis_start, vis_end, pos_pt_np):
    embed, layers, norm, lm_head = pt
    L = len(seq_tokens)
    ids = torch.tensor([seq_tokens], dtype=torch.long)
    te = embed(ids).to(torch.bfloat16)
    emb = torch.cat([te[:, :vis_start, :], visual_pt_bf16.unsqueeze(0), te[:, vis_end:, :]], dim=1)
    cos_pt, sin_pt = X.mrope_cos_sin(torch.from_numpy(pos_pt_np).long(),
                                     head_dim=128, base=1_000_000.0, mrope_section=[16, 24, 24])
    bm = create_block_mask(create_sparse_mask([L], [L], ["causal"], torch.device("cpu")),
                           B=1, H=16, Q_LEN=L, KV_LEN=L, device=torch.device("cpu"),
                           BLOCK_SIZE=128, _compile=False)
    h = emb[0]
    all_idx = torch.arange(L, dtype=torch.long)
    with torch.no_grad():
        for layer in layers:
            h = layer(packed_sequence=h, sample_lens=[L], attention_mask=bm,
                      packed_position_embeddings=(cos_pt, sin_pt), packed_und_token_indexes=all_idx,
                      packed_gen_token_indexes=torch.empty(0, dtype=torch.long), mode_forward="validation")
        return lm_head(norm(h)).to(torch.float32).cpu().numpy()    # (L, V)


def run_case(name, tok, mlx_model, pt_llm, mlx_vit, pt_vit,
             patches_fixed, patches_raster, norm, grid, instruction, question,
             placeholder_id, temporal_scale):
    T_g, H_g, W_g = grid
    h_lat, w_lat = H_g // SPATIAL_MERGE_SIZE, W_g // SPATIAL_MERGE_SIZE
    n_vis = T_g * h_lat * w_lat
    grid_mlx = mx.array(np.array([[T_g, H_g, W_g]], np.int32))
    grid_pt = torch.from_numpy(np.array([[T_g, H_g, W_g]], np.int64))

    # patch independence (non-blind): our patches vs PT's own patchify
    pt_patches = patchify_video_with_merge(
        torch.from_numpy(np.transpose(norm, (3, 0, 1, 2))), PATCH_SIZE, TEMPORAL_PATCH_SIZE).numpy()
    tie = np.array_equal(np.asarray(patches_fixed), pt_patches)

    visual_fixed = mlx_vit(patches_fixed, grid_mlx)
    visual_raster = mlx_vit(patches_raster, grid_mlx)
    with torch.no_grad():
        visual_pt = pt_vit(hidden_states=torch.from_numpy(pt_patches), grid_thw=grid_pt)
    visual_pt_bf16 = visual_pt.to(torch.bfloat16)
    vit_cos_fixed = _cos(visual_pt.cpu().numpy(), np.asarray(visual_fixed))

    seq, vis_start, vis_end = _build_seq(tok, instruction, question, n_vis, placeholder_id)
    L = len(seq)
    span = lambda nseq: VisionSpec(start=vis_start - 1, length=n_vis, t=T_g, h=h_lat, w=w_lat,
                                   temporal_scale=temporal_scale)

    # ---- MLX-fixed greedy decode K (production path) -> R + MLX_fixed per-step logits ----
    R, mlx_fixed_step = [], []
    cur = list(seq)
    for _k in range(K):
        Lc = len(cur)
        pos = build_positions_for_layout(Lc, [span(Lc)])
        mask = build_lance_attention_mask(seq_len=Lc, split_lens=[Lc], attn_modes=["causal"])
        lg = _mlx_logits_all(mlx_model, cur, visual_fixed, vis_start, vis_end, pos, mask)  # (Lc,V)
        last = np.asarray(lg[-1])
        mlx_fixed_step.append(last)
        nxt = int(np.argmax(last)); R.append(nxt); cur.append(nxt)

    seq_ext = cur
    Lx = len(seq_ext)
    pos_ours = np.asarray(build_positions_for_layout(Lx, [span(Lx)]))           # (3,1,Lx)
    pos_pt = _pt_positions(seq_ext, T_g, H_g, W_g)                              # (3,1,Lx)
    pos_id = np.array_equal(pos_ours.astype(np.int64), pos_pt.astype(np.int64))

    # ---- PT teacher-forced over seq_ext (PT's own positions) ----
    mask_mlx = build_lance_attention_mask(seq_len=Lx, split_lens=[Lx], attn_modes=["causal"])
    pos_mlx = build_positions_for_layout(Lx, [span(Lx)])
    pt_all = _pt_logits_all(pt_llm, seq_ext, visual_pt_bf16, vis_start, vis_end, pos_pt)        # (Lx,V)
    # ---- MLX-raster teacher-forced over seq_ext ----
    mlx_raster_all = np.asarray(_mlx_logits_all(mlx_model, seq_ext, visual_raster, vis_start, vis_end, pos_mlx, mask_mlx))

    # gen-step logits at positions L-1 .. L+K-2
    steps = []
    for k in range(K):
        p = L - 1 + k
        pt_k = pt_all[p]; rf_k = mlx_raster_all[p]; fx_k = mlx_fixed_step[k]
        steps.append({
            "k": k, "tok": int(R[k]), "tok_str": tok.decode([R[k]]),
            "cos_fixed": _cos(pt_k, fx_k), "cos_raster": _cos(pt_k, rf_k),
            "pt_top": int(np.argmax(pt_k)), "fixed_top": int(np.argmax(fx_k)),
            "raster_top": int(np.argmax(rf_k)),
        })

    cos_fixed_min = min(s["cos_fixed"] for s in steps)
    fixed_top_agree = all(s["pt_top"] == s["fixed_top"] for s in steps)
    cos_raster_min = min(s["cos_raster"] for s in steps)
    raster_diverges = (cos_raster_min < 0.99) or any(s["raster_top"] != s["pt_top"] for s in steps)

    ok = tie and pos_id and (cos_fixed_min >= 0.999) and fixed_top_agree and raster_diverges
    return {
        "name": name, "grid": [T_g, H_g, W_g], "n_vis": n_vis, "L": L, "K": K,
        "patches_byte_identical": bool(tie), "positions_byte_identical_to_PT": bool(pos_id),
        "vit_cos_fixed": vit_cos_fixed,
        "answer_fixed": tok.decode(R),
        "cos_fixed_min": cos_fixed_min, "fixed_top1_all_agree": bool(fixed_top_agree),
        "cos_raster_min": cos_raster_min, "raster_discriminates": bool(raster_diverges),
        "steps": steps, "pass": bool(ok),
    }


def main() -> None:
    tok = AutoTokenizer.from_pretrained("checkpoints/Lance-3B-MLX", trust_remote_code=True)
    print("[build] MLX LLM + PT LLM (shared backbone) ...")
    mlx_model = LanceLLM(LanceTextConfig()); load_full_lance(mlx_model, MLX_WEIGHTS); mlx_model.eval()
    pt_embed, pt_layers, pt_norm, pt_lmhead, pt_image_vit = X.build_pt_x2t_model()
    pt_llm = (pt_embed, pt_layers, pt_norm, pt_lmhead)

    results = []

    # ---- image ----
    img_vit = LanceViT(); load_lance_vit(img_vit, VIT_WEIGHTS); img_vit.eval()
    pf, grid_i = preprocess_image(IMAGE)
    norm_i = _norm_image(IMAGE)
    results.append(run_case(
        "image", tok, mlx_model, pt_llm, img_vit, pt_image_vit,
        pf, _raster_patches(norm_i), norm_i, grid_i,
        "Look at the image carefully and answer the question.",
        "Is the largest segment greater than sum of all the other segments?",
        IMG_TOKEN_ID, 1))

    # ---- video ----
    vid_vit = LanceViT(); load_video_vit(vid_vit, VIDEO_W); vid_vit.eval()
    full = mx.load(VIDEO_W)
    vt = {"vision_tower." + k[len("vit_model."):]: v for k, v in full.items() if k.startswith("vit_model.")}
    pt_vid_vit = S7.Qwen2_5_VisionTransformerPretrainedModel(S7.Qwen2_5_VLVisionConfig(
        depth=32, hidden_size=1280, intermediate_size=3420, num_heads=16, in_channels=3, patch_size=14,
        spatial_patch_size=14, spatial_merge_size=2, temporal_patch_size=2, window_size=112,
        layer_norm_eps=1e-6, tokens_per_second=2, out_hidden_size=2048,
        fullatt_block_indexes=[7, 15, 23, 31], hidden_act="silu"))
    pt_vid_vit.load_state_dict(S7.mlx_to_pt_vit_state(vt), strict=False); pt_vid_vit.eval()
    clip = np.load(FRAMES)[:8]
    pv, grid_v = preprocess_video(clip, max_pixels=MAX_PIX_VIDEO)
    norm_v = _norm_video(clip, MAX_PIX_VIDEO)
    results.append(run_case(
        "video", tok, mlx_model, pt_llm, vid_vit, pt_vid_vit,
        pv, _raster_patches(norm_v), norm_v, grid_v,
        "Watch the video carefully and answer the question.",
        "How many times did the person launch objects on the table? Options: (A) 3 (B) 2 (C) 4",
        VIDEO_PAD_ID, VIDEO_TEMPORAL_SCALE))

    with open("out/stage11_x2t_verify.json", "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 78)
    print("SCOPE: given identical resized+normalized frames & grid, MLX path == PT path")
    print("       (PT's real bucket-resize vit_transform is out of scope — see module docstring)")
    print("-" * 78)
    allok = True
    for r in results:
        allok &= r["pass"]
        print(f"\n[{r['name']}] grid={tuple(r['grid'])} n_vis={r['n_vis']} K={r['K']}  "
              f"answer={r['answer_fixed']!r}")
        print(f"  patches byte-id={r['patches_byte_identical']}  positions byte-id(PT)={r['positions_byte_identical_to_PT']}  "
              f"ViT cos={r['vit_cos_fixed']:.6f}")
        print(f"  FIXED : per-step logit-cos min={r['cos_fixed_min']:.6f}  top1-all-agree={r['fixed_top1_all_agree']}")
        print(f"  RASTER: per-step logit-cos min={r['cos_raster_min']:.6f}  discriminates={r['raster_discriminates']}")
        bad = [s["k"] for s in r["steps"] if s["raster_top"] != s["pt_top"]]
        print(f"          (raster top-1 diverges from PT at steps {bad})")
        print(f"  => {'PASS' if r['pass'] else 'FAIL'}")
    print("=" * 78)
    print("GATE stage11_x2t_verify:", "PASS — hardened (multi-token K=8 + discriminative + non-blind + production prompt)"
          if allok else "FAIL")
    print("[log] out/stage11_x2t_verify.json")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
