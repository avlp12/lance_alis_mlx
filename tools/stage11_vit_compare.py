"""STAGE 11 step 2B (b) — ViT byte-diff vs PT's REAL pipeline (image + video).

Supersedes the STAGE-7 ViT check for *preprocessing* correctness.  The STAGE-7
harness (stage7_x2t_compare) built patches once via our preprocess_image and fed
the SAME tensor to both PT and MLX ViT — so both mis-grouped identically and
agreed at cos 1.0, structurally BLIND to patch token-order.  This harness instead
makes each side use its OWN preprocessing and pins the reference to PT's real
pipeline: `patchify_video_with_merge` (2×2 merge-grouped) → PT ViT.

Two layers (isolate; doctrine "돈다 ≠ 맞다"):
  L0  preprocessing parity — our _patchify_frames patches == PT
      patchify_video_with_merge patches (same normalized frames), byte-identical.
  L1  ViT forward parity   — truth = PT_ViT(PT patches);  ours = MLX_ViT(our
      patches);  cos ≥ 0.999.  A `raster` column re-runs the OLD pre-fix order so
      the bug→fix delta is visible (raster ≈ 0.29–0.36, fixed ≈ 1.0).

Both cover the SAME LanceViT class; only the weights differ (image ViT
vit.safetensors `vision_tower.*`  vs  video weight `vit_model.*` remapped).

Reuses tools/stage7_vit_compare for the PT env (flash_attn shim, refs/Lance
path, PT ViT class, mlx_to_pt_vit_state layout transpose, _cosine).
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

import numpy as np
import torch
from PIL import Image

import mlx.core as mx

# Importing stage7_vit_compare installs the flash_attn shim, puts refs/Lance on
# the path, and imports the PT ViT class + config (all at module load).
import tools.stage7_vit_compare as S7
from data.data_utils import patchify_video_with_merge

from lance_mlx.pipelines.x2t import (
    preprocess_image, preprocess_video, _patchify_frames,
    QWEN_VL_IMAGE_MEAN, QWEN_VL_IMAGE_STD,
    PATCH_SIZE, SPATIAL_MERGE_SIZE, TEMPORAL_PATCH_SIZE,
)
from lance_mlx.vit import LanceViT

IMAGE_VIT = "checkpoints/Lance-3B-MLX/vit.safetensors"        # image ViT, vision_tower.*
VIDEO_W   = "out/lance_3b_video_mlx/model.safetensors"        # video weight, vit_model.*
IMG       = "refs/Lance/assets/image-understanding/cases/image-understanding-case-01.png"
FRAMES    = "out/stage11_assets/vqa01_frames.npy"

_cos = S7._cosine


def _resize_hw(H0: int, W0: int, max_pixels: int) -> tuple[int, int]:
    """preprocess_*'s resize rule: round down to multiples of 28, fit budget."""
    step = PATCH_SIZE * SPATIAL_MERGE_SIZE                    # 28
    H = max(step, (H0 // step) * step)
    W = max(step, (W0 // step) * step)
    if H * W > max_pixels:
        s = (max_pixels / (H * W)) ** 0.5
        H = max(step, int(H * s) // step * step)
        W = max(step, int(W * s) // step * step)
    return H, W


def _normalize(frames_uint8: np.ndarray, H: int, W: int) -> np.ndarray:
    """Resize+normalize a stack of uint8 frames to (N, H, W, 3) — same math as
    preprocess_image / preprocess_video (the normalized intermediate they hide)."""
    out = np.empty((len(frames_uint8), H, W, 3), np.float32)
    for i, f in enumerate(frames_uint8):
        im = Image.fromarray(f).resize((W, H), Image.BICUBIC)
        out[i] = (np.asarray(im, np.float32) / 255.0 - QWEN_VL_IMAGE_MEAN) / QWEN_VL_IMAGE_STD
    return out


def _raster_patches(norm: np.ndarray) -> np.ndarray:
    """The OLD pre-fix (plain raster) patchify — kept ONLY to show the delta."""
    p, tp = PATCH_SIZE, TEMPORAL_PATCH_SIZE
    T, H, W = norm.shape[:3]
    Tg, Hg, Wg = T // tp, H // p, W // p
    a = norm.reshape(Tg, tp, Hg, p, Wg, p, 3).transpose(0, 2, 4, 6, 1, 3, 5)
    return a.reshape(Tg * Hg * Wg, 3 * tp * p * p).astype(np.float32)


def _load_vit_both(weights_path: str, prefix: str):
    """Load a LanceViT (MLX) and an equivalent PT ViT from the same weights.
    prefix='vision_tower.' (image) keeps keys; 'vit_model.' (video) remaps."""
    full = mx.load(weights_path)
    if prefix == "vision_tower.":
        vt = dict(full)
    else:
        vt = {"vision_tower." + k[len(prefix):]: v
              for k, v in full.items() if k.startswith(prefix)}
    mlx_vit = LanceViT()
    mlx_vit.load_weights(list(vt.items()), strict=True)
    mx.eval(mlx_vit.parameters())
    mlx_vit.eval()
    pt_vit = S7.Qwen2_5_VisionTransformerPretrainedModel(S7.Qwen2_5_VLVisionConfig(
        depth=32, hidden_size=1280, intermediate_size=3420, num_heads=16, in_channels=3,
        patch_size=14, spatial_patch_size=14, spatial_merge_size=2, temporal_patch_size=2,
        window_size=112, layer_norm_eps=1e-6, tokens_per_second=2, out_hidden_size=2048,
        fullatt_block_indexes=[7, 15, 23, 31], hidden_act="silu"))
    miss, unexp = pt_vit.load_state_dict(S7.mlx_to_pt_vit_state(vt), strict=False)
    pt_vit.eval()
    return mlx_vit, pt_vit, len(vt), len(miss), len(unexp)


def run_case(name, weights_path, prefix, frames_uint8, ours_patches, grid):
    """ours_patches/grid come from the REAL production preprocess_* (ties the test
    to deployed code).  We rebuild the normalized frames to feed PT's patchify."""
    Tg, Hg, Wg = grid
    H, W = Hg * PATCH_SIZE, Wg * PATCH_SIZE
    norm = _normalize(frames_uint8, H, W)
    # Tie replication to production: our patchify(norm) must equal production patches.
    repro = np.asarray(_patchify_frames(norm)[0])
    tie = np.array_equal(repro, ours_patches)
    pt_patches = patchify_video_with_merge(
        torch.from_numpy(np.transpose(norm, (3, 0, 1, 2))), PATCH_SIZE, TEMPORAL_PATCH_SIZE).numpy()
    raster = _raster_patches(norm)

    l0 = np.array_equal(ours_patches, pt_patches)            # preprocessing parity

    mlx_vit, pt_vit, nkeys, miss, unexp = _load_vit_both(weights_path, prefix)
    gp = torch.from_numpy(np.array([list(grid)], np.int64))
    gm = mx.array(np.array([list(grid)], np.int32))
    with torch.no_grad():
        truth = pt_vit(hidden_states=torch.from_numpy(pt_patches), grid_thw=gp).cpu().numpy()
    ours_out   = np.asarray(mlx_vit(mx.array(ours_patches), gm))
    raster_out = np.asarray(mlx_vit(mx.array(raster), gm))

    return {
        "name": name, "grid": grid, "keys": nkeys, "miss": miss, "unexp": unexp,
        "tie": tie, "l0": l0,
        "cos_fixed": _cos(truth, ours_out),
        "cos_raster": _cos(truth, raster_out),
        "n_tokens": truth.shape[0],
    }


def main() -> None:
    cases = []

    # ---- image (image ViT) ----
    ours_img, grid_img = preprocess_image(IMG)
    ours_img = np.asarray(ours_img)
    img_pil = np.asarray(Image.open(IMG).convert("RGB"))
    cases.append(run_case("image", IMAGE_VIT, "vision_tower.",
                          np.stack([img_pil, img_pil]),   # the T=2 dup preprocess_image uses
                          ours_img, grid_img))

    # ---- video (video ViT = vit_model) ----
    clip = np.load(FRAMES)[:8]
    mp = 14 * 14 * 12 * 12
    ours_vid, grid_vid = preprocess_video(clip, max_pixels=mp)
    ours_vid = np.asarray(ours_vid)
    cases.append(run_case("video", VIDEO_W, "vit_model.", clip, ours_vid, grid_vid))

    print("=" * 74)
    print(f"{'case':6s} {'grid(T,H,W)':14s} {'keys':5s} {'tie':4s} {'L0':4s} "
          f"{'cos(truth,FIXED)':17s} {'cos(truth,raster)':17s}")
    print("-" * 74)
    ok = True
    for c in cases:
        l1 = c["cos_fixed"] >= 0.999
        ok = ok and c["tie"] and c["l0"] and l1
        print(f"{c['name']:6s} {str(c['grid']):14s} {c['keys']:<5d} "
              f"{'OK' if c['tie'] else 'X':4s} {'OK' if c['l0'] else 'X':4s} "
              f"{c['cos_fixed']:<17.6f} {c['cos_raster']:<17.6f}")
        if c["miss"] or c["unexp"]:
            print(f"       (PT load miss={c['miss']} unexp={c['unexp']})")
    print("=" * 74)
    print("L0 preprocessing parity = our patches BYTE-identical to PT patchify_video_with_merge")
    print("L1 ViT forward parity   = cos(PT-real-pipeline, our preprocess+ViT) ≥ 0.999")
    print("raster column           = OLD pre-fix order, for the bug→fix delta")
    print("GATE step 2B (b):", "PASS — our vision stream matches PT's real pipeline (image+video)"
          if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
