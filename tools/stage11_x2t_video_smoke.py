"""STAGE 11 step 2A — does the x2t_video forward RUN end-to-end?

Two gates:
  (i)  video ViT: vit_model.* -> vision_tower.* remap load + T>1 grid forward.
  (ii) full x2t_video: shared LLM backbone + video ViT + chat seq + AR decode.

The LLM backbone is byte-identical between the image and video weights (the
video weight = image backbone (+) supplement; the supplement only adds
vit_model + the 31-frame latent_pos_embed, which x2t does not use), so we load
the image backbone for the LLM and the video ViT (vit_model) for the vision.

"Does it run + shapes sane", NOT correctness — PT byte-diff (step 2B/C) proves it.
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, ".")

import numpy as np
import mlx.core as mx
import mlx.utils as mu
from transformers import AutoTokenizer

from lance_mlx.backbone import LanceLLM, LanceTextConfig
from lance_mlx.vit import LanceViT
from lance_mlx.pipelines.x2t import (
    preprocess_video, load_video_vit, x2t_video, SPATIAL_MERGE_SIZE,
)

VIDEO_WEIGHT = "out/lance_3b_video_mlx/model.safetensors"
IMG_BACKBONE = "out/lance_3b_mlx/model.safetensors"   # shared LLM (== video's)
FRAMES_NPY = "out/stage11_assets/vqa01_frames.npy"
TOK_DIR = "checkpoints/Lance-3B-MLX"


def main() -> None:
    # ---- gate (i): video ViT ----
    vit = LanceViT()
    n = load_video_vit(vit, VIDEO_WEIGHT)
    frames = np.load(FRAMES_NPY)
    clip = frames[:6]
    patches, (T_g, H_g, W_g) = preprocess_video(clip, max_pixels=56 * 56)
    visual = vit(patches, mx.array([[T_g, H_g, W_g]], dtype=mx.int32))
    mx.eval(visual)
    vit_ok = visual.shape[1] == 2048 and bool(np.isfinite(np.asarray(visual)).all())
    print(f"[2A-i ] video ViT: remap {n} keys, clip6->grid({T_g},{H_g},{W_g}) "
          f"-> {tuple(visual.shape)}  {'OK' if vit_ok else 'FAIL'}")

    # ---- gate (ii): full x2t_video end-to-end ----
    print("[2A-ii] building LanceLLM + loading shared backbone (24.7GB) ...")
    t0 = time.time()
    model = LanceLLM(LanceTextConfig())
    bw = mx.load(IMG_BACKBONE)
    ours = set(dict(mu.tree_flatten(model.parameters())).keys())
    to_load = {k: v for k, v in bw.items() if k in ours}
    model.load_weights(list(to_load.items()), strict=True)
    mx.eval(model.parameters())
    tok = AutoTokenizer.from_pretrained(TOK_DIR, use_fast=True)
    print(f"[2A-ii] backbone {len(to_load)} keys loaded in {time.time()-t0:.0f}s")

    question = "Watch the video and answer: what is happening? Options: (A) cooking (B) sports (C) talking"
    t1 = time.time()
    res = x2t_video(model, vit, tok, clip, question, max_new_tokens=12, max_pixels=56 * 56)
    print(f"[2A-ii] x2t_video ran in {time.time()-t1:.0f}s")
    print(f"[2A-ii] n_visual_tokens={res.n_visual_tokens}  out_tokens={res.tokens}")
    print(f"[2A-ii] decoded text: {res.text!r}")

    run_ok = vit_ok and len(res.tokens) > 0
    print("=" * 56)
    print("GATE step 2A:", "PASS — x2t_video forward runs end-to-end (ViT T>1 + LLM + AR)"
          if run_ok else "FAIL")


if __name__ == "__main__":
    main()
