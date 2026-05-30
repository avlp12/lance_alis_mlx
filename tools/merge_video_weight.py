"""Merge image backbone + video supplement → standalone Lance_3B_Video MLX weight.

Replicates the STAGE 9 verified merge (stage9_mlx_30step.py:56-74):

    merged = dict(image_backbone)
    merged.update(video_supplement)              # supplement overrides
    to_load = {k: v for k, v in merged if k in model.params}

The override matters: the image backbone's `latent_pos_embed.pos_embed` is the
1-frame image variant, which would shape-mismatch the 31-frame video model; the
supplement's 31-frame version replaces it.  `model.load_weights(strict=True)`
validates that to_load covers the video model's params exactly (keys + shapes).

The result is serialized as the standalone Lance_3B_Video/model.safetensors
(F32), matching ByteDance's one-file-per-variant layout.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, ".")

import mlx.core as mx
import mlx.utils as mu

from lance_mlx.backbone import LanceLLM, LanceTextConfig, PositionEmbedding3D

MAX_NUM_LATENT_FRAMES = 31   # video (vs 1 for image)
MAX_LATENT_SIZE = 64


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="image backbone MLX safetensors")
    ap.add_argument("--sup", required=True, help="video supplement MLX safetensors")
    ap.add_argument("--dst", required=True, help="output standalone video weight")
    args = ap.parse_args()

    # Build the video model purely to obtain its exact parameter key/shape set.
    cfg = LanceTextConfig()
    model = LanceLLM(cfg)
    model.latent_pos_embed = PositionEmbedding3D(
        max_num_latent_frames=MAX_NUM_LATENT_FRAMES,
        max_latent_size=MAX_LATENT_SIZE,
        hidden_size=cfg.hidden_size,
    )
    ours = set(dict(mu.tree_flatten(model.parameters())).keys())

    img_w = mx.load(args.img)
    sup_w = mx.load(args.sup)
    merged = dict(img_w)
    for k, v in sup_w.items():
        merged[k] = v                       # supplement overrides on conflict

    to_load = {k: v for k, v in merged.items() if k in ours}

    not_in_model = sorted(set(merged) - ours)
    missing = sorted(ours - set(merged))
    print(f"[merge] img {len(img_w)} + sup {len(sup_w)} -> merged {len(merged)}")
    print(f"[merge] video model params {len(ours)} / to_load {len(to_load)}")
    print(f"[merge] merged keys dropped (not in model): {len(not_in_model)}  {not_in_model[:6]}")
    print(f"[merge] model params missing from merged:   {len(missing)}  {missing[:6]}")
    lpe = to_load.get("latent_pos_embed.pos_embed")
    print(f"[merge] latent_pos_embed.pos_embed: {tuple(lpe.shape) if lpe is not None else None}")
    print(f"[merge] vit_model keys: {sum(1 for k in to_load if k.startswith('vit_model'))}")

    # Strict load of the t2v subset validates keys + shapes against the t2v
    # model exactly (this is the part STAGE 9 verified end-to-end).
    model.load_weights(list(to_load.items()), strict=True)
    print("[merge] strict load OK — t2v subset (1021) covers the t2v model exactly")

    # The STANDALONE is the FULL merged state (ByteDance Lance_3B_Video bundles
    # the video ViT into one file): LLM backbone + adapters + 31-frame
    # latent_pos_embed (overridden) + video ViT (vit_model).  t2v consumes the
    # 1021-key subset; x2t_video / video_edit also use the vit_model keys.
    os.makedirs(os.path.dirname(args.dst), exist_ok=True)
    mx.save_safetensors(args.dst, merged)
    print(f"[save] {args.dst}  ({os.path.getsize(args.dst)/1e9:.2f} GB, {len(merged)} keys "
          f"= {len(to_load)} t2v + {len(merged)-len(to_load)} vit_model)")


if __name__ == "__main__":
    main()
