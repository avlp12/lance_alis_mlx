"""STAGE 2 smoke test: tokenize → forward → inspect top-k next-token logits."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlx.core as mx

from lance_mlx.backbone import LanceTextConfig, LanceLLM, load_text_backbone


def _build_text_position_ids(seq_len: int) -> mx.array:
    """For a pure-text prefix, 3D mRoPE collapses to (3, 1, L) of identical rows."""
    pos = mx.arange(seq_len, dtype=mx.int32).reshape(1, -1)         # (1, L)
    pos = mx.broadcast_to(pos[None, ...], (3, 1, seq_len))           # (3, 1, L)
    return pos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="checkpoints/Lance-3B-MLX/model.safetensors")
    ap.add_argument("--tokenizer-dir", default="checkpoints/Lance-3B-MLX")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--top-k", type=int, default=8)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)

    # Apply chat template if the tokenizer has one — Lance uses Qwen chat
    # format; for a "next-token" smoke test we deliberately *don't* apply
    # the chat template so we can compare logits at a known position.
    ids = tok(args.prompt, return_tensors=None)["input_ids"]
    print(f"[tok] {len(ids)} tokens: {ids}")
    print(f"[tok] decoded: {[tok.decode([i]) for i in ids]}")

    cfg = LanceTextConfig()
    print(f"[build] LanceLLM ({cfg.num_hidden_layers} layers, head_dim={cfg.head_dim}) ...")
    t0 = time.time()
    m = LanceLLM(cfg)
    stats = load_text_backbone(m, args.weights)
    print(f"[load] {stats['loaded_keys']} keys, skipped {stats['skipped_outer_keys']} outer, {time.time()-t0:.1f}s")

    input_ids = mx.array([ids], dtype=mx.int32)                # (1, L)
    position_ids = _build_text_position_ids(len(ids))

    print("[fwd ] forward pass ...")
    t0 = time.time()
    m.eval()
    logits = m(input_ids, position_ids)                        # (1, L, V)
    mx.eval(logits)
    print(f"[fwd ] done in {time.time()-t0:.2f}s   logits shape: {tuple(logits.shape)}   dtype: {logits.dtype}")

    last = logits[0, -1]                                       # (V,)
    print(f"[fwd ] last-token logit stats: min={last.min().item():.3f} max={last.max().item():.3f} "
          f"mean={last.mean().item():.3f} std={last.std().item():.3f}")

    # mx.argsort gives ascending; flip to descending and take first K.
    desc_idx = mx.argsort(-last)
    top_idx = desc_idx[: args.top_k].tolist()
    top_v = [float(last[i].item()) for i in top_idx]
    print(f"[fwd ] top-{args.top_k} next-token predictions:")
    for rank, (idx, score) in enumerate(zip(top_idx, top_v)):
        print(f"        #{rank+1}  id={idx:6d}  logit={score:7.3f}  → {tok.decode([idx])!r}")


if __name__ == "__main__":
    main()
