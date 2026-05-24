"""STAGE 2 verification battery.

Runs N diverse prompts through LanceLLM (chat-template formatted) and
prints the top-5 next-token candidates for each.  Pass = the answer's
expected token appears in top-5 for ≥4 of 5 prompts.

This is the *qualitative* leg of STAGE 2 verification.  The quantitative
PyTorch-cosine leg is parked: it requires a PT reference setup that
duplicates Lance's qk_norm injection on top of a transformers
Qwen2.5-VL-3B-Instruct, and offers diminishing return until STAGE 4
exercises the MoE routing — at which point the same harness handles
both checks.
"""
from __future__ import annotations

import time

import mlx.core as mx
from transformers import AutoTokenizer

from lance_mlx.backbone import LanceTextConfig, LanceLLM, load_text_backbone


PROMPTS = [
    ("What is the capital of France? Answer in one word.",      ["Paris"]),
    ("What is 2 + 2? Answer with just the number.",             ["4", " 4"]),
    ("Complete: 'roses are red, violets are' — one word.",      ["blue", "Blue"]),
    ("What animal says 'meow'? One word.",                       ["cat", "Cat", " cat"]),
    ("What color is the sky on a clear day? One word.",         ["blue", "Blue", " blue"]),
]


def main() -> None:
    tok = AutoTokenizer.from_pretrained(
        "checkpoints/Lance-3B-MLX", trust_remote_code=True
    )
    cfg = LanceTextConfig()
    m = LanceLLM(cfg)
    t0 = time.time()
    load_text_backbone(m, "checkpoints/Lance-3B-MLX/model.safetensors")
    m.eval()
    print(f"[load] strict-load OK in {time.time()-t0:.1f}s")

    passes = 0
    for prompt, expected in PROMPTS:
        msg = [{"role": "user", "content": prompt}]
        formatted = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        ids = tok(formatted, return_tensors=None)["input_ids"]
        input_ids = mx.array([ids], dtype=mx.int32)
        L = len(ids)
        pos = mx.broadcast_to(
            mx.arange(L, dtype=mx.int32).reshape(1, -1)[None], (3, 1, L)
        )
        t0 = time.time()
        logits = m(input_ids, pos)
        mx.eval(logits)
        dt = time.time() - t0

        last = logits[0, -1]
        desc = mx.argsort(-last)[:5].tolist()
        top_decoded = [tok.decode([i]) for i in desc]
        hit = any(any(e.strip() == d.strip() for d in top_decoded) for e in expected)
        passes += int(hit)
        mark = "✓" if hit else "✗"

        print(f"\n[{mark}] {prompt!r}  ({L} tok, {dt:.2f}s)")
        print(f"    expected ∈ {expected!r}")
        for rank, idx in enumerate(desc):
            print(f"      #{rank+1}  {tok.decode([idx])!r:25s}  logit={last[idx].item():7.3f}")

    print(f"\n[result] {passes}/{len(PROMPTS)} prompts hit expected token in top-5")


if __name__ == "__main__":
    main()
