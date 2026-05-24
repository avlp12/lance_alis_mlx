"""STAGE 2 quantitative verification.

(A) Architectural cosine: our LanceLLM(has_qk_norm=False, has_moe_gen=False)
    vs mlx-vlm's stock Qwen2.5-VL LanguageModel, both loaded from the *same*
    Lance language_model.* weights filtered down to the standard Qwen2.5-VL
    key set.  Expectation: cosine ≥ 0.99999 — proves our skeleton (RoPE,
    attention shape ops, MLP, layer norm, lm_head) is bit-equivalent to the
    accepted reference.

(B) qk_norm contribution: same Lance weights, our LanceLLM with
    has_qk_norm=True vs has_qk_norm=False, identical chat prompts.
    Reports cosine + top-1 stability so we can attribute the 3/5 imperfect
    knowledge probes to qk_norm vs Lance fine-tune distribution.
"""
from __future__ import annotations

import time

import mlx.core as mx
from mlx.utils import tree_flatten
from transformers import AutoTokenizer

from lance_mlx.backbone import LanceTextConfig, LanceLLM, load_text_backbone


PROMPTS = [
    "What is the capital of France? Answer in one word.",
    "What is 2 + 2? Answer with just the number.",
    "Complete: 'roses are red, violets are' — one word.",
    "What animal says 'meow'? One word.",
    "What color is the sky on a clear day? One word.",
]


def _tokenize_chat(tok, prompt: str) -> mx.array:
    msg = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors=None)["input_ids"]
    return mx.array([ids], dtype=mx.int32)


def _pos_ids(L: int) -> mx.array:
    return mx.broadcast_to(
        mx.arange(L, dtype=mx.int32).reshape(1, -1)[None], (3, 1, L)
    )


def _cosine(a: mx.array, b: mx.array) -> float:
    af = a.flatten().astype(mx.float32)
    bf = b.flatten().astype(mx.float32)
    return float(
        (mx.sum(af * bf) / (mx.linalg.norm(af) * mx.linalg.norm(bf) + 1e-12)).item()
    )


# ---------------------------------------------------------------------------
# (A) Architectural cosine vs mlx-vlm stock backbone
# ---------------------------------------------------------------------------
def part_a_arch_cosine() -> None:
    from mlx_vlm.models.qwen2_5_vl.language import LanguageModel as RefLM
    from mlx_vlm.models.qwen2_5_vl.config import TextConfig, ModelConfig, VisionConfig

    print("\n========== (A) Architectural cosine vs mlx-vlm stock ==========")

    # Build mlx-vlm reference config (no qk_norm, no moe_gen in module tree).
    ref_text_cfg = TextConfig(
        model_type="qwen2_5_vl",
        hidden_size=2048,
        num_hidden_layers=36,
        intermediate_size=11008,
        num_attention_heads=16,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=151936,
        max_position_embeddings=128000,
        rope_theta=1_000_000.0,
        rope_scaling={"type": "mrope", "mrope_section": [16, 24, 24]},
        tie_word_embeddings=True,
    )
    ref_model_cfg = ModelConfig(
        text_config=ref_text_cfg,
        vision_config=VisionConfig(
            model_type="qwen2_5_vl",
            depth=32,
            hidden_size=1280,
            intermediate_size=3420,
            num_heads=16,
            in_channels=3,
            out_hidden_size=2048,
        ),
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        vision_token_id=151654,
        model_type="qwen2_5_vl",
    )
    ref = RefLM(ref_text_cfg, ref_model_cfg)

    # Build our model in *stock* mode (no qk_norm, no moe_gen).
    our_cfg = LanceTextConfig(has_qk_norm=False, has_moe_gen=False)
    ours = LanceLLM(our_cfg)

    # Load full Lance MLX checkpoint and filter to *standard* Qwen keys
    # only (drop qk_norm, moe_gen, norm_moe_gen).  Build two key views:
    #   - `ours_pairs`: keep `language_model.` prefix (our module tree)
    #   - `ref_pairs`:  strip `language_model.` prefix (mlx-vlm's tree)
    # Also drop lm_head for ref since mlx-vlm ties at forward (no lm_head
    # attribute when tie_word_embeddings=True).
    all_w = mx.load("checkpoints/Lance-3B-MLX/model.safetensors")
    DROP_TOKENS = ("_moe_gen", "q_norm", "k_norm")
    keep_lance_keys = [
        k for k in all_w
        if k.startswith("language_model.")
        and not any(tok in k for tok in DROP_TOKENS)
    ]
    print(f"[load] kept {len(keep_lance_keys)} standard-Qwen keys "
          f"(dropped {sum(1 for k in all_w if k.startswith('language_model.')) - len(keep_lance_keys)} qk_norm/moe_gen)")

    ours_pairs = [(k, all_w[k]) for k in keep_lance_keys]
    # Strip "language_model." prefix and drop lm_head for the mlx-vlm reference.
    ref_pairs = []
    for k in keep_lance_keys:
        if k.endswith("lm_head.weight"):
            continue
        assert k.startswith("language_model.")
        ref_pairs.append((k[len("language_model."):], all_w[k]))

    ours_keys = set(dict(tree_flatten(ours.parameters())).keys())
    ref_keys  = set(dict(tree_flatten(ref.parameters())).keys())
    print(f"[load] ours params: {len(ours_keys)}   ref params: {len(ref_keys)}   "
          f"ours_pairs: {len(ours_pairs)}   ref_pairs: {len(ref_pairs)}")

    only_in_ours = ours_keys - {k for k, _ in ours_pairs}
    only_in_ref  = ref_keys  - {k for k, _ in ref_pairs}
    print(f"[load] ours param keys not covered by load: {len(only_in_ours)}")
    print(f"[load] ref  param keys not covered by load: {len(only_in_ref)}")

    ours.load_weights(ours_pairs, strict=False)
    ref.load_weights(ref_pairs,  strict=False)
    mx.eval(ours.parameters()); mx.eval(ref.parameters())
    ours.eval(); ref.eval()
    print("[load] both loaded with same standard-Qwen subset (prefixes aligned)")

    tok = AutoTokenizer.from_pretrained("checkpoints/Lance-3B-MLX", trust_remote_code=True)

    cosines = []
    for prompt in PROMPTS[:3]:                     # 3 prompts is enough for arch identity
        input_ids = _tokenize_chat(tok, prompt)
        L = input_ids.shape[1]
        pos = _pos_ids(L)

        t0 = time.time()
        ref_out = ref(input_ids, position_ids=pos).logits
        mx.eval(ref_out)
        t_ref = time.time() - t0

        t0 = time.time()
        our_out = ours(input_ids, pos)
        mx.eval(our_out)
        t_our = time.time() - t0

        # Compare last-token logits.
        c_last = _cosine(ref_out[0, -1], our_out[0, -1])
        # Also full-sequence logits for sanity.
        c_all  = _cosine(ref_out, our_out)
        max_abs = float(mx.abs(ref_out - our_out).max().item())
        cosines.append(c_last)
        print(f"  prompt({L} tok):  cos(last)={c_last:.6f}  cos(all)={c_all:.6f}  "
              f"max|Δ|={max_abs:.3e}  ours={t_our:.2f}s ref={t_ref:.2f}s")

    print(f"\n[A] mean cos(last)={sum(cosines)/len(cosines):.6f}  "
          f"min={min(cosines):.6f}  max={max(cosines):.6f}")
    print(f"[A] pass criterion ≥ 0.99999: "
          f"{'PASS' if min(cosines) >= 0.99999 else 'FAIL'}")


# ---------------------------------------------------------------------------
# (B) qk_norm contribution measurement
# ---------------------------------------------------------------------------
def part_b_qk_norm_contribution() -> None:
    print("\n========== (B) qk_norm contribution (ON vs OFF) ==========")

    tok = AutoTokenizer.from_pretrained("checkpoints/Lance-3B-MLX", trust_remote_code=True)

    # ON model: full Lance config
    on = LanceLLM(LanceTextConfig())  # has_qk_norm=True default
    load_text_backbone(on, "checkpoints/Lance-3B-MLX/model.safetensors")
    on.eval()

    # OFF model: same architecture minus qk_norm; load same weights with qk_norm filtered.
    off = LanceLLM(LanceTextConfig(has_qk_norm=False))
    all_w = mx.load("checkpoints/Lance-3B-MLX/model.safetensors")
    off_keys = {k for k in all_w
                if k.startswith("language_model.")
                and "q_norm" not in k and "k_norm" not in k}
    off.load_weights([(k, all_w[k]) for k in off_keys], strict=False)
    mx.eval(off.parameters())
    off.eval()

    rows = []
    for prompt in PROMPTS:
        input_ids = _tokenize_chat(tok, prompt)
        L = input_ids.shape[1]
        pos = _pos_ids(L)
        on_out  = on(input_ids, pos)
        off_out = off(input_ids, pos)
        mx.eval(on_out); mx.eval(off_out)

        on_last  = on_out[0, -1]
        off_last = off_out[0, -1]
        c = _cosine(on_last, off_last)
        on_top1  = int(mx.argmax(on_last).item())
        off_top1 = int(mx.argmax(off_last).item())
        same_top1 = on_top1 == off_top1
        rows.append((prompt, c, on_top1, off_top1, same_top1, tok.decode([on_top1]), tok.decode([off_top1])))

    print(f"\n{'prompt':50s} {'cos':>8s}  {'on top1':>10s}  {'off top1':>10s}  same?")
    print("-" * 100)
    for p, c, on_t, off_t, same, on_d, off_d in rows:
        flag = "✓" if same else "✗"
        print(f"{p[:48]:50s} {c:8.4f}  {on_d!r:>10s}  {off_d!r:>10s}  {flag}")

    print(f"\n[B] mean cosine = {sum(r[1] for r in rows)/len(rows):.4f}")
    print(f"[B] top-1 stable across qk_norm toggle: {sum(int(r[4]) for r in rows)}/{len(rows)}")


def main():
    part_a_arch_cosine()
    part_b_qk_norm_contribution()


if __name__ == "__main__":
    main()
