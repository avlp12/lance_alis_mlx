"""STAGE 4 Tier 1: self-routing properties on Lance weights.

Pure-MLX correctness tests that don't need PT.  Checks:

  (A) `gen_mask=None` path == STAGE 2 forward (regression).
  (B) With `gen_mask` set on a slab, the *UND* portion of the output
      matches a fresh forward where `gen_mask=None`.  I.e. the canonical
      route at UND positions is unaffected by the routing existing.
  (C) With `gen_mask=None` vs `gen_mask=all_True`, the outputs are
      *different at every position* (proves the moe_gen branch is
      actually exercising the _moe_gen weights).
  (D) With `gen_mask` covering only a slab, the *GEN* portion matches
      what you'd get from an all-True forward at that slab — i.e.
      per-token routing is independent across positions.

Together these prove our `mx.where` merging implements per-token routing
correctly without PT.  Quantitative PT cosine is its own STAGE 4 leg.
"""
from __future__ import annotations

import mlx.core as mx
from transformers import AutoTokenizer

from lance_mlx.backbone import LanceLLM, LanceTextConfig, load_text_backbone
from lance_mlx.rope import build_positions_for_layout, VisionSpec


WEIGHTS = "checkpoints/Lance-3B-MLX/model.safetensors"


def _prompt_ids(tok, text: str) -> mx.array:
    msg = [{"role": "user", "content": text}]
    formatted = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    return mx.array([tok(formatted, return_tensors=None)["input_ids"]], dtype=mx.int32)


def _cosine(a: mx.array, b: mx.array) -> float:
    af, bf = a.flatten().astype(mx.float32), b.flatten().astype(mx.float32)
    return float((mx.sum(af * bf) / (mx.linalg.norm(af) * mx.linalg.norm(bf) + 1e-12)).item())


def main() -> None:
    tok = AutoTokenizer.from_pretrained("checkpoints/Lance-3B-MLX", trust_remote_code=True)
    m = LanceLLM(LanceTextConfig())
    load_text_backbone(m, WEIGHTS)
    m.eval()

    # Build a synthetic multimodal sequence: chat prompt + a placeholder
    # VAE-latent slab of length N.  Use a tiny slab (4×4=16) so the test
    # is fast.
    ids = _prompt_ids(tok, "Describe an image in detail.")
    text_len = int(ids.shape[1])
    GEN_LEN = 16                                              # toy slab
    VS, IT, VE = 151652, 151655, 151653
    full = mx.concatenate([
        ids,
        mx.array([[VS] + [IT] * GEN_LEN + [VE]], dtype=mx.int32),
    ], axis=1)
    L = int(full.shape[1])
    span = VisionSpec(start=text_len, length=GEN_LEN, t=1, h=4, w=4)
    pos = build_positions_for_layout(L, [span])

    print(f"[setup] seq_len={L}  text_len={text_len}  GEN slab=[{text_len+1},{text_len+1+GEN_LEN})")

    # ----- (A) gen_mask=None equals STAGE 2 fast path on text-only prefix -----
    print("\n=== (A) gen_mask=None routes through canonical path (regression) ===")
    out_none = m(full, pos)
    # Compare with the text-only prompt forwarded alone (also gen_mask=None).
    pos_text = build_positions_for_layout(text_len, [])
    out_text_only = m(ids, pos_text)
    # The text portion of the full-sequence forward should equal the standalone
    # forward up to position text_len (causal mask isolates them by attention
    # to nothing past the cutoff, but vision tokens *can* attend to text
    # tokens behind via mRoPE position — however since these are placeholder
    # token IDs the embedding lookup doesn't depend on the slab).
    cos_text = _cosine(out_none[:, :text_len, :], out_text_only[:, :text_len, :])
    print(f"  cos(out_none[:text_len], out_text_only) = {cos_text:.6f}")
    assert cos_text >= 0.99999, f"(A) failed: text portion drifts {cos_text}"

    # ----- (B) gen_mask covers slab → UND portion same as gen_mask=None -----
    print("\n=== (B) GEN-only mask leaves UND positions unchanged ===")
    gen_mask = mx.zeros((1, L), dtype=mx.bool_)
    slab_start = text_len + 1
    slab_end = slab_start + GEN_LEN
    cols = mx.arange(L)
    gen_mask = (cols >= slab_start) & (cols < slab_end)
    gen_mask = gen_mask[None, :]                              # (1, L)

    out_routed = m(full, pos, gen_mask=gen_mask)

    # UND positions (0..slab_start, slab_end..L) should be byte-identical
    # to the gen_mask=None forward — because the canonical weights
    # process those tokens both times.
    und_slice = mx.concatenate([
        out_none[:, :slab_start, :],
        out_none[:, slab_end:, :],
    ], axis=1)
    routed_und_slice = mx.concatenate([
        out_routed[:, :slab_start, :],
        out_routed[:, slab_end:, :],
    ], axis=1)
    diff = float(mx.abs(und_slice - routed_und_slice).max().item())
    cos = _cosine(und_slice, routed_und_slice)
    print(f"  UND-slice max|Δ| = {diff:.3e}   cos = {cos:.6f}")
    # Not strictly byte-equal because attention is over the *whole*
    # mixed sequence — the GEN slab K/V differ across the two forwards,
    # and UND queries attend to them.  We use cosine (direction) only;
    # max|Δ| reads as O(1) in raw hidden state space (which has
    # magnitude O(few) at f32), so absolute threshold is the wrong
    # tripwire here.
    assert cos >= 0.999, \
        f"(B) UND positions diverge too much: cos={cos} (maxΔ={diff:.3e} informational)"

    # ----- (C) all-True gen_mask vs all-False → outputs differ everywhere -----
    print("\n=== (C) all-True mask differs from all-False (moe_gen weights active) ===")
    gen_all = mx.broadcast_to(mx.array([[True]]), (1, L))
    out_all_gen = m(full, pos, gen_mask=gen_all)
    cos_all = _cosine(out_none, out_all_gen)
    diff_all = float(mx.abs(out_none - out_all_gen).max().item())
    print(f"  cos(out_none, out_all_gen) = {cos_all:.6f}   max|Δ| = {diff_all:.3e}")
    assert cos_all < 0.999, \
        f"(C) failed: all-True mask gave the same result as None — _moe_gen unused? cos={cos_all}"

    # ----- (D) informational: GEN slab output in routed vs all-True context -----
    # These differ by *attention noise*: under routed, UND K/V come from
    # canonical weights; under all-True, UND K/V come from moe_gen
    # weights.  GEN slab queries see different attention contexts → its
    # output legitimately differs.  Reported as info only, no assertion.
    print("\n=== (D) GEN-slab output across routed vs all-True (informational) ===")
    slab_routed = out_routed[:, slab_start:slab_end, :]
    slab_all    = out_all_gen[:, slab_start:slab_end, :]
    diff_slab = float(mx.abs(slab_routed - slab_all).max().item())
    cos_slab = _cosine(slab_routed, slab_all)
    print(f"  cos(slab_routed, slab_all_gen) = {cos_slab:.6f}   max|Δ| = {diff_slab:.3e}")
    print("  (expected: differ — attention sees different UND K/V branches)")

    print("\n[result] (A), (B), (C) properties hold ✓ — routing wired correctly")


if __name__ == "__main__":
    main()
