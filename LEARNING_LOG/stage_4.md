# STAGE 4 — Mixture-of-Tokens Routing ★ Lance의 심장

**Status:** ✅ PASSED  (2026-05-22)
**Deliverable:** `lance_mlx/moe_gen.py` + routing wired into `lance_mlx/backbone.py`
**Verification:** Tier 1 self-routing properties ✓ + Tier 2 `mx.where ≡ scatter` (f32 floor) ✓ + Tier 3 cosine vs clean PT reimplementation: **1.000000 at layers 0/12/24** (workorder criterion ≥ 0.999 PASS)
**Settles:** [STAGE 2 → STAGE 4] PT-Lance layer-0 cosine — closed.

## 1. What this stage did

Wired per-token MoE-gen routing through the entire backbone — every
projection (q/k/v/o), every qk_norm, every layernorm, every MLP, and
the final `model.norm` now has a `_moe_gen` sibling that fires for
positions where `gen_mask` is True.  Routing strategy: both branches
computed on the full sequence, merged with `mx.where`.  Attention itself
(sdpa) is NOT split — the mixed sequence attends to itself; only
projections/norms differ per token.

## 2. Block highlights (the "why")

- **4.1 — `moe_gen.py`: SequenceLayout + build_gen_mask + route helpers.**
  Single file dedicated to the routing concept, separate from `backbone.py`
  where the actual wiring lives.  Reason: routing is *one idea*
  (per-token mask + merge), and the module exists primarily as a
  documentation home for the "Mixture-of-Tokens (RockTalk) =
  Mixture-of-Tasks (Reza2kn)" naming clash and the wiring contract.
- **4.2 — `LanceAttention` routing.** New `gen_mask` parameter.  Both
  branches of q/k/v projection + qk_norm computed independently, then
  merged.  qk_norm applied *per branch before merge* — order matters
  (PT does the same).  sdpa runs once on the merged q/k/v.  o_proj
  branch split at the end.
- **4.3 — `LanceDecoderLayer` + `model.norm` paired routing.**
  input_layernorm pair, post_attention_layernorm pair, mlp pair —
  each splits + merges per token.  Final `model.norm` / `norm_moe_gen`
  was the surprise added at STAGE 2; STAGE 4 just exercises the merge.
- **4.4 — Tier 1 self-routing properties.** Four-test sanity (one
  intentionally informational after recognising a wrong hypothesis):
  - (A) `gen_mask=None` ⇒ STAGE 2 fast path (cos=1.000000).
  - (B) GEN-only mask: UND positions cos=0.999968 — minute attention
    K/V drift, expected.
  - (C) all-True mask vs all-False: cos=-0.05 — orthogonal, proving the
    `_moe_gen` weights actively change the forward.
  - (D) GEN-slab vs all-True at GEN positions: cos=0.33 — *expected* to
    differ because attention reads different UND K/V across the two
    contexts.  Initial assertion was wrong; reframed as informational.
- **4.5 — Tier 2 `mx.where ≡ scatter` proof.** Three synthetic shapes
  (32/128/97 sequence × varying spans, dummy random weights) — both
  routing impls produce cos≥1-1e-6, max|Δ| at f32 matmul precision
  floor.  Proves the merge strategy is *mathematically equivalent* to
  PT's scatter — different reduction order, same result.
- **4.6 — Tier 3 PT cosine.** `stage4_pt_cosine.py` builds a *clean PT
  reimplementation* of `Qwen2MoTDecoderLayer.forward_inference` (from
  refs source, NOT imported — to avoid flash_attn/flex_attention deps).
  Loads PT Lance_3B weights into layers 0, 12, 24.  Runs same synthetic
  input through PT + our MLX layer.  Numpy cosine on output.
  **All three: cos = 1.000000.**

## 3. Verification

### Tier 1 — wiring sanity (real Lance weights, full forward)

| Test | Result | Interpretation |
|---|---|---|
| (A) gen_mask=None ⇒ STAGE 2 fast path | cos=1.000000 | regression-free |
| (B) GEN-slab mask: UND positions | cos=0.999968 | attention K/V drift, expected |
| (C) all-True vs all-False | cos=-0.05 | _moe_gen weights actively used |
| (D) slab routing vs all-True at slab | cos=0.33 (info) | attention sees different context |

### Tier 2 — `mx.where` merge ≡ PT scatter (synthetic dummy weights)

| L | D | H | spans | max\|Δ\| | cos |
|---|---|---|---|---|---|
| 32 | 64 | 64 | 1 | 0.000e+00 | 0.999999940 |
| 128 | 256 | 128 | 2 | 1.526e-05 | 0.999999881 |
| 97 | 333 | 211 | 3 | 0.000e+00 | 1.000000119 |

`max|Δ|` non-zero on the middle case is f32 matmul reduction-order
noise (scatter computes per-slab matmul, where computes full matmul —
different intermediate dimensions ⇒ different rounding).  cos≥1−1e-7
confirms direction identity.

### Tier 3 — clean PT vs ours (real Lance_3B weights, 3 layers)

Input: synthetic B=1, L=48 sequence with GEN slab [24, 40), random
hidden state injected directly at each layer (bypasses embed_tokens to
isolate the layer under test).

| Layer | cos | max\|Δ\| |
|---|---|---|
| 0 | **1.000000** | 4.028e-03 |
| 12 | **1.000000** | 7.080e-03 |
| 24 | **1.000000** | 2.002e-02 |

`max|Δ|` grows with layer depth — classic accumulated f32 reduction
noise from non-associative matmul order between PT and MLX.  cos=1.0
across all depths confirms direction identity throughout.

**Workorder criterion (≥ 0.999 at 3 layers): PASS.**

### Independent verification provenance

The PT side in `stage4_pt_cosine.py` is a *fresh, hand-written*
translation of `refs/Lance/modeling/lance/qwen2_navit.py:575-740`
(Qwen2MoTDecoderLayer.forward_inference, "und"/"gen" mode).  It shares
*no code* with our MLX backbone.py — it uses `torch.nn.functional.scaled_dot_product_attention`,
not flex_attention; it has no `flash_attn` dependency; it doesn't
import refs/Lance.  Its faithfulness to PT Lance rests on the
line-by-line translation of PT source, which is observable in the
diff.  If both impls had the same routing bug, cos=1.0 could still hold
— but the routing pattern is documented in RockTalk's README and
matches the PT source structure, so a shared bug would require both
the README and PT source to be misleading the same way.  Strong, not
absolute, evidence of correctness.

**⚠ STAGE 5 retroactive note (2026-05-22):** This stage closed with the
informal expectation that "STAGE 6 first-image will be the final
behavioural check that retires Tier 3's limitation."  STAGE 5 *broke*
that assumption.  Two of the three bugs caught at STAGE 5 (patchify
channel order, AvgDown3D group axis) were the kind that produce
"slightly off" forward output — cos 0.97–0.99 — which would render
as a *visually plausible but degraded* image.  Behavioural verification
would have *passed them*.  Only the layer-wise PT cosine compare caught
them.  ⇒ STAGE 4 owes a re-verification using STAGE 5's stronger
pattern (*direct import of `refs/Lance/modeling/lance/qwen2_navit.py`*,
not our hand reimpl).  Logged at `VERIFICATION_BACKLOG.md` as
"[opened STAGE 4 → settle by STAGE 6] STAGE 4 백본 재검증".

## 4. STAGE 2/3 regression

After STAGE 4 wiring:
- STAGE 2 (A) cosine vs mlx-vlm stock: **1.000000 across 3 prompts**.
- STAGE 3 verifier: **12/12 byte-identical**.

Fast path (`gen_mask=None`) is untouched at the code level — STAGE 2
text-only forward routes through the original code path with the
identical computation graph.

## 5. Carried forward

- `gen_mask` plumbing through LanceLLM → STAGE 6 (T2I pipeline) will
  construct gen_mask from the VAE-latent slab indices and pass through.
- `SequenceLayout` + `build_gen_mask` helpers ready for use.
- PT reference module `PtMotLayer` in `tools/stage4_pt_cosine.py` —
  reusable for STAGE 5/6 deeper-layer parity checks.
- 3-layer cosine harness — STAGE 6 t2i numerical correctness will
  extend this to full-model forward (all 36 layers).

## 6. Open items / not yet addressed

- **MoE routing performance**: computing both branches doubles
  projection FLOPs.  At Lance_3B scale on M3 Ultra this isn't a
  bottleneck (pre-fill stays sub-second), but STAGE 8/9 video may
  surface it.  Log as candidate IMPROVEMENT.
- The PT reference module is *currently* a verification artifact, not
  production code.  If we want even stronger guarantees we could run
  full-model PT cosine on a real prompt; that's a higher-cost run
  (instantiate all 36 layers in PT, ~3 GB of CPU memory + slow matmul).
  Logged as optional follow-up.

## 7. Code-reviewer pass (workorder §5.7)

- **BLOCKING resolved:**
  - PT shim docstring now explicit that it tests *fp32 algorithmic shape
    parity*, NOT bf16 mixed-precision parity (PT Lance's q/k upcast path
    is intentionally not exercised — logged for STAGE 9 bf16 mode).
  - `_scatter_route` adds two tripwires (coverage `written[]` flag +
    `max(abs)>0` post-check) so a future MLX version that silently
    breaks in-place index assignment is caught (verified: MLX 0.31.2
    *does* support `out[i] = v` for both scalar and array rows).
- **SUGGESTED A applied:**
  - `_make_mrope_cos_sin` now takes `rope_base` + `mrope_section` from
    `LanceTextConfig` instead of hardcoded — prevents silent
    desynchronisation if config ever changes.
  - Tier 3 also reports `rel_L2 = ||Δ||/||pt||` (~2.6e-6 at layer 24)
    alongside cos and max|Δ| — clean view of energy drift.
  - `build_gen_mask` early-returns zero mask when `gen_spans` is empty.
  - `LanceDecoderLayer`/`LanceQwen2Model` use the same `self._has_moe_gen`
    flag as `LanceAttention` instead of `getattr(..., None)`.  Single
    pattern across the file.
  - `LanceQwen2Model` has a `TODO` comment marking where the explicit
    prefix-LM attention mask needs to land for STAGE 6/7 T2I/TI2I.
  - Position IDs explicitly typed `int32` on both PT and MLX side.
  - Tier 1 (B) assertion simplified to cosine-only (`cos >= 0.999`)
    after observing max|Δ| is correctly O(1) in raw hidden state space.
- **SUGGESTED B logged to IMPROVEMENTS.md:**
  - Defensive `gen_mask.any().item()` short-circuit (sync cost
    tradeoff).
  - Tier 1 (D) → pre-sdpa Q assertion at layer-internal hook.
  - `build_gen_mask` multi-span vectorisation for spans > 10.
  - bf16 mixed-precision parity check for STAGE 9.
- **Regression check:** Tier 1 (A)(B)(C)(D) all hold; Tier 2 3/3 pass;
  Tier 3 layer 0/12/24 cos = 1.000000 (rel_L2 ≈ 2.5e-6, stable across
  depth — energy drift well-bounded).  STAGE 2 cos still 1.000000.
  STAGE 3 still 12/12.
