# STAGE 1 — Weight Conversion (PyTorch → MLX)

**Status:** ✅ PASSED  (2026-05-21)
**Deliverable:** `tools/convert_weights.py`
**Output:** `out/lance_3b_mlx/model.safetensors` (24.74 GB, 1021 tensors, 6.185B params, float32)

## 1. What this stage did

Wrote a single-file converter that ingests `bytedance-research/Lance/Lance_3B/model.safetensors`
(PT, bf16) and emits an MLX-compatible safetensors file under arbitrary dtype with
optional key rename / prefix drop, plus self-check and cross-check against a
reference. Ran it against the real Lance_3B checkpoint and achieved bit-exact
parity with `RockTalk/Lance-3B-MLX/model.safetensors`.

## 2. Block highlights (the "why")

- **1.1 — inspect-only first.** Refused to write conversion rules until we'd
  seen the PT key list. Confirmed top-level buckets (`language_model`, `time_embedder`,
  `llm2vae`, `vae2llm`, `latent_pos_embed`) and per-bucket counts before deciding
  the rename map.
- **1.2 — bf16 bit-preservation path.** PT bf16 → NumPy bypasses bf16 (NumPy has
  no bf16 dtype). We view-as-int16, pass bits through, then `mx.array.view(bf16)`.
  Same bit pattern, no precision loss, no upcast.
- **1.3 — two-tier verification.** Self-check (round-trip load: key count + param
  count preserved) is local. Cross-check (`--verify-against`) diffs against an
  oracle (RockTalk MLX). Tier 1 catches IO bugs, tier 2 catches naming bugs.
- **1.4 — conv-layout helper.** `conv_pt_to_mlx(w)` permutes axis 1 to the end
  for rank-4 and rank-5. **Unused in STAGE 1** (Lance_3B is pure Linear/RMSNorm)
  but parked here because STAGE 2 (ViT `patch_embed` 2D conv) and STAGE 5
  (Wan 2.2 VAE 3D conv) need exactly the same rule.
- **1.5 — dtype + drop policy.** Default `--dtype float32` matches RockTalk's
  choice. `--drop-prefix` reserved for `connector.*` when we get there
  (Lance_3B PT release doesn't ship one; the T2I path doesn't use it).
- **1.6 — rename map.** Only one rename was needed for Lance_3B: PT's
  `TimestepEmbedder` stores its two Linear layers under `nn.Sequential`
  indices (`.mlp.0.*`, `.mlp.2.*`); RockTalk (and our MLX class at STAGE 6)
  use the cleaner `fc1/fc2` naming, mirroring the `MLPconnector` in the
  same PT file. Four keys renamed, byte-for-byte parity achieved.

## 3. What we learned vs the assumed plan

| Assumption (from workorder §6) | Reality |
|---|---|
| Conv layout transform is core to STAGE 1 | Lance_3B body has no conv. The helper is real but parked for STAGE 2/5. |
| `lm_head.weight` is tied → not in safetensors | Both PT *and* MLX store it physically; tying is a runtime detail of the model class. We copy it. |
| RockTalk source code is available as a "정답지" | github.com/RockTalk/Lance-MLX is non-public. Reference is the HF MLX weights + the README prose. PT is the actual source-of-truth. |
| `connector.*` (MLPconnector ViT→LLM) ships with Lance_3B | Not in `Lance_3B/model.safetensors`. Only built by the Lance class when `vit_type=="qwen2_5_vl"`; the T2I-only PT release omits it. ViT bundle has `vision_tower.merger.mlp.*` instead. |

## 4. Verification

### Conversion + structural diff (vs `RockTalk/Lance-3B-MLX/model.safetensors`)

| metric | value |
|---|---|
| tensors converted | 1021 / 1021 |
| total params | 6.185B |
| keys dropped | 0 |
| keys renamed | 4 (`time_embedder.mlp.{0,2}.{w,b}` → `time_embedder.fc{1,2}.{w,b}`) |
| shared keys with reference | **1021 / 1021** |
| keys only in ours | 0 |
| keys only in reference | 0 |
| shape mismatches on intersection | **0** |
| file size (ours) | 24,740,958,849 B |
| file size (reference) | 24,740,958,849 B |
| file size delta | **0 bytes** |

### Element-wise spot check (random 12 tensors, weights of varied shape)

| tensor | shape | max\|Δ\| | cosine |
|---|---|---|---|
| `language_model.model.layers.1.mlp_moe_gen.up_proj.weight` | (11008, 2048) | 0.000e+00 | 1.000000 |
| `language_model.model.layers.3.self_attn.q_proj.weight`    | (2048, 2048)  | 0.000e+00 | 1.000000 |
| `language_model.model.layers.23.self_attn.o_proj_moe_gen.weight` | (2048, 2048) | 0.000e+00 | 1.000000 |
| `language_model.model.layers.11.self_attn.v_proj.weight`   | (256, 2048)   | 0.000e+00 | 1.000000 |
| `language_model.model.layers.20.input_layernorm.weight`    | (2048,)       | 0.000e+00 | 1.000000 |
| `language_model.model.layers.1.input_layernorm.weight`     | (2048,)       | 0.000e+00 | 1.000000 |
| `language_model.model.layers.2.self_attn.k_proj_moe_gen.bias` | (256,)      | 0.000e+00 | 1.000000 |
| `language_model.model.layers.30.self_attn.v_proj.bias`     | (256,)        | 0.000e+00 | 1.000000 |
| `language_model.model.layers.24.self_attn.q_norm.weight`   | (128,)        | 0.000e+00 | 1.000000 |
| `language_model.model.layers.25.self_attn.q_norm.weight`   | (128,)        | 0.000e+00 | 1.000000 |
| `language_model.model.layers.7.self_attn.q_norm.weight`    | (128,)        | 0.000e+00 | 1.000000 |
| `language_model.model.layers.34.self_attn.k_norm_moe_gen.weight` | (128,)  | 0.000e+00 | 1.000000 |

Interpretation: bf16 → f32 upcast is exact (bf16 mantissa fits in f32 mantissa
exactly), and RockTalk went through the same path. The two files are
literally identical.

## 5. Model facts gleaned for downstream stages

(from `Lance_3B/llm_config.json` + observed tensor shapes)

- Backbone: Qwen2.5-VL, `num_hidden_layers=36`, `hidden_size=2048`,
  `intermediate_size=11008`, `num_attention_heads=16`,
  `num_key_value_heads=2` (GQA 8:1, head_dim=128), `vocab_size=151936`,
  `rms_norm_eps=1e-6`, `rope_theta=1e6`.
- mRoPE sections: `[16, 24, 24]` → 16 text + 24 H + 24 W rotary dims per head.
- `tie_word_embeddings=true` but the checkpoint stores `lm_head.weight` anyway.
- Special tokens: `bos=151643`, `eos=151645`, `vision_start=151652`,
  `vision_end=151653`, `vision=151654`, `image=151655`, `video=151656`.
- ViT (separate file): `depth=32`, `hidden=1280`, `out_hidden=2048`,
  `patch=14`, `spatial_merge=2`, `temporal_patch=2`,
  `fullatt_block_indexes=[7,15,23,31]`.
- VAE/latent adapters present in this LLM checkpoint:
  `vae2llm` (48→2048), `llm2vae` (2048→48), `time_embedder` (256→2048→2048),
  `latent_pos_embed` (1 tensor — the lookup table).

## 6. Carried forward to STAGE 2

- The bit-exact MLX checkpoint we produced — usable directly as the LLM
  weights for the backbone implementation.
- `conv_pt_to_mlx` helper, ready for ViT patch_embed.
- The 36-layer / 16-head / GQA-8 / head_dim-128 / mrope [16,24,24] config
  that the MLX backbone class has to match.
- The `_moe_gen` key inventory (PackedAttentionMoT + Qwen2MLP_moe_gen +
  paired RMSNorms in every layer) — this is what STAGE 4 will route per-token.

## 7. Code-reviewer pass (workorder §5.7)

Run: code-reviewer agent (Opus, xhigh-equivalent) on `tools/convert_weights.py`.

- **BLOCKING:** none — bit-exact parity already achieved.
- **SUGGESTED applied now:**
  - `.detach()` symmetry on bf16 view-as-int16 path (footgun for future
    reuse on autograd-tracked tensors).
  - One-line docstring on `conv_pt_to_mlx` documenting the wiring contract
    (must be called *before* `_torch_to_mlx`, returns torch).
  - Warn when `lm_head.weight` is absent (forces a tie at load time).
  - Replaced `kept_src_keys: list[str]` with running int counter.
  - Guarded `__doc__.splitlines()[0]` against `-OO`.
- **SUGGESTED deferred to STAGE 2 start (intentional):**
  - Refactor `_RENAME_PREFIXES` tuple into a `Rule` dataclass with
    `match(key) -> Optional[list[str]]` semantics, supporting fan-out
    (single PT key → multiple MLX keys, needed for ViT `attn.qkv` →
    `q_proj`/`k_proj`/`v_proj`) and regex on layer index. Doing it before
    ViT/VAE rename rules accumulate is the cheap moment.
  - Streaming write / per-tensor `del` for VAE-sized checkpoints
    (irrelevant on this 512 GB box, real on smaller boxes).
  - Int64-narrowing warning in `_DTYPE_MAP` (not triggered by Lance_3B).
- **Re-run after fixes:** identical result — 1021/1021 keys, 0 mismatches,
  PARITY OK.

## 7b. Workorder v2.1 absorption (no impact on STAGE 1)

Workorder updated to v2.1 mid-stage. New items relevant to downstream stages:

- **Topography clarified**: RockTalk = MLX port (정답지). Reza2kn = quantization
  (separate work on the original PT). They are *not* the same project; search
  hits for "Lance MLX" may surface either.
- **STAGE 4 qk_norm gotcha**: PT modified-Qwen2.5-VL has `qk_norm` weights;
  Reza2kn dropped them to fit mlx-lm's stock qwen2 class. We're a full port,
  so we *keep* them. STAGE 1 already preserves these keys verbatim — sample:
  `language_model.model.layers.24.self_attn.q_norm.weight (128,)`,
  `*_moe_gen` variants too. No action needed.
- **Naming**: RockTalk "Mixture-of-Tokens" = Reza2kn "Mixture-of-Tasks".
- **STAGE 10 (optional)**: quantization track parked after STAGE 9.

## 8. Open items / not yet addressed

- ViT conversion (`vit.safetensors` → `vision_tower.*` mlx-vlm naming) is its
  own script, queued for STAGE 2.
- VAE conversion is its own script, queued for STAGE 5.
- The `connector.*` rename rule (`connector.fc{1,2}.*` →
  `vision_tower.merger.mlp.{0,2}.*` or whatever mlx-vlm expects) — TBD when
  STAGE 2 lands.
