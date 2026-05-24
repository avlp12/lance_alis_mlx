# STAGE 3 — 3D mRoPE Position Generation

**Status:** ✅ PASSED  (2026-05-21)
**Deliverable:** `lance_mlx/rope.py` (text_positions, build_positions_for_layout, lance_pos_shift, shift_positions)
**Verification:** 12/12 byte-identical tests (text-only / single image /
multi-image hand-verified / video T>1 / Lance pos_shift slab masking)

## 1. What this stage did

Moved position-ID generation out of `backbone.py`'s callers and into its
own module.  The function `build_positions_for_layout(seq_len,
vision_spans)` takes a flat sequence length plus a list of `VisionSpec`s
describing where each image/video block sits and produces the
`(3, 1, L)` mRoPE position triple Qwen2.5-VL expects.  Text-only collapses
to `arange × 3 broadcast` (one-liner).  Image/video patches get their
`(t, h, w)` grid coordinates, with the position-counter cursor
advancing by `max(t, h, w)` per the Qwen2.5-VL extent rule (the
non-obvious "share the right corner of the previous grid" semantic).
A separate `shift_positions(positions, shift, col_start, col_end)`
helper plus `lance_pos_shift(max_latent_size, max_num_latent_frames)`
constant let later stages disambiguate VAE-latent slabs from ViT slabs.

## 2. Block highlights (the "why")

- **3.1 — VisionSpec dataclass + text_positions.** Per-image bundle of
  `(start, length, t, h, w)` lets us assemble interleaved sequences
  without per-call ad-hoc arithmetic.
- **3.2 — `build_positions_for_layout`.** Distilled mlx-vlm's
  `get_rope_index` (~150 lines of `.tolist()` and Python branches) into
  ~30 lines of MLX-friendly broadcasts.  Single batch dim assumed
  (Lance is packed-sequence single-batch).  Cursor advance rule
  `max(t,h,w)` documented inline — most surprising bit of the algorithm.
- **3.3 — verifier vs mlx-vlm.** Text and single-image cases:
  byte-identical (9/9).
- **3.3 — finding: mlx-vlm multi-image bug.** Their `get_rope_index`
  computes `vision_start_indices = mx.sum(...)` instead of an array of
  positions, so multi-image inputs get most-of-img2's placeholders mapped
  to text positions.  Filed in `IMPROVEMENTS.md`, upstream candidate.
  Our impl is correct (matches transformers).
- **3.4 — multi-image vs hand ground truth.** Replaced mlx-vlm oracle
  with a paper-and-pencil expected array for the two-image case.
  Byte-identical (10/10).
- **3.5 — video (T>1) + Lance pos_shift.** T_lat=3, h_lat=w_lat=16
  (Lance T2V 9-frames-256px shape) and `lance_pos_shift(max_latent_size=32,
  max_num_latent_frames=7) → 8192` slab-only application.  Both ✓ (12/12).
- **regression** — STAGE 2 cosine test re-run: still 1.000000 across all
  three prompts.  STAGE 3 additions don't disturb the text-only forward.

## 3. What we learned / RockTalk gap

- mlx-vlm is *not* a trustworthy oracle for multi-image position
  generation.  We have to use transformers' `Qwen2_5_VLModel.get_rope_index`
  or hand-built ground truth.  PT Lance correctly uses the transformers
  version (`refs/Lance/modeling/lance/lance.py:241`), so our `build_positions_for_layout`
  is on the right side.
- PT Lance applies an *additional* `shift_position_ids` (from
  `data/common.py`) that has more dimensions (attn_mode, pro_type,
  i_sample_modality) than our simple constant slab shift.  The simple
  shift handles the T2I "VAE-latent slab needs to be distinct from
  any ViT positions" case, which is everything STAGE 5/6 needs.  Full
  exact-replay of `shift_position_ids` is deferred to STAGE 6/7 — logged
  in `VERIFICATION_BACKLOG.md`.

## 4. Verification

```
=== (1) text-only ===
  ✓ L=1, L=7, L=31, L=128  (all byte-identical to mlx-vlm)

=== (2) text + one image + text ===
  ✓ 1×4×4, 1×8×8, 1×16×16, 1×13×21 (non-square), edge-no-pad

=== (3) text + img + text + img + text (hand ground truth) ===
  ✓ two images interleaved  (mlx-vlm bugged, hand-verified)

=== (4) text + video(T>1) + text ===
  ✓ T_lat=3 × 16 × 16  (Lance T2V 9-frame 256px shape)

=== (5) Lance pos_shift on VAE-latent span ===
  ✓ pos_shift(8192) applied only to slab, 0 elsewhere
```

**STAGE 2 regression check:** mean `cos(last) = 1.000000`, min/max
unchanged.

## 5. Carried forward

- `build_positions_for_layout` — direct input for STAGE 4/5/6 sequence
  assembly.
- `lance_pos_shift` + `shift_positions` — STAGE 6 (T2I) uses pos_shift
  on the VAE-latent slab; STAGE 7 (TI2I) uses it on the noise-target
  slab to keep cond-VAE / noise-VAE positions distinct.
- VisionSpec datatype — shared protocol between rope.py callers and any
  sequence-builder in later stages.

## 6. Open items / not yet addressed

- Full PT `shift_position_ids` replay (attn_modes / pro_type branches)
  → `VERIFICATION_BACKLOG.md` (settle at STAGE 6).
- mlx-vlm `get_rope_index` multi-image bug → `IMPROVEMENTS.md` (upstream
  issue + minimal repro).
- KV-cache-aware position generation (incremental positions for AR
  decoding) — needed at STAGE 7 X→T.  Today's helper is pre-fill-only.

## 7. Code-reviewer pass (workorder §5.7)

- **BLOCKING:** none.
- **A applied:** (1) `_image_position_block` docstring rewritten — the old
  text described a +1 offset that the actual caller doesn't apply (the
  caller pre-increments past `<vision_start>`).  New docstring matches
  the implementation: "i-th placeholder sits at `base + (t_i, h_i,
  w_i)`".  (2) `shift_positions` early-returns when `shift == 0`.
- **B → IMPROVEMENTS.md:** (1) `shift_positions` more idiomatic via
  `mx.where(mask, positions+shift, positions)` — apply when STAGE 6/7
  calls it in tight loops, with baseline-vs-after timing.  (2) AR-decode
  per-step position emission (don't call `build_positions_for_layout`
  per token at STAGE 7 X→T — track cursor and emit a (3,1,1) directly).
- **B → VERIFICATION_BACKLOG.md:** Property-based randomized mRoPE
  verifier (N=1000 random multi-image / asymmetric layouts vs
  transformers).  Test-strength upgrade, not a porting gate.
- **Regression check:** 12/12 STAGE 3 tests still pass, STAGE 2 cosine
  still 1.000000.  No behavioural drift.
