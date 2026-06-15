# lance_alis_mlx

Hand-port of [ByteDance Lance](https://github.com/bytedance/Lance) multimodal
model from PyTorch to Apple MLX.  Every block is ported and cross-validated
against the original PT via byte-diff (cosine ≥ 0.999) before the next block
starts.

## Status

| Stage | Component | Gate |
|---|---|---|
| 1 | PT → MLX weight conversion | ✓ bit-exact vs RockTalk reference (same SHA256) |
| 2 | Qwen2.5-VL text backbone (36-layer Lance LLM) | ✓ cos = 1.000000 vs mlx-vlm stock |
| 3 | 3D mRoPE positions | ✓ byte-identical 12/12 |
| 4 | MoE-gen routing (MoT) | ✓ cos = 1.000000 vs clean PT re-impl |
| 5 | Wan 2.2 VAE image path (T = 1) | ✓ ~40 dB PSNR round-trip vs PT |
| 6 | Flow matching + CFG denoising loop | ✓ end-to-end cos ≥ 0.999 vs PT 30-step |
| 7 | ViT + X→T + TI2I (3 pipelines) | ✓ forward parity; ⚠ preprocessing bug found post-release — re-verified non-blind in Stage 11 (see correction note) |
| 8 | 3D Causal Video VAE (T > 1) | ✓ 4 gates cos = 1.000000 (encode + decode) |
| 9 | Video DiT + t2v (text-to-video) | ✓ 30-step per-step cos ≥ 0.999, video pixel cos = 0.999338 |
| 11a | x2t (image + video) — non-blind re-verification | ✓ patches + positions PT-recomputed byte-identical; K=8 top-1 8/8, min logit cos 0.999124 (image) / 0.999437 (video) |
| 11b | image_edit (TI2I) — velocity non-blind re-verification | ✓ ViT-cond PT-recomputed; 3-comp velocity min cos 0.999640; raster control collapses v_full |
| 11c | video_edit (TIV2V) — non-blind re-verification | ✓ ViT + 3-slab positions byte-identical; velocity min cos 0.99991; 5-step accumulation cos 0.999999; cond scale cos 1.0 |

*Numbering note: stages 1–9 are the port. **Stage 10 (quantization) is planned, not yet
started** — hence 9 → 11. **Stage 11** is the post-release non-blind re-verification; its
three parts (11a / 11b / 11c) cover x2t, image_edit, and video_edit.*

**Stages 1–9 (the port) + Stage 11 (re-verification) complete; all six Lance tasks ported
and verified** (t2i / t2v / x2t image+video / image_edit / video_edit).  Every core path of
Lance — image / video generation, editing, understanding — is ported to MLX and byte-diff
verified against the original PyTorch.

Stage 7 numbers *(as originally measured — see the Post-release correction below)*:

- ViT (Qwen2.5-VL vision tower): cos = 1.000000
- X→T first-token logits: cos = 0.999923, top-1 token match
- TI2I 3-forward (v_full / v_t_uncond / v_tv_uncond): all cos ≥ 0.999
- TI2I 30-step PT-vs-MLX final latent: cos = 0.997340
- Real-photo edits (orange cat → black panther, + bow tie) match RockTalk reference outputs

> These confirm **forward parity** — our MLX forward matches PT's forward — but the
> input the two sides compared was *our own* `preprocess_image` output, not a patch
> tensor PT recomputed from the raw image. A patch-token-ordering bug (see below) was
> therefore shared by both sides and agreed at cos = 1.0. Read the numbers as
> "MLX matches PT given the same (then-wrong) preprocessing," not as "the preprocessing
> matches PT's real pipeline." The Stage 11 re-verification (table row 11) closes that gap.

Stage 8 numbers (3D Causal Video VAE, bottom-up then top-level):

- CausalConv3d `cache_x`, Resample 3D, ResidualBlock, Down/Up_ResidualBlock,
  Encoder3d / Decoder3d — each block byte-equal incl. feat_cache *state*
- WanVAE_ top-level T = 5 round-trip vs PT: mu / log_var / xhat / xhat* all cos = 1.000000

Stage 9 numbers (Video DiT + t2v, production text_template=True):

- single-step PT byte-diff: v_full cos = 0.999916, v_unc cos = 0.999848, v_blend cos = 0.999452
- 30-step per-step latent cos ≥ 0.999 (min 0.999437), CFG on→off transition stable (cos = 0.999602)
- video pixel cos = 0.999338 (PT VAE decode vs MLX) — caught a silent VAE-scale bug
  that the latent-level gate (0.999437) did *not*; see Lesson 23
- t2v inference is **pure MLX** — the sequence builder was migrated from a PT
  wrapper to a manual token concat, verified byte-identical against PT in 5
  gates (input_ids / modality / split_lens / attn_modes / vae_token_indexes)

## Post-release correction (Stage 11, 2026-06-08)

After release we found, fixed, and re-verified two bugs in our own "verified" code.
We record them here because catching them — twice, by two independent methods, after
release — is the verification culture working, not failing.

**The weights are unaffected.** Both bugs are in our MLX *pipeline / preprocessing* and
in a verification *harness*, not in the safetensors. The published weight stays
byte-identical to `RockTalk/Lance-3B-MLX` (same SHA256). `t2i` / `t2v` never touch the
ViT and are unaffected. Affected paths: `x2t` (image VQA), `image_edit` (ViT-conditioning),
`x2t_video`.

1. **ViT patch-token order** (raster → 2×2 merge-grouped). `preprocess_image` emitted
   patch tokens in plain raster (T,H,W) order; PT's `patchify_video_with_merge` and the
   mlx-vlm ViT both expect 2×2 spatial-merge-grouped order. Channels were identical — a
   pure token-order bug. Against PT's real pipeline, the raster order scores cos ≈ 0.29
   (image) / 0.36 (video); the corrected merge-grouped order scores cos = 1.000000.

2. **x2t_video temporal mRoPE multiplier** (×`tokens_per_second` = 2). PT `get_rope_index`
   scales the video time axis by `tokens_per_second`; our position builder used unit steps.
   Video-only (T > 1); image (T = 1) is immune. Our `t2v` already had the multiplier — only
   the x2t path missed it.

**Why Stage 7's "PT direct import" gates didn't catch them.** They fed PT *our* intermediates
instead of letting PT recompute from the raw input: the x2t / TI2I gates passed our
`preprocess_image` patches to the PT ViT (TI2I copied our ViT output wholesale), and our
positions to PT. Both sides inherited the same misunderstanding and agreed at cos ≈ 1.0 —
the import was blind. This refines the doctrine below: a direct PT import is only independent
when **PT recomputes from the raw input with its own code**.

**A wrong reference hid bug 2 at Stage 3.** Stage 3 byte-checked our positions against
mlx-vlm's `get_rope_index`, but mlx-vlm also drops the video multiplier, so our unit-step
matched it and looked correct. PT-Lance's `get_rope_index` is the real truth.

**Re-verification (non-blind).** The Stage 11 gate (`tools/stage11_x2t_verify.py`,
`out/stage11_x2t_verify.json`) has PT recompute patches and positions from the raw frames and
byte-asserts them against ours before any forward, over a production prompt and K = 8 tokens,
with the old raster order kept as a discriminative control:

| | image | video |
|---|---|---|
| patches / positions PT-recomputed byte-identical | ✓ | ✓ |
| ViT cos vs PT real patchify | 1.000000 | 1.000000 |
| K = 8 top-1 vs PT | 8 / 8 | 8 / 8 |
| K = 8 logit cos (min) | 0.999124 | 0.999437 |
| raster control cos (min) | 0.553 (collapses) | 0.968 |

The temporal-mRoPE bug flipped a real output token on the x2t_video probe ('Nothing' → 'In',
matching PT's true output after the fix), confirming it was a real bug, not cosmetic.

**Honest scope.** The gate injects identical pre-resized frames to both sides; PT's real
`vit_transform` bucket resize is *not* exercised (claim scope = "given identical resized /
normalized frames and grid"). `image_edit`'s 3-component CFG velocity **is now re-verified
non-blind** (Stage 11): PT recomputes the ViT-conditioning from the raw image and the velocity
matches at min cos 0.999640, with the old raster order kept as a discriminative control that
collapses `v_full` to 0.996. `video_edit` is now **implemented and re-verified non-blind** too
— it is `image_edit`'s method on the video path (PT `tiv2v_sample`). PT recomputes patches, the
video ViT, and the 3-slab mRoPE positions (`get_rope_index` + `shift_position_ids`,
byte-identical); the 3-component velocity matches at min cos 0.99991 (raster control collapses
`v_full` to 0.994), and a 5-step trajectory holds at latent cos 0.999999 (per-step error does
not compound). The cond VAE-encode scale is separately de-blind against a PT Wan VAE at
cos 1.000000. Full pixel decode is not measured directly — it is implied by the per-step +
accumulation cos and the byte-clean Stage 8 VAE decode. **This completes all six Lance tasks**
(t2i / t2v / x2t image+video / image_edit / video_edit).

## Verification doctrine

Every block:

1. **Original PT direct import** (not a clean re-implementation).  `refs/Lance/`
   is imported under a flash_attn / flex_attention shim so we run the upstream
   code on CPU.  A direct import only stays independent when **PT recomputes from
   the raw input with its own code**; feeding PT *our* intermediate (preprocessed
   patches, ViT output, positions) re-shares the very misunderstanding the import
   is meant to rule out, and both sides then agree at cos ≈ 1.0 while wrong.  Stage 7's
   x2t / TI2I gates did exactly that and stayed blind to a patch-ordering bug until
   Stage 11 — see *Post-release correction* above.  The corrected gates have PT derive
   patches and positions from the raw frames and byte-assert them against ours.
2. **Same PRNG** on both sides (NumPy `default_rng`) so initial noise is
   bit-identical.
3. **Byte-diff** at every layer / step / forward variant.  `cos ≥ 0.999` is the
   gate; `max|Δ|` is reported alongside (f32 dot-product noise floor ≈ 1e-6).
4. **Behavioural cross-check** on production-realistic inputs (real photos,
   real edit instructions).  Numerical pass without behavioural pass = not
   done.

## Pure-MLX inference, PyTorch only at verification time

The `lance_mlx/` package never imports `torch` or `refs/Lance` at runtime.
Inference (`t2i`, `t2v`, `x2t` image+video, `image_edit`, `video_edit`) runs on
MLX + the HF Qwen2 fast tokenizer only — verified by tracing `sys.modules` after
import (zero `torch` / `refs` / `flash_attn` modules loaded).

The PT byte-diff harnesses in `tools/stage*_compare.py` *do* import upstream
PyTorch under a shim — that is the entire point.  PyTorch is the source of
truth at verification time and disappears at inference time.  When the t2v
sequence builder still depended on PT at runtime (a leftover from Stage 9 §0),
it was migrated to a pure-MLX manual token concat and re-verified byte-identical
against the PT `ValidationDataset.t2v_sample` output before this release.

## Lessons

23 lessons distilled across the stages, plus 2 added post-release (Stage 11); see the
`§ Lessons` section of each `LEARNING_LOG/stage_*.md`.  Selected:

- *Single-step byte-diff ≠ multi-step correctness* (Stage 7) — chunking and
  accumulation are separate gates.
- *"Same as PT" ≠ "working correctly"* (Stage 7) — production-realistic
  inputs are required.  Synthetic gradient + "saturated" instruction makes
  both PT and MLX hallucinate identically; that is reproduction, not correctness.
- *Verification tools themselves can lie* (Stage 7, bug E) — when the
  hypothesis disagrees with the harness, both are suspect, not just the
  hypothesis.
- *Manual ground truth is an unverified hypothesis* (Stage 9, L18) — a fixture
  we *hand-simulated* from reading PT code is not proof PT produces it.  Call
  the PT code directly; our manual t2v sequence was wrong three different ways.
- *Different inputs → same output = the strongest bug signature* (Stage 9, L19)
  — two genuinely different inputs giving a byte-identical result means the
  forward is input-independent; this is how Lesson E (a flex-attention mask
  polarity inversion) was re-caught after it had already been fixed once.
- *Pin the lesson in code, not just docs* (Stage 9, L21) — Lesson E re-fired
  on reused code because the original fix was harness-local.  `pt_layer_mask()`
  now `raise`s (not `assert`s, so `-O` can't strip it) on the wrong mask dtype.
- *Gate the final output, not the intermediate* (Stage 9, L23) — the latent
  cos passed at 0.999437 while the decoded video diverged at 0.948 because a
  VAE normalization scale was dropped.  Checking pixels + the transform chain,
  not just latents, caught it.
- *A PT-direct-import gate is only non-blind if PT recomputes from raw inputs*
  (Stage 11) — feeding PT our intermediates (preprocessed patches, ViT output,
  positions) re-shares the misunderstanding the direct import was meant to rule
  out.  A patch-ordering bug had agreed at cos = 1.0, blind, since release.  This
  is Lesson 19 / bug E firing at the *harness-architecture* level, not just a mask.
- *A wrong reference makes both sides wrong together* (Stage 11) — Stage 3
  byte-checked our mRoPE positions against mlx-vlm's `get_rope_index`, which drops
  the video `tokens_per_second` multiplier too, so our matching unit-step looked
  correct.  PT-Lance is the real truth; against it the x2t_video temporal positions
  were off by ×2.

## Layout

```
lance_mlx/       MLX implementations (backbone, rope, attention mask, pipelines, vae)
                   — pure MLX, no torch/refs import at runtime
tools/           Cross-validation harnesses + smoke tests (one per stage / block)
                   — these *do* import upstream PT (the verification source of truth)
LEARNING_LOG/    Per-stage notes, 25 lessons distilled, audit trail of wrong hypotheses
out/audit_manual_v_t/  intentionally *wrong* manual ground-truth fixtures, kept as the
                   byte-level evidence behind Lesson 18 (see its README)
archive/         superseded working notes (kept for trace, not load-bearing)
IMPROVEMENTS.md          deferred improvements (B-class deviations from RockTalk)
UPSTREAM.md              upstream bugs found along the way (e.g. mlx-vlm multi-image RoPE)
VERIFICATION_BACKLOG.md  deferred verification items
LEARNING_WORK_ORDER_Lance_MLX_v2.md  project workorder
```

Fetched separately (not committed):

- `refs/Lance/` — upstream PT code snapshot.  `./tools/fetch_refs.sh` fetches
  our mirrored snapshot (Apache 2.0).  We can't pin an exact upstream commit:
  our snapshot's file hashes match no commit in the current
  `bytedance-research/Lance` HF history (likely an upstream force-push or a
  GitHub-vs-HF divergence), so we mirror the exact files we verified against
  for reproducibility.  Needed only for the `tools/` harnesses, not for inference.
- `checkpoints/` — ~30 GB model weights (HuggingFace, see Setup)
- `out/` (except the curated `audit_manual_v_t/`) — intermediate tensors,
  generated images, logs
- `.venv/` — Python virtual environment

## Setup

Apple Silicon required (developed on M3 Ultra 512 GB).  Python 3.12.

```bash
git clone https://github.com/avlp12/lance_alis_mlx
cd lance_alis_mlx

python3.12 -m venv .venv && source .venv/bin/activate
pip install mlx mlx-vlm torch transformers safetensors einops pillow \
            huggingface_hub numpy

mkdir -p checkpoints
# Our verified MLX conversion, F32 (same SHA256 as the PT source).  The repo
# now holds image (Lance_3B/) + video (Lance_3B_Video/) weights in subdirs;
# RockTalk/Lance-3B-MLX supplies the matching ViT + tokenizer for the harnesses.
hf download avlp12/Lance-3B-Alis-MLX-Traced --local-dir checkpoints/Lance-Alis
hf download RockTalk/Wan2.2-VAE-MLX --local-dir checkpoints/Wan2.2-VAE-MLX

# Optional: original PT Lance + upstream PT code — needed only for the
# PT-direct-import byte-diff harnesses in tools/.  Skip for MLX inference.
hf download bytedance-research/Lance --local-dir checkpoints/Lance
./tools/fetch_refs.sh   # fetches our mirrored refs/Lance snapshot (see Layout note)
```

Run a cross-validation harness (needs the PT side from the optional step):

```bash
PYTHONPATH=. .venv/bin/python tools/stage7_ti2i_compare.py        # TI2I 3-forward
PYTHONPATH=. .venv/bin/python tools/stage8_wanvae_compare.py      # video VAE 4-gate
PYTHONPATH=. .venv/bin/python tools/stage9_single_step_compare.py # t2v v_full/v_unc/v_blend
PYTHONPATH=. .venv/bin/python tools/stage9_per_step_cos.py        # t2v 30-step per-step cos
```

Generate an image:

```bash
PYTHONPATH=. .venv/bin/python tools/stage6_t2i_smoke.py        # T2I (text-to-image)
PYTHONPATH=. .venv/bin/python tools/stage7_x2t_smoke.py         # X→T (image-to-text)
PYTHONPATH=. .venv/bin/python tools/stage7_ti2i_smoke.py        # TI2I (image edit)
```

## Web UI (Gradio)

A browser front-end over the six verified pipelines lives in `app.py` — a pure UI
layer that calls the pipeline functions verbatim (no new model logic):

```bash
.venv/bin/pip install gradio
.venv/bin/python app.py          # open the printed http://127.0.0.1:7860
```

Six tabs — **t2i**, **image_edit**, **x2t** (image→text), **t2v**, **x2t_video**,
**video_edit**.  The image bundle (LLM + ViT) and video bundle are lazy-loaded once on
first use and cached; the Wan VAE is shared.

Practical settings (measured, not assumed):

- **Resolution.** Lance images are trained at **768** (`image_768res`); below ~512 the
  output is out-of-distribution and breaks (seams / oversaturation).  Video runs at a
  **video preset** (~480 = 360p) — lower than image.  The UI defaults to these.
- **Decode scale.** Generated latents live in the Wan-VAE normalized space, so decode
  must un-normalize with the per-channel `(mean, 1/std)` scale (PT's `WanVAE.decode`
  always does); omitting it oversaturates.
- **Prompts.** Lance is very prompt-sensitive — short prompts look rough; detailed
  prompts (cf. `refs/Lance/config/examples/t2i_example.json`) are much better.
- **Quality (honest).** Lance-3B is a **3B unified any-to-any** model; its image / video
  *generation* is modest — clearly below dedicated diffusion engines (SDXL / FLUX).  Its
  strengths are breadth (six tasks in one model) and the understanding / editing paths,
  not photorealism.  We confirmed PT and our MLX produce the *same* modest quality at 768
  (latent cos 0.94 = bf16-vs-f32 precision over the long flow trajectory, not a port bug);
  the official showcase images are best-case.
- **Cost.** Generation is heavy — t2v ≈ 90 s at 480 / 13 frames; image_edit and
  video_edit are several minutes (3-component CFG).  Keep video clips short.

## Project context

This is an *observation learning* project.  One person (Alis) sets the
strategy at each stage and judges results, while Claude (Opus) writes the
code, runs harnesses, and reports findings.  The audit trail in
`LEARNING_LOG/*.md` includes both passed gates and Claude's wrong hypotheses
that the byte-diff caught — kept verbatim as a record of how hand-porting
this scale of model actually goes.

## License

This project is licensed under the Apache License 2.0 — same as the upstream
ByteDance Lance and Alibaba Wan 2.2 VAE.  See [LICENSE](LICENSE).

Why Apache 2.0:

- Matches upstream — no license-compatibility friction with `refs/Lance/`
  and the Wan VAE code we port.
- Patent grant (Section 3) is meaningful for ML code in a way MIT's silence
  on patents is not.

`refs/` contents are upstream snapshots and remain under their original
licenses (Apache 2.0 for ByteDance Lance, see each subdirectory's `LICENSE`
for the others).  Apache 2.0 headers preserved verbatim in each upstream
file.

## On the weights

Our published MLX weight (`avlp12/Lance-3B-Alis-MLX-Traced`) is **bit-identical**
(same SHA256, `5ede2f0a…`) with `RockTalk/Lance-3B-MLX` — both follow the same
deterministic **F32** conversion path from `bytedance-research/Lance`, so the
bytes match by construction.  (A separate bf16 build is `mlx-community/Lance-3B-bf16`.)
The video weight (`Lance_3B_Video/`) is the standalone F32 build, verified
end-to-end against PT t2v (video pixel cos 0.999338).  **The weight is not the
contribution.**
What this repo adds is the *verification* — the byte-diff harnesses
(`tools/stage*_compare.py`), the per-stage lesson trail (`LEARNING_LOG/`), and
the kept-verbatim wrong hypotheses (`out/audit_manual_v_t/`).  "It works" is
cheap; "here is exactly how we checked it matches PT, including where we were
wrong" is the point.

## Acknowledgments

- ByteDance Lance team — original PyTorch model and research
- RockTalk — MLX checkpoint conversion and standalone weight layout; our
  conversion reproduces theirs byte-for-byte and we use it as a parity reference
- Alibaba Wan 2.2 team — 3D Causal VAE architecture
- Qwen team — Qwen2.5-VL backbone + tokenizer
