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
| 7 | ViT + X→T + TI2I (3 pipelines) | ✓ cos ≥ 0.999 + real-photo perceptual |
| 8 | 3D Causal Video VAE (T > 1) | ✓ 4 gates cos = 1.000000 (encode + decode) |
| 9 | Video DiT + t2v (text-to-video) | ✓ 30-step per-step cos ≥ 0.999, video pixel cos = 0.999338 |

**STAGE 1–9 complete.**  Every core path of Lance — image / video generation,
editing, understanding — is ported to MLX and byte-diff verified against the
original PyTorch.

Stage 7 numbers:

- ViT (Qwen2.5-VL vision tower): cos = 1.000000
- X→T first-token logits: cos = 0.999923, top-1 token match
- TI2I 3-forward (v_full / v_t_uncond / v_tv_uncond): all cos ≥ 0.999
- TI2I 30-step PT-vs-MLX final latent: cos = 0.997340
- Real-photo edits (orange cat → black panther, + bow tie) match RockTalk reference outputs

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

## Verification doctrine

Every block:

1. **Original PT direct import** (not a clean re-implementation).  `refs/Lance/`
   is imported under a flash_attn / flex_attention shim so we run the upstream
   code on CPU.  Algorithmic misunderstandings can't be shared between the two
   sides this way.
2. **Same PRNG** on both sides (NumPy `default_rng`) so initial noise is
   bit-identical.
3. **Byte-diff** at every layer / step / forward variant.  `cos ≥ 0.999` is the
   gate; `max|Δ|` is reported alongside (f32 dot-product noise floor ≈ 1e-6).
4. **Behavioural cross-check** on production-realistic inputs (real photos,
   real edit instructions).  Numerical pass without behavioural pass = not
   done.

## Pure-MLX inference, PyTorch only at verification time

The `lance_mlx/` package never imports `torch` or `refs/Lance` at runtime.
Inference (`t2i`, `t2v`, `x2t`, `image_edit`) runs on MLX + the HF Qwen2 fast
tokenizer only — verified by tracing `sys.modules` after import (zero
`torch` / `refs` / `flash_attn` modules loaded).

The PT byte-diff harnesses in `tools/stage*_compare.py` *do* import upstream
PyTorch under a shim — that is the entire point.  PyTorch is the source of
truth at verification time and disappears at inference time.  When the t2v
sequence builder still depended on PT at runtime (a leftover from Stage 9 §0),
it was migrated to a pure-MLX manual token concat and re-verified byte-identical
against the PT `ValidationDataset.t2v_sample` output before this release.

## Lessons

23 lessons distilled across the stages; see the `§ Lessons` section of each
`LEARNING_LOG/stage_*.md`.  Selected:

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

## Layout

```
lance_mlx/       MLX implementations (backbone, rope, attention mask, pipelines, vae)
                   — pure MLX, no torch/refs import at runtime
tools/           Cross-validation harnesses + smoke tests (one per stage / block)
                   — these *do* import upstream PT (the verification source of truth)
LEARNING_LOG/    Per-stage notes, 23 lessons distilled, audit trail of wrong hypotheses
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
# Our verified MLX conversion (byte-identical / same SHA256 as the original PT source).
hf download avlp12/Lance-3B-Alis-MLX-Traced --local-dir checkpoints/Lance-3B-MLX
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
deterministic bf16-preserving conversion path from `bytedance-research/Lance`,
so the bytes match by construction.  **The weight is not the contribution.**
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
