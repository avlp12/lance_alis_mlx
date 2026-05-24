# lance_alis_mlx

Hand-port of [ByteDance Lance](https://github.com/bytedance/Lance) multimodal
model from PyTorch to Apple MLX.  Every block is ported and cross-validated
against the original PT via byte-diff (cosine ≥ 0.999) before the next block
starts.

## Status

| Stage | Component | Gate |
|---|---|---|
| 1 | PT → MLX weight conversion | ✓ bit-exact vs RockTalk reference |
| 2 | Qwen2.5-VL text backbone (36-layer Lance LLM) | ✓ cos = 1.000000 vs mlx-vlm stock |
| 3 | 3D mRoPE positions | ✓ byte-identical 12/12 |
| 4 | MoE-gen routing (MoT) | ✓ cos = 1.000000 vs clean PT re-impl |
| 5 | Wan 2.2 VAE image path (T = 1) | ✓ ~40 dB PSNR round-trip vs PT |
| 6 | Flow matching + CFG denoising loop | ✓ end-to-end cos ≥ 0.999 vs PT 30-step |
| 7 | ViT + X→T + TI2I (3 pipelines) | ✓ cos ≥ 0.999 + real-photo perceptual |
| 8 | 3D Causal Video VAE (T > 1) | **in progress** |

Stage 7 numbers:

- ViT (Qwen2.5-VL vision tower): cos = 1.000000
- X→T first-token logits: cos = 0.999923, top-1 token match
- TI2I 3-forward (v_full / v_t_uncond / v_tv_uncond): all cos ≥ 0.999
- TI2I 30-step PT-vs-MLX final latent: cos = 0.997340
- Real-photo edits (orange cat → black panther, + bow tie) match RockTalk reference outputs

Stage 8 progress (bottom-up):

- CausalConv3d `cache_x` (streaming temporal causal conv) — cos = 1.0, cache state max\|Δ\| = 0
- Resample `upsample3d` / `downsample3d` (pixel-shuffle T expansion, stride-2 T contraction) — cos = 1.0, frame-order verified
- ResidualBlock feat_cache pass-through — slot-by-slot byte-equal
- Down_ResidualBlock (resblock × 2 + Resample slot mixing) — cos = 1.0, all 5 slots byte-equal
- Next: Up_ResidualBlock, Encoder3d / Decoder3d, then full T = 5 encode/decode

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

13 lessons distilled across the stages; see the `§4 Lessons` section of each
`LEARNING_LOG/stage_*.md`.  Selected:

- *Single-step byte-diff ≠ multi-step correctness* (Stage 7 §3) — chunking and
  accumulation are separate gates.
- *"Same as PT" ≠ "working correctly"* (Stage 7 §3) — production-realistic
  inputs are required.  Synthetic gradient + "saturated" instruction makes
  both PT and MLX hallucinate identically; that is reproduction, not correctness.
- *Verification tools themselves can lie* (Stage 7 §3, bug E) — when the
  hypothesis disagrees with the harness, both are suspect, not just the
  hypothesis.
- *Visual seams ≠ tile bugs* (Stage 6 §2) — 512² grid lines turned out to be
  PRNG-identity mismatch, not VAE tile blending.

## Layout

```
lance_mlx/       MLX implementations (backbone, rope, attention mask, pipelines, vae)
tools/           Cross-validation harnesses + smoke tests (one per stage / block)
LEARNING_LOG/    Per-stage notes, lessons distilled, audit trail of wrong hypotheses
refs/            Frozen snapshots of upstream PT (Apache 2.0; re-fetchable)
                   Lance              https://github.com/bytedance/Lance
                   Lance-3B-MLX       https://huggingface.co/RockTalk/Lance-3B-MLX
                   Lance-3B-Video-MLX https://huggingface.co/RockTalk/Lance-3B-Video-MLX
                   Wan2.2-VAE-MLX     https://huggingface.co/RockTalk/Wan2.2-VAE-MLX
IMPROVEMENTS.md          deferred improvements (B-class deviations from RockTalk)
UPSTREAM.md              upstream bugs found along the way
VERIFICATION_BACKLOG.md  deferred verification items
LEARNING_WORK_ORDER_Lance_MLX_v2.md  project workorder
```

Not in the repo (`.gitignore`):

- `checkpoints/` — ~30 GB model weights (fetch from HuggingFace, see Setup)
- `out/`         — intermediate cross-validation tensors and generated images
- `.venv/`       — Python virtual environment

## Setup

Apple Silicon required (developed on M3 Ultra 512 GB).  Python 3.12.

```bash
git clone https://github.com/avlp12/lance_alis_mlx
cd lance_alis_mlx

python3.12 -m venv .venv && source .venv/bin/activate
pip install mlx mlx-vlm torch transformers safetensors einops pillow \
            huggingface_hub numpy

mkdir -p checkpoints
huggingface-cli download RockTalk/Lance-3B-MLX   --local-dir checkpoints/Lance-3B-MLX
huggingface-cli download RockTalk/Wan2.2-VAE-MLX --local-dir checkpoints/Wan2.2-VAE-MLX

# Optional: original PT Lance — needed only for the PT-direct-import byte-diff
# harnesses in tools/.  Skip if you only want to run MLX inference.
huggingface-cli download bytedance-research/Lance --local-dir checkpoints/Lance
```

Run a cross-validation harness:

```bash
PYTHONPATH=. .venv/bin/python tools/stage7_ti2i_compare.py
PYTHONPATH=. .venv/bin/python tools/stage8_causal_conv3d_compare.py
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

Code we wrote (`lance_mlx/`, `tools/`, `LEARNING_LOG/`, top-level docs):
choose your license — none committed yet; pick MIT or Apache 2.0 to match
upstream.

`refs/` contents are upstream snapshots and remain under their original
licenses:

- ByteDance Lance — Apache 2.0
- RockTalk MLX ports (Lance-3B-MLX, Lance-3B-Video-MLX, Wan2.2-VAE-MLX) —
  see each repo's LICENSE
- Wan 2.2 VAE (Alibaba Wan team) — Apache 2.0

Apache 2.0 headers preserved verbatim in each upstream file.

## Acknowledgments

- ByteDance Lance team — original PyTorch model and research
- RockTalk — MLX checkpoint conversion and standalone weight layout used as
  parity reference
- Alibaba Wan 2.2 team — 3D Causal VAE architecture
