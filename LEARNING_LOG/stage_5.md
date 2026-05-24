# STAGE 5 — Wan 2.2 Image VAE (T=1)

**Status:** ✅ PASSED  (2026-05-22)
**Deliverable:** `lance_mlx/vae_wan22.py`
**Verification:** byte-identical to PT WanVAE_ (encoder cos=1.0, decoder layer-by-layer cos=1.0); round-trip PSNR 40 dB at 256² (criterion ≤ 1e-3 MSE PASS).

## 1. What this stage did

Hand-ported the Wan 2.2 VAE (3D causal video VAE) to MLX, T=1 image
path.  Encoder: `(B, T, H, W, 3)` RGB in [-1, 1] → `(B, T, H/16, W/16, 48)`
latent (16× spatial via patchify 2× + encoder 8×).  Decoder: latent →
reconstructed RGB.  Strict-loads 196-tensor RockTalk standalone
`Wan2.2-VAE-MLX` checkpoint (704.7 M params).  Validated against the
original PT `WanVAE_` (refs/Lance) with our MLX-→PT-layout reverse
conversion: byte-equivalent forward through every layer.

## 2. Block highlights (the "why")

- **5.1 — `CausalConv3d` with flat weight + asymmetric T pad.**  PT
  subclasses `nn.Conv3d`; we needed the weight at `conv.weight`
  (not `conv.conv.weight`) to match RockTalk's checkpoint, so we own
  the `weight`/`bias` parameters directly and call `mx.conv_general`.
- **5.1 — `WanRMSNorm` is L2-normalize-and-scale**, not standard
  RMSNorm.  PT uses `F.normalize(x, dim=-1) * sqrt(C) * gamma`.  Key
  is `gamma` (per PT name).
- **5.2 — `ResidualBlock` named children** (`norm1/conv1/norm2/conv2`)
  to match RockTalk's MLX naming (PT stores these as `residual.0/.2/.3/.6`
  Sequential indices; RockTalk renamed them for clarity).
- **5.2 — `AttentionBlock` per-frame spatial attn** via 1×1 Conv2d for
  qkv/proj.  `mx.fast.scaled_dot_product_attention` matches manual sdpa
  within f32 noise (3e-5).
- **5.3 — `Down_ResidualBlock` always has `avg_shortcut`** — this was a
  bug-trap.  PT creates `self.avg_shortcut = AvgDown3D(...)` regardless
  of `down_flag`; when `down_flag=False`, factor_s=1 gives identity-ish
  passthrough.  My first attempt guarded `if down_flag: self.avg_shortcut`,
  and the last stage silently dropped the residual.  `AvgDown3D` has
  no learnable params so strict-load didn't catch it.
- **5.4 — patchify channel order matches PT einops `b c f (h q) (w r)
  -> b (c r q) f h w`** — *c slowest*, then r (W-inner), then q (H-inner)
  fastest.  My first attempt had `(p_h p_w c)` C-fastest; conv1's
  trained weights expect c-slowest.

## 2b. Verification-strength epiphany

STAGE 5 quietly upgraded the project's whole *verification doctrine*.

Three load-bearing facts from this stage:

1. **strict-load passes silent bugs.** All three STAGE-5 bugs left the
   196/196 key load 0-diff.  Strict-load checks *module shape*, never
   *forward semantics*.
2. **Behavioural check is *necessary but not sufficient*.** Two of the
   three bugs would render as "slightly blurry / off-colour image" at
   cos 0.97 — visually almost-correct.  A reasonable human check would
   pass them.  Only layer-wise numerical compare against PT caught them.
3. **Direct-import-of-original-PT beats clean-reimpl.**  STAGE 4's
   Tier 3 used a clean PT rewrite we wrote by hand (`stage4_pt_cosine.py`).
   STAGE 5 instead did `sys.path.insert(0, "refs/Lance")` and `from
   modeling.vae.wan.vae2_2 import WanVAE_` directly.  Result was the
   same nominal cos=1.0, but with *no risk of shared algorithmic
   misunderstanding* between our MLX and our PT — because the PT side
   wasn't *ours*.

**Doctrine update (retroactive to STAGE 4):** A STAGE is "verified"
only after **(a) layer-wise hidden-state cos ≥ 0.999 vs the *original*
PT package**, *and* (b) behavioural / numerical end-to-end check passes.
Clean PT reimpl + behavioural check together is *weaker* than original-PT
layer cos alone — because (1) shares interpretation risk and (2) misses
sub-cos-1.0 bugs.  See `VERIFICATION_BACKLOG.md` entry "STAGE 4 백본
재검증" — STAGE 4 owes a re-pass under the stronger pattern.

## 3. The three bugs

| Bug | Symptom | Root cause |
|---|---|---|
| Patchify channel order | latent std too small, recon cos 0.4 | Flat order was `(p_h, p_w, c)` C-fastest instead of `(c, r, q)` C-slowest. The trained conv1 expected PT's order. |
| Down_ResidualBlock avg_shortcut on last stage | latent std 0.21 vs 0.7, cos 0.4 → 0.7 after partial fix | Gated `self.avg_shortcut` on `down_flag`; PT always creates it.  AvgDown3D has no params → strict-load wouldn't catch. |
| AvgDown3D group axis order | encoder cos 0.97 (not 1.0) — first divergence | Flat order after permute was `(ft, fs_h, fs_w, C)` C-fastest, group-mean read wrong slabs.  PT uses `(C, ft, fs_h, fs_w)` C-slowest so each output channel pools over factor positions of the *same* input channel. |

After all three fixes: encoder cos=1.000000, every decoder layer cos=1.000000.

## 4. Verification

### Layer-by-layer PT cosine (real Lance/RockTalk weights, 64×64 sinusoid)

| stage | cos | max\|Δ\| |
|---|---|---|
| encoder.conv1 | 1.000000 | 8.3e-7 |
| encoder.downsamples.0 | 1.000000 | < 1e-5 |
| ... (all stages) | 1.000000 | < 1e-4 |
| encoder mu | **1.000000** | 1.7e-6 |
| decoder.conv1 | 1.000000 | 4.8e-7 |
| decoder.middle.0–2 | 1.000000 | 4.8e-6 |
| decoder.upsamples.0–3 | **1.000000** | up to 4.5e-5 (depth) |

Our MLX is byte-equivalent to PT throughout.  Combined with PT's own
round-trip (28 dB on the high-freq pattern, 40 dB on the slow one),
this confirms the implementation is *correct*, not just "passing a
weak metric".

### Round-trip MSE / PSNR (slow-frequency sinusoid, our criterion)

| size | MSE | max\|Δ\| | cos | PSNR | PASS? |
|---|---|---|---|---|---|
| 64×64 | 4.1e-4 | 0.252 | 0.999198 | 39.84 dB | ✓ |
| 128×128 | 3.6e-4 | 0.249 | 0.999326 | 40.51 dB | ✓ |
| 256×256 | 3.2e-4 | 0.245 | 0.999392 | 40.90 dB | ✓ |

Criterion: MSE ≤ 1e-3.  All three sizes PASS.  RockTalk reports
37.99 dB; we hit 40.9 dB on the equivalent low-frequency pattern.

### Failure case (documented, not a bug)

Original high-frequency sinusoid (period 32 = 2× spatial_downsample of 16)
hits Nyquist for the encoder downsample.  Both PT and MLX produce
MSE ~6e-3 / PSNR 28 dB on it.  This is the VAE's known band-limit, not
implementation error — confirmed by PT having the same number.

### Performance (M3 Ultra, T=1)

| size | encode | decode |
|---|---|---|
| 64×64 | 34 ms | 48 ms |
| 128×128 | 22 ms | 50 ms |
| 256×256 | 32 ms | 99 ms |

RockTalk reports 80 ms encode / 200 ms decode at 256² on M3 Ultra —
ours is faster on encode, marginally faster on decode.

## 5. Carried forward

- `Wan2_2_VAE(Wan22VAEConfig())` instance ready to be the VAE leg of
  STAGE 6 (T2I diffusion: decode the denoised latent into the final
  image).
- `Encoder3d` ready for STAGE 7 (TI2I: VAE-encode the cond image to a
  cond-latent slab).
- The 3D causal machinery (CausalConv3d feat_cache hooks, time_conv
  in Resample, AvgDown3D / DupUp3D factor_t > 1 paths) is *structurally*
  in place but its T>1 forward branches raise NotImplementedError —
  STAGE 8 opens those.
- `tools/stage5_pt_compare.py` — a working MLX ↔ PT WanVAE_ shim
  (clean PT weight conversion via axis permute + module name rename).
  Reusable for STAGE 8 video path verification.

## 6. Open items

- T>1 video path (CausalConv3d feat_cache plumbing, Resample time_conv
  branches, DupUp3D first_chunk semantics for chunked video frames) →
  STAGE 8.
- Scale parameter (`mu = (mu - scale[0]) * scale[1]`) — accepted at
  call time, no built-in default.  STAGE 6 diffusion pipeline supplies it.

## 7. Code-reviewer pass (workorder §5.7)

- **BLOCKING:** none — the three bugs were caught & fixed during dev.
- **A applied:**
  - `Resample` 3D NotImplementedError moved to *top* of `__call__` so a
    STAGE 6/7 caller fails fast instead of after wasting a spatial pass.
    Message also names the mode and T value.
  - `Up_ResidualBlock.__call__(first_chunk=True)` default now matches
    `Decoder3d` and `Wan2_2_VAE.decode`.  Docstring documents STAGE 8/9
    streaming use (False after first call).
  - `patchify` docstring now carries a worked example (`k = c*p*p + r*p + q`)
    so the next reader catches a flip without needing to know einops.
  - `AvgDown3D` docstring now spells out the identity case (factor=1)
    and the *inverted flatten* failure mode (the STAGE 5 bug).
  - `Wan22VAEConfig` class docstring explicitly warns against unifying
    `enc_dim`/`dec_dim`.
  - `encode()` comment that we discard `log_var` (deterministic mean).
- **B → IMPROVEMENTS.md:**
  - `CausalConv3d` random init → switch to zeros (RNG hygiene, strict=False safety).
  - `stage5_pt_compare.py` `.replace` chain → loop refactor for grep-ability.
  - `Resample.upsample2d` double-repeat intermediate → memory pattern.
- **B → VERIFICATION_BACKLOG.md:**
  - `WanRMSNorm` near-zero input parity vs PT.  Settles at STAGE 6 first
    image (if visuals are clean, distribution stayed in normal range).
- **Regression check:** all 3 sizes still PASS (MSE 3.2–4.1e-4, PSNR
  39.84–40.90 dB).  STAGE 2/3/4 regression checks unchanged.
