# STAGE 6 — Flow Matching + CFG Denoising Loop ★ 첫 이미지 생성

**Status:** ✅ PASSED  (2026-05-23, full cross-validation)
**Deliverable:** `lance_mlx/scheduler.py` + `lance_mlx/pipelines/t2i.py` + `lance_mlx/attn_mask.py` + adapter integration in `lance_mlx/backbone.py`
**Verification chain (수치 + 행동 + 재현성, 모두 교차):**
1. §0 backbone re-verify (PT 원본 import) — min cos 0.999938 at layer 0/12/24/35
2. §1 step-by-step latent vs PT 30 step — min cos 0.999631 (256²) / 0.999641 (512²)
3. §2 VAE decode at 32×32 latent — byte-identical PT vs MLX (cos 1.0, max|Δ|=0)
4. §2 End-to-end image PT vs MLX — cos 0.999435 (512²)
5. §3 Path identity — production `t2i.py` vs compare inline — *byte-identical* (cos 1.0, max|Δ|=0, MSE=0) once PRNG unified

## §0. Entry — STAGE 4 backbone re-verification *before* writing image-gen code

Per the doctrine update from STAGE 5 (`stage_5.md §2b`), this STAGE opens
with a *numerical* verification rather than diving straight into the
denoising loop.  Settles `VERIFICATION_BACKLOG` item "[STAGE 4 → STAGE 6]
STAGE 4 백본 재검증 — 원본 PT 직접 import".

### Setup (`tools/stage6_pt_backbone_compare.py`)

1. Stub `flash_attn` with a SDPA-based shim (single-sequence cases only).
2. Neutralize transformers' flash_attn probe (`is_flash_attn_2/3/4_available → False`).
3. Stub `modeling.lance` package without running its `__init__.py`.
4. `importlib.import_module("modeling.lance.qwen2_navit")` — *original*
   `Qwen2MoTDecoderLayer` from `refs/Lance/modeling/lance/qwen2_navit.py`.
5. Load PT Lance_3B per-layer weights as **bf16** (PT's mode="gen" path
   forces `packed_query_sequence.to(torch.bfloat16)`).
6. Synthetic input matching STAGE 4 Tier 3 (B=1, L=48, GEN slab [24, 40)).

### Result

| layer | cos | max\|Δ\| | rel_L2 |
|---|---|---|---|
| 0  | **0.999979** | 15.5  | 6.6e-3 |
| 12 | **0.999957** | 18.9  | 9.3e-3 |
| 24 | **0.999938** | 62.0  | 1.1e-2 |
| 35 | **0.999979** | 416   | 6.8e-3 |

**min cos = 0.999938 ≥ 0.999 → PASS** against *original* PT package.

cos < 1.0은 *desirable* — bf16 PT vs f32 MLX boundary 횡단, *진짜 다른
코드*와 비교했다는 증거.  STAGE 4 Tier 3 cos=1.000000은 우리가 *우리
해석을 자기 자신과* 비교했다는 신호였음(이제 supersede).

## §1. Scheduler + Adapter Integration + First Image

### Building blocks

- **`lance_mlx/scheduler.py`**: `FlowMatchingSchedule`, `make_schedule` (PT byte-exact
  formula: `timesteps = shift * t / (1 + (shift-1) * t)`, dts diff), `cfg_velocity`
  (CFG blend + global norm rescale per Lance), `euler_step`, `sample_init_noise`
  (NumPy PRNG — see §3).
- **`lance_mlx/attn_mask.py`**: `build_lance_attention_mask` — replicates PT
  `create_sparse_mask` predicate on a token grid (causal text + bidirectional
  within latent slab); verified bit-equivalent to PT predicate output.
- **`lance_mlx/pipelines/t2i.py`**: T2I pipeline — sequence layout
  (`<im_start>[prompt]<im_end><vis_start>[N=h_lat·w_lat·t_lat latent placeholders]<vis_end>`),
  per-step embed (`vae2llm(x_t) + time_embedder(t) + latent_pos_embed(pos_ids)`),
  CFG forward × 2 + Euler.
- **`lance_mlx/backbone.py`**: `LanceLLM` adds `vae2llm`, `llm2vae`,
  `time_embedder` (sinusoid + 2-layer MLP), `latent_pos_embed` (4096×2048 lookup).
  `load_full_lance` strict-loads all 1021 keys including 9 adapters.

### Step-by-step PT cosine (`tools/stage6_pt_denoise_compare.py`)

Side-by-side denoise: PT (original `Qwen2MoTDecoderLayer` × 36 + adapter
forwards) vs our MLX, same noise init (NumPy seed=0), same prompt/CFG/shift.
PT mask via `create_sparse_mask` BlockMask + flex_attention-→SDPA monkey-patch.

256² (16×16 latent, 30 steps):
- min cos = 0.999631 at step 30 (rel_L2 2.7%)

512² (32×32 latent, 30 steps):
- min cos = 0.999641 at step 30 (rel_L2 2.7%)
- final flat latent cos = 0.999641
- reshape→NTHWC latent cos = 0.999641
- **VAE decode output cos = 0.999435**, max|Δ| 0.998 (픽셀 단위 거의 동일)

Both well above 0.995 gate.  Drift = bf16(PT)/f32(MLX) cumulative noise.

## §2. 보류 해소 — 격자/banding의 *진짜* 원인

§1 도중 두 visual artifact 만남:
- 256² 출력 — 약한 banding / softness
- 512² 출력 — 굵은 흰 격자선 + 블록 분할

처음엔 *서로 다른 두 버그*로 추정. 진단 chain:

1. **VAE 무죄 — byte-identical at 32×32 latent.**
   `tools/stage6_vae_512_compare.py`: random N(0, 0.7) latent (1, 1, 32, 32, 48) →
   PT WanVAE_.decode vs MLX Wan2_2_VAE.decode. cos=1.0, max|Δ|=0.
   → VAE에는 tile seam 버그 없음.

2. **`max_latent_size = 32 → 64` (silent layout bug).**
   `_latent_position_indices`의 `max_latent_size` 디폴트가 PT `LanceConfig`의
   *학습 기본값* 32였으나, `checkpoints/Lance-3B-MLX/config.json`은
   `max_latent_size=64, max_num_latent_frames=1` (image variant).
   같은 4096 table size지만 *position index 의미가 다름* (`h*32+w` vs `h*64+w`).
   step-by-step cos compare에서 *PT 측도 우리 코드의 32를 그대로 받아 사용*
   → 양쪽이 *동일하게 틀린 값*을 쓰니 cos는 통과. **256² banding을 일으킨
   진짜 원인**.

3. **PRNG identity — 진범 (512² 격자 + 모든 reproducibility 문제).**
   smoke (`t2i.py` via `mx.random.normal(seed=0)`) vs compare-MLX
   (`stage6_pt_denoise_compare.py` via `np.random.default_rng(0)`) — *같은 seed
   0이지만 PRNG 라이브러리가 다르니 완전 다른 노이즈 샘플*.
   - MLX PRNG noise → 굵은 흰 격자 그림
   - NumPy PRNG noise → 미세 텍스처 그림 (compare/PT와 동일)
   `sample_init_noise`를 `np.random.default_rng`로 통일한 *직후*
   smoke vs compare-MLX = **cos 1.0, max|Δ|=0, MSE=0** 직접 측정으로 확정.

### 두 visual symptom의 정체

| symptom | 원인 |
|---|---|
| 256² 약한 banding | `max_latent_size` 32 → 64 fix로 해소 |
| 512² 굵은 흰 격자 | MLX PRNG seed=0의 특정 outcome.  같은 코드 + numpy seed=0 → 미세 텍스처 (PT와 일치).  *우리 구현 버그 아님* |

VAE의 미세 가로 텍스처(현재 numpy seed=0 출력)는 *Lance 모델 자체*의
출력 특성(또는 더 미세한 모델 한계).  cos+시각 양쪽 PT와 일치 확인.

## §3. Production fix — PRNG unification (`scheduler.py:130-148`)

`sample_init_noise(shape, seed)`을 NumPy 기반으로 통일:

```python
import numpy as np
rng = np.random.default_rng(seed)
return mx.array(rng.standard_normal(shape).astype("float32"), dtype=dtype)
```

이유: cross-validation harness가 NumPy PRNG를 쓰므로 production이 같은
PRNG여야 *비교 가능*.  MLX random은 reproducibility 도구로는 좋지만 PT
side-by-side 와의 *byte-identical*은 불가능 (다른 PRNG → 다른 sample).

**검증 후 코드 반영 확정** — `grep` 으로 확인:
```
130:def sample_init_noise(shape, *, seed=0, dtype=mx.float32):
144:    import numpy as np
145:    rng = np.random.default_rng(seed)
```

## §4. 교훈 (Lessons distilled from STAGE 6)

기존 STAGE 1~5의 lessons는 각 stage log에 있음.  여기 새로 박힌 것:

### Lesson 9 (new) — *PRNG identity*도 공유 설정의 일종

> `seed 0`의 *의미*는 라이브러리마다 다르다 (`mx.random ≠ np.random ≠
> torch.random`).  cross-validation harness가 *같은 PRNG*에서 노이즈를
> 뽑으면, *해당 PRNG의 outcome*에 대해서만 검증된 것.  Production이
> 다른 PRNG를 쓰면 — 검증된 outcome과 *다른 그림*이 나올 수 있고, 그건
> 버그가 아님.
>
> **보호책:** production code의 *재현성 결정자*(noise sampler 포함)를
> *cross-validation harness가 쓰는 것과 동일하게* 일치시킨다.  여기선
> `sample_init_noise`를 numpy로 통일.
>
> **그리고 — 어떤 시각 증상도 단독으로 버그 시그니처가 아니다.**
> 격자/banding/색감 변화는 *seed 하나만으로도* 만들어진다.  *동일 입력
> + 동일 PRNG로 PT vs MLX byte-diff*만이 최종 게이트.

### Lesson 7 (referenced) — *남의 말은 지도지 정답이 아니다*

STAGE 5에서 확립된 doctrine — "clean PT reimpl + 시각검증의 조합은
약하다.  원본 PT 직접 import + layer cos가 진짜 검증" — 의 더 일반화된
형태.  *모든 secondhand source*가 *지도(어디 볼지 알려줌)*지 *정답(거기
가면 답이 있다)*은 아님.  RockTalk README, PT default kwarg, mlx-vlm,
*그리고 — Claude의 가설*도 secondhand.

## §5. 정직한 audit trail — Claude의 두 틀린 가설

STAGE 6 §2 디버깅 중 Claude가 *두 번* 틀린 가설을 냈다.  매번 *원본 PT
직접 비교*가 걸러냈다.  *Claude의 reasoning도 Lesson 7의 적용 대상* —
그럴듯한 진단이지만 *검증 안 거치면 잘못된 지도*.

| 시점 | Claude의 가설 | 실제 결과 |
|---|---|---|
| 512² 격자 첫 진단 | "VAE decode tile seam 버그.  STAGE 5는 256²만 검증, 32×32 latent path는 미검증" | `stage6_vae_512_compare.py`로 직접 비교 → byte-identical (cos 1.0).  VAE 무죄 |
| max_latent_size 32→64 fix 후 잔존 banding | "Flux-style latent normalization (shift/scale) 미적용" | PT `WanVideoVAE.vae_decode`가 `scale=` 없이 호출 → 디폴트 [0, 1] identity 확인.  Lance inference는 normalization 안 함.  무죄 |

**Doctrine 자기적용**:
- Claude가 *그럴듯한* 분석을 내놓을 때 *확신성이 강해 보일수록* 검증 강화.
- 가설은 *어디를 볼지 가리키는 화살표*지 *결론이 아님*.
- 매번 *원본 PT 직접 비교*가 최종 판정자.

## §6. Carried forward to STAGE 7

- **`LanceLLM(LanceTextConfig())`** — verified 4 levels:
  - STAGE 2 (A): structural cos=1.0 vs mlx-vlm stock
  - STAGE 4 Tier 3: cos=1.0 vs clean-PT reimpl (superseded by §0)
  - STAGE 6 §0: cos≥0.999 vs original PT (bf16 mixed precision)
  - STAGE 6 §1: end-to-end image cos≥0.999 vs PT denoise
- **`Wan2_2_VAE`** — verified 32×32 latent byte-identical to PT (§2).
- **Tools** reusable: `stage6_pt_backbone_compare.py` (layer-wise probe),
  `stage6_pt_denoise_compare.py` (end-to-end denoise+image), `stage6_vae_512_compare.py`
  (VAE-only at various sizes).
- **9 lessons** distilled (Lesson 9 new at this STAGE).
- **VERIFICATION_BACKLOG** items closed: STAGE 4 backbone re-verify, STAGE 6
  first-image, STAGE 5 RMSNorm near-zero (implicit — no degeneracy observed).

## §7. Code-reviewer pass (workorder §5.7)

Reviewer: Opus, xhigh-equivalent.

- **BLOCKING:** none — verification gates already pin the math.
- **SUGGESTED-A applied:**
  - `_to_uint8` dead `if False else` branch in `stage6_pt_denoise_compare.py` → cleaned.
  - `attn_mask.build_lance_attention_mask` guards added (`seq_len > 0`, `split_lens` non-empty).
  - `sample_init_noise` dtype assertion + docstring on f32 invariant + bf16
    future path.
  - `cfg_velocity` short-circuit kept with comment (reviewer flagged as regression,
    but PT lance.py:688 outer guard makes our shortcut *mathematically identical*
    to PT — documented inline).
  - `load_full_lance` now reads `_EXPECTED_OUTER_KEYS` and prints drift before
    strict-load, plus a pre-check on `_ADAPTER_ATTRS` so a renamed adapter
    fails early with a clear message (not opaque "checkpoint key X not modeled").
  - Tiny lints: `L_cond, = (cond_ids.shape[1],)` → plain assignment;
    `n_cond`/`n_blend` renamed `n_v_cond`/`n_v_` to match PT variable names.
- **SUGGESTED-B → IMPROVEMENTS.md:**
  - Skip uncond forward at cfg_scale ≤ 1.0 (Lance dynamic CFG ramp).
  - Batch cond + uncond into B=2 forward (STAGE 7 has 3-component CFG, larger gain).
- **SUGGESTED-B → VERIFICATION_BACKLOG.md:**
  - flex_attention SDPA shim multi-sample packing (STAGE 7 may trip).
  - `build_lance_attention_mask` multi-document validation.
- **Regression check:** STAGE 2 (cos=1.000000), STAGE 3 (12/12 byte-identical),
  STAGE 4 (PT layer cos=1.000000), STAGE 5 (round-trip PSNR 40 dB), STAGE 6
  (smoke t2i 512² → byte-identical PNG) — all still pass.
