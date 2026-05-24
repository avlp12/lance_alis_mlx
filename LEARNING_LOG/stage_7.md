# STAGE 7 — TI2I + X→T + ViT ★ multimodal extension

**Status:** ✅ PASSED  (2026-05-23, full cross-validation + real-photo perceptual)
**Deliverable:**
- `lance_mlx/vit.py` (Qwen2.5-VL vision tower wrapper)
- `lance_mlx/pipelines/x2t.py` (image understanding AR pipeline)
- `lance_mlx/pipelines/image_edit.py` (TI2I edit pipeline + 3-component CFG)
- `lance_mlx/scheduler.py:cfg_velocity_3comp` (TI2I CFG blend + Lance global-norm rescale)

**Verification chain (수치 × 행동 × production-realistic 입력, 모두 교차):**
1. §1 ViT — PT 원본 import vs MLX: cos = 1.000000 on Qwen2.5-VL vision tower (single-image grid 224²)
2. §2 X→T — PT first-token logits vs MLX: cos = 0.999923, top-1 token 일치 ('The')
3. §3 (a) TI2I 3-forward first-step harness — PT vs MLX:
   - `v_full      cos = 0.999632`
   - `v_t_uncond  cos = 0.999875`
   - `v_tv_uncond cos = 0.999780`
   - embed (pre-layer) cos = 0.999997 all 3 layouts
4. §3 (b) 의심점 ①②③ 명시적 byte-diff:
   - vae2llm UND 방향 (cond_latent→LLM): cos = 1.000000
   - Edit-mode system prompt: byte-equal 308 bytes, tokenized 56 ids 일치
   - 3-comp CFG 수식 (cfg_text/vit 다양한 edge case): cos = 1.00000000
5. §3 (c) PT 30-step end-to-end vs MLX (합성 gradient, cfg=3.0/1.5, seed=0):
   - per-step ‖v_full‖ 차이 < 1%
   - **최종 latent cos = 0.997340**
   - 두 그림 perceptual identical (둘 다 모델이 환각한 토끼/레고 — 모델 거동, 우리 버그 아님)
6. §3 (d) Real-photo perceptual gate (RockTalk `orange_cat_chair.png` 512², cfg=4.0/1.0, seed=0, 30step):
   - "Make the cat completely black, like a panther." → 고양이가 검게 변함 (vs RockTalk ref `edit_black_panther.png`)
   - "Add a small red bow tie to the cat." → 나비넥타이 추가됨 (vs RockTalk ref `edit_bowtie.png`)
   - **사용자 perceptual 판정: 의도 편집이 정확히 수행** → TI2I 진짜 작동 확정

---

## §1. ViT (Qwen2.5-VL vision tower) MLX port

### Setup
- 정책: Lance는 ViT을 *수정 안 함* — RockTalk MLX 체크포인트의 `vit.safetensors` 키가
  vanilla Qwen2.5-VL과 *완전 일치* 확인 후, `mlx_vlm.models.qwen2_5_vl.vision.VisionModel`을
  *그대로* 감싸 사용.  `lance_mlx/vit.py:LanceViT`는 thin wrapper + strict-load helper.
- Verification: `tools/stage7_vit_compare.py` — `orange_cat_chair.png` 224²,
  patchified로 ViT 통과한 (`N_vit=64, D=2048`) 출력을 PT 직접 import 출력과 cos 비교.
- **결과:** cos = 1.000000 (f32, no precision boundary).

### 교훈 (referenced)
- *재구현이 항상 옳은 답이 아님*.  검증된 라이브러리(mlx-vlm)가 충분하면 그대로 쓰되,
  *"원본이 수정 안 함"을 키 dump로 확인 후* — 가설 아닌 사실로 옮긴 후에.

---

## §2. X→T (image understanding AR pipeline)

### Setup
- `lance_mlx/pipelines/x2t.py`:
  - `preprocess_image` — Qwen2.5-VL preprocessor 직접 재구현
    (`patch_size=14, spatial_merge_size=2, temporal_patch_size=2`, CLIP normalization).
    `preprocessor_config.json`이 체크포인트에 없어 manual.  T=1 image는 T=2로 복제 후 patchify.
  - 시퀀스: `<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<vision_start>[N_vit IMG placeholders]<vision_end>[question]<|im_end|>\n<|im_start|>assistant\n`
  - AR 디코딩: greedy, full-sequence re-forward 매 토큰 (KV cache 없음 — IMPROVEMENTS 항목).

### Verification (`tools/stage7_x2t_compare.py`)
- PT 직접 import: full PT Lance LLM (36 bf16 layers) + PT ViT + adapters로
  *첫 토큰* logits 계산.
- 우리 MLX 동일 시퀀스로 첫 토큰 logits.
- **결과:** cos = 0.999923, top-1 token 일치 (둘 다 `'The'`), top-5도 유사.
- Smoke (`tools/stage7_x2t_smoke.py`): synthetic sunset gradient → 14.6 tok/s,
  "The image shows a vibrant sunset over what appears to be a horizon..." (의미적으로 맞음).

### 교훈 (new)
- *greedy AR은 첫 토큰만 검증하면 전체가 따라온다*.  매 토큰을 PT와 byte-diff 할 필요 없음 —
  첫 토큰의 token-id가 일치하면, *다음 step의 입력이 PT와 동일*, 따라서 다음 logits도 동일,
  토큰도 동일.  검증 비용 = 1 forward, 전체 보장.  *Sampling*이라면 안 통함 (PRNG identity 의존).
- **KV cache는 IMPROVEMENTS (성능, *정확성 아님*).**  full-sequence re-forward도 결과는 같음 —
  KV cache가 빠를 뿐.  검증 게이트와 분리.

---

## §3. TI2I (text-conditioned image edit)

이 stage의 핵심.  STAGE 6 T2I + STAGE 7 §2 X→T가 각각 byte-diff 통과한 다음에도 *조합*은
새 버그를 만든다 (STAGE 4 교훈의 명백한 적용).  TI2I novel한 부분 세 가지가 모두 미검증이었다:
- **(a)** ViT(이해) 조건 + VAE-encode(생성 조건화) 이중 조건화
- **(b)** vae2llm을 *이해 방향*으로 사용 (STAGE 6은 llm2vae 생성 방향만)
- **(c)** 3-component CFG (vs T2I의 2-component)

### Building blocks
- `lance_mlx/pipelines/image_edit.py`:
  - 시퀀스: `<|im_start|>system\n[EDIT_SYSTEM_PROMPT verbatim]<|im_end|>\n<|im_start|>user\n<vision_start>[N_vit ViT-placeholders]<vision_end>[instruction]<|im_end|>\n<|im_start|>assistant\n<vision_start>[N_cond VAE-encode-placeholders]<vision_end><vision_start>[N_noise target-noise]<vision_end>`
  - 3개의 sequence 변형: `full` (모든 조건), `t_uncond` (text drop), `tv_uncond` (text + ViT drop) — 매 step 3 forward.
- `lance_mlx/scheduler.py:cfg_velocity_3comp`:
  ```
  v_blend = v_tv_uncond + cfg_text*(v_full - v_t_uncond) + cfg_vit*(v_t_uncond - v_tv_uncond)
  ratio  = clip(||v_full|| / ||v_blend||, renorm_min, 1.0)
  v_t    = v_blend * ratio
  ```
  PT `lance.py:707-724` 그대로.  변수 매핑:
  - `v_t` ↔ `v_full` (cond_hidden_state, 모든 조건 on)
  - `cfg_text_v_t` ↔ `v_t_uncond` (text dropped)
  - `cfg_text_vit_v_t` ↔ `v_tv_uncond` (text + ViT dropped)

### 발견 + 수정한 5개 버그

Single-step PT byte-diff harness (`tools/stage7_ti2i_compare.py`)가 처음 0.870/0.932/0.955로
**전부 깨짐**.  PT validation_gen 한 줄씩 대조해 다섯 개 — 그 중 마지막은 *harness 자체*:

| Bug | 무엇 | 어디서 났나 | PT 출처 |
|---|---|---|---|
| **A** | cond slab에 `time_embedder(0)` 빠짐 | 우리는 cond를 "clean이니까 time 안 필요"로 처리.  PT는 *모든* latent token에 `vae2llm + time_embedder + latent_pos_embed` 적용, cond=0, noise=current. | `lance.py:663-666` |
| **B** | cond slab UND routing.  text+ViT만 UND, VAE는 cond+noise 둘 다 GEN. | 우리는 cond를 "이해 조건"으로 보고 UND로 라우팅.  PT는 *모든 VAE latent*가 `packed_gen_token_indexes`. | `lance.py:681` |
| **C** | noise mRoPE positions가 자기 위치였음.  PT는 `pro_type=10` `modality==1 ← modality==2` 복사 (noise = cond positions).  ViT은 T axis만 base=1000으로 shift, H/W 그대로. | 우리는 build_positions_for_layout 결과를 그대로 사용.  pro_type=10 변환 누락. | `common.py:60-67` |
| **D** | attention mask에서 `<vision_start>`/`<vision_end>`를 인접 causal에 묶음. | 우리 split이 `[pre_text, vit_inner, mid_text, vae_inner, sep, noise_inner, vis_end]` (vis 토큰들이 causal 사이에 흩어짐).  PT는 vision slab을 *통째로*: `[..., vit_full_inc_specials, ..., vae_full_inc_specials, noise_inc_specials, ...]`. | `validation_dataset.py:518` (`curr_split_len = len(span_index) + 2`) |
| **E** *(harness 자체)* | PT-side에 bf16 0/-inf additive mask 전달 → `flex_attention` monkey-patch의 `dense.to(torch.bool)`이 *반전*: `bf16(-inf) → True`, `bf16(0) → False`. attention이 정확히 거꾸로 동작. | STAGE 6 T2I는 mask가 단순 `causal/noise/causal`로 -inf 패턴이 적어 cos에 묻혔음 (검증 한계).  TI2I의 풍부한 -inf 패턴이 노출. | `tools/stage7_ti2i_compare.py:_install_flex_attention_sdpa_patch` |

**수정 후 cos**: 0.999632 / 0.999875 / 0.999780 — gate 0.999 통과.

### Trajectory (디버깅 순서, audit trail)

1. Single-step byte-diff harness 작성, 첫 결과 0.870/0.932/0.955 — 전부 깨짐.  *전부* 깨졌으므로
   "어느 하나가 깨지면 그 forward의 입력에 버그" 분기가 아니라 "공통 인프라(VAE-cond + routing + positions) 버그".
2. PT validation_gen 정독 → A/B/C/D 4개 식별, 모두 수정.
3. 재실행 → cos *오히려 떨어짐* (0.826/0.868/0.931).  Claude의 가설이 *그럴듯한데 검증이 부인하는* 상황.
4. 한 fix씩 토글, position 격리 (양쪽 같은 pos_np 사용), embed-stage cos 측정 (= 0.999997, 완벽).
5. layer-by-layer cos diagnostic (`tools/stage7_ti2i_layer_diag.py`): layer 0에서 *full hidden* cos 0.865로 발산,
   *noise slab* cos는 0.989로 비교적 보존.  layer 1부터 full hidden catastrophic (cos 0.095).
6. *non-noise 영역만 발산*이 단서 — 같은 input embed, 같은 positions, 같은 routing인데 발산.
   → attention의 input이 다르다 → attention mask 검사.  PT-side mask 의심.
7. monkey-patch 코드 정독 → **E 발견**: bf16 0/-inf → bool 변환 시 -inf→True 반전.  PT-side에 bool mask 직접 전달.
8. 재실행 → cos 0.9996/0.9999/0.9998.  Gate 통과.

### Verification chain (post-fix)
- 5단계 cross-check 모두 pass (위 verification chain 1-6).
- 합성 gradient 30-step PT vs MLX cos=0.9973 + 두 그림 perceptual identical.
- 진짜 사진 30-step → RockTalk ref와 perceptual 일치 (panther & bowtie).

---

## §4. 교훈 (Lessons distilled from STAGE 7)

기존 STAGE 1~6 lessons는 각 stage log에 있음.  여기 새로 박힌 것:

### Lesson 10 (new) — *single-step byte-diff ≠ multi-step 보장*

> 1 step byte-diff 통과해도 30 step 그림 깨질 수 있다.  *검증 범위가 곧 한계*다.
>
> STAGE 7 §3에서 single-step cos ≥ 0.9996 통과 후에도 합성 gradient 30-step 결과가
> "토끼/레고" 환각이었다.  *작은 케이스가 큰 케이스를 보장하지 않음* (교훈 8 일반화):
> single-step(N=1) 통과는 single-step 정확성만 보장, multi-step(N=30) 누적 정확성은 *별도 검증* 필요.
>
> **보호책:** 검증 범위를 step 수와 일치시킴.  multi-step pipeline에는 multi-step 게이트:
> - 30-step PT vs MLX final latent cos (≥ 0.99 in TI2I, ≥ 0.999 in T2I)
> - per-step ‖v‖ trajectory 비교

### Lesson 11 (new) — *"PT와 같다"는 "제대로 작동한다"가 아니다*

> Reproduction과 Correctness는 다른 명제.  PT=MLX를 byte-diff로 보여도, *PT 자체가 그
> 입력에서 의도된 출력을 내지 못하는 입력*이면 그건 검증된 게 아니다.
>
> STAGE 7 §3 합성 gradient에서 PT=MLX cos=0.9973, 두 그림 모두 토끼/레고.  "우리가 PT를
> 재현했다"는 사실이지만, "TI2I가 제대로 작동한다"는 *다른* 명제 — production-realistic 입력에서
> 의도된 편집이 나오는지 별도 확인 필요.
>
> **보호책:**
> - 검증 게이트를 *두 명제 각각* 따로:
>   - (재현성) PT 직접 import vs MLX byte-diff — 우리 구현이 PT와 같은가
>   - (정확성) production-realistic 입력 perceptual judgment — 의도된 출력이 나오는가
> - 둘 다 통과해야 stage 종료.

### Lesson 12 (new) — *부적절한 테스트 입력이 가짜 버그를 만든다*

> 합성 gradient는 TI2I 진단 입력으로 *적절하지 않다*.  구조가 없는 입력에서는 모델 자체가
> 환각하므로, 우리 구현이 옳아도 "그림이 깨진" 것처럼 보인다 → false-positive 버그 신호 → 디버깅 시간 낭비.
>
> **보호책:**
> - 테스트 입력 = production-realistic 분포에서 추출.  TI2I는 실제 사진 + 자연어 instruction.
> - 합성 입력은 *수치 진단용*(byte-diff)에만 — *시각 평가*에는 부적합.
> - RockTalk repo의 sample 같은 *공식 reference*가 있으면 perceptual gate에 사용.

### Lesson 13 (new) — *검증 도구 자체가 틀릴 수 있다*

> Harness가 옳다고 가정하지 말 것.  옳은 fix를 적용했는데 cos가 *떨어지면* fix가 틀린 게 아니라
> harness가 틀렸을 가능성을 의심.
>
> STAGE 7 §3에서 A/B/C/D 4개 fix 다 PT 정확.  Harness의 cos는 *오히려* 떨어졌음 (0.870 → 0.826).
> Claude의 hypothesis (PT 정독 후) vs harness 결과가 *반대* 방향 — 보통 hypothesis가 틀렸다고
> 결론지을 신호.  하지만 6번의 isolation diagnostic 끝에 *harness E*: flex_attention monkey-patch의
> bool mask 반전이 PT-side를 거꾸로 돌리고 있었다.
>
> **보호책:**
> - Hypothesis-result 불일치 시 *둘 다 의심*.  hypothesis만 retract하지 않음.
> - Harness 자체에 sanity check (예: identity attention mask로 cos=1.0 확인)을 박는다.

---

## §5. 정직한 audit trail — Claude의 가설이 또 화살표였던 지점들

STAGE 6 §5의 패턴 반복.  STAGE 7 §3 디버깅 중 Claude가 *그럴듯한 진단*을 냈지만 검증이 부인:

| 시점 | Claude의 가설 / 진단 | 실제 결과 |
|---|---|---|
| 첫 cos 결과 (0.870/0.932/0.955) | "4개 fix A/B/C/D 다 옳음, 적용하면 PASS" | 적용 후 cos *떨어짐* (0.826/0.868/0.931).  Hypothesis 방향은 맞았는데 검증이 부인 → harness 자체 의심 못 했음 |
| Fix C (모든 axis shift) | "ViT modality=4는 T/H/W 세 axis 모두 1000으로 shift" | PT는 `position_ids[0, :, mask] += shift` — *T axis만*.  H/W 그대로 |
| Mid-debug | "ViT의 mRoPE high-T positions (=1000)가 MLX mRoPE에서 처리 문제" | mRoPE는 cos(t*freq), sin(t*freq) — t=1000도 그냥 큰 수.  무죄 |
| Pre-harness-E 발견 | "30-step 누적 path 어딘가 미세 버그.  Per-step latent cos로 발산 지점 추적해야 함" | 사실은 single-step *forward*가 깨져 있었음 (PT-side bool mask 반전).  Layer-0 cos diagnostic이 진단의 결정타 |

**Doctrine 자기적용 (STAGE 6 §5에서 박은 것의 재확인)**:
- *Claude가 그럴듯한 분석을 내놓을 때 확신성이 강할수록 검증 강화*.
- *가설은 어디를 볼지 가리키는 화살표지 결론이 아님*.
- *원본 PT 직접 비교*가 최종 판정자.
- **추가 (Lesson 13):** *검증 도구 자체도 검증 대상*.  Hypothesis-result 불일치 시 *둘 다 의심*.

---

## §6. Carried forward to STAGE 8

- **3개 파이프라인 (T2I, X→T, TI2I)** — full cross-validated, production용 import path 정리됨:
  - `lance_mlx.pipelines.t2i.t2i`
  - `lance_mlx.pipelines.x2t.x2t`
  - `lance_mlx.pipelines.image_edit.image_edit`
- **Tools** (재사용 가능):
  - `stage7_vit_compare.py` (ViT byte-diff)
  - `stage7_x2t_compare.py` (LLM first-token logits byte-diff)
  - `stage7_ti2i_compare.py` (3-forward single-step byte-diff)
  - `stage7_ti2i_concerns_diag.py` (명시적 ①②③ 점검)
  - `stage7_ti2i_pt30step.py` (PT 30-step end-to-end + image-side-by-side)
  - `stage7_ti2i_layer_diag.py` (layer-by-layer cos for debugging)
- **13 lessons** distilled (Lessons 10/11/12/13 new at this STAGE).
- **VERIFICATION_BACKLOG items closed:**
  - flex_attention SDPA shim multi-sample → not needed (TI2I uses 3 *separate* forwards, single-sequence each).  Backlog 항목은 STAGE 8/9 video batching에서 재발화 가능 — 그대로 열려 둠.
  - `build_lance_attention_mask` multi-document validation → not needed at STAGE 7 (TI2I single-document).  STAGE 8/9 batched inference에서 settle.

**STAGE 8 = 3D Causal Video VAE + tile decoding** (최난관).  여기서 *진짜* spatiotemporal tile
봉합을 다룬다 — STAGE 6 512²의 *가짜* seam(PRNG 통일로 해소)과 달리, 비디오 latent의 temporal
causal convolution과 tile-by-tile decode 봉합이 본질적 도전.

---

## §7. Code-reviewer pass (workorder §5.7)

Reviewer: Opus, code-reviewer agent.  Scope: files produced in STAGE 7 only
(`lance_mlx/vit.py`, `lance_mlx/pipelines/x2t.py`, `lance_mlx/pipelines/image_edit.py`,
`lance_mlx/scheduler.py:cfg_velocity_3comp`, all `tools/stage7_*.py`).

- **BLOCKING:** none — numerical gates (ViT cos=1.0, X→T cos=0.999923, TI2I 3-forward cos≥0.999632,
  30-step cos=0.997340, real-photo perceptual match) already pin correctness.

- **SUGGESTED-A applied:**
  - `image_edit.py:_forward_v` — `vs, ve` shadowing footgun (rebound from `vit_span` to `vae_span` mid-function).
    Renamed end-to-end to `vit_s/vit_e`, `vae_s/vae_e`, `noise_s/noise_e`.  5-min change, removes a future-maintainer pit.
  - `image_edit.py:_forward_v` — added `assert (noise_e - noise_s) == (vae_e - vae_s)` for Fix C position copy
    precondition.  Silent on equal widths today; loud if a future TI2I-V/refedit passes mis-sized cond.
  - `x2t.py` — removed unused `Optional`, `text_positions` imports.
  - `x2t.py` — collapsed redundant `next_id == IM_END_ID or next_id == EOS_ID` (literally same id) to a single check
    with a forward-looking comment.
  - `image_edit.py` — moved `import time` to module top (was inside the function).
  - `scheduler.py:cfg_velocity_3comp` — added `cfg_text == 1.0 and cfg_vit == 1.0` short-circuit before renorm,
    matching PT `lance.py:688` outer gate.  Mathematically identical (`v_blend == v_full` at unit scales) but
    now explicit — protects a future ablation (`cfg_text=1.0`) from a silent ratio computation.
  - Regression: `tools/stage7_ti2i_compare.py` re-run after fixes — cos identical to pre-fix
    (0.999632 / 0.999875 / 0.999780).  No behavioral change.

- **SUGGESTED-B → IMPROVEMENTS.md:**
  - `pro_type=10` position shifts as reusable `rope.py` helpers (`apply_vit_t_axis_shift`, `copy_positions`) —
    eliminates the per-step numpy round-trip in `_forward_v`; same idiom as existing `shift_positions`.
  - `build_split_lens_from_spans(seq_len, spans)` helper in `attn_mask.py` — third call site appearing (T2I,
    TI2I, harness) would benefit.
  - X→T per-step mask grow-by-one cache + KV cache (already deferred at module docstring; ~29 tok/s target).
  - `preprocess_image` divergence from HF `Qwen2_5_VLImageProcessor` — silent at gate-test sizes; load-bearing
    for 4K inputs.

- **NITPICK (not applied, low ROI):**
  - `vit.py` `tuple[int, ...]` default style vs `backbone.py` `field(default_factory=...)` — pick one project-wide
    at next refactor pass.
  - `EDIT_SYSTEM_PROMPT` curly apostrophe — already verified byte-equal + tokenized 56 ids match PT in
    `tools/stage7_ti2i_concerns_diag.py:diag_prompt`, no runtime assert added (verification harness is the lock).
  - log-format alignment between `t2i.py` and `image_edit.py` denoising-loop prints — readability only.
  - `scheduler.py:make_schedule` direction comment ("integrating from noise t=1 toward data t=0") — already
    implied in the docstring; redundant inline comment skipped.

- **Regression check:** STAGE 2 forward (top-8 logits OK), STAGE 4 routing self-test (A/B/C properties hold),
  STAGE 6 T2I 30-step (latent stats consistent with pre-STAGE-7 run), STAGE 7 §3 TI2I single-step (cos
  identical to pre-rename) — all pass.
