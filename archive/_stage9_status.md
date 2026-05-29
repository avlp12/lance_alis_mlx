# STAGE 9 진행 보고서 — Video DiT + t2v

**작성:** 2026-05-26 14:30 KST
**상태:** §1 진입 직전 — production 시퀀스 빌드 점근 결정 대기

---

## 완료된 검증 ✓

### §0 게이트 PASS
- cos(v_t_pt, v_t_mlx) = **0.999918** (text_template=False 단순 케이스)
- A.2 단계 1 PT validation_dataset 코드 그대로 사용 (manual 제거)
- ★ v1/v2 byte-identical 교차비교로 **Lesson E 재발화 발견 + 수정** (`patched_flex_attention` 의 `dense.to(bool)` polarity 반전)

### 단계 3 — 공용 헬퍼 `tools/_pt_smoke_common.py` ✓
- `install_pt_smoke_env()` — env shim 통합
- `pt_layer_mask(dense_bool)` — **`assert dtype == torch.bool`** Lesson E 영구 차단
- 단계 4 + v2 PT smoke + MLX harness 헬퍼 사용 리팩터 → 모든 통과 유지 (cos=0.999918)

### 단계 4 — full positions byte-diff ✓
- PT Lance 자체 `Qwen2ForCausalLM.get_rope_index` 직접 호출 (transformers 5.9.0 아님 — 시그너처 다름)
- self spoof (`MockConfig`)
- 결과 (text_template=False 시퀀스 기준, apply_qwen=True):
  ```
  text [0..10]:        (0..10, 0..10, 0..10)
  vis_start (idx 11):  (11, 11, 11)
  IMG[0]    (idx 12):  (12, 12, 12)
  IMG[last] (idx 139): (12, 19, 19)   ← t=12 const (image case, second_per_grid_t=0)
  vis_end   (idx 140): (20, 20, 20)
  ```
- MLX `build_t2v_positions` ~15 줄, PT 정답지 **byte-identical (0 diff)** ✓
- shift_position_ids (pro_type=10) = **no-op for t2v**

### 단계 4-2 — uncond positions + uncond mask byte-diff ✓
- uncond_mask = `modality != 0` → text(0) drop, noise(1) keep
- uncond_text_ids = `[vis_start, IMG×128, vis_end]` (130 tokens)
- uncond split-level filter: text split drop, noise split keep → `[130], ["noise"]`
- **uncond positions:**
  ```
  vis_start (idx 0):   (0, 0, 0)
  IMG[0]    (idx 1):   (1, 1, 1)   ← text_len=1 offset
  IMG[127]  (idx 128): (1, 8, 8)
  vis_end   (idx 129): (9, 9, 9)
  ```
- MLX `build_t2v_uncond_positions` byte-identical ✓
- **uncond attn_mask** = all-True (16900 = 130×130) — noise self-attention. MLX `build_lance_attention_mask(130, [130], ["noise"])` byte-identical ✓

### 검증된 컴포넌트 매트릭스 (text_template=False 시퀀스 기준)

| 영역 | MLX 함수 | 정답지 | 검증 |
|---|---|---|---|
| Full positions | `build_t2v_positions` | 단계 4 | ✓ 0 diff |
| Uncond positions | `build_t2v_uncond_positions` | 단계 4-2 | ✓ 0 diff |
| Full attn_mask | `build_lance_attention_mask(L, [text,vid], ["causal","noise"])` | PT predicate | ✓ STAGE 4 |
| Uncond attn_mask | `build_lance_attention_mask(130, [130], ["noise"])` | PT predicate (all-True) | ✓ 0 diff |
| VAE latent pos ids | `coords_t·M² + coords_h·M + coords_w` | `get_flattened_position_ids_extrapolate_video` | 동일 산술 |
| CFG (2-comp + interval + global renorm) | `v=v_unc+cfg*(v_full-v_unc); scale=clamp(\|v_full\|/\|v\|,min,1)` | PT lance.py:707-724 | ✓ PT 정독 |
| Schedule | `make_schedule(steps=30, shift=3.5)` | STAGE 6 | ✓ |
| VAE decode T_lat>1 | `Wan2_2_VAE.decode` (chunked) | STAGE 8 | ✓ cos=1.0 |
| Unpatchify (DiT→VAE) | `vae_wan22.unpatchify`, pt=ph=pw=1 | STAGE 8 | ✓ |

---

## 사용자 결정 대기 — text_template=True 시퀀스 점근

### 발견 + 정리

**상호작용 (`text_template` × `apply_qwen_2_5_vl_pos_emb`):**

| 조합 | packed_position_ids 출처 |
|---|---|
| `text_template=*`, `apply_qwen=False` | `process_text_template` 의 1D running counter |
| `text_template=*`, **`apply_qwen=True` (production)** | **`get_rope_index` 호출 (text_template 결과 무시)** |

→ **production (둘 다 True) 에선 positions 가 `input_ids` 만의 함수**. text_template 의 packed_position_ids 무시.

**MLX builder 의 자동 적응:**
`build_t2v_positions(text_split_len, t_lat, h_lat, w_lat, L)` 가 *시퀀스 길이만* 의존. text_template=True 면 text_split_len 이 더 길어지지만 builder logic 변경 불필요 — 새 값만 인자로.

**추가 검증 필요한 단일 항목: `input_ids` (시퀀스 자체)**
1. text_template=True 의 chat template 적용 후 PT 시퀀스 정답지 dump
2. MLX 측 시퀀스 build → byte-diff (manual interpretation 위험 차단)
3. 통과하면 PT `get_rope_index` 재호출 → 새 positions 정답지
4. MLX builder (`text_split_len_new`) 호출 → byte-diff
5. uncond 도 동일 (text drop 후 시퀀스 — 동일)

### `process_text_template` 분리 어려움 — 세 옵션

PT `process_text_template` 가 `self.sample` instance state 강하게 의존. 직접 함수 호출 안 됨.

**옵션 A (권장)** — ValidationDataset minimal instance build + `t2v_sample(idx=0)` 호출
- PT 코드 **100% 직접 사용**. 우리 해석 0
- dependencies: `DataConfig`, dummy `jsonl_path`, tokenizer + `new_token_ids`
- 작업 크기: dependencies build 중간. 결과 fixture 한 번 저장 → MLX 측 byte-diff
- doctrine 충실 (STAGE 1~8 패턴)

**옵션 B** — `tokenizer.apply_chat_template` (HF Qwen2.5) + PT 출력과 byte-diff 검증
- HF Qwen2.5 chat template 이 PT `render_qwenvl_prompt` 결과와 byte-identical 가능성 (Qwen2.5 표준)
- 구현 ~5 줄
- 위험: byte-diff 안 통과면 manual interpretation. *반드시 byte-diff 검증 후 사용*
- 빠르지만 PT 메소드 자체 호출 아님

**옵션 C** — `process_text_template` 코드 body 추출 (self.sample → local dict)
- PT 코드 *논리는 그대로*, instance state 제거 변형
- self_state-free 함수로 재작성. PT 의 buffer layout 따라 감
- 중간 크기 작업. 결과 PT byte-diff 필수
- *부분적 manual interpretation* — A 보다 위험

### 권고: 옵션 A
- doctrine 충실 (PT 자체 호출, 해석 0)
- dependencies build 가 일회성 작업
- 결과 fixture 저장 후 단계 2 v3 production 정답지 + t2v.py 둘 다 활용
- 옵션 B 는 byte-diff 통과 시 추가 fast path 로 적용 가능 (사후 검증)

---

## 진행 plan (사용자 결정 후)

1. **시퀀스 정답지 단계** (옵션 결정 후):
   - PT 시퀀스 정답지 dump (`out/stage9_pt_t2v_seq_real.npy`, text_template=True production)
   - MLX 시퀀스 builder → byte-diff
2. **새 positions 정답지 (production 기준):**
   - PT `get_rope_index(new_input_ids, ...)` → 새 full positions
   - PT `get_rope_index(uncond_input_ids, ...)` → 새 uncond positions
   - MLX `build_t2v_positions(new_text_split_len, ...)` byte-diff
   - MLX `build_t2v_uncond_positions` byte-diff
3. **단계 1** — `lance_mlx/pipelines/t2v.py` 본격 작성 (검증된 컴포넌트 조립)
4. **단계 2** — `tools/stage9_pt_video_dit_smoke_v3.py` (production 정답지: full + uncond v_t intercept, 헬퍼 사용)
5. **단계 5** — single-step PT byte-diff:
   - `cos(v_full_pt, v_full_mlx) ≥ 0.999`
   - `cos(v_unc_pt, v_unc_mlx) ≥ 0.999`
   - `cos(v_blend_pt, v_blend_mlx) ≥ 0.999`
6. **단계 6** — 30-step + 첫 영상 + per-step PT cos diagnostic (교훈 11 video 버전)

---

## 안전망 (사용자 명시 교훈, 매 단계 유지)

- **PRNG numpy 통일** (교훈 9) — `np.random.default_rng(seed).standard_normal(...)`, `mx.random` 금지
- **step별 PT cos 전 통과선언 금지** (교훈 11 video 버전) — 첫 영상 나와도
- **§0 단순 케이스 통과 ≠ production 통과** (교훈 8) — 매 단계 production 케이스로 재검증

---

## TaskList 현재 상태

```
#22 [completed] 단계 3: tools/_pt_smoke_common.py 공용 헬퍼
#21 [completed] 단계 4: get_rope_index_video 단독 byte-diff
#27 [completed] 단계 4-2: uncond positions + uncond attn_mask byte-diff
#23 [in_progress] 단계 1: lance_mlx/pipelines/t2v.py (production)
#24 [pending]   단계 2: stage9_pt_video_dit_smoke_v3 (production 정답지)
#25 [pending]   단계 5: single-step PT byte-diff (v_full/v_unc 각각 + blend)
#26 [pending]   단계 6: 30-step + 첫 영상 + per-step PT cos
[새]           text_template=True 시퀀스 정답지 + production positions 재검증
```

다음 결정: text_template=True 시퀀스 빌드 점근 (옵션 A / B / C). 결정 후 본격 진입.
