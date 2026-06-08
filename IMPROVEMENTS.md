# IMPROVEMENTS — backlog of B-class deviations from RockTalk reference

Workorder §4.5 channel.  Anything that would change *behavior, numerical
output, or performance* compared to the RockTalk MLX port lands here as
a discovery, NOT in the porting code.  Apply only after the owning
STAGE passes parity, and only with baseline-vs-after measurements.

Format:

```
## [STAGE N] Title
- 발견: …
- 분류: 성능 / 메모리 / 수치정확도 / 구조
- 리스크: …
- 상태: 기록됨 / 검토중 / 적용됨(measurement) / 기각(reason)
```

---

## [STAGE 3] `shift_positions` more idiomatic via `mx.where`
- **발견:** Current impl builds a zero-buffer + mask + multiply + add (4
  intermediate tensors).  More idiomatic and slightly cheaper:
  `mx.where(mask[None, None, :], positions + shift, positions)`.  Same
  semantics, no `zeros_like` allocation, no `mx.array(shift, dtype=...)`
  wrap (MLX broadcasts Python ints).
- **분류:** 성능 / 가독성
- **리스크:** None — identical numerics; trivial to verify.
- **상태:** 기록됨.  Apply when STAGE 6/7 begins calling `shift_positions`
  in tight loops, with baseline-vs-after timing.

## [STAGE 3] AR-decode per-step position emission for STAGE 7 X→T
- **발견:** `build_positions_for_layout` is appropriate for prefill but
  shouldn't be called per decode step — at AR time we only need
  `mx.array([[[c]],[[c]],[[c]]])` where `c = text_cursor` increments per
  generated token.  STAGE 7 X→T should track the cursor itself rather
  than rebuilding the prefix.
- **분류:** 성능 (avoid recomputation)
- **리스크:** None — STAGE 3 helper is correct for prefill; STAGE 7 just
  needs a different helper for the AR loop.
- **상태:** 기록됨.  Apply when implementing STAGE 7 X→T decode loop.

## [STAGE 4] Defensive `gen_mask.any()` short-circuit in routing
- **발견:** `LanceAttention/Layer/Model.__call__`의 `routed` 판정은
  `gen_mask is not None`만 본다.  Caller가 *all-zero* `gen_mask`를 *defensive*하게
  넘기면 (예: T2I 파이프라인이 GEN slab 없는 단계에서) 우리는 양 branch 다
  계산하고 머지 — 2× projection FLOPs 무료 발생.  방어책:
  `routed = self._has_moe_gen and gen_mask is not None and bool(gen_mask.any().item())`.
  단점: `.item()` sync 비용 (decode 마다 호출 시 누적).
- **분류:** 성능 (방어적 caller 흐름)
- **리스크:** None — semantics 동일.  단 `.item()` sync가 decode loop hot path에
  들어가면 오히려 더 느려질 수 있어 측정 필요.
- **상태:** 기록됨.  Apply when 실제 caller가 all-zero gen_mask를 넘기는
  코드 경로가 등장하면 (STAGE 6/7 파이프라인).

## [STAGE 4] Tier 1 (D)를 sdpa 이전 단계에서 단정 가능
- **발견:** Tier 1 (D)는 "GEN-slab routing이 attention context와 *독립*"이라는
  가설을 forward 끝단에서 단정하려 했다 — attention global mixing 때문에
  실패가 *예상됨*.  진짜 단정 가능 위치는 *sdpa 이전*: GEN 토큰의 Q는
  routed/all-True 양쪽 모두 `q_proj_moe_gen(x)` → 동일.  layer 내부에
  hook을 걸어 pre-sdpa Q를 dump하면 단정으로 승격 가능.
- **분류:** 검증 (수치정확도 — 측정도구)
- **리스크:** None — 더 엄밀한 단정 추가, 기존 결과 그대로.
- **상태:** 기록됨.  Apply: STAGE 4 verification re-visit 또는 회귀 디버깅 시점.

## [STAGE 4] `build_gen_mask` 다수 span 벡터화
- **발견:** 현재 Python `for span in layout.gen_spans` 루프로 `mask | slab`
  반복.  spans=1~3은 무시 가능, 그러나 1k+ 스팬(예: multi-image-edit 시퀀스)에서
  O(n_spans · L).  벡터화: `mx.array(starts), mx.array(ends)` broadcast 한 번.
- **분류:** 성능
- **리스크:** None — 결과 동일.
- **상태:** 기록됨.  Apply when 실제 sequence가 spans>10이 되면 (현재 단계에선
  필요 없음).

## [STAGE 4] PT shim의 bf16 mixed-precision 경로 미커버
- **발견:** `stage4_pt_cosine.py`는 fp32 end-to-end.  PT Lance 실제 운용은
  bf16 with q/k upcast for moe_gen qk_norm — refs/Lance qwen2_navit.py:418-424.
  우리 cos=1.0은 *알고리즘 shape* parity만 입증하고 bf16 fidelity는 미검증.
- **분류:** 수치정확도 (검증 강화)
- **리스크:** Low — Lance MLX는 STAGE 1에서 f32로 변환했으므로 inference도 f32 운용
  예정.  bf16 운용을 도입할 때(혹시 STAGE 9 video 메모리 압박) 검증 필요.
- **상태:** 기록됨.  Apply when bf16 inference 모드 도입 시.

## [STAGE 5] `CausalConv3d.__init__` random weight init
- **발견:** Currently uses `mx.random.uniform(-scale, scale, …)` for the
  initial weight tensor.  Fine when load_weights overwrites everything,
  but: (a) consumes the global RNG stream every model build (downstream
  `mx.random` becomes build-order-dependent), and (b) silently masks any
  missed key if a caller later uses `strict=False`.
- **분류:** 안전성 / 결정성
- **리스크:** Low — STAGE 5–8 callers all use `strict=True`.
- **상태:** 기록됨.  Apply by replacing with `mx.zeros((O, kT, kH, kW, I))`
  when next touching CausalConv3d (no behaviour change after load).

## [STAGE 5] `stage5_pt_compare.py` `.replace` chain refactor
- **발견:** Module name remapping is a 30-line chain of literal
  `.replace(...)` calls.  Works but brittle and grep-unfriendly.  Cleaner:
  loop `for prefix in ("downsamples.0", …): for src,dst in [("norm1",
  "residual.0"), …]: k = k.replace(f"{prefix}.{src}", f"{prefix}.{dst}")`.
- **분류:** 가독성
- **리스크:** None — pure refactor, same string output.
- **상태:** 기록됨.  Apply when STAGE 8 extends the PT shim (more layer
  prefixes to map).

## [STAGE 5] `Resample.upsample2d` double-repeat materialises 2× intermediate
- **발견:** `mx.repeat(x, 2, axis=2)` then `mx.repeat(x, 2, axis=3)`
  builds an `(B, T, 2H, W, C)` tensor before the second repeat reaches
  `(B, T, 2H, 2W, C)`.  Memory waste of 2× the intermediate at the
  largest stage.
- **분류:** 메모리
- **리스크:** None — same numerics.
- **상태:** 기록됨.  Apply if profiling at STAGE 7/9 shows peak-mem
  pressure (M3 Ultra 512 GB makes this unlikely in STAGE 5).

## [STAGE 6] Skip uncond forward at cfg_scale ≤ 1.0
- **발견:** `lance_mlx/pipelines/t2i.py`의 per-step loop이 `cfg_scale` 값과
  무관하게 *항상* uncond forward를 한 번 더 돌린다.  PT는
  `if cfg_text_scale_ > 1.0:` 가드 안에서만 uncond forward를 실행 — scale ≤ 1.0
  이면 그 절반 forward 비용을 절약.
- **분류:** 성능
- **리스크:** None — semantics 동일.  단 cfg_scale 동적으로 변동되는 경우
  (cfg_interval 활용) 분기 추가 필요.
- **상태:** 기록됨.  Apply when t2i.py가 dynamic cfg_interval을 도입할 때
  (현재 STAGE 6 디폴트는 전 step CFG=4.0이라 항상 uncond 필요).

## [STAGE 6] Batch cond + uncond into one forward (B=2)
- **발견:** Per-step loop이 cond / uncond를 두 별도 forward로 부른다.
  CFG=4.0의 30 step → 60 forward.  cond/uncond를 단일 (B=2) forward로 묶으면
  ~2× speed-up 가능 (MLX kernel launch overhead 제거).  단 mask + position이
  세트별로 다르니 batched 패딩 필요.
- **분류:** 성능
- **리스크:** mask broadcasting 정확성.  cond와 uncond는 L이 다른 시퀀스 (L_cond=1035,
  L_unc=1028), 패딩 + attention mask로 *각 sample만* 자기 자리 보게 해야.
- **상태:** 기록됨.  STAGE 7 TI2I 진입 시 cond/uncond/ti2i 3개 forward이라 더 의미.

## [STAGE 2] Richer ablation metric than logit-cosine
- **발견:** `stage2_cosine.py` part (B) uses logit cosine to measure
  qk_norm contribution.  With vocab=151936 and a long tail, cosine
  collapses to near-1.0 across most prompts even when argmax flips —
  cosine under-reports how disruptive an ablation actually is.  The
  `same_top1` column carried the real signal.
- **분류:** 수치정확도 (measurement)
- **리스크:** None — purely a tool quality issue; existing measurements
  are valid (mean cos 0.50 is unambiguously poor here), just imprecise.
- **상태:** 기록됨.  Apply: add `top5_jaccard` and `KL(on||off)` columns
  to the (B)-style ablation in any future STAGE that re-runs the same
  probe.  Don't retroactively patch STAGE 2; just enrich at next use.

## [STAGE 7] `pro_type=10` position shifts as reusable `rope.py` helpers
- **발견:** `lance_mlx/pipelines/image_edit.py:_forward_v` does `pos_np = np.asarray(pos)` → mutate → `mx.array(pos_np)` for the pro_type=10 shifts (ViT T-axis to 1000, noise pos ← cond pos).  Three round-trips per CFG forward × per step.  Cost is trivial in absolute terms but it's the *only* numpy bounce in the per-step path, and the same logic will be needed for STAGE 8/9 video (TI2V, refedit).
- **분류:** 구조 (코드 재사용) — 성능 영향은 작음.
- **리스크:** None.  Need to express `pos[0, :, vit_s:vit_e] += shift` and `pos[:, :, noise_s:noise_e] = pos[:, :, vae_s:vae_e]` purely in MLX (`mx.where` + `mx.concatenate`).  Same idiom as `rope.shift_positions` at `lance_mlx/rope.py:186`.
- **상태:** 기록됨.  Helpers: `apply_vit_t_axis_shift(pos, vit_span, base=1000)`, `copy_positions(pos, src_span, dst_span)`.  Apply at STAGE 8 video entry when same pattern recurs.

## [STAGE 7] Split-builder helper from explicit slab boundaries
- **발견:** `image_edit.py:_forward_v` builds `split_lens / attn_modes` from `(vit_span, vae_span, noise_span)` via 8 lines of arithmetic (`sl_pre_vit / sl_vit / sl_mid / sl_vae / sl_noise / sl_tail`).  `t2i.py:_build_t2i_sequence` does the same shape with 2-slab text+noise.  `tools/stage7_ti2i_compare.py:build_sequences` does it a third time.
- **분류:** 구조 (코드 재사용).
- **리스크:** None.  `build_split_lens_from_spans(seq_len, spans=[(start, end, mode), ...])` in `lance_mlx/attn_mask.py` would shrink each call site from ~10 lines to ~2.
- **상태:** 기록됨.  Apply when a 4th call site appears (STAGE 8 video TI2V or refedit will probably need it).

## [STAGE 7] X→T per-step mask grow-by-one + KV cache
- **발견:** `lance_mlx/pipelines/x2t.py:x2t` rebuilds the full `(L_new, L_new)` attention mask in numpy every token.  Two layered wins:
  1. Mask is monotone: each step adds one row + one column of causal=True.  A grow-by-one cache cuts mask cost ~30× over a 30-token generation.
  2. KV cache (already deferred at module docstring): each step today re-forwards 36 layers over the full prefix.  Target ~29 tok/s (RockTalk reported speed) needs KV cache.
- **분류:** 성능 (정확성 영향 없음 — 1과 2 모두 동일 결과 보장).
- **리스크:** None numeric.  KV cache adds state, but `LanceAttention` already has a `cache` arg in its `__call__` signature (currently unused at inference).  Validation: first-token logits should still match PT (cos≥0.999).
- **상태:** 기록됨.  Apply at STAGE 7§2b reviewer pass already noted.  Backlog stays open.

## [STAGE 7] `preprocess_image` divergence from HF `Qwen2_5_VLImageProcessor`
- **발견:** `lance_mlx/pipelines/x2t.py:preprocess_image` handcrafts the resize budget (floor-then-fit) vs HF's iterative round-to-multiple-of-`step`.  Honors `max_pixels` but clamps `min_pixels` only loosely.  Test image (224²/512²) is well in-budget so STAGE 7 gates passed; a 4K input would silently produce a different patch count than PT.
- **분류:** 수치정확도 (boundary inputs).
- **리스크:** Low for current scope (test images in-budget).  Either delegate to `transformers.Qwen2_5_VLImageProcessor` (one extra dep call, eliminates discrepancy) or add explicit `assert H_target * W_target >= min_pixels`.
- **상태:** 기록됨.  Apply if X→T is extended to large/small images, or before any "production-realistic input" benchmark.
  ★ 별개 버그 주의: 이 항목은 *resize budget*(max/min_pixels) 건이다.  실제 release-blocking 이었던 것은
  *patch-token 순서*(raster vs 2×2 merge-grouped) — 아래 STAGE 11 항목 참조.

## [STAGE 8] `Up_ResidualBlock` / `Decoder3d` first_chunk default mismatch with PT
- **발견:** STAGE 8 §1.5 wiring 중.  PT `Up_ResidualBlock.forward(..., first_chunk=False)` and `Decoder3d.forward(..., first_chunk=False)` default to **False**.  Our MLX `Up_ResidualBlock.__call__` / `Decoder3d.__call__` default to **True** (set during STAGE 5 image path convenience: single T=1 call always has first_chunk=True).
- **분류:** 수치정확도 (잠재 silent bug).
- **리스크:** Today not load-bearing — STAGE 5 image callers pass `first_chunk=True` explicitly (`Wan2_2_VAE.decode` line 825), STAGE 8 standalone tests pass it explicitly too.  But the *default* divergence is a landmine: a future caller that omits the kwarg (e.g., a STAGE 5 path that gets re-entered, or a copy-paste from PT code) would get DIFFERENT semantics on the two sides — DupUp3D would drop frames on MLX side but not on PT.  Silent because shapes still align in some configurations.
- **상태:** 기록됨.  Either flip MLX defaults to False (match PT, requires audit of STAGE 5 image-path callers and explicit `first_chunk=True` at every call site) or leave the divergence and add a `__call__` runtime warning when the default fires.  Decide at STAGE 5 image-path re-entry or before any STAGE 9 video changes.

## [STAGE 9] flex_attention shim의 bool 변환 (Lesson E) 공용 헬퍼화
- **발견:** STAGE 9 §0 에서 STAGE 7 §3 의 Lesson E (`patched_flex_attention` 의 `dense.to(torch.bool)` 가 bf16 0/-inf 의 polarity 반전) 가 *재발화*.  STAGE 7 ti2i_compare 는 mask 를 *bool dense* 로 layer 에 전달해 회피 (`attn_mask_LL_pt = dense_bool` line 759).  STAGE 9 PT smoke 는 *bf16 additive* 를 직접 전달 → patched_flex_attention 의 `dense.to(torch.bool)` 가 -inf 를 True (truthy) 로 변환 → attention 패턴 반전 → 모든 입력에 대해 같은 v_t 출력 (sequence/positions/mask 영향 무관).  두 시간 디버깅 후 `attention_mask=attn_dense_bool` 직접 전달로 fix.
- **분류:** 수치정확도 (silent bug; 코드 재사용 시 매번 재발화 위험).
- **리스크:** PT smoke 도구를 *새로 작성할 때마다* 같은 lesson 재발화 가능성.  STAGE 7 의 fix 는 그 harness 한정 → STAGE 9 PT smoke 가 *같은 shim 코드* 사용함에도 *mask 전달 방식* 만 다르게 해서 silent 발화.  미래 STAGE 9+ (video_edit, x2t_video, t2v fine-tune) PT smoke 작성 시 동일 재발화 위험.
- **상태:** 기록됨.  공용 헬퍼 작성 후보:
  ```python
  # tools/_pt_smoke_common.py (가칭)
  def install_pt_smoke_env() -> None:
      """모든 STAGE 9+ PT smoke 가 호출하는 단일 환경 셋업 헬퍼.

      - flash_attn stub (single-sequence shim)
      - modeling.lance namespace stub
      - flex_attention SDPA patch  ← Lesson E 처리 위치 통일
      - transformers utils flash-attn availability disable

      Lesson E 처리 contract: layer 에 mask 를 전달할 때는 *반드시* bool dense
      (`attn_dense_bool` 형식). additive (bf16 0/-inf) 전달 금지 — silent
      polarity 반전.  공용 헬퍼 안에 `def pt_layer_mask(dense_bool) -> torch.Tensor`
      assertion 추가 (`assert dtype == bool`).
      ```
  STAGE 7 ti2i_compare + STAGE 9 PT smoke v2 가 이 헬퍼 import 하도록 리팩토링.
  Apply: STAGE 9 §1 t2v 단계 2 production 정답지 도구 작성 시점.
  **상태 갱신 (STAGE 9 종료):** `tools/_pt_smoke_common.py` 작성 완료, `assert` → `raise TypeError/RuntimeError` (reviewer BLOCKING D fix). STAGE 7 ti2i_compare 는 자체 inline 환경 셋업 사용 중 (STAGE 9 도구만 헬퍼 채택) — 미래 STAGE 10+ PT smoke 추가 시 STAGE 7 도 헬퍼로 리팩터.

## [STAGE 9] `Wan2_2_VAE.decode` 의 production scale 을 클래스 default 로
- **발견:** STAGE 9 §1 단계 6 closing 에서 발견 — `vae.decode(latent)` (scale 인자 생략) 호출 시 *identity scale* 적용 → video dynamic range 1.5× 발산 (cos 0.948 FAIL, latent cos 0.999437 통과인데도). t2v.py 에 `VAE_SCALE_MEAN/STD = [...]` 하드코딩 + 호출 site 에서 명시 전달로 회피.
- **분류:** 수치정확도 (silent dynamic range 발산) + 구조 (PT 의 `Wan2_2_VAE` 인스턴스 attr 을 MLX 가 pipeline layer 에 duplicate).
- **리스크:** 미래 video pipeline (`video_edit`, `tv2v`, etc.) 가 `vae.decode(latent)` 호출 시 동일 silent 발산. *identity scale 이 default* 인 contract 자체가 함정.
- **상태:** 기록됨.  Apply 후보:
  1. `Wan2_2_VAE` 에 `default_video_scale` (classmethod 또는 attr) 추가, `VAE_SCALE_MEAN/std` 옮김.
  2. `decode(scale="identity"|"video"|tuple)` 형태로 변경, T_lat>1 + "identity" 조합 시 raise.
  Apply: STAGE 10+ video pipeline (video_edit, tv2v) 도입 시점.  당장은 t2v.py 의 하드코딩 fix 로 동작 OK.

## [STAGE 9] `_forward_v` 의 contiguous-span 가정 (TI2I video-edit 위험)
- **발견:** `lance_mlx/pipelines/t2v.py:_forward_v` 의 vae_token_indices 가 *single contiguous span* 가정 (`assert np.array_equal(vi, arange(...))`).  현재 t2v full + uncond 모두 single noise slab 이라 OK.  미래 video_edit 의 *cond slab + noise slab* (TI2I 의 video 변형) 에서는 *두 contiguous span* — assertion 으로 알려서 silent 발화 차단.
- **분류:** 구조 (미래 case 차단).
- **리스크:** 미래 video_edit pipeline 작성자가 assert message "must be contiguous" 만 보고 *re-sort* 로 우회 시도 → 잘못된 위치 splice → silent.
- **상태:** 기록됨.  Apply 후보: assertion message 에 *실패 모드* 명시 + image_edit.py 의 multi-slab scatter 패턴 pointer 추가.  Apply: video_edit pipeline 작성 시점.

## [STAGE 11] ViT patch-token order — raster → 2×2 merge-grouped (★ release-blocking, FIXED)
- **발견:** `preprocess_image`/`_patchify_frames`가 ViT 패치를 plain raster (T,H,W)로 냈다.
  PT `data_utils.patchify_video_with_merge` + mlx-vlm `VisionModel`은 2×2 merge-grouped를 기대
  (연속 4토큰=한 2×2 spatial-merge 블록).  채널순서 동일 — 순수 토큰순서 버그.
- **분류:** 수치정확도 (release-blocking; x2t / image_edit / x2t_video ViT-cond 전부).
- **리스크:** parity 깸 → *고침*.  t2i/t2v 무관(ViT 미사용).  weight 무관(전처리 버그).
- **상태:** **적용됨(measurement).**  `_patchify_frames` merge-grouped permute
  (THWC `transpose(0,2,5,3,6,8,1,4,7)`), `preprocess_image`는 위임.  측정: PT-real 대비
  raster cos 0.29(image)/0.36(video) → merge-grouped cos 1.000000.  비-장님 재검증
  `tools/stage11_x2t_verify.py`: K=8 top-1 8/8, min cos 0.999124(image)/0.999437(video),
  raster 대조군 min 0.553/0.968 (discriminative).

## [STAGE 11] x2t_video temporal mRoPE multiplier (×tokens_per_second=2) (FIXED)
- **발견:** PT `get_rope_index`(qwen2_navit.py:1258) video 분기는 시간축에 ×`tokens_per_second`=2.
  우리 `_image_position_block`(rope.py)는 unit step.  T>1(video)만 영향, T=1(image) 면역.
  우리 t2v.py는 승수가 맞게 있었다 — x2t 경로만 누락.
- **분류:** 수치정확도 (video-only).
- **리스크:** parity 깸 → *고침*.  scale=1 신·구 byte-identical(무회귀).
- **상태:** **적용됨(measurement).**  `rope.py` `VisionSpec.temporal_scale`(default 1),
  `x2t.py` `VIDEO_TEMPORAL_SCALE=2`.  측정: 우리 위치 vs PT `get_rope_index` byte-identical
  (`stage11_x2t_video_positions_compare.py`); top-1 'Nothing'→'In'(버그가 실제 출력 변경, 수정 후
  PT 진짜 출력과 일치).

## [STAGE 11] 검증 harness 클래스 결함 — PT-direct-import 에 우리 중간결과 주입 = 장님 (방법론)
- **발견:** STAGE 7 `stage7_x2t_compare.py`(:240-242,265)는 우리 `preprocess_image` 패치를 PT ViT에,
  `stage7_ti2i_compare.py`(:685)는 우리 ViT 출력을 PT에 통째 주입.  "PT 원본 import" 인데 PT가 원시
  입력에서 다시 계산하지 않음 → 양쪽이 같은 오해 공유 → cos≈1.0 합의로 전처리/위치 버그 은폐.
- **분류:** 검증 (방법론 결함 — 모든 multimodal 게이트에 잠재).
- **리스크:** 미래 게이트가 같은 패턴 재현 시 silent.
- **상태:** **교정됨(방법론).**  비-장님 계약: PT는 *원시 입력*에서 patches·positions를 자기 코드로
  산출 + byte-assert 후 forward.  `stage11_x2t_verify.py`가 이 계약 구현.  공개문서 doctrine 보정.

## [STAGE 11] 교훈 — 레퍼런스가 틀리면 두 쪽이 같이 틀린다 (Lesson 25)
- **발견:** STAGE 3이 build_positions 를 mlx-vlm `get_rope_index`와 byte-검증했으나 mlx-vlm 도 video
  `tokens_per_second` 승수 드롭 → 우리 unit-step 과 일치해 ②를 못 봄.
- **분류:** 검증 (레퍼런스 선택).
- **리스크:** "검증된 라이브러리"가 같은 누락을 공유하면 parity 가 거짓 안심.
- **상태:** 기록됨(교훈).  진짜 truth = PT-Lance `get_rope_index`.  LEARNING_LOG stage_7 §8 + 공개문서.
