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
