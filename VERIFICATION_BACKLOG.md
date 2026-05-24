# VERIFICATION BACKLOG — deferred checks owed to later STAGEs

Items here are not improvements (those go to `IMPROVEMENTS.md`).  They are
*verifications we explicitly postponed* during a STAGE — the STAGE passed
its own criterion through a different leg, but a deeper or more direct
check is owed at a later STAGE where the harness is needed anyway.

Settling an item means: running the check, recording the result in the
owning STAGE's log, and crossing it off here.  An unsettled item at the
end of the porting run is a known unknown.

Format:

```
## [opened STAGE N → settle by STAGE M] Title
- 무엇을 미뤘나: …
- 왜 미뤘나: …
- 어떻게 갚나: …
- 통과 기준: …
- 상태: 열림 / 진행중 / 통과 / 기각(사유)
```

---

## [opened STAGE 5 → settle by STAGE 6] WanRMSNorm near-zero input parity
- **무엇을 미뤘나:** PT `F.normalize` 와 우리 `mx.linalg.norm / mx.maximum(., 1e-12)` 의 *분모 floor* 동작이 `||x||_2 → 0` 입력에서 byte-identical인지 직접 측정. 현재 cos=1.0 결과는 *통상* 입력에서의 일치고, 극단(dead-channel) 경계에서 diverge 가능성 미확인.
- **왜 미뤘나:** Lance forward pass에서 *실제로* 극단 case 발생 가능성이 낮음. STAGE 6 첫 이미지 생성에서 denoising loop이 진짜 입력 분포를 통과하니, 거기서 안정성 확인 가능.
- **어떻게 갚나:** STAGE 6 t2i 통합 후 hidden state 분포 측정 + PT 동일 시점 비교. 분포 꼬리에 ||x|| < 1e-6 entry가 있는지, 있다면 cos 영향이 있는지.
- **통과 기준:** STAGE 6 시각 출력이 정상이면 무력화 (분포가 정상 영역에 머묾을 입증).
- **상태:** 열림.

## [opened STAGE 6 → settle by STAGE 7] flex_attention SDPA shim — multi-sample (cu_seqlens > 2) 미커버
- **무엇을 미뤘나:** `tools/stage6_pt_denoise_compare.py`의 `_install_flash_attn_stub`이 단일 sequence packing (`cu_seqlens_q.numel() == 2`)만 처리.  STAGE 7 TI2I/CFG batching이 multi-sample packing을 시도하면 `NotImplementedError`로 죽음.
- **왜 미뤘나:** STAGE 6 cond + uncond는 *별도 forward*로 처리 (batched 아님), 그래서 항상 single-sequence packing.  STAGE 7 어디서 multi-sample이 들어올지는 진입 후 결정.
- **어떻게 갚나:** STAGE 7 첫 multi-sample case에서 shim을 block-diagonal mask로 확장.  단일 sequence는 `cu_seqlens=[0, L]`, multi-sample은 `cu_seqlens=[0, L1, L1+L2, ...]` → 각 sample을 (L_i, L_i) 블록으로 자르고 cross-sample은 -inf.
- **통과 기준:** Multi-sample input에서 SDPA shim이 flex_attention 결과와 동일 (블록별 분리 마스킹).
- **상태:** 열림.

## [opened STAGE 6 → settle anytime] `build_lance_attention_mask` multi-document path
- **무엇을 미뤘나:** `lance_mlx/attn_mask.py`의 `build_lance_attention_mask` 가 `document_lens=None` (single document) 케이스만 STAGE 6에서 검증됨.  PT `create_sparse_mask` 와 *바이트 동치*를 다중 document (다중 sample packed 시퀀스)에서도 확인 안 함.
- **왜 미뤘나:** STAGE 6 t2i는 single document.  multi-document는 STAGE 7 (TI2I = cond image + target) 또는 batched inference에서 등장.
- **어떻게 갚나:** PT `create_sparse_mask` predicate를 multi-doc 예제로 호출 + 우리 helper 결과와 셀별 bit-diff.
- **통과 기준:** N=5 random multi-doc configs에서 셀별 diff=0.
- **상태:** 열림.

## [opened STAGE 3 → settle anytime] Property-based mRoPE position generator
- **무엇을 미뤘나:** 무작위 N개 이미지 / 무작위 (t,h,w) grid / 무작위 텍스트 간격으로 `build_positions_for_layout` 결과를 transformers Qwen2.5-VL 또는 hand-formula로 대량 검증.  현재 12-test battery는 3-image+, 비대칭 gap, edge-case 시퀀스를 커버 못함.
- **왜 미뤘나:** STAGE 3 통과 기준은 byte-identical on representative cases였고 12/12로 충족.  property-based check는 *추가 안전망*이지 게이트 조건은 아니다.
- **어떻게 갚나:** Hypothesis 또는 단순 시드 루프로 100~1000 케이스 생성 → 우리 출력 vs transformers `get_rope_index` 동등성.
- **통과 기준:** N=1000에서 mismatch=0.
- **상태:** 열림.  실제로 multi-image / unusual layout 회귀가 발견되면 우선순위 상승.

## [opened STAGE 3 → settle by STAGE 6] PT Lance `shift_position_ids` exact replay for T2I/TI2I layouts
- **무엇을 미뤘나:** PT Lance의 `data.common.shift_position_ids` 전체 동작(`attn_modes` / `split_lens` / `pro_type` / `shift_attn_mode` 인자에 따라 다른 shift 양을 적용)을 우리 `shift_positions`가 *모두* 재현하는지 직접 검증.  지금은 "한 슬랩에 상수 더하기" 케이스까지만 hand-verified.
- **왜 미뤘나:** STAGE 3 검증 기준은 "mRoPE 적용 forward가 PyTorch와 일치"였고, position generation의 *기본 알고리즘* (텍스트, 단일/다중 이미지, 비디오, 단순 슬랩 shift)은 byte-identical로 통과 (12/12).  T2I 및 TI2I 시퀀스의 *완전* 조립은 STAGE 6/7에서 파이프라인 코드 짤 때 의미를 가지며, 그때 PT inference_lance.py의 실제 position 출력과 직접 비교가 자연스럽다.
- **어떻게 갚나:** STAGE 6 첫 T2I 통합 시점에 (i) PT Lance를 동일 프롬프트로 한 step run, (ii) `packed_position_ids` 덤프, (iii) 우리 조립과 cosine/exact-match.
- **통과 기준:** byte-identical on T2I, TI2I, T2V 각 layout.  실패 시 `shift_position_ids`의 `pro_type` 분기 또는 attn_mode 매핑 보완.
- **상태:** 열림.

## [opened STAGE 4 → ~~settle by STAGE 6 first-image~~ → 가정 폐기] STAGE 4 Tier 3 *clean PT reimpl* 한계 — 진짜 독립 검증 미수행
- **무엇을 미뤘나:** Tier 3에서 사용한 `stage4_pt_cosine.py`의 PT side는 *우리가 손으로 옮긴* clean PT 재구현이다.  refs/Lance source의 `Qwen2MoTDecoderLayer.forward_inference`를 line-by-line 번역했으나, *알고리즘 오해가 양쪽 impl에 동시에 박힐 위험*은 원리적으로 남는다 (예: qk_norm 적용 시점/축에 대한 우리의 *해석*이 PT 의도와 다른 경우 — 우리 MLX와 우리 PT가 둘 다 같은 잘못된 해석을 따르면 cos=1.0이 나옴).  *진짜* 독립 검증은 *원본 PT package* (refs/Lance/modeling/* 그대로 import)로 forward를 돌리고 그 hidden state와 cosine.
- **왜 미뤘나 (당시 가정):** 원본 PT 환경 구축이 무겁다 — flash_attn는 CUDA-only (Mac 불가), flex_attention path 우회, common/data 패키지 shim 필요, refs/Lance의 절대 import 경로 패치.  STAGE 4 통과 기준은 cos ≥ 0.999인데 clean PT 재구현으로 1.000000 통과 — 알고리즘 shape parity는 확보됨.  *최종 행동 검증*은 STAGE 6 첫 이미지 생성: 그림이 나오면 routing + qk_norm + 전 backbone이 의도대로 작동한 것.  그림이 garbage면 어딘가 깨졌다 — 그때 PT 환경 본격 셋업.
- **🚫 가정 폐기 (2026-05-22, STAGE 5 반증):** "STAGE 6 행동 검증이 충분조건"이 *틀렸음을* STAGE 5가 증명했다.  STAGE 5 세 버그 중 두 개(patchify 채널 순서, AvgDown3D 그룹 축)는 cos 0.97~0.99 구간에서 "약간 흐릿한 그림"으로 나타났을 가능성이 높다 — *시각 검증만으론 통과했을 것이다*.  실제로는 PT layer-wise cos 비교만이 그 버그들을 잡았다.  **행동 검증은 필요조건이지 충분조건이 아니다.  수치 검증(원본 PT layer cos)이 더 강하다.**  이 항목 자체는 *새 항목* "[STAGE 4 → STAGE 6] STAGE 4 백본 재검증"으로 대체됨.
- **상태:** ❌ 기각 — 가정 폐기됨. 새 항목으로 *승계*.

## [opened STAGE 4 (재발급 2026-05-22) → settled at STAGE 6 entry 2026-05-22] STAGE 4 백본 재검증 — 원본 PT 직접 import
- **무엇을 미뤘나:** STAGE 4 Tier 3가 *우리 손으로 재구현한* clean PT (`stage4_pt_cosine.py:PtMotLayer`)로 cos=1.0을 받았다.  이게 정말로 옳은지 확인하려면 *우리 손이 닿지 않은* 원본 코드(`refs/Lance/modeling/lance/qwen2_navit.py`)를 직접 `import`해서 같은 forward.  STAGE 5에서 이 패턴(`sys.path.insert(0, "refs/Lance"); from modeling.vae.wan.vae2_2 import WanVAE_`)을 확립했고 — 가중치 layout만 역변환하면 *원본 코드가 직접 forward*. 알고리즘 오해 공유 위험이 원리적으로 *0*.
- **갚은 방법 (`tools/stage6_pt_backbone_compare.py`):**
  1. `refs/Lance`를 PYTHONPATH에 + `flash_attn` stub(SDPA shim, 단일 시퀀스 케이스 한정) + transformers의 flash_attn probe 무력화 + `modeling.lance.__init__.py` 우회용 stub 패키지 등록.
  2. `importlib.import_module("modeling.lance.qwen2_navit")`로 *원본* `Qwen2MoTDecoderLayer` 직접 import.
  3. PT Lance_3B 가중치를 bf16으로 layer 0/12/24/35에 strict-load (PT 코드가 mode="gen"에서 bf16 mixed precision 가정).
  4. 동일 합성 입력(B=1, L=48, GEN slab [24,40)) 양쪽 forward → numpy cosine.
- **결과:**
  | layer | cos | max\|Δ\| | rel_L2 |
  |---|---|---|---|
  | 0  | 0.999979 | 15.5  | 6.6e-3 |
  | 12 | 0.999957 | 18.9  | 9.3e-3 |
  | 24 | 0.999938 | 62.0  | 1.1e-2 |
  | 35 | 0.999979 | 416   | 6.8e-3 |
  - **min cos = 0.999938 (PASS, criterion ≥ 0.999).**
- **추가 학습 — cos<1.0의 *의미*:** clean-reimpl 비교(STAGE 4 Tier 3)는 cos=1.000000을 줬다.  원본 PT 비교는 cos≈0.99994.  *이 차이가 검증의 진정성을 입증한다.*  cos=1.0은 "둘 다 f32, 같은 우리 해석" — 사일런트 버그를 못 잡는다.  cos≈0.99994는 "PT bf16-mixed vs 우리 f32, 알고리즘 동치 + quantization noise" — 실제로 다른 코드와 비교했다는 증거.  rel_L2가 0.7~1.1%로 일정하고 layer depth가 깊어져도 폭주 안 함 → 누적 bf16 정밀도 손실 한도 내.
- **상태:** ✅ 통과 (2026-05-22).  세부는 `LEARNING_LOG/stage_6.md` §0 (STAGE 6 entry).  STAGE 5 doctrine(원본 PT import > clean reimpl)이 실전에서 작동함을 입증.

## [opened STAGE 2 → settled at STAGE 4] PT-Lance layer-0 attention direct cosine
- **무엇을 미뤘나:** PT Lance 첫 어텐션 블록 출력 vs 우리 LanceLLM 같은 블록 출력 cosine.
- **갚은 방법:** `tools/stage4_pt_cosine.py` — refs/Lance source의 `Qwen2MoTDecoderLayer.forward_inference` 경로를 *처음부터 다시* clean PT로 옮긴 후(flash_attn / flex_attention 회피, common/data shim 불필요), PT Lance_3B 가중치를 layer 0/12/24에 로드하여 합성 입력 + GEN slab mask로 forward → 우리 MLX layer i와 numpy로 cosine 비교.
- **결과:** layer 0/12/24 모두 **cos = 1.000000**, max|Δ| 4e-3 / 7e-3 / 2e-2 (f32 reduction-order 누적). 통과 기준 ≥0.999 충족.
- **상태:** ✅ 통과 (2026-05-22).  세부는 `LEARNING_LOG/stage_4.md` §3.
