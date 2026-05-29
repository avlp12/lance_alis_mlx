# audit_manual_v_t — 의도적으로 *틀린* 정답지 (Lesson 18 물증)

이 디렉토리의 fixture 는 STAGE 9 §0 에서 **우리가 수동 (manual) 으로 잘못
시뮬레이션한 PT 정답지** 입니다.  **추론/검증에 사용하지 마세요.**  교훈 18
("manual 정답지는 검증 안 된 가설") 의 *물증* 으로 보존됩니다.

## 어디서 잘못 됐나

STAGE 9 §0 진입 시 우리는 PT `validation_dataset.t2v_sample` 의 출력을
직접 호출하지 않고 *PT 코드를 정독한 후 수동 재현* 했습니다.  결과:

| 항목 | manual (이 디렉토리) | PT 진짜 (out/stage9_pt_video_*_production.npy) |
|---|---|---|
| sequence | chat template (system + user + assistant + IMG slab) | bos + prompt + eos + IMG slab (text_template=False 시) 또는 production chat template |
| video positions | 3D coords (t, h, w 각자 변화) | constant text_split_len (apply_qwen=False) 또는 video case 3D 패턴 |
| vis_start/vis_end modality | 0 (text) | 1 (noise) — PT 의도 |

**우리 manual 이 PT 와 *근본 차이* 였음** — 정독 후 손으로 재현하는 과정이
*검증 안 된 해석* 의 누적.

## 어떻게 발견 됐나

manual v1 (chat-template) vs v2 (raw text_template=False) — 두 *완전히 다른*
시퀀스인데 *v_t byte-identical* 시그너처 (max-abs-diff = 0).  "다른 입력
같은 출력" → forward 자체가 입력 무관 → 진짜 원인은 *Lesson E 재발화*
(`patched_flex_attention` 의 `dense.to(bool)` polarity 반전).  fix 후 PT
진짜 정답지 (`out/stage9_pt_video_v_t_step0.npy`) cos=0.999918 PASS.

상세: `LEARNING_LOG/stage_9.md` §0 정답지 사건.

## 진짜 정답지

추론/검증 용 PT 정답지는 *이 디렉토리가 아닌* `out/stage9_pt_video_*_production.npy`
시리즈입니다:
- `out/stage9_pt_video_pos_ids_production.npy` — full positions (production)
- `out/stage9_pt_video_uncond_pos_ids_production.npy` — uncond positions
- `out/stage9_pt_video_attn_mask_production.npy` — full mask
- `out/stage9_pt_video_uncond_attn_mask_production.npy` — uncond mask
- `out/stage9_pt_video_v_{full,unc,blend}_step0.npy` — v3 PT smoke 출력
- `out/stage9_pt_30step_{latent,final_x_t,video}.npy` — 30-step + video pixel

## 왜 삭제 안 하나

교훈은 *코드*(`pt_layer_mask` assertion) + *문서*(`LEARNING_LOG`) + *물증*
(이 디렉토리) 세 곳에 박힙니다.  fixture 가 사라지면 "manual 시뮬레이션 이
얼마나 PT 와 달랐는지" 가 *bytes 단위로* 재현 불가능.  교훈 18 의 도큐먼트는
물증이 있어야 weight 가 실립니다.

`out/audit_manual_v_t/` 는 의도적인 audit trail.  공개 repo 에서도 유지.
