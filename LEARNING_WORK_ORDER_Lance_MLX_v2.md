# LEARNING WORK_ORDER: Lance-MLX 재구현 (관찰 학습 모드)

**Version:** 2.2 (Coder+Narrator Edition)
**Owner:** Alis (@avlp12)
**Mode:** Claude Code가 짜고, 블록마다 해설. Alis는 흐름을 보며 학습.
**Target backend:** MLX on M3 Ultra 512GB
**정답지(reference):** RockTalk/Lance-3B-MLX + github.com/RockTalk/Lance-MLX
**원본:** github.com/bytedance/Lance

> **지형도 (혼동 방지):** 같은 모델을 두 사람이 독립 작업했다.
> - **RockTalk** = 풀 MLX **포팅** (생성+편집+이해+비디오 전부 직접 구현). → **이 워크오더의 정답지.**
> - **Reza2kn** = **양자화** (원본 PyTorch 기반. understanding 추출판 + 풀 AWQ/NVFP4). RockTalk와 무관, 원본에서 독립 작업. → STAGE 10 양자화 참고용.
> 두 작업은 성격이 다르다. 우리는 *포팅을 손으로 따라가며 배우는* 것이므로 정답지는 RockTalk다.

---

## 0. 이 워크오더의 본질

동작하는 포팅은 RockTalk가 이미 만들었다. **이 과정의 목적은 Alis가 멀티모달 MLX 포팅의 흐름을 관찰하며 이해하는 것이다.**

따라서 Claude Code의 임무는 두 가지다:

1. **코드를 짠다** — 컴포넌트를 처음부터 끝까지 직접 작성한다. Alis는 타이핑하지 않는다.
2. **짜면서 가르친다** — 각 블록을 짠 직후, 방금 무엇을 했고 *왜* 그렇게 했는지 설명한다. Alis가 흐름을 따라오게 한다.

코드 없이 설명만 하지 마라(그건 강의지 관찰 학습이 아니다). 설명 없이 코드만 쏟지도 마라(그건 그냥 포팅이지 학습이 아니다). **둘을 블록 단위로 번갈아라.**

## 1. 작업 리듬 — 블록 단위 짜고-해설

각 컴포넌트(STAGE)를 *의미 있는 블록*으로 나눈다. 블록은 "하나의 개념이 완결되는 코드 단위" (함수 하나, 또는 긴밀히 묶인 몇 함수). 그리고:

### 블록마다 이 순서:
1. **짠다.** 해당 블록의 MLX 코드를 작성한다 (create_file 또는 str_replace).
2. **해설한다.** 코드 직후, 다음을 3~7줄로:
   - **방금 뭐 했나** — 이 블록이 하는 일 한 줄.
   - **왜 이렇게** — 핵심 결정의 이유. 특히 MLX 특유의 처리(레이아웃, dtype, 연산 순서)나 Lance 특유의 구조.
   - **정답지 대비** — RockTalk가 같은 부분을 어떻게 했는지 한 줄. "동일함(MLX 필연)" / "우린 이래서 다름" / "스타일 차이".
3. **다음 블록으로.** Alis가 멈추라거나 물어보지 않으면 계속 흐른다.

### 해설의 톤:
- 교과서 말고 *옆에서 짜며 중얼거리는 시니어 엔지니어*처럼.
- 자명한 건 건너뛴다 (`import numpy as np`에 해설 금지). 비자명한 결정에만.
- 길게 늘어놓지 마라. 형님은 흐름을 본다. 핵심만.

### STAGE 끝:
- 그 STAGE 전체를 3~5줄로 요약 (무엇이 됐고, 검증 통과했고, 다음 STAGE에서 뭐가 이어지나).
- numerical 검증 실행 + 결과 표.
- **사람 승인 게이트.** Alis "다음" 하기 전 다음 STAGE 진입 금지.

## 2. Alis의 개입 지점

Alis는 평소엔 흐름을 본다. 다음 때 끼어든다:

- "잠깐, 그 블록 더 설명해" → Claude가 그 블록만 깊이 판다.
- "거기 RockTalk랑 왜 달라?" → Claude가 차이를 상세 대조.
- "이거 다른 방법은 없어?" → Claude가 대안과 트레이드오프 제시.
- "그냥 넘어가, 자명해" → Claude가 해설 생략하고 코딩 속도 올림.

Claude는 이 신호에 즉시 반응한다. **개입이 없으면 블록 리듬을 유지하며 흐른다.**

## 3. STAGE 순서 (학습 곡선 순, 처음→끝)

쉬운 것부터. 앞 STAGE가 뒤의 전제가 되도록.

### STAGE 1 — Weight 변환 파이프라인
**짤 것:** `convert_weights.py` — PyTorch safetensors → MLX.
**해설 초점:**
- conv weight `(O, I, [T,]H, W)` → MLX `(O, [T,]H, W, I)` 변환이 왜 필요한지 (MLX는 channels-last)
- `lm_head.weight`가 `embed_tokens.weight`에 tied되는 Qwen 관습
- `*_moe_gen.*` 키를 verbatim 보존하는 이유 (STAGE 4 예고)
**검증:** 변환 후 safetensors 로딩, 텐서 수·shape 일치.
**정답지:** RockTalk `tools/convert_weights.py`.

### STAGE 2 — Qwen2.5-VL 텍스트 백본
**짤 것:** `model/backbone.py` — mlx-vlm 재사용 + Lance 변경분.
**해설 초점:**
- mlx-vlm qwen2_5_vl을 어디까지 그대로 쓰고 어디서 Lance가 갈라지나
- KV cache가 AR 디코딩에서 어떻게 작동하나 (X→T 예고)
**검증:** 텍스트 in→out, PyTorch와 next-token logits cosine sim ≥ 0.999.

### STAGE 3 — 3D mRoPE
**짤 것:** `model/rope.py`.
**해설 초점:**
- 텍스트 토큰 vs 이미지 토큰의 position 좌표 차이
- 이미지 patch (h,w) grid → position 매핑
- temporal 확장 시 무엇이 더해지나 (비디오 예고)
**검증:** mRoPE 적용 forward가 PyTorch와 일치.

### STAGE 4 — Mixture-of-Tokens 라우팅 ★ Lance의 심장
**짤 것:** `model/moe_gen.py` (또는 backbone 내 라우팅부).
**해설 초점 (여기 가장 길게 설명):**
- 블록마다 일반 weight + `_moe_gen` weight가 *둘 다* 존재
- 텍스트(UND) 토큰→일반, 생성(GEN/VAE-latent) 토큰→`_moe_gen`, *같은 forward에서*
- 시퀀스를 GEN slab vs UND slab으로 슬라이싱→각각 expert→concat
- attention/MLP/layernorm 전부에 `_moe_gen` 짝
- (참고: 사전 추정 "dual-stream MoE"는 부정확. 실제는 토큰 종류별 라우팅. 이 차이를 짚으면 좋은 학습 포인트.)
- **명명 주의:** 이 구조를 RockTalk는 "Mixture-of-Tokens", Reza2kn은 "Mixture-of-Tasks"로 부른다. 같은 것을 독립적으로 명명한 것. 검색·문서 읽을 때 둘 다 같은 라우팅을 가리킨다.
- 원본 modified-Qwen2.5-VL에는 `qk_norm` weight가 있다. (Reza2kn은 mlx-lm의 표준 qwen2 클래스가 이걸 정의 안 해서 드롭했고 품질 손실을 감수함 — 우리는 풀포팅이므로 *드롭하지 말고 살린다*. 이게 양자화 추출판과 풀포팅의 갈림길.)
**검증:** 멀티모달 forward(텍스트+생성 토큰 혼합), PyTorch와 hidden state cosine sim ≥ 0.999 at 3 layers.
**정답지:** RockTalk `qwen2_navit_mlx.py` 라우팅부.

### STAGE 5 — Wan 2.2 이미지 VAE (T=1)
**짤 것:** `model/vae_wan22.py` (T=1 경로 먼저).
**해설 초점:**
- VAE latent space, scale factor
- conv 레이아웃 변환 실전 (STAGE 1 복습)
- `z_dim=48, c_dim=160, dim_mult=(1,2,4,4)` config 의미
**검증:** encode→decode round-trip MSE ≤ 1e-3 (단일 이미지).
**정답지:** RockTalk/Wan2.2-VAE-MLX.

### STAGE 6 — Flow Matching + CFG Denoising Loop
**짤 것:** `model/scheduler.py` + denoising loop.
**해설 초점:**
- flow matching vs DDPM (velocity prediction)
- `timestep_shift=3.5` 의미
- CFG `v = v_uncond + scale*(v_cond - v_uncond)` 직관
- step마다 forward 2번(cond/uncond)인 이유
**검증:** **첫 이미지 생성.** t2i 동작, PyTorch와 최종 latent cosine sim ≥ 0.995.
**★ 여기서 그림이 나온다. 첫 큰 성취.**

### STAGE 7 — 이미지 편집(TI2I) + 이해(X→T)
**짤 것:** `pipelines/image_edit.py`, `pipelines/x2t_image.py`.
**해설 초점 (TI2I):**
- ViT(UND) + VAE-encode(cond latent) 이중 조건화
- 3-component CFG: `v_final = v_tv_uncond + cfg_text*(v_full - v_t_uncond) + cfg_vit*(v_t_uncond - v_tv_uncond)`
- edit-mode 시스템 프롬프트 verbatim 필요 이유
**해설 초점 (X→T):**
- 같은 백본이 AR 디코딩으로 텍스트 출력
- KV cache 복습, ~29 tok/s
**검증:** 편집 5샘플 + VQA 10샘플.
**★ 여기까지가 이미지 트랙 완주. RockTalk Lance-3B-MLX와 동등.**

### STAGE 8 — 3D Causal Video VAE + Tile Decoding ★ 최난관
**짤 것:** `vae_wan22.py`의 video 경로 (T>1).
**해설 초점:**
- 3D causal convolution — 시간축 인과 패딩
- temporal compression/decompression
- **tile decoding** — 시공간 분할→오버랩→봉합. seam 없애기.
- `mx.conv3d` 성능·정확도 한계 (짜기 전 측정)
- `temperal_downsample=(False,True,True)` 의미
**검증:** 짧은 클립 round-trip MSE ≤ 1e-3 + seam 시각 검사.
**현실 인정:** RockTalk도 9프레임 256px에 머묾. 이 한계의 *이유*를 이해하는 게 목표. 그 너머는 보너스.

### STAGE 9 — Video DiT + t2v 통합
**짤 것:** `model/video_dit.py` + `pipelines/t2v.py`.
**해설 초점:**
- spatiotemporal attention (full 3D? factorized?)
- video mRoPE 3D 확장 (STAGE 3 복습)
- STAGE 6 denoising loop의 temporal 확장
**검증:** t2v 동작. RockTalk 스펙(T_lat=3, 9프레임 256×256) 재현. 가능하면 그 이상.

### STAGE 10 (선택) — 양자화 ★ Alis 특기 영역
**왜 선택:** 풀포팅(STAGE 1~9) 완주 후, 형님이 원하면 추가. 학습이 아니라 *기여* 단계.
**짤 것:** 완성된 MLX 포팅의 양자화 버전.
**참고처:**
- Reza2kn `github.com/Reza2kn/lance-quant` — 원본 PyTorch 기반 추출+양자화 툴킷. 단 이건 *포팅이 아니라 양자화*라 우리 MLX 코드와 경로가 다름.
- Reza2kn DWQ 기법: bf16 teacher로부터 KL-divergence distillation으로 per-group scale/bias 최적화, 같은 비트 예산에서 PTQ 대비 ~0.6 bpw 품질 회복. 형님의 GLM-5.1·Qwen3.5 dynamic quant 경험과 같은 계열.
**빈자리(기여 기회):** Reza2kn은 균일 4bit affine / NVFP4만 했다. **MLX dynamic quant(레이어별 혼합 비트)** 는 아무도 안 함 — 형님 특기와 정확히 겹치는 공백. 풀포팅 위에 dynamic quant를 얹으면 `avlp12/Lance-MLX-Dynamic` 자리가 비어 있다.
**현실 인정:** 이건 학습 워크오더의 본 범위 밖. STAGE 9까지가 "포팅 학습 완주"이고, STAGE 10은 별도 동기·별도 예산으로 판단.

## 4. 보고/로그

각 STAGE 완료 시 `LEARNING_LOG/stage_N.md`에 누적:
1. **이 STAGE가 한 일** (요약)
2. **핵심 해설 모음** — 블록별 "왜"의 정수만 추림
3. **RockTalk 대비 차이점** (있었다면)
4. **검증 결과** (numerical 표)
5. **다음 STAGE로 이어지는 것**

이 로그는 형님 X/Substack 콘텐츠 소재가 된다. "Claude Code로 멀티모달 MLX 포팅 따라가며 배운 것" 시리즈.

## 4.5 개선점 트랙 (IMPROVEMENTS.md) — 격리가 핵심

작업하다 보면 "RockTalk보다 이렇게 하는 게 낫겠다"는 개선점이 보인다. 환영한다 — 단 **즉흥 적용은 금지**다. 이유: 포팅의 정답지는 *검증된 동작 구현(RockTalk)과의 일치*다. 개선이 포팅 코드에 섞이면, numerical 불일치가 났을 때 "개선 탓인지 / 포팅 버그인지 / MLX op 차이인지" 구분 불가 — 디버깅 지옥. (규칙 3 외과적, 규칙 7 평균내지 마라)

따라서 개선을 두 종류로 나눈다:

**A. 우리 코드 품질 개선** (변수명, 구조, 헬퍼 추출, 주석, 테스트 보강)
→ 기존 code-reviewer 게이트(§5.7)에서 처리. 그대로 진행. baseline 동작을 안 바꾸므로 즉시 적용 OK.

**B. RockTalk보다 나은 *동작/수치/성능* 발견** (더 빠른 MLX op, 메모리 절감, 더 정확한 수치, qk_norm·RoPE·VAE 처리 개선 등)
→ **즉시 적용 금지. `IMPROVEMENTS.md`에 기록만.** 다음 절차를 따른다:

1. **발견 즉시 기록** — 해당 STAGE 진행 중 개선점이 보이면 흐름을 멈추지 말고 IMPROVEMENTS.md에 한 항목 추가하고 계속 포팅 흐름 유지. (포팅이 1순위, 개선은 백로그)
2. **STAGE 검증 통과 후 판단** — 그 STAGE가 RockTalk parity/cosine 기준을 통과한 *다음*에, 백로그 항목을 형님과 함께 검토.
3. **적용 시 baseline 대비 측정** — 적용하기로 하면, parity 통과한 baseline을 기준선으로 두고 개선을 *측정 가능하게* 검증한다 (예: "latency 1.4초→0.9초", "cosine 0.9991→0.9997", "peak mem 38GB→29GB"). 측정 없는 "개선"은 적용 안 함. (규칙 4: 성공 기준 정의 후 반복, 규칙 9: 의도를 검증)
4. **적용 후 재-parity** — 개선이 RockTalk와의 동등성을 *의도적으로 깨는* 것이면(즉 우리가 더 낫다고 판단), 그 사실을 IMPROVEMENTS.md에 "divergence from reference + 근거 + 측정값"으로 명시. 조용히 분기 금지. (규칙 11: 충돌은 드러내라)

**IMPROVEMENTS.md 항목 형식:**
```
## [STAGE N] 제목
- 발견: 무엇을 봤나 (RockTalk/PT는 어떻게 했고, 우리가 본 더 나은 길)
- 분류: 성능 / 메모리 / 수치정확도 / 구조
- 리스크: parity를 깨나? MLX 버전 의존? 검증 난이도?
- 상태: 기록됨 / 검토중 / 적용됨(측정값) / 기각(사유)
```

**왜 이게 가치 있나:** IMPROVEMENTS.md는 (a) 포팅을 오염 없이 지키고, (b) 개선 아이디어를 휘발 안 시키고, (c) STAGE 10 양자화·비디오 프레임 확장 같은 *진짜 기여*의 씨앗이 되며, (d) 그 자체로 형님 콘텐츠 소재("RockTalk 포팅을 따라가며 발견한 N가지 개선점")가 된다. 풀포팅 완주 후 IMPROVEMENTS.md의 적용 가능 항목들이 `avlp12` 고유 버전의 차별점이 된다.

## 5. 규칙 (Alis 규칙 적용)

1. **추측 금지.** PyTorch/RockTalk 동작이 불명확하면 직접 실행해 확인 후 짠다. 추측으로 짜고 해설하지 마라. (규칙 1, 8)
2. **외과적.** 필요한 것만 짠다. 투기적 추상화·"나중에 확장 대비" 금지. (규칙 2, 3)
3. **해설은 정직하게.** 확신 없는 부분은 "이건 RockTalk를 따랐는데 이유는 추정"이라고 드러내라. 아는 척 금지. (규칙 12)
4. **순서 지켜라.** 앞 STAGE 검증 통과 + Alis 승인 없이 다음 진입 금지. (규칙 10, 12)
5. **토큰 예산.** STAGE가 길면 블록 묶음으로 세션 분할. 초과를 드러내라. (규칙 6)
6. **충돌 판정.** Alis 코드 아닌 *우리 구현*과 RockTalk가 다르면 동치/버그/스타일 판정해 해설에 포함. (규칙 7)
7. **code-reviewer 게이트.** STAGE 종료 시 code-reviewer.md(Opus, xhigh) 호출. BLOCKING/SUGGESTED/NITPICK.
8. **개선점 격리.** 동작/수치/성능 개선(B종)을 발견하면 즉시 적용하지 말고 IMPROVEMENTS.md에 기록. STAGE 검증 통과 후 baseline 대비 측정해 적용 판단. (§4.5)

## 6. 첫 응답 형식

이 워크오더를 읽었으면 Claude Code의 첫 응답:

1. 역할 1줄 확인: "나는 짜면서 블록마다 해설한다. 형님은 흐름을 본다."
2. 준비 상태 확인: 원본 repo, RockTalk repo, 체크포인트, MLX 환경이 있는지.
3. **STAGE 1 시작 선언** + 첫 블록 짜기 진입. (바로 코드 흐름 시작. 형님이 보고 싶은 게 그거다.)

준비물:
- 원본: `git clone github.com/bytedance/Lance`
- 정답지: `git clone github.com/RockTalk/Lance-MLX`
- 체크포인트(검증용): `huggingface-cli download bytedance-research/Lance`
- `mlx>=0.29`, `mlx-vlm>=0.3`, `numpy`, `einops`, `transformers`, `pillow`
- PyTorch reference 환경 (CPU 또는 두 번째 머신)

---

**End of LEARNING WORK_ORDER v2.**

Claude Code에게: 너는 코더이자 해설자다. 코드를 짜라 — 단, 블록마다 멈춰서 "방금 뭐 했고 왜"를 짧게 설명하라. 형님이 흐름을 보며 배운다. §6대로 STAGE 1부터 시작하라.
