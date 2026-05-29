# STAGE 8 — 3D Causal Video VAE + temporal chunked decode ★

**Status:** ✅ PASSED  (2026-05-25, T=5 round-trip 네 cos 모두 1.000000)
**Deliverable:**
- `lance_mlx/vae_wan22.py` (Wan 2.2 VAE — T=1 image path는 STAGE 5에서 상속,
  T>1 video path는 STAGE 8 §1에서 building block 7개, §2에서 WanVAE_ 최상위 봉합)
- `tools/stage8_causal_conv3d_compare.py` (`CausalConv3d.cache_x` 스트리밍 검증)
- `tools/stage8_resample_compare.py` (upsample3d "Rep" sentinel + downsample3d shift)
- `tools/stage8_residual_compare.py` (`ResidualBlock` 2-slot feat_cache 검증)
- `tools/stage8_down_residual_compare.py` (`Down_ResidualBlock` 한계 게이트)
- `tools/stage8_up_residual_compare.py` (`Up_ResidualBlock` + `DupUp3D` first_chunk)
- `tools/stage8_encoder3d_compare.py` / `stage8_decoder3d_compare.py` (블록 통합 검증)
- `tools/stage8_pt_video_smoke.py` (§0 PT 정답지 생성: `out/stage8_pt_video_*.npy`)
- `tools/stage8_wanvae_compare.py` (§2 최종 게이트 — T=5 round-trip mu/logvar/xhat cos)

**Verification chain (bottom-up 빌딩 블록 7개 → 최상위 round-trip):**

| Block | Tool | Gate | Result |
|---|---|---|---|
| §0 PT 정답지 | `stage8_pt_video_smoke.py` | PT shapes/no-NaN | PT end-to-end forward OK |
| §1.1 CausalConv3d cache_x | `stage8_causal_conv3d_compare.py` | cos vs PT both branches | cos = 1.0 (stateless), cos = 1.0 (streaming) |
| §1.2 Resample 3D | `stage8_resample_compare.py` | "Rep" sentinel + downsample3d shift | cos = 1.0 |
| §1.3 ResidualBlock | `stage8_residual_compare.py` | 2-slot feat_cache, output + STATE byte-diff | cos = 1.0, cache STATE byte-equal |
| §1.4 Down_ResidualBlock | `stage8_down_residual_compare.py` | (한계 게이트 — §3 교훈 참조) | cos = 1.0 on solvable subset |
| §1.5 Up_ResidualBlock + DupUp3D | `stage8_up_residual_compare.py` | first_chunk 비대칭 (frame drop) | cos = 1.0; first_chunk=True/False 양쪽 |
| §1.6 Encoder3d | `stage8_encoder3d_compare.py` | conv1 + downsamples + middle + head | cos = 1.0 (chunked over T=5) |
| §1.7 Decoder3d | `stage8_decoder3d_compare.py` | conv1 + middle + upsamples + head | cos = 1.0 (per-frame iter) |
| §2 WanVAE_ top-level | `stage8_wanvae_compare.py` | 4-gate cos vs §0 fixtures | **cos = 1.0 × 4 ✓** |

**최종 §2 게이트 (T=5 video, scale=mean/inv_std 적용):**
- `mu     ` cos = 1.000000  mse = 1.45e-12  maxabs = 6.4e-6  p50 = 7.2e-7   p90 = 1.9e-6
- `log_var` cos = 1.000000  mse = 8.14e-11  maxabs = 3.1e-5  p50 = 5.7e-6   p90 = 1.5e-5
- `xhat*  ` cos = 1.000000  mse = 4.92e-11  maxabs = 6.9e-5  p50 = 4.0e-6   p90 = 1.1e-5  (PT mu → MLX decode, decoder 격리)
- `xhat   ` cos = 1.000000  mse = 6.67e-11  maxabs = 9.5e-5  p50 = 4.6e-6   p90 = 1.3e-5  (MLX 풀 round-trip)

---

## §0. 진입 매핑 — "tile = temporal, NOT spatial"

진입 지시 문구는 "tile decoding" / "spatial tile seam 봉합"이었다. PT 코드 직접 매핑 후
이 전제가 잘못이라는 것 확인 → 정정.

**실제로 PT가 청크하는 축:**
- `WanVAE_.encode(x, scale)` (vae2_2.py:759) — patchify 후 *T축*을 chunk:
  ```
  iter_ = 1 + (t - 1) // 4
  chunk 0 = x[:, :, :1, :, :]              # 첫 1프레임
  chunk i = x[:, :, 1+4(i-1):1+4i, :, :]   # 이후 4프레임씩
  ```
  T=5 → iter_=2.  *공간 H/W는 자르지 않음.*
- `WanVAE_.decode(z, scale)` (vae2_2.py:787) — latent T_lat를 1프레임씩:
  ```
  for i in range(iter_=z.shape[2]):
      first_chunk = (i == 0)
      decode 1 latent frame → expand to T_pixel frames per iter
  ```

**Tile은 *temporal*이고 봉합은 *causal* (과거→현재 단방향)** — 공간 seam(STAGE 6 512²의
가짜 seam)과 본질이 다르다.  Causal pad는 future leak이 없으므로 *왼쪽 boundary는 zero pad*,
*오른쪽 boundary는 인접 청크의 마지막 CACHE_T 프레임 (= 2)을 다음 청크 conv에 넘겨주는
feat_cache 스트리밍*.

이 정정 후 STAGE 8 작업 구조 결정:
- "공간 tile 봉합 알고리즘 설계" → 불필요
- "temporal causal conv + feat_cache 상태 머신 정확 재현" → STAGE 8 본질

---

## §1. Building block 검증 체인 (bottom-up)

PT `vae2_2.py`의 7개 빌딩 블록을 의존성 순서대로 *각 블록별 PT vs MLX byte-diff* 거친 후
상위로 올라간다.  STAGE 1~7과 다른 점: 각 블록이 *상태*(`feat_cache`)를 가져, 출력뿐 아니라
*post-call cache 내용*까지 PT와 같은지 확인.

### §1.1 `CausalConv3d` — `cache_x` 인자 (asymmetric T pad + 스트리밍)

- PT signature: `forward(x, cache_x=None)` — `cache_x`가 있으면 T축에 concat, "before" pad를
  `cache_x.shape[T]`만큼 줄임.  STAGE 5 image path(T=1)에선 `cache_x=None`만 호출 → 정확히
  단일 frame 0-pad.
- MLX (`vae_wan22.py:122-142`): `mx.concatenate([cache_x, x], axis=1)` 후
  `pad_t_before -= cache_x.shape[1]`.  *bool branch 단순 — 한 줄 fix.*
- 검증 (`stage8_causal_conv3d_compare.py`):
  - **Stateless 경로** (cache_x=None, T=1): cos = 1.0 (STAGE 5 regression 확인)
  - **스트리밍 경로** (cache_x = prev[-2:], T=4): cos = 1.0
- 교훈: kernel=1 인 CausalConv3d는 `2*padding[0] = 0` 이라 cache_x 분기 자체가 작동 안 함 →
  자동으로 stateless.  WanVAE_ 최상위의 `conv1/conv2` (k=1)가 이 케이스 — feat_cache slot
  소비 안 함.  PT `count_conv3d`도 conv1/conv2를 *encoder/decoder 외부*에서 호출, 슬롯 카운트
  영향 없음.

### §1.2 `Resample` — 3D 모드 (`upsample3d` / `downsample3d`)

가장 복잡한 블록.  두 경로 모두 *PT 라인 정독 후* MLX 재현:

- **`upsample3d` (PT line 123-153):**
  - 첫 호출: `feat_cache[idx] is None` → 센티널 `"Rep"` 저장, time_conv *skip*.
  - 다음 호출: `prev == "Rep"`이면 time_conv를 cache_x 없이 호출, 아니면 prev를 cache_x로.
  - 그 다음 pixel-shuffle 2× T 확장 (`(B, 2C, T, H, W) → (B, C, T*2, H, W)`):
    PT는 `stack(dim=3)`, MLX는 `reshape (T, H, W, 2, C) → transpose (0,1,4,2,3,5) → reshape (T*2, H, W, C)`.
    프레임 ordering: `idx = t*2 + g` (outer T, inner group) — *PT와 일치 확인이 핵심*.
- **`downsample3d` (PT line 159-170):**
  - 첫 호출: `feat_cache[idx] is None` → input 전체를 cache로 저장, time_conv 그 청크에선 skip.
  - 이후: `cache_x = x[:, -1:, ...]`, time_conv를 `concat(prev[-1:], x)`에 적용.
- 검증 (`stage8_resample_compare.py`):
  - 두 청크 시나리오 (chunk-0 sentinel + chunk-1 streaming) PT byte-diff.
  - `"Rep"` 전이 시점에서 post-call `feat_cache` 내용까지 PT와 비교 (출력만이 아니라 *STATE*).
  - cos = 1.0.

### §1.3 `ResidualBlock` — 2 slot 정확 소비

- PT (line 213-229): 한 호출에서 `conv1`이 슬롯 1개, `conv2`가 슬롯 1개 소비.  *shortcut conv는
  k=1 → 슬롯 소비 안 함* (자기-스트리밍 불필요).
- MLX (`vae_wan22.py:199-249`): 두 conv 각각 `cache_x = h[:, -CACHE_T:, ...]` 저장 +
  `if cache_x.shape[1] < 2 and feat_cache[idx] is not None: prepend prev[:, -1:, ...]` 경계 보정.
- 검증 (`stage8_residual_compare.py`): 청크 1, 청크 2 순차 호출, 매 호출 후
  `feat_cache[idx_conv1]`, `feat_cache[idx_conv2]` 내용을 PT 직접 호출 결과와 byte-diff.
  cos = 1.0.

### §1.4 `Down_ResidualBlock` — *게이트 한계* 발견

게이트 시도 1: "chunked encode 결과 vs stateless full-T encode 결과"가 PT 측에서도 동일한지
직접 비교 → **PT 측에서도 다름**.  이유: `avg_shortcut = AvgDown3D` 가 *입력 T 전체*에 대해
average pool 하는데, chunked로 부르면 매 청크 입력 T가 다르므로 shortcut 출력이 본질적으로
다르다.  Main path는 stateless가 아니라 *streaming 전용*으로 설계됨.

→ 게이트 정의를 *PT가 실제로 호출하는 방식*(chunked)에 맞춤: chunked PT vs chunked MLX cos.
이 정정 후 cos = 1.0.

이 발견이 **§3 Lesson 14**의 원천: *"게이트 자체를 PT로 검증" — 피드백이 준 게이트 정의도
가설이지 결론 아님.*

### §1.5 `Up_ResidualBlock` + `DupUp3D` — `first_chunk` 비대칭

`DupUp3D`는 `factor_t`만큼 T축을 repeat-expand 후, `first_chunk=True`일 때만 처음
`(factor_t - 1)` 프레임을 *drop* (PT line 404).  이유: causal conv의 첫 frame은 *과거가 없는*
경계 조건이라 한 frame만 출력해야 PT와 align.  이후 청크(first_chunk=False)는 모든
expanded frame을 보존.

- MLX (`vae_wan22.py:510-526`): `if first_chunk: x = x[:, self.factor_t - 1:]`.
- 검증 (`stage8_up_residual_compare.py`): first_chunk=True/False 양쪽 매트릭스, T_in=1/2,
  factor_t=1/2 조합 모두 cos=1.0.  *별도로* 전체 Decoder3d에서 first_chunk=True only 케이스
  재현 확인 (cos=1.0).

PT default는 `first_chunk=False` (line 470), MLX default는 `first_chunk=True` (STAGE 5
image 편의).  **현재 모든 call site가 명시적으로 kwarg 넘김** → 동작 차이 없음.  미래 silent
landmine으로 `IMPROVEMENTS.md [STAGE 8]`에 기록 (line 174-178).

### §1.6 `Encoder3d` — 상위 슬롯 순서

- 슬롯 walking 순서:
  1. `conv1` (1 slot)
  2. `downsamples[i]` for i in 0..3 — 각 `Down_ResidualBlock`이 자기 ResidualBlock×mult +
     선택적 Resample만큼 슬롯 소비
  3. `middle = [RB, Attn, RB]` (2 + 0 + 2 = 4 slots; Attn은 슬롯 안 씀)
  4. `head_conv` (1 slot)
- `_count_causal_conv3d(self.encoder) = 26` (cf. PT `count_conv3d(self.encoder) = 26` — 일치).
- 검증 (`stage8_encoder3d_compare.py`): T=5 chunked PT vs chunked MLX cos=1.0.

### §1.7 `Decoder3d` — 대칭 + `first_chunk` 전파

- 슬롯 walking 순서: conv1 → middle (4 slots) → upsamples[i] for i in 0..3 → head_conv.
- `first_chunk` 인자는 모든 `Up_ResidualBlock`에 전파, `DupUp3D` first-chunk frame drop이 발화.
- `_count_causal_conv3d(self.decoder) = 34`.
- 검증 (`stage8_decoder3d_compare.py`): T_lat=2, per-frame iter, first_chunk=(i==0) — cos=1.0.

---

## §2. WanVAE_ 최상위 round-trip — STAGE 8 진짜 게이트

§1의 빌딩 블록 7개가 byte-clean이어도 최상위 *조립*에서 새 버그가 가능 (STAGE 4/7 교훈).
사용자가 §0 PT 정답지를 fixtures로 박은 이유.

### 진입 시점에서 점검한 §0 위험 3개

1. **`conv1` / `conv2` (mu/logvar 분리 + pre-decode):** PT `WanVAE_.__init__` line 742-743 →
   k=1 CausalConv3d.  k=1 이므로 causal pad = 0 → stateless.  encoder/decoder 외부에 있어
   `count_conv3d`가 슬롯을 안 셈.  *full T-axis tensor에 한 번에 적용*, chunked 루프 외부.
2. **Scale 적용:** §3(T2I)에서 *이미지 inference는 normalization 안 함*(scale=identity)
   확인했지만 *비디오는 다르다*.  PT `Wan2_2_VAE` wrapper(line 875-885)가 *하드코딩된*
   per-channel mean[48] + std[48] 배열로 `[mean_t, 1.0/std_t]` 형태 scale 호출.
   encode: `mu = (mu - mean) * (1/std)`, decode: `z = z * std + mean`.  **log_var은 스케일 안 됨.**
3. **`patchify` / `unpatchify` 채널 순서:** STAGE 5 silent bug class.  einops
   `"b c f (h q) (w r) -> b (c r q) f h w"` 의 *flatten tail이 (c, r, q)* — channel slowest, r
   (W-inner) middle, q (H-inner) fastest.  MLX reshape+transpose 직접 검증:
   `arange(1,3,2,4,4)` 입력에서 PT einops 결과와 max-abs-diff = 0 (byte-exact).

세 점 모두 *PT 직접 확인 후* 진입 — 가정 없이.

### MLX 구현 (`vae_wan22.py:909-993`)

```
encode(image, scale, return_logvar):
    image (B, T, H, W, 3) 또는 (B, H, W, 3)
    x = patchify(image, 2)               # (B, T, H/2, W/2, 12)
    if T == 1: enc_out = encoder(x)      # STAGE 5 fast path 보존
    else:
        n_slots = _count_causal_conv3d(encoder)
        feat_cache = [None] * n_slots
        feat_idx = [0]
        iter_ = 1 + (T - 1) // 4
        chunk-0: x[:, :1]; chunk-i>0: x[:, 1+4(i-1):1+4i]
        per chunk: feat_idx[0] = 0  # reset
        enc_out = concat(chunks, axis=1)
    mu_logvar = conv1(enc_out)            # stateless, k=1
    mu = mu_logvar[..., :z_dim]
    log_var = mu_logvar[..., z_dim:]
    if scale: mu = (mu - scale[0]) * scale[1]
    return (mu, log_var) if return_logvar else mu

decode(z, scale):
    if scale: z = z / scale[1] + scale[0]
    x = conv2(z)
    if T_lat == 1: decoder(x, first_chunk=True)
    else: 1프레임씩 iter, first_chunk=(i==0)
    return unpatchify(out, 2)
```

### 게이트 결과

위 표 참조.  네 cos 모두 1.000000.  maxabs는 mu에서 6.4e-6 (encoder만 통과),
xhat에서 9.5e-5 (encoder + conv1 + conv2 + decoder + unpatchify 5단 누적) — *FP32 numerical
floor*.  

p90/max 비율: mu에서 6.4e-6 / 1.9e-6 ≈ 3.4×, xhat에서 9.5e-5 / 1.3e-5 ≈ 7.3× — 둘 다
정규분포의 tail 비율(가우시안이면 ~3.5× at p90 vs max in finite sample).  *폭주(10×+)
아님, 단순 누적 노이즈*.

---

## §3. 교훈 (STAGE 8 고유)

STAGE 1~7에서 박힌 13개 Lesson에 더해, STAGE 8에서 새로 박힌 것:

### Lesson 14 (new) — *feat_cache 는 상태 머신이다.  출력만 byte-diff 하면 절반만 검증*

> STAGE 1~7의 검증은 모두 *stateless* — 입력 → 출력 cos.  STAGE 8 streaming VAE의 conv는
> *호출 후 cache state*가 다음 호출 입력의 일부.  같은 입력으로 cos=1.0 통과해도 *cache가
> PT와 다르면* 다음 청크에서 silent 발산.
>
> 구체:
> - 청크-0 forward: 출력 byte-diff cos=1.0 → ✓
> - 그러나 청크-0 종료 시점에 `feat_cache[idx]`가 PT 와 다른 값 저장 → 청크-1 forward에서
>   PT는 prev[-1:]를 cache_x로 사용, MLX는 *다른* prev[-1:]를 사용 → 청크-1 출력 발산
> - 그 시점에 잡으면 청크-1 cos 분석으로 청크-0 cache 버그를 거꾸로 추적해야 함 (비싼 디버깅)
>
> **보호책:** state-bearing 블록은 *출력 + 상태 모두* PT와 byte-diff.  STAGE 8 §1.3
> `stage8_residual_compare.py`에 박은 패턴:
> ```python
> mlx_out = block(x, feat_cache=mlx_cache, feat_idx=mlx_idx)
> pt_out  = pt_block(x_pt, feat_cache=pt_cache, feat_idx=pt_idx)
> assert cos(mlx_out, pt_out) > 0.999
> for slot in range(slot_count):
>     assert cos(mlx_cache[slot], pt_cache[slot]) > 0.999  # ← STATE 도 비교
> ```
> 특히 sentinel transition (`"Rep"` → 실제 tensor)이 가장 위험.  String/None/Tensor 세 타입
> 분기가 silent 한 비교 실패를 만든다.

### Lesson 15 (new) — *수치 노이즈의 "분포 형태"가 폭주 vs 정상을 가른다*

> Block-by-block 검증에서 maxabs가 *크게 나와도* 그게 곧 버그는 아니다.  Decoder maxabs가
> Encoder의 ~10× 크게 나오는 건 단순히 *upsample depth* (decoder가 spatial 8× + temporal
> 4× expansion으로 누적 stage 더 많음) 결과 — 폭주 아님.
>
> 폭주와 단순 누적의 분포 차:
> - 폭주: 한두 outlier가 max를 끌어올림.  나머지는 작음.  **p90/max 비율 10×+**.
> - 단순 누적: 가우시안적, 모든 frame에 고른 노이즈.  **p90/max 비율 ~3× (가우시안 tail).**
>
> STAGE 8 §1 Decoder 첫 측정에서 maxabs=1e-3 (Encoder의 ~10×) — *처음엔 버그처럼 보임*.
> p90/max 보니 ~3.5× → 폭주 없음.  **임계값을 미리 박지 말 것**: maxabs<1e-5 같은 절대값
> 임계는 stage 종속.  분포 형태 (p90/max, 또는 std/max)를 보고 *그 stage 적정* 결정.
>
> **보호책:** 모든 cos check에 maxabs + p50 + p90 출력.  비율로 폭주 판별.

### Lesson 16 (new) — *게이트 정의도 가설이지 결론 아님*

> STAGE 8 §1.4 `Down_ResidualBlock` 검증 시 제안된 게이트: "chunked decode vs stateless
> full-T decode가 같은가".  *이게 PT 측에서도 안 됨* — `avg_shortcut`(AvgDown3D)가 입력 T
> 전체에 대해 pool 하므로 chunked vs full-T가 PT 자체적으로 다른 출력.
>
> Main path는 streaming 전용 설계: stateless full-T 호출 자체가 PT가 의도한 사용법이 아님.
> 게이트 정의가 *PT 행동 분석 없이* 제안된 것 → 수학적으로 불가능한 비교.
>
> **보호책:** 게이트 정의가 들어오면 *그 게이트 자체*를 PT 측에서 먼저 시연. "이 비교는
> PT 측에서도 hold 하는가?"  Hold 안 하면 게이트 정의 수정.  *피드백이 준 게이트도
> 가설*임을 의식할 것.

### Lesson 17 (new) — *scale 은 이미지/비디오가 다르다.  T2I 가정 비디오 이전 금지*

> STAGE 6 §3 (T2I)에서 *Lance inference는 latent normalization 안 함* (scale=identity)
> 직접 확인.  이걸 비디오 VAE 에 가정하면 **무성 silent 버그**: encode 결과는 같은 dtype
> 같은 shape 으로 나오지만 *수치적으로* PT 와 다른 분포.
>
> PT 비디오 wrapper는 *하드코딩 per-channel scale* (`Wan2_2_VAE.__init__` mean[48], std[48]).
> Image 와 video 가 같은 VAE를 쓰지만 *호출 wrapper*가 scale 정책을 다르게 박았다.
>
> **보호책:** 모달리티 바뀔 때마다 *그 모달리티의 PT call site 직접 확인*.  "이전 모달리티에서
> scale=identity 였으니 이번에도"는 안 됨.  특히 normalization, position embedding, mask 패턴
> 등 *호출 측 정책*은 모달리티마다 재확인.

---

## §4. 정직한 audit trail — Claude 가설/전제가 정정된 지점

STAGE 6 §5 / STAGE 7 §5 패턴 반복.  Claude/Claude-피드백 측이 *그럴듯한 진단*을 냈지만
PT 직접 매핑이 부인한 지점:

| 시점 | 가설/전제 | 실제 |
|---|---|---|
| STAGE 8 진입 지시 | "tile = spatial seam 봉합 (STAGE 6 512² 후속)" | PT 직접 매핑: tile = *temporal causal* chunk, spatial은 자르지 않음.  봉합은 feat_cache 스트리밍, 가짜 seam 아님 |
| §1.4 게이트 정의 | "chunked vs stateless full-T 가 같아야 한다" | PT 측에서도 다름 (`AvgDown3D`가 입력 T 전체에 의존).  Main path가 streaming 전용 설계.  게이트 정의 자체가 잘못 |
| §0 위험 점검 시 (가정 차단) | "T2I 에서 scale=identity 였으니 비디오도 그럴 것" | PT `Wan2_2_VAE` wrapper가 하드코딩 mean/std 배열.  *모달리티별 다름* — 가정 전 PT 직접 확인 |
| §2 구현 중 mid-flight | "`vars(self)`로 MLX module 자식 walk 가능" | MLX `nn.Module`은 `__setattr__` 오버라이드로 자식을 internal store에 숨김.  `vars(self)`는 `{'_no_grad', '_training'}` 만 반환 → `_count_causal_conv3d`가 0 반환 → 빈 cache list → 첫 청크에서 IndexError.  *Canonical traversal*: `module.modules()` (PT 와 같은 이름).  검증: PT count_conv3d 26/34 ↔ MLX 동일 |

이 표가 보여주는 패턴 (STAGE 6 §5, STAGE 7 §5 의 누적):
- *피드백 (사용자/세션) 측의 게이트 정의/전제도 화살표지 결론 아님*.  PT 측에서 hold 하는지
  먼저 확인.
- *이전 stage 의 발견을 다음 stage 에 그대로 이전 금지*.  모달리티/호출 wrapper 별 재확인.
- *MLX 환경 특이사항*은 PT-MLX 등가성과 별개로 한 번 더 확인 필요 (Module traversal,
  parameter tracking, list-of-Module 처리 등).

**Doctrine 자기적용:**
- Claude 그럴듯한 분석/전제 → 확신성이 강할수록 검증 강화.
- 게이트도 검증 대상 — *게이트 정의가 PT 측에서 hold 하는가?* 먼저 확인.
- *모달리티 전환은 호출 wrapper 정책의 재매핑*.  Lesson 17 의 일반화.

---

## §5. STAGE 9 로 가져갈 자산

### 검증된 코드 자산
- **`lance_mlx/vae_wan22.py`** — image (T=1) + video (T>1) 양 path 모두 PT byte-clean.
  STAGE 9 video DiT 의 *encode 측 (이해 조건)* / *decode 측 (생성 출력)* 모두 그대로 사용.
- **`feat_cache` 메커니즘** — temporal causal conv 의 streaming pattern 추상화 정확 작동.
  STAGE 9 video DiT 에 *spatiotemporal attention* / *3D RoPE* 신규 영역과 분리, VAE 쪽은
  이미 닫힌 박스.

### 검증 도구 (재사용 가능)
- `stage8_pt_video_smoke.py` — PT 정답지 생성 패턴 (synthetic N(0,1) 입력 + 4개 npy fixture).
  STAGE 9 에서 video DiT 첫 step PT 정답지 생성 시 같은 패턴 차용.
- `stage8_wanvae_compare.py` — 4-gate cos harness.  STAGE 9 t2v 첫 step 검증 시
  (latent_first / latent_uncond / latent_blend / x_hat) 같은 4-gate 패턴 적용 가능.
- §1 의 7개 block-by-block compare tool — STAGE 9 video DiT 의 block-by-block 검증 (예:
  spatiotemporal attention block) 패턴 차용.

### 교훈 (17개 누적; 14~17 STAGE 8 신규)
- Lesson 14 (state-bearing block: 출력 + 상태 모두 byte-diff)
- Lesson 15 (분포 형태로 폭주 vs 정상 판별, 절대 임계 박지 말 것)
- Lesson 16 (게이트 정의 자체가 가설, PT 측 hold 확인 선행)
- Lesson 17 (모달리티별 호출 wrapper 정책 재확인, 이전 stage 발견 silent 이전 금지)

### VERIFICATION_BACKLOG 변경
- STAGE 8 진입 시 열려 있던 `Wan2_2_VAE` T>1 path 미검증 — *closed*.
- 새로 열린 항목 (STAGE 9 진입에서 발화 가능):
  - Video DiT *spatiotemporal attention*: STAGE 8 VAE 의 spatial seam 가짜 봉합과 달리,
    attention 의 spatial chunking 은 *실재 seam* (window attention).  봉합 알고리즘 필요할 수
    있음 — 그러나 Lance 가 video 에 window attention 쓰는지 PT 확인 선행.
  - 3D mRoPE (T-axis 확장): STAGE 3 helpers (text+image 2D) 를 3D 로 확장.  
  - bf16 inference 모드 (만약 video latent 메모리 압박 시): 기존 IMPROVEMENTS 항목과 합류.

### IMPROVEMENTS 변경
- STAGE 8 §1.5 `first_chunk` default 불일치 (PT False vs MLX True) — *기록됨* (line 174-178).
  STAGE 5 image-path 재진입 또는 STAGE 9 video 진입 시점에 결정.

---

## §6. Code-reviewer pass

Reviewer: Opus, code-reviewer agent.  Scope: STAGE 8 §2 additions
(`vae_wan22.py:884-993` — `_count_causal_conv3d`, `Wan2_2_VAE.encode/decode` chunked
extension) + `tools/stage8_wanvae_compare.py` 전체.  §1 의 building blocks 는 STAGE 8 §1
(commit 7a9a581) 시점에 이미 cross-validated 되어 *이번 pass 범위 외*.

- **BLOCKING:** none.  Reviewer 검토 결과 — chunked 코드가 PT `vae2_2.py:759-813` 와
  line-for-line 일치, T==1 fast path 가 STAGE 5 caller contract 보존, slot allocation 이
  `mod.modules()` 로 PT `count_conv3d` 와 동일 traversal, `feat_idx[0]=0` reset 이 모든
  CausalConv3d 호출이 T 와 무관 unconditional 인 한 안전.

- **SUGGESTED 적용:**
  - S1 — `encode` return type hint `-> mx.array | tuple[mx.array, mx.array]` 추가 (정적 분석
    명확화).
  - S2 — `scale` direction (INV_STD, NOT std) 명시적 경고 docstring 강화 (encode + decode 양
    docstring 에 박음).  caller 가 `(mean, std)` 를 잘못 넘기는 silent 발산 방지.
  - S5 — `feat_idx[0] = 0` reset 라인에 *"conv-walk order must match chunk-0 every chunk"*
    inline comment + encode docstring 에 *"unconditional-conv-walk invariant"* 경고 추가.
    미래 누군가가 `if T > k: skip conv` 같은 가드를 추가했을 때 silent slot drift 방지.

- **SUGGESTED 미적용 (낮은 ROI):**
  - S3 — `return_logvar` boolean toggle: 현재 모든 caller 가 `mu` 만 원함, idiomatic 충분.
  - S4 — `_count_causal_conv3d` 결과 cache: encoder/decoder 각 26/34 modules walk, 호출당
    cheap.  JIT 도입 시 재평가.
  - S6 — `load_weights` try/except: dev harness, 스택트레이스로 충분.
  - S7 — T=1 path 와 chunked path 의 등가성 별도 테스트: 현재 둘이 다른 코드 경로, 만약 future
    fold 하면 의미 있음.  현재는 STAGE 5 fixtures (image path) + STAGE 8 fixtures (video path)
    각각 별도 정답지로 격리됨.

- **NITPICK** (미적용): typing style mixed (`Optional[X]` vs `X | None`), single-element
  `mx.concatenate` 최적화, harness `sys.path.insert` 의 cwd 의존성.

- **Regression check (실행 검증, read-only signature check 아님):**
  - STAGE 5 `stage5_roundtrip.py` — T=1 image VAE round-trip 64²/128²/256² 모두 PASS
    (cos 0.999198/0.999326/0.999392, PSNR 39.84/40.51/40.90 dB).  *원본 STAGE 5
    LEARNING_LOG 의 "PSNR 40 dB at 256²" 와 일치.*
  - STAGE 6 `stage6_t2i_smoke.py` — 30 step T2I 완료, ||v_cond|| 진행 패턴 정상
    (250→270→210), latent shape (1, 1, 32, 32, 48), VAE decode range [-1.481, +1.477] 정상.
  - STAGE 7 `stage7_ti2i_compare.py` — 3-forward PT vs MLX:
    `v_full=0.999632 / v_t_uncond=0.999875 / v_tv_uncond=0.999780`.
    *원본 STAGE 7 §3 LEARNING_LOG line 105-107 의 숫자와 정확히 일치 (byte-for-byte 보존).*
  - STAGE 8 `stage8_wanvae_compare.py` self-regression — 4 gates cos=1.0 재확인.
  - **결론:** T=1 fast path가 STAGE 8 확장으로 인해 변경되지 않음을 *동작 검증*으로 확정.
    signature 호환성 검증만으로는 STAGE 5 의 3개 silent bug 가 통과했었던 전례 (Lesson 1)
    감안 — 실행 검증이 필수.

---

## §7. Status / Next

**STAGE 8 종료.**  3D Causal Video VAE + temporal chunked decode 양방향 byte-clean.
4 gates × cos = 1.0.  17 lessons distilled.

**STAGE 9 진입:** Video DiT + t2v.  STAGE 6 무기 (flow matching, CFG 3-comp, per-step PT cos
diagnostic) 거의 그대로 재사용.  신규 영역:
- *Spatiotemporal attention* (PT `lance.py` video branch 직접 매핑 선행)
- *3D mRoPE* (T-axis 확장; STAGE 3 helpers 위에)
- *t2v wrapper* (`pipelines/t2v.py` 신설; t2i 의 video 확장)
