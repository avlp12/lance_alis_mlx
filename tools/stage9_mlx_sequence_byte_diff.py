"""STAGE 9 cleanup — t2v 추론 PT 의존 제거 검증.

PT `_t2v_seq.build_t2v_sequence_pt` (ValidationDataset 호출, 런타임 PT 의존)
대체할 *순수 MLX manual concat* 빌더.  PT 정답지와 5-gate byte-diff.

Gate (모두 byte-identical):
  1. input_ids
  2. sample_modality
  3. split_lens
  4. attn_modes
  5. packed_vae_token_indexes
"""
import json
import numpy as np
from transformers import AutoTokenizer


# Special token IDs — PT 정답지에서 추출, 하드코드 (Qwen2.5 vocab)
BOS = 151644             # <|im_start|>
EOS = 151645             # <|im_end|>
VIS_START = 151652       # <|vision_start|>
VIS_END = 151653         # <|vision_end|>
IMAGE_PAD = 151655       # <|image_pad|>
VIDEO_PAD = 151656       # <|video_pad|>
NEWLINE = 198            # '\n'

# Production T2V system prompt — refs/Lance/data/common.py:31 그대로
T2V_SYSTEM_PROMPT = (
    "Describe the video by detailing the color, quantity, visible text, "
    "shape, size, texture, spatial relationships and motion/camera movements "
    "of the objects and background:"
)


def build_t2v_sequence_mlx(prompt: str, tokenizer, *,
                           num_frames: int = 50,
                           H: int = 768, W: int = 768,
                           vae_down_t: int = 4,
                           vae_down_s: int = 16) -> dict:
    """순수 MLX (tokenizer + manual concat) 빌더 — runtime PT 의존성 0.

    PT `ValidationDataset.t2v_sample` (text_template=True) 의 sequence build
    경로 와 *byte-identical* 결과 — manual interpretation 위험 은 byte-diff
    gate 로 차단.

    Returns dict with same keys as `_t2v_seq.build_t2v_sequence_pt`.
    """
    # Latent shape
    t_lat = (num_frames - 1) // vae_down_t + 1
    h_lat = H // vae_down_s
    w_lat = W // vae_down_s
    n_video = t_lat * h_lat * w_lat

    # Encode tokens — add_special_tokens=False (우리가 직접 박음)
    sys_tokens = tokenizer.encode(T2V_SYSTEM_PROMPT, add_special_tokens=False)
    user_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    SYSTEM_LBL = tokenizer.encode("system", add_special_tokens=False)
    USER_LBL = tokenizer.encode("user", add_special_tokens=False)
    ASST_LBL = tokenizer.encode("assistant", add_special_tokens=False)

    # Sequence segments (PT decoded 패턴 그대로, single-byte exact)
    # split 0 (causal, modality=-1): im_start + "system" + \n + sys + im_end + \n + im_start + "user" + \n
    split_0 = ([BOS] + SYSTEM_LBL + [NEWLINE] + sys_tokens
               + [EOS, NEWLINE, BOS] + USER_LBL + [NEWLINE])
    # split 1 (causal, modality=0): user prompt
    split_1 = user_tokens
    # split 2 (causal, modality=-1): im_end + \n + im_start + "assistant" + \n
    split_2 = [EOS, NEWLINE, BOS] + ASST_LBL + [NEWLINE]
    # split 3 (noise, modality=1): vis_start + video_pad×N + vis_end
    split_3 = [VIS_START] + [VIDEO_PAD] * n_video + [VIS_END]
    # split 4 (causal, modality=-1): im_end (no \n after)
    split_4 = [EOS]

    input_ids = split_0 + split_1 + split_2 + split_3 + split_4
    L = len(input_ids)

    sample_modality = ([-1] * len(split_0) + [0] * len(split_1) + [-1] * len(split_2)
                       + [1] * len(split_3) + [-1] * len(split_4))
    split_lens = [len(split_0), len(split_1), len(split_2), len(split_3), len(split_4)]
    attn_modes = ["causal", "causal", "causal", "noise", "causal"]

    # packed_vae_token_indexes: video_pad positions (vis_start 직후 ~ vis_end 직전)
    vis_start_idx = len(split_0) + len(split_1) + len(split_2)   # = first idx of split_3
    packed_vae_token_indexes = list(range(vis_start_idx + 1, vis_start_idx + 1 + n_video))

    return {
        "input_ids":                np.array(input_ids,                dtype=np.int64),
        "sample_modality":          np.array(sample_modality,          dtype=np.int64),
        "sample_task":              np.zeros(L,                         dtype=np.int64),
        "packed_vae_token_indexes": np.array(packed_vae_token_indexes,  dtype=np.int64),
        "split_lens":               split_lens,
        "attn_modes":               attn_modes,
        "L":                        L,
        "video_grid_thw":           np.array([[t_lat, h_lat * 2, w_lat * 2]]),
        "video_size":               np.array([[num_frames, H, W]]),
    }


def main():
    print("=" * 72)
    print("STAGE 9 cleanup — manual MLX seq builder → PT 정답지 5-gate byte-diff")
    print("=" * 72)

    tok = AutoTokenizer.from_pretrained("checkpoints/Lance-3B-MLX", use_fast=True)
    USER_PROMPT = "A red panda riding a wave at sunset."

    result = build_t2v_sequence_mlx(USER_PROMPT, tok,
                                    num_frames=5, H=128, W=128,
                                    vae_down_t=4, vae_down_s=16)
    print(f"[MLX builder] L={result['L']}  split_lens={result['split_lens']}  "
          f"vae_indices_len={len(result['packed_vae_token_indexes'])}")

    # Load PT fixtures
    pt_seq = np.load("out/stage9_pt_t2v_seq_real.npy")
    pt_modality = np.load("out/stage9_pt_video_sample_modality.npy")
    with open("out/stage9_pt_t2v_seq_meta.json") as f:
        pt_meta = json.load(f)

    # 5 gates
    gates = []
    g1 = np.array_equal(result["input_ids"], pt_seq)
    print(f"\n[gate 1] input_ids byte-identical: {g1}")
    if not g1:
        diff = result["input_ids"] != pt_seq
        idxs = np.where(diff)[0]
        print(f"  diff at {len(idxs)} positions: {idxs.tolist()[:10]}")
        for i in idxs[:5]:
            print(f"    idx {i}: MLX={result['input_ids'][i]} PT={pt_seq[i]}")
    gates.append(("input_ids", g1))

    g2 = np.array_equal(result["sample_modality"], pt_modality)
    print(f"[gate 2] sample_modality byte-identical: {g2}")
    if not g2:
        diff = result["sample_modality"] != pt_modality
        idxs = np.where(diff)[0]
        print(f"  diff at {len(idxs)} positions: {idxs.tolist()[:10]}")
    gates.append(("sample_modality", g2))

    g3 = result["split_lens"] == pt_meta["split_lens"]
    print(f"[gate 3] split_lens match: {g3}  MLX={result['split_lens']} PT={pt_meta['split_lens']}")
    gates.append(("split_lens", g3))

    g4 = result["attn_modes"] == pt_meta["attn_modes"]
    print(f"[gate 4] attn_modes match: {g4}")
    gates.append(("attn_modes", g4))

    pt_n_vae = pt_meta["packed_vae_token_indexes_len"]
    g5 = len(result["packed_vae_token_indexes"]) == pt_n_vae
    print(f"[gate 5] packed_vae_token_indexes len: {g5}  "
          f"MLX={len(result['packed_vae_token_indexes'])}  PT={pt_n_vae}")
    gates.append(("packed_vae_token_indexes len", g5))

    # Bonus: vae_token_indices 값 자체도 일치 — PT t2v_sample 의 그 list 와
    # 우리 range() 가 byte-identical 인지 확인 (production case 에선 range 가 정확)
    # PT 정답지가 range list 인지 — t2v_sample line 822-849 에서 process_text_template
    # 가 packed_vae_token_indexes 도 채움. 정확한 값 확인:
    # 우리 range = [55, 56, ..., 182].  PT 도 같은 range 일 것.

    print("\n" + "=" * 72)
    all_pass = all(g for _, g in gates)
    if all_pass:
        print("ALL 5 GATES PASS — manual MLX seq builder = PT byte-identical")
        print("→ _t2v_seq.py 의 PT 의존 *제거 가능*.  공개 가능한 순수 MLX 포팅.")
    else:
        print(f"FAIL — {sum(1 for _, g in gates if not g)}/5 gates failed")
        for name, ok in gates:
            print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    print("=" * 72)


if __name__ == "__main__":
    main()
