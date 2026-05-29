"""STAGE 9 — t2v sequence helper (순수 MLX, runtime PT 의존성 0).

이전 (v1) 은 `ValidationDataset.__new__` + `t2v_sample(0)` 호출 — runtime
에 `refs/Lance` PT 코드 + flash_attn shim 필요.  공개 모순.

현재 (v2): tokenizer(HF) + manual token concat.  PT `t2v_sample(text_template=True)`
출력과 *byte-identical* 검증됨 (5-gate, `tools/stage9_mlx_sequence_byte_diff.py`):
  - input_ids ✓
  - sample_modality ✓
  - split_lens ✓
  - attn_modes ✓
  - packed_vae_token_indexes ✓

PT 정답지는 검증 시점에만 필요 — 추론 런타임은 순수 MLX.

PT decoded sequence layout:
  split 0 (40, modality=-1): [im_start, "system", \n, *sys_tokens, im_end, \n, im_start, "user", \n]
  split 1 ( 9, modality= 0): user prompt tokens
  split 2 ( 5, modality=-1): [im_end, \n, im_start, "assistant", \n]
  split 3 (130, modality= 1): [vis_start, *video_pad×N, vis_end]
  split 4 ( 1, modality=-1): [im_end]   (no \n after)
"""
from __future__ import annotations

import numpy as np


# Special token IDs — Qwen2.5 vocab, byte-identical with PT validation_dataset.
BOS_TOKEN_ID = 151644       # <|im_start|>
EOS_TOKEN_ID = 151645       # <|im_end|>
VIS_START_ID = 151652       # <|vision_start|>
VIS_END_ID   = 151653       # <|vision_end|>
IMAGE_PAD_ID = 151655       # <|image_pad|>
VIDEO_PAD_ID = 151656       # <|video_pad|>
NEWLINE_ID   = 198          # '\n'

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
    """순수 MLX (HF tokenizer + manual concat) t2v 시퀀스 빌더.

    PT `ValidationDataset.t2v_sample(text_template=True)` 와 byte-identical
    (5-gate verified at `tools/stage9_mlx_sequence_byte_diff.py`).

    Args:
        prompt: user text input.
        tokenizer: HF Qwen2 tokenizer.
        num_frames / H / W: video shape (production defaults 50/768/768).
        vae_down_t / vae_down_s: VAE downsample factors (temporal/spatial).

    Returns dict:
        {
          'input_ids':                np.ndarray (L,) int64,
          'sample_modality':          np.ndarray (L,) int64 — values: -1 (system_prompt), 0 (text), 1 (noise),
          'sample_task':              np.ndarray (L,) int64 — all zeros (task t2v=0),
          'packed_vae_token_indexes': np.ndarray (n_video,) int64,
          'split_lens':               list[int] — [40, 9, 5, 130, 1] for production case,
          'attn_modes':               list[str] — ['causal', 'causal', 'causal', 'noise', 'causal'],
          'L':                        int,
          'video_grid_thw':           np.ndarray (1, 3),
          'video_size':               np.ndarray (1, 3),
        }
    """
    # Latent shape
    t_lat = (num_frames - 1) // vae_down_t + 1
    h_lat = H // vae_down_s
    w_lat = W // vae_down_s
    n_video = t_lat * h_lat * w_lat

    # Encode content tokens (add_special_tokens=False — boundary 직접)
    sys_tokens  = tokenizer.encode(T2V_SYSTEM_PROMPT, add_special_tokens=False)
    user_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    SYSTEM_LBL  = tokenizer.encode("system",    add_special_tokens=False)
    USER_LBL    = tokenizer.encode("user",      add_special_tokens=False)
    ASST_LBL    = tokenizer.encode("assistant", add_special_tokens=False)

    # PT 정답지와 byte-identical 패턴 (5-gate 검증됨)
    split_0 = ([BOS_TOKEN_ID] + SYSTEM_LBL + [NEWLINE_ID] + sys_tokens
               + [EOS_TOKEN_ID, NEWLINE_ID, BOS_TOKEN_ID] + USER_LBL + [NEWLINE_ID])
    split_1 = user_tokens
    split_2 = [EOS_TOKEN_ID, NEWLINE_ID, BOS_TOKEN_ID] + ASST_LBL + [NEWLINE_ID]
    split_3 = [VIS_START_ID] + [VIDEO_PAD_ID] * n_video + [VIS_END_ID]
    split_4 = [EOS_TOKEN_ID]

    input_ids = split_0 + split_1 + split_2 + split_3 + split_4
    L = len(input_ids)

    sample_modality = (
        [-1] * len(split_0) + [0] * len(split_1) + [-1] * len(split_2)
        + [1] * len(split_3) + [-1] * len(split_4)
    )
    split_lens = [len(split_0), len(split_1), len(split_2), len(split_3), len(split_4)]
    attn_modes = ["causal", "causal", "causal", "noise", "causal"]

    vis_start_idx = len(split_0) + len(split_1) + len(split_2)
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


# Backward compat alias (v1 caller가 있으면 동일 결과 — byte-identical, 단 PT 의존 0)
build_t2v_sequence_pt = build_t2v_sequence_mlx
