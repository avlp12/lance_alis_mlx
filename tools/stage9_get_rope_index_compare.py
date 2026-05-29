"""STAGE 9 §1 단계 4 — get_rope_index 단독 byte-diff harness.

옵션 B (사용자 결정): PT Lance 자체 `Qwen2ForCausalLM.get_rope_index`
(refs/Lance/modeling/lance/qwen2_navit.py:1120) 를 직접 import + 호출.
우리 해석 0. self.config 만 spoof.

NOTE: transformers 5.9.0 의 `Qwen2_5_VLModel.get_rope_index` 와는 *시그너처가
다름* (mm_token_type_ids 신규 인자) — 사용 안 함.

검증 시나리오:
  1. PT t2v_sample text_template=False 케이스의 input_ids + video_grid_thw 로
     get_rope_index 호출 → position_ids (3, 1, L)
  2. shift_position_ids (pro_type=10) 적용 (validation_gen line 616-625 와 동일)
  3. v2 PT smoke 의 manual positions 와 byte-diff (audit — manual 의 차이 정확 확인)
  4. 결과 dump → out/stage9_pt_video_pos_ids_real.npy
     이게 §1 t2v.py 가 사용할 production positions 정답지.
"""
from __future__ import annotations

import os
import sys
import importlib

# STAGE 9+ PT smoke 공용 환경 셋업 (Lesson E containment)
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("refs/Lance"))
from tools._pt_smoke_common import install_pt_smoke_env, pt_layer_mask  # noqa: F401
install_pt_smoke_env()

import numpy as np
import torch
from transformers import AutoTokenizer

qwen2_navit = importlib.import_module("modeling.lance.qwen2_navit")
PtQwen2ForCausalLM = qwen2_navit.Qwen2ForCausalLM
from data.common import shift_position_ids


# ---- production config (refs/Lance-3B-Video-MLX/config.json + tokens_per_second) ----
class MockVisionConfig(dict):
    """Spoof for self.config.vision_config dict access (e.g. ['spatial_merge_size'])."""
    pass


class MockConfig:
    """Spoof for self.config — only the attrs get_rope_index reads."""
    def __init__(self):
        self.image_token_id = 151655          # <|image_pad|>
        self.video_token_id = 151656          # <|video_pad|>
        self.vision_start_token_id = 151652   # <|vision_start|>
        self.vision_config = MockVisionConfig({
            "spatial_merge_size": 2,
            "tokens_per_second": 2,            # from Lance-3B-Video-MLX vit_config
            "temporal_patch_size": 2,
        })


class MockSelf:
    def __init__(self):
        self.config = MockConfig()


# ---- build a PT t2v_sample text_template=False sequence (matches our v2 PT smoke) ----
USER_PROMPT = "A red panda riding a wave at sunset."
T_VIDEO = 5
H_PIX = W_PIX = 128
VAE_DOWN_TEMPORAL = 4
VAE_DOWN_SPATIAL = 16

t_lat = (T_VIDEO - 1) // VAE_DOWN_TEMPORAL + 1            # 2
h_lat = H_PIX // VAE_DOWN_SPATIAL                          # 8
w_lat = W_PIX // VAE_DOWN_SPATIAL                          # 8
num_vid_tokens = t_lat * h_lat * w_lat                     # 128
SPATIAL_MERGE_SIZE = 2

tok = AutoTokenizer.from_pretrained("checkpoints/Lance-3B-MLX", use_fast=True)

BOS_TOKEN_ID = tok.convert_tokens_to_ids("<|im_start|>")
EOS_TOKEN_ID = tok.convert_tokens_to_ids("<|im_end|>")
START_OF_IMAGE = tok.convert_tokens_to_ids("<|vision_start|>")
END_OF_IMAGE = tok.convert_tokens_to_ids("<|vision_end|>")
IMG_TOKEN_ID = 151655

# PT validation_dataset.t2v_sample (text_template=False) 그대로
text_ids_raw = tok.encode(USER_PROMPT)
text_ids = [BOS_TOKEN_ID] + text_ids_raw + [EOS_TOKEN_ID]
text_split_len = len(text_ids)
text_ids.append(START_OF_IMAGE)
text_ids.extend([IMG_TOKEN_ID] * num_vid_tokens)
text_ids.append(END_OF_IMAGE)
video_split_len = num_vid_tokens + 2
L = text_split_len + video_split_len

print(f"[seq] L={L}  text_split_len={text_split_len}  video_split_len={video_split_len}")
print(f"      t_lat={t_lat}  h_lat={h_lat}  w_lat={w_lat}  num_vid_tokens={num_vid_tokens}")

# ---- get_rope_index 호출 ----
input_ids = torch.tensor(text_ids, dtype=torch.long).unsqueeze(0)   # (1, L)
# PT validation_dataset.t2v_sample line 856-857:
#   vae_video_grid_thw = [[t, h * spatial_merge_size, w * spatial_merge_size]]
#   video_grid_thw    = [[[t, h * spatial_merge_size, w * spatial_merge_size]]]
# PT validation_gen line 605: grid_thw_rope = video_grid_thw[i_sample]
#   → grid_thw_rope = [[t, h*2, w*2]] = [[2, 16, 16]]
video_grid_thw = torch.tensor([[t_lat, h_lat * SPATIAL_MERGE_SIZE, w_lat * SPATIAL_MERGE_SIZE]])
second_per_grid_ts = torch.tensor([1.0])

mock_self = MockSelf()

print("\n[call] PT Lance Qwen2ForCausalLM.get_rope_index (unbound, spoof self) ...")
position_ids, mrope_deltas = PtQwen2ForCausalLM.get_rope_index(
    mock_self,
    input_ids=input_ids,
    image_grid_thw=video_grid_thw,   # PT validation_gen line 610: 둘 다 같은 값 전달
    video_grid_thw=video_grid_thw,
    second_per_grid_ts=second_per_grid_ts,
    attention_mask=None,
)
print(f"[out] position_ids: shape={tuple(position_ids.shape)} dtype={position_ids.dtype}")
print(f"      rope_deltas:  shape={tuple(mrope_deltas.shape)}  value={mrope_deltas.flatten().tolist()}")

# Inspect: text positions, video positions, post-video text positions
pos_np = position_ids.numpy()
print(f"\n[positions] (3, 1, L) shape — first axis = (t, h, w)")
print(f"  text head [0..5]:        t={pos_np[0,0,:6].tolist()}, "
      f"h={pos_np[1,0,:6].tolist()}, w={pos_np[2,0,:6].tolist()}")
print(f"  text tail (BEFORE vis):  t={pos_np[0,0,text_split_len-1]}, "
      f"h={pos_np[1,0,text_split_len-1]}, w={pos_np[2,0,text_split_len-1]}")
print(f"  vis_start (idx={text_split_len}): t={pos_np[0,0,text_split_len]}, "
      f"h={pos_np[1,0,text_split_len]}, w={pos_np[2,0,text_split_len]}")
vis_first_img = text_split_len + 1
print(f"  IMG[0] (idx={vis_first_img}): t={pos_np[0,0,vis_first_img]}, "
      f"h={pos_np[1,0,vis_first_img]}, w={pos_np[2,0,vis_first_img]}")
print(f"  IMG[1] (idx={vis_first_img+1}): t={pos_np[0,0,vis_first_img+1]}, "
      f"h={pos_np[1,0,vis_first_img+1]}, w={pos_np[2,0,vis_first_img+1]}")
last_img = vis_first_img + num_vid_tokens - 1
print(f"  IMG[last] (idx={last_img}): t={pos_np[0,0,last_img]}, "
      f"h={pos_np[1,0,last_img]}, w={pos_np[2,0,last_img]}")
print(f"  vis_end (idx={L-1}): t={pos_np[0,0,L-1]}, h={pos_np[1,0,L-1]}, w={pos_np[2,0,L-1]}")

# ---- shift_position_ids (validation_gen line 616-625) ----
# t2v 의 attn_modes = ["causal", "noise"]; sample_modality: text=0, video=1
# shift_attn_mode=["full_noise", "full"] — t2v 의 attn_modes 어느 것도 매칭 안 됨 → shift 안 함
# pro_type=10 의 modality 기반: modality==1 (noise) ← modality==2 (cond) copy
#   → t2v 는 modality 2 (ref_source) 가 없음 → noise 자기 자신 그대로
# 즉 t2v 에서 shift_position_ids 가 *완전 no-op* 일 가능성. 확인.
split_lens = [text_split_len, video_split_len]
attn_modes = ["causal", "noise"]
sample_modality = [0] * text_split_len + [1] * video_split_len
sample_task = [0] * L                  # TASK_T2V=0

shifted = shift_position_ids(
    position_ids.clone(),
    pos_shift=1000,
    attn_modes=attn_modes,
    split_lens=split_lens,
    shift_attn_mode=["full_noise", "full"],
    pro_type=10,
    i_sample_task=torch.tensor(sample_task),
    i_sample_modality=torch.tensor(sample_modality),
)
shifted_np = shifted.numpy()
is_no_op = np.array_equal(shifted_np, pos_np)
print(f"\n[shift_position_ids] no-op for t2v (attn_modes=['causal','noise'])? {is_no_op}")
if not is_no_op:
    print(f"  max-abs diff: {int(np.abs(shifted_np - pos_np).max())}")

# ---- 비교: v2 PT smoke 의 manual positions (apply_qwen_2_5_vl_pos_emb=False 케이스) ----
v2_pos = np.load("out/stage9_pt_video_current_pos_ids.npy")
print(f"\n[byte-diff] PT get_rope_index 결과 (apply_qwen_2_5_vl_pos_emb=TRUE) "
      f"vs v2 manual (apply_qwen_2_5_vl_pos_emb=FALSE):")
print(f"  PT shape: {pos_np.shape}  v2 shape: {v2_pos.shape}")
print(f"  byte-identical: {np.array_equal(pos_np, v2_pos)}")
if not np.array_equal(pos_np, v2_pos):
    # 차이 표
    diff_axes = (pos_np != v2_pos).any(axis=(0, 1))
    diff_idxs = np.where(diff_axes)[0]
    print(f"  positions different at {len(diff_idxs)} sequence positions: idxs={diff_idxs.tolist()[:10]}{'...' if len(diff_idxs)>10 else ''}")
    # text region
    text_diff = (pos_np[:,:,:text_split_len] != v2_pos[:,:,:text_split_len]).sum()
    video_diff = (pos_np[:,:,text_split_len:] != v2_pos[:,:,text_split_len:]).sum()
    print(f"  text region (0..{text_split_len}) diff entries: {int(text_diff)}")
    print(f"  video region ({text_split_len}..{L}) diff entries: {int(video_diff)}")

# ---- save fixture ----
os.makedirs("out", exist_ok=True)
np.save("out/stage9_pt_video_pos_ids_real.npy", shifted_np)
print(f"\n[save] out/stage9_pt_video_pos_ids_real.npy  shape={shifted_np.shape}")
print("       (이게 §1 t2v.py 의 production positions 정답지)")
