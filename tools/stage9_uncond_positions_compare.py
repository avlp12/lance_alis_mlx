"""STAGE 9 §1 단계 4-2 — UNCOND positions/mask PT 정답지 + MLX 빌더 byte-diff.

uncond sequence = full_text_ids[modality != 0] = [vis_start, IMG×N, vis_end]
PT path:
  uncond_mask         = i_sample_modality != 0
  uncond_text_ids     = current_text_ids[uncond_mask]
  uncond_split_lens   = split-level filter (text split drop, noise split keep)
  uncond_attn_modes   = ["noise"] (+ pad)
  uncond_positions    = get_rope_index(uncond_text_ids, image/video_grid_thw, ...)
                        → shift_position_ids (no-op for t2v)
  uncond_attn_mask    = process_attention_mask(modes, lens, BLOCK_SIZE=128)

MLX builder: PT get_rope_index 의 t2v 분기 (image case, second_per_grid_t=0) 의
*text 없는* 변형. text_len=1 (vis_start), st_idx=0, video span 모두 +1 offset.

검증: PT 단독 호출 결과 vs MLX 빌더 byte-diff → 0 diff 가 정답.
"""
from __future__ import annotations

import os
import sys
import importlib

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("refs/Lance"))
from tools._pt_smoke_common import install_pt_smoke_env
install_pt_smoke_env()

import numpy as np
import torch
from transformers import AutoTokenizer

qwen2_navit = importlib.import_module("modeling.lance.qwen2_navit")
PtQwen2ForCausalLM = qwen2_navit.Qwen2ForCausalLM
from data.common import shift_position_ids


# ---- spoof self (matches 단계 4) ----
class MockVisionConfig(dict):
    pass

class MockConfig:
    def __init__(self):
        self.image_token_id = 151655
        self.video_token_id = 151656
        self.vision_start_token_id = 151652
        self.vision_config = MockVisionConfig({
            "spatial_merge_size": 2,
            "tokens_per_second": 2,
            "temporal_patch_size": 2,
        })

class MockSelf:
    def __init__(self):
        self.config = MockConfig()


# ---- build full t2v sequence (text_template=False, matches v2) ----
USER_PROMPT = "A red panda riding a wave at sunset."
T_VIDEO = 5
H_PIX = W_PIX = 128
VAE_DOWN_TEMPORAL = 4
VAE_DOWN_SPATIAL = 16
SPATIAL_MERGE_SIZE = 2

t_lat = (T_VIDEO - 1) // VAE_DOWN_TEMPORAL + 1
h_lat = H_PIX // VAE_DOWN_SPATIAL
w_lat = W_PIX // VAE_DOWN_SPATIAL
num_vid_tokens = t_lat * h_lat * w_lat

tok = AutoTokenizer.from_pretrained("checkpoints/Lance-3B-MLX", use_fast=True)
BOS = tok.convert_tokens_to_ids("<|im_start|>")
EOS = tok.convert_tokens_to_ids("<|im_end|>")
VIS_START = tok.convert_tokens_to_ids("<|vision_start|>")
VIS_END = tok.convert_tokens_to_ids("<|vision_end|>")
IMG = 151655

prompt_ids = tok.encode(USER_PROMPT)
full_text_ids = [BOS] + prompt_ids + [EOS]
text_split_len = len(full_text_ids)
full_text_ids.append(VIS_START)
full_text_ids.extend([IMG] * num_vid_tokens)
full_text_ids.append(VIS_END)
video_split_len = num_vid_tokens + 2
L = text_split_len + video_split_len

# modality: text=0, video span (vis_start/IMG/vis_end)=1
modality = [0] * text_split_len + [1] * video_split_len

# ---- uncond_mask + uncond_text_ids ----
modality_t = torch.tensor(modality, dtype=torch.long)
uncond_mask_t = (modality_t != 0)
print(f"[uncond_mask] sum={int(uncond_mask_t.sum())} (expected video_split_len={video_split_len})")

full_text_ids_t = torch.tensor(full_text_ids, dtype=torch.long)
uncond_text_ids = full_text_ids_t[uncond_mask_t]
uncond_L = int(uncond_text_ids.shape[0])
print(f"[uncond_text_ids] shape={tuple(uncond_text_ids.shape)} L={uncond_L}")
print(f"  first 3: {uncond_text_ids[:3].tolist()}  (expected: [vis_start, IMG, IMG])")
print(f"  last 2:  {uncond_text_ids[-2:].tolist()}  (expected: [IMG, vis_end])")

# ---- PT get_rope_index for uncond ----
video_grid_thw = torch.tensor([[t_lat, h_lat * SPATIAL_MERGE_SIZE, w_lat * SPATIAL_MERGE_SIZE]])
mock = MockSelf()
print("\n[PT call] get_rope_index for uncond sequence ...")
uncond_pos_ids_pt, uncond_deltas = PtQwen2ForCausalLM.get_rope_index(
    mock,
    input_ids=uncond_text_ids.unsqueeze(0),
    image_grid_thw=video_grid_thw,
    video_grid_thw=video_grid_thw,
    second_per_grid_ts=torch.tensor([1.0]),
    attention_mask=torch.ones((1, uncond_L), dtype=torch.long),
)
pos_pt_np = uncond_pos_ids_pt.numpy()
print(f"[PT] uncond_positions: shape={pos_pt_np.shape}")
print(f"  vis_start (idx=0):     (t,h,w) = ({pos_pt_np[0,0,0]}, {pos_pt_np[1,0,0]}, {pos_pt_np[2,0,0]})")
print(f"  IMG[0]    (idx=1):     ({pos_pt_np[0,0,1]}, {pos_pt_np[1,0,1]}, {pos_pt_np[2,0,1]})")
print(f"  IMG[1]    (idx=2):     ({pos_pt_np[0,0,2]}, {pos_pt_np[1,0,2]}, {pos_pt_np[2,0,2]})")
print(f"  IMG[8]    (idx=9):     ({pos_pt_np[0,0,9]}, {pos_pt_np[1,0,9]}, {pos_pt_np[2,0,9]})")
print(f"  IMG[127]  (idx=128):   ({pos_pt_np[0,0,128]}, {pos_pt_np[1,0,128]}, {pos_pt_np[2,0,128]})")
print(f"  vis_end   (idx=129):   ({pos_pt_np[0,0,129]}, {pos_pt_np[1,0,129]}, {pos_pt_np[2,0,129]})")

# shift_position_ids — t2v 의 uncond_attn_modes=["noise"] 도 no-op
uncond_split_lens = [video_split_len]
uncond_attn_modes = ["noise"]
uncond_modality = modality_t[uncond_mask_t]
shifted = shift_position_ids(
    uncond_pos_ids_pt.clone(),
    pos_shift=1000,
    attn_modes=uncond_attn_modes,
    split_lens=uncond_split_lens,
    shift_attn_mode=["full_noise", "full"],
    pro_type=10,
    i_sample_task=torch.zeros(uncond_L, dtype=torch.long),
    i_sample_modality=uncond_modality,
)
no_op = np.array_equal(shifted.numpy(), pos_pt_np)
print(f"\n[shift] no-op for uncond? {no_op}")

# ---- MLX uncond_position builder ----
def build_t2v_uncond_positions(t_lat: int, h_lat: int, w_lat: int, uncond_L: int) -> np.ndarray:
    """MLX side uncond position builder.

    uncond_text_ids = [vis_start, IMG×(t·h·w), vis_end].  text 가 없는 시퀀스.

    PT get_rope_index 의 t2v 분기 (image case, second_per_grid_t=0):
      text_len = ed - st = (vis_start) → 1 token (only vis_start before IMG)
      st_idx   = 0
      llm_pos_ids_list[0] = arange(1) + 0 = [0]          # vis_start
      llm_pos_ids_list[1] = stack(t,h,w) + 1 + 0          # video, all + text_len
      post-video (vis_end): st_idx = max + 1
    """
    n_video = t_lat * h_lat * w_lat
    pos = np.zeros((3, 1, uncond_L), dtype=np.int64)

    # 1) vis_start (idx 0): (0, 0, 0)
    # already 0 from zeros

    # 2) video span (idx 1 .. n_video)
    offset = 1   # = text_len + st_idx = 1 + 0
    t_index = np.zeros(n_video, dtype=np.int64)
    h_index = np.tile(np.repeat(np.arange(h_lat), w_lat), t_lat)
    w_index = np.tile(np.arange(w_lat), t_lat * h_lat)
    pos[0, 0, 1:1 + n_video] = t_index + offset
    pos[1, 0, 1:1 + n_video] = h_index + offset
    pos[2, 0, 1:1 + n_video] = w_index + offset

    # 3) vis_end (post-video): st_idx = max + 1
    if 1 + n_video < uncond_L:
        # video max = offset + max(h_lat, w_lat) - 1 = 1 + 7 = 8 (for h=w=8)
        # vis_start (0,0,0) is below this, so max = video max
        max_pos = offset + max(h_lat, w_lat) - 1
        st_idx_post = max_pos + 1
        post_len = uncond_L - (1 + n_video)
        for axis in range(3):
            pos[axis, 0, 1 + n_video:] = np.arange(post_len) + st_idx_post
    return pos

pos_mlx = build_t2v_uncond_positions(t_lat, h_lat, w_lat, uncond_L)

# byte-diff
diff_mask = pos_pt_np != pos_mlx
print(f"\n[byte-diff] MLX uncond builder vs PT get_rope_index:")
print(f"  byte-identical: {np.array_equal(pos_pt_np, pos_mlx)}")
if not np.array_equal(pos_pt_np, pos_mlx):
    rows = np.where(diff_mask.any(axis=(0, 1)))[0]
    print(f"  diff at {len(rows)} sequence positions: idxs={rows.tolist()[:10]}")
    for r in rows[:5]:
        print(f"    idx {r}: PT={pos_pt_np[:,0,r].tolist()}  MLX={pos_mlx[:,0,r].tolist()}")
else:
    print("  ✓ MLX uncond builder = PT 정확 일치 (단계 4-2 정답지)")

# ---- save fixture ----
np.save("out/stage9_pt_video_uncond_pos_ids.npy", pos_pt_np)
np.save("out/stage9_pt_video_uncond_text_ids.npy", uncond_text_ids.numpy())
print(f"\n[save] out/stage9_pt_video_uncond_pos_ids.npy   shape={pos_pt_np.shape}")
print(f"       out/stage9_pt_video_uncond_text_ids.npy  shape={uncond_text_ids.shape}")
print(f"       (uncond_L={uncond_L}, uncond_split_lens={uncond_split_lens}, uncond_attn_modes={uncond_attn_modes})")
