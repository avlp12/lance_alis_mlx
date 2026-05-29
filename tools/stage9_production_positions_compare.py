"""STAGE 9 §1 단계 4-3 — production 시퀀스(L=185) 기준 positions/mask 재검증.

§0 단계 4 의 positions 는 *text_template=False* 시퀀스(L=141) 기준.
production 은 text_template=True (L=185, split_lens=[40,9,5,130,1], modality 에 system(-1) 추가).
교훈 8 (단순 ≠ production) → 재검증 필수.

검증:
1. PT get_rope_index(production input_ids=185, video_grid_thw) → 새 full positions
2. shift_position_ids 적용 (production attn_modes)
3. MLX build_t2v_positions(text_split_len_new, t_lat, h_lat, w_lat, L=185) → byte-diff
4. uncond_mask = modality != 0 → 어떤 split keep (production 에선 system+noise keep?)
5. uncond input_ids + uncond positions 재호출
6. MLX uncond positions byte-diff
7. full attn_mask + uncond attn_mask 재계산 + byte-diff (split_lens 새 값 기준)
"""
from __future__ import annotations

import json
import os
import sys
import importlib

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("refs/Lance"))
from tools._pt_smoke_common import install_pt_smoke_env
install_pt_smoke_env()

import numpy as np
import torch
import mlx.core as mx
from data.data_utils import create_sparse_mask
from data.common import shift_position_ids
from lance_mlx.attn_mask import build_lance_attention_mask

qwen2_navit = importlib.import_module("modeling.lance.qwen2_navit")
PtQwen2ForCausalLM = qwen2_navit.Qwen2ForCausalLM


# ---- production fixture load ----
print("[load] production sequence fixture (단계 4-3 입력) ...")
pt_seq = np.load("out/stage9_pt_t2v_seq_real.npy")
with open("out/stage9_pt_t2v_seq_meta.json") as f:
    meta = json.load(f)
L = meta["L_sequence"]
split_lens = meta["split_lens"]
attn_modes = meta["attn_modes"]
t_lat, h_lat, w_lat = meta["video_grid_thw"]
SPATIAL_MERGE_SIZE = 2
print(f"  L={L}  split_lens={split_lens}  attn_modes={attn_modes}")
print(f"  video_grid_thw={meta['video_grid_thw']}")


# ---- spoof for PT get_rope_index ----
class MockConfig:
    image_token_id = 151655
    video_token_id = 151656
    vision_start_token_id = 151652
    vision_config = {"spatial_merge_size": 2, "tokens_per_second": 2, "temporal_patch_size": 2}

class MockSelf:
    def __init__(self):
        self.config = MockConfig()

# ---- modality reconstruction from sequence ----
# sample_modality 는 t2v_sample 안에서 process_text_template 가 채움.
# 우리는 fixture 에 modality_unique 만 받음 — 직접 reconstruct 필요.
# Strategy: PT t2v_sample 재호출해서 sample_modality dump.
# Or: derive from PT sequence by inspecting span boundaries.
# 가장 깨끗: 재호출 (cheap, 1초). 이미 fixture build harness 가 있음.
import types as _types
sys.modules.setdefault("decord", _types.ModuleType("decord"))
import decord
decord.cpu = lambda x=0: None
decord.VideoReader = object
_video_dir = os.path.abspath("refs/Lance/data/video")
_pkg_video = _types.ModuleType("data.video"); _pkg_video.__path__ = [_video_dir]
sys.modules["data.video"] = _pkg_video
_pkg_sampler = _types.ModuleType("data.video.sampler"); _pkg_sampler.__path__ = [_video_dir + "/sampler"]
sys.modules["data.video.sampler"] = _pkg_sampler

vd_mod = importlib.import_module("data.datasets_custom.validation_dataset")
ValidationDataset = vd_mod.ValidationDataset

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("checkpoints/Lance-3B-MLX", use_fast=True)
new_token_ids = {
    "bos_token_id": tok.convert_tokens_to_ids("<|im_start|>"),
    "eos_token_id": tok.convert_tokens_to_ids("<|im_end|>"),
    "start_of_image": tok.convert_tokens_to_ids("<|vision_start|>"),
    "end_of_image": tok.convert_tokens_to_ids("<|vision_end|>"),
    "image_token_id": 151655,
}
class MockDataConfig:
    task = "t2v"; num_frames = 5; H = 128; W = 128
    vae_downsample = (4, 16, 16); text_template = True
    resolution = "video_480p"; max_duration = 6.0

ds = ValidationDataset.__new__(ValidationDataset)
ds.tokenizer = tok; ds.new_token_ids = new_token_ids
ds.bos_token_id = new_token_ids["bos_token_id"]; ds.eos_token_id = new_token_ids["eos_token_id"]
ds.start_of_image = new_token_ids["start_of_image"]; ds.end_of_image = new_token_ids["end_of_image"]
ds.image_token_id = new_token_ids["image_token_id"]
ds.data_config = MockDataConfig(); ds.system_prompt_type = "SP0"; ds.sample_task = "t2v"
ds.data = [{"data": "A red panda riding a wave at sunset.", "index": "0", "original_prompt_en": "A red panda riding a wave at sunset."}]
ds.sample = ds.set_sequence_status(); ds.frame_condition_idx = []
result = ds.t2v_sample(0)

sample_modality_pt = result["sample_modality"]
sample_modality = sample_modality_pt.numpy() if hasattr(sample_modality_pt, "numpy") else np.array(sample_modality_pt)
sample_task = np.array(result["sample_task"].numpy() if hasattr(result["sample_task"], "numpy") else result["sample_task"])
packed_vae_token_indexes = np.array(result["packed_vae_token_indexes"])
print(f"\n[modality] unique={sorted(set(sample_modality.tolist()))}")
# Split-level breakdown
cursor = 0
print(f"[split-level modality]")
for i, (sl, am) in enumerate(zip(split_lens, attn_modes)):
    sub_mod = sample_modality[cursor:cursor + sl]
    uniq = sorted(set(sub_mod.tolist()))
    print(f"  split {i} (len={sl}, mode={am}): modality unique={uniq}")
    cursor += sl

# Find vis_start position → text_split_len_new
vs_idx = int(np.where(pt_seq == new_token_ids["start_of_image"])[0][0])
text_split_len_new = vs_idx     # vis_start idx = text_split_len_new
n_video = sum(1 for x in packed_vae_token_indexes)
print(f"\n[derived] text_split_len_new (vis_start idx) = {text_split_len_new}")
print(f"           n_video (= packed_vae_token_indexes len) = {n_video}")


# ---- PT get_rope_index (production input_ids) ----
input_ids_pt = torch.tensor(pt_seq, dtype=torch.long).unsqueeze(0)
video_grid_thw = torch.tensor([[t_lat, h_lat * SPATIAL_MERGE_SIZE, w_lat * SPATIAL_MERGE_SIZE]])
mock = MockSelf()
print("\n[PT call] get_rope_index for L=185 production ...")
pos_ids_pt, _ = PtQwen2ForCausalLM.get_rope_index(
    mock, input_ids=input_ids_pt,
    image_grid_thw=video_grid_thw, video_grid_thw=video_grid_thw,
    second_per_grid_ts=torch.tensor([1.0]),
    attention_mask=None,
)
pos_pt_np = pos_ids_pt.numpy()
print(f"[PT] positions: shape={pos_pt_np.shape}")
print(f"  text head [0..3]:           t={pos_pt_np[0,0,:4].tolist()}")
print(f"  text tail (idx={vs_idx-1}): t={pos_pt_np[0,0,vs_idx-1]}")
print(f"  vis_start (idx={vs_idx}):   t={pos_pt_np[0,0,vs_idx]}, h={pos_pt_np[1,0,vs_idx]}, w={pos_pt_np[2,0,vs_idx]}")
print(f"  IMG[0]    (idx={vs_idx+1}): t={pos_pt_np[0,0,vs_idx+1]}, h={pos_pt_np[1,0,vs_idx+1]}, w={pos_pt_np[2,0,vs_idx+1]}")
print(f"  IMG[last] (idx={vs_idx+n_video}): t={pos_pt_np[0,0,vs_idx+n_video]}, h={pos_pt_np[1,0,vs_idx+n_video]}, w={pos_pt_np[2,0,vs_idx+n_video]}")
print(f"  vis_end   (idx={vs_idx+n_video+1}): t={pos_pt_np[0,0,vs_idx+n_video+1]}, h={pos_pt_np[1,0,vs_idx+n_video+1]}, w={pos_pt_np[2,0,vs_idx+n_video+1]}")
if vs_idx + n_video + 2 < L:
    print(f"  tail      (idx={vs_idx+n_video+2}): t={pos_pt_np[0,0,vs_idx+n_video+2]}, h={pos_pt_np[1,0,vs_idx+n_video+2]}, w={pos_pt_np[2,0,vs_idx+n_video+2]}")

# shift_position_ids
shifted_pt = shift_position_ids(
    pos_ids_pt.clone(),
    pos_shift=1000,
    attn_modes=attn_modes,
    split_lens=split_lens,
    shift_attn_mode=["full_noise", "full"],
    pro_type=10,
    i_sample_task=torch.from_numpy(sample_task),
    i_sample_modality=torch.from_numpy(sample_modality),
)
no_op = np.array_equal(shifted_pt.numpy(), pos_pt_np)
print(f"\n[shift] no-op for production t2v? {no_op}")
pos_pt_final = shifted_pt.numpy()


# ---- MLX build_t2v_positions (v2 — video case 지원) ----
def build_t2v_positions(text_split_len, t_lat, h_lat, w_lat, L,
                        *, second_per_grid_t=1.0, tokens_per_second=2):
    """PT get_rope_index 의 t2v 분기 — text_template 에 따라 image/video case 분기.

    text_template=False → image_token_id 사용 → PT image case → second_per_grid_t=0
                          → t_index = [0]*n_video (t const)
    text_template=True  → video_token_id 사용 → PT video case → second_per_grid_t=1
                          → t_index = repeat(arange(t_lat) * tps, h*w)
                          → IMG[k of frame f] t = f * second_per_grid_t * tps + offset

    h_index / w_index 는 양쪽 동일.  tokens_per_second 는 Lance vit config = 2.
    """
    pos = np.zeros((3, 1, L), dtype=np.int64)
    n_video = t_lat * h_lat * w_lat
    text_len_inc_vs = text_split_len + 1
    v_start = text_len_inc_vs
    for axis in range(3):
        pos[axis, 0, :text_len_inc_vs] = np.arange(text_len_inc_vs)
    offset = text_len_inc_vs

    # PT lance.py:1255-1261 그대로
    range_t = np.arange(t_lat, dtype=np.float64)            # (t_lat,)
    time_long = (range_t * second_per_grid_t * tokens_per_second).astype(np.int64)
    t_index = np.repeat(time_long, h_lat * w_lat)            # shape (n_video,)
    h_index = np.tile(np.repeat(np.arange(h_lat, dtype=np.int64), w_lat), t_lat)
    w_index = np.tile(np.arange(w_lat, dtype=np.int64), t_lat * h_lat)

    pos[0, 0, v_start:v_start + n_video] = t_index + offset
    pos[1, 0, v_start:v_start + n_video] = h_index + offset
    pos[2, 0, v_start:v_start + n_video] = w_index + offset

    if v_start + n_video < L:
        max_t = int(t_index.max())
        max_h = h_lat - 1
        max_w = w_lat - 1
        max_pos = offset + max(max_t, max_h, max_w)
        st_idx_post = max_pos + 1
        post_len = L - (v_start + n_video)
        for axis in range(3):
            pos[axis, 0, v_start + n_video:] = np.arange(post_len) + st_idx_post
    return pos

pos_mlx = build_t2v_positions(text_split_len_new, t_lat, h_lat, w_lat, L)
identical = np.array_equal(pos_pt_final, pos_mlx)
print(f"\n[byte-diff] MLX build_t2v_positions(text_split_len={text_split_len_new}) vs PT (production L={L}): {identical}")
if not identical:
    rows = np.where((pos_pt_final != pos_mlx).any(axis=(0, 1)))[0]
    print(f"  diff at {len(rows)} positions: idxs={rows.tolist()[:10]}")
    for r in rows[:5]:
        print(f"    idx {r}: PT={pos_pt_final[:,0,r].tolist()}  MLX={pos_mlx[:,0,r].tolist()}")
else:
    print("  ✓ production full positions byte-identical")


# ---- uncond (modality != 0) ----
uncond_mask = sample_modality != 0
uncond_text_ids = pt_seq[uncond_mask]
uncond_L = int(uncond_mask.sum())
print(f"\n[uncond] uncond_L={uncond_L}  (text drop, system + noise keep)")
print(f"  first 5: {uncond_text_ids[:5].tolist()}")
print(f"  last 3:  {uncond_text_ids[-3:].tolist()}")

# uncond split-level (PT uncond_split_pro_new)
cursor = 0
uncond_split_lens, uncond_attn_modes = [], []
for sl, am in zip(split_lens, attn_modes):
    sub_mod = sample_modality[cursor:cursor + sl]
    keep = int((sub_mod != 0).sum())
    cursor += sl
    if keep == 0:
        continue
    uncond_split_lens.append(keep)
    uncond_attn_modes.append(am)
print(f"[uncond] split_lens={uncond_split_lens}  attn_modes={uncond_attn_modes}")

# PT get_rope_index for uncond
uncond_input_ids_pt = torch.from_numpy(uncond_text_ids).long().unsqueeze(0)
print(f"\n[PT call] get_rope_index for uncond L={uncond_L} ...")
uncond_pos_pt, _ = PtQwen2ForCausalLM.get_rope_index(
    mock, input_ids=uncond_input_ids_pt,
    image_grid_thw=video_grid_thw, video_grid_thw=video_grid_thw,
    second_per_grid_ts=torch.tensor([1.0]),
    attention_mask=None,
)
uncond_pos_pt_np = uncond_pos_pt.numpy()
# find vis_start in uncond
u_vs_idx = int(np.where(uncond_text_ids == new_token_ids["start_of_image"])[0][0])
print(f"  uncond vis_start at idx={u_vs_idx}  → uncond text_split_len = {u_vs_idx}")
print(f"  IMG[0] (idx={u_vs_idx+1}): {uncond_pos_pt_np[:,0,u_vs_idx+1].tolist()}")
print(f"  IMG[last] (idx={u_vs_idx+n_video}): {uncond_pos_pt_np[:,0,u_vs_idx+n_video].tolist()}")

# MLX uncond build (text_split_len = u_vs_idx)
pos_mlx_uncond = build_t2v_positions(u_vs_idx, t_lat, h_lat, w_lat, uncond_L)
ident_uncond = np.array_equal(uncond_pos_pt_np, pos_mlx_uncond)
print(f"\n[byte-diff] MLX uncond positions (text_split_len={u_vs_idx}) vs PT: {ident_uncond}")
if not ident_uncond:
    rows = np.where((uncond_pos_pt_np != pos_mlx_uncond).any(axis=(0, 1)))[0]
    print(f"  diff at {len(rows)} positions: idxs={rows.tolist()[:10]}")
    for r in rows[:5]:
        print(f"    idx {r}: PT={uncond_pos_pt_np[:,0,r].tolist()}  MLX={pos_mlx_uncond[:,0,r].tolist()}")
else:
    print("  ✓ production uncond positions byte-identical")


# ---- attn_mask byte-diff (full + uncond) ----
print("\n=== attn_mask byte-diff ===")
# full mask
predicate = create_sparse_mask([L], split_lens,
    ["full" if m in ("full_noise", "full_noise_target") else m for m in attn_modes],
    device=torch.device("cpu"))
q = torch.arange(L)[:, None]; k = torch.arange(L)[None, :]
b = torch.tensor(0); h = torch.tensor(0)
pt_dense_full = predicate(b=b, h=h, q_idx=q, kv_idx=k).numpy()
mlx_full = build_lance_attention_mask(L, split_lens, attn_modes)
mlx_full_bool = np.asarray(mlx_full) == 0.0
ident_full_mask = np.array_equal(pt_dense_full, mlx_full_bool)
print(f"[full mask] PT True count={int(pt_dense_full.sum())}  MLX={int(mlx_full_bool.sum())}  byte-identical: {ident_full_mask}")

# uncond mask
predicate_u = create_sparse_mask([uncond_L], uncond_split_lens,
    ["full" if m in ("full_noise", "full_noise_target") else m for m in uncond_attn_modes],
    device=torch.device("cpu"))
q_u = torch.arange(uncond_L)[:, None]; k_u = torch.arange(uncond_L)[None, :]
pt_dense_uncond = predicate_u(b=b, h=h, q_idx=q_u, kv_idx=k_u).numpy()
mlx_uncond = build_lance_attention_mask(uncond_L, uncond_split_lens, uncond_attn_modes)
mlx_uncond_bool = np.asarray(mlx_uncond) == 0.0
ident_uncond_mask = np.array_equal(pt_dense_uncond, mlx_uncond_bool)
print(f"[uncond mask] PT True count={int(pt_dense_uncond.sum())}  MLX={int(mlx_uncond_bool.sum())}  byte-identical: {ident_uncond_mask}")


# ---- save fixtures ----
np.save("out/stage9_pt_video_pos_ids_production.npy", pos_pt_final)
np.save("out/stage9_pt_video_attn_mask_production.npy", pt_dense_full)
np.save("out/stage9_pt_video_uncond_pos_ids_production.npy", uncond_pos_pt_np)
np.save("out/stage9_pt_video_uncond_attn_mask_production.npy", pt_dense_uncond)
np.save("out/stage9_pt_video_sample_modality.npy", sample_modality)
np.save("out/stage9_pt_video_uncond_text_ids_production.npy", uncond_text_ids)
prod_meta = {
    **meta,
    "text_split_len_new": int(text_split_len_new),
    "uncond_L": uncond_L,
    "uncond_text_split_len": u_vs_idx,
    "uncond_split_lens": uncond_split_lens,
    "uncond_attn_modes": uncond_attn_modes,
}
with open("out/stage9_pt_video_production_meta.json", "w") as f:
    json.dump(prod_meta, f, indent=2)
print(f"\n[save] 6 production fixtures + meta")
print(f"[summary] full positions byte-identical: {identical}")
print(f"[summary] uncond positions byte-identical: {ident_uncond}")
print(f"[summary] full mask byte-identical:      {ident_full_mask}")
print(f"[summary] uncond mask byte-identical:    {ident_uncond_mask}")
