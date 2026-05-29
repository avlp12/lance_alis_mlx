"""STAGE 9 §1 단계 4-3 — text_template=True production 시퀀스 정답지 + MLX byte-diff.

옵션 A (doctrine 게이트):
  ValidationDataset.__new__ 우회 minimal init + t2v_sample(0) 직접 호출.
  PT 코드 100% 그대로, 우리 해석 0.
  (FRAME_SAMPLER / VideoTransform 등 video-file 의존성은 t2v_sample 이
   안 쓰니까 __init__ 우회로 회피.)

옵션 B (production fast path):
  tokenizer.apply_chat_template (HF Qwen2.5) 사용 → A 정답지와 byte-diff.
  통과하면 t2v.py 의 런타임 시퀀스 빌드에 사용 (런타임 PT 의존성 제거).

검증 시퀀스: text_template=True + apply_qwen_2_5_vl_pos_emb=True production case.

Output fixtures:
  - out/stage9_pt_t2v_seq_real.npy           — A 정답지 (packed_text_ids)
  - out/stage9_pt_t2v_seq_meta.json          — text_split_len, video_split_len, modality, ...
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
from transformers import AutoTokenizer

# Import PT ValidationDataset + helpers (코드 그대로)
sys.modules.setdefault("decord", __import__("types").ModuleType("decord"))
# stub decord to skip its heavy import
import decord  # already stubbed
decord.cpu = lambda x=0: None
decord.VideoReader = object


# Stub data.video.* (VideoTransform / FrameSampler) — t2v_sample 안 쓰지만 import 차단
_video_dir = os.path.abspath("refs/Lance/data/video")
import types as _types
_pkg_video = _types.ModuleType("data.video"); _pkg_video.__path__ = [_video_dir]
sys.modules["data.video"] = _pkg_video
_pkg_sampler = _types.ModuleType("data.video.sampler"); _pkg_sampler.__path__ = [_video_dir + "/sampler"]
sys.modules["data.video.sampler"] = _pkg_sampler

vd_mod = importlib.import_module("data.datasets_custom.validation_dataset")
ValidationDataset = vd_mod.ValidationDataset


# ---- production config ----
USER_PROMPT = "A red panda riding a wave at sunset."
T_VIDEO = 5
H_PIX = W_PIX = 128
VAE_DOWN_T = 4
VAE_DOWN_S = 16

t_lat = (T_VIDEO - 1) // VAE_DOWN_T + 1
h_lat = H_PIX // VAE_DOWN_S
w_lat = W_PIX // VAE_DOWN_S
num_vid_tokens = t_lat * h_lat * w_lat


class MockDataConfig:
    task = "t2v"
    num_frames = T_VIDEO
    H = H_PIX
    W = W_PIX
    vae_downsample = (VAE_DOWN_T, VAE_DOWN_S, VAE_DOWN_S)
    text_template = True
    resolution = "video_480p"
    max_duration = 6.0
    # system_prompt_type 없음 → SP0 default


# ---- ValidationDataset.__new__ 우회 build ----
print("[build] ValidationDataset.__new__ minimal init ...")
tok = AutoTokenizer.from_pretrained("checkpoints/Lance-3B-MLX", use_fast=True)
new_token_ids = {
    "bos_token_id":   tok.convert_tokens_to_ids("<|im_start|>"),
    "eos_token_id":   tok.convert_tokens_to_ids("<|im_end|>"),
    "start_of_image": tok.convert_tokens_to_ids("<|vision_start|>"),
    "end_of_image":   tok.convert_tokens_to_ids("<|vision_end|>"),
    "image_token_id": 151655,
}

ds = ValidationDataset.__new__(ValidationDataset)
ds.tokenizer = tok
ds.new_token_ids = new_token_ids
ds.bos_token_id = new_token_ids["bos_token_id"]
ds.eos_token_id = new_token_ids["eos_token_id"]
ds.start_of_image = new_token_ids["start_of_image"]
ds.end_of_image = new_token_ids["end_of_image"]
ds.image_token_id = new_token_ids["image_token_id"]
ds.data_config = MockDataConfig()
ds.system_prompt_type = "SP0"
ds.sample_task = "t2v"
ds.data = [{"data": USER_PROMPT, "index": "0", "original_prompt_en": USER_PROMPT}]
ds.sample = ds.set_sequence_status()
ds.frame_condition_idx = []
print("[build] done.")

# ---- A: t2v_sample(0) 직접 호출 ----
print("\n[A] PT ValidationDataset.t2v_sample(0) 호출 (text_template=True production) ...")
result = ds.t2v_sample(0)
print(f"[A] return keys: {sorted(result.keys())[:8]}...")
packed_text_ids = result["packed_text_ids"]
packed_text_indexes = result["packed_text_indexes"]
packed_vae_token_indexes = result["packed_vae_token_indexes"]
split_lens = result["split_lens"]
attn_modes = result["attn_modes"]
sample_lens = result["sample_lens"]
sample_modality = result["sample_modality"]
sample_task = result["sample_task"]

pt_seq = packed_text_ids.numpy() if hasattr(packed_text_ids, "numpy") else np.array(packed_text_ids)
L = int(sample_lens[0])
print(f"[A] sequence: shape={pt_seq.shape}  sample_lens={sample_lens}")
print(f"    split_lens={split_lens}")
print(f"    attn_modes={attn_modes}")
print(f"    first 5 tokens: {pt_seq[:5].tolist()}")
print(f"    last 5 tokens:  {pt_seq[-5:].tolist()}")
print(f"    sample_modality unique: {sorted(set(np.array(sample_modality).tolist()))}")
print(f"    L={L}  packed_vae_token_indexes len={len(packed_vae_token_indexes) if hasattr(packed_vae_token_indexes, '__len__') else 'NA'}")

# ---- B: tokenizer.apply_chat_template fast path ----
print("\n[B] HF tokenizer.apply_chat_template 시도 ...")
T2V_SYSTEM_PROMPT = ("Describe the video by detailing the color, quantity, "
                    "visible text, shape, size, texture, spatial relationships "
                    "and motion/camera movements of the objects and background:")

# PT side renders {"role":"user","content":[{"type":"text","text":prompt}]},
#                 {"role":"assistant","content":[{"type":"video"}]}
# with default_system=T2V_SYSTEM_PROMPT (Qwen2.5-VL style).
# HF Qwen2.5 tokenizer chat template — best-effort match.
messages_hf = [
    {"role": "system", "content": T2V_SYSTEM_PROMPT},
    {"role": "user", "content": USER_PROMPT},
    {"role": "assistant", "content": ""},
]
try:
    hf_rendered = tok.apply_chat_template(messages_hf, tokenize=False, add_generation_prompt=False)
    print(f"[B] rendered preview (first 200): {hf_rendered[:200]!r}")
    hf_seq = tok(hf_rendered, add_special_tokens=False, return_tensors="np")["input_ids"][0]
    print(f"[B] HF tokenized: shape={hf_seq.shape}")

    # Note: HF gives us only text. PT t2v_sample inserts vis_start + IMG*N + vis_end at the
    # video span. We'd need to inject those at the right position. The expected location is
    # where {"type": "video"} placeholder resolves to.

    # Quick byte-diff with PT sequence (text-only portion)
    # PT pt_seq includes vis_start + IMG*128 + vis_end. HF hf_seq is text-only.
    # Find vision_start in PT
    vs_idx = int(np.where(pt_seq == new_token_ids["start_of_image"])[0][0])
    pt_text_before_vs = pt_seq[:vs_idx]
    pt_text_after_ve = pt_seq[vs_idx + 1 + num_vid_tokens + 1:]
    pt_text_only = np.concatenate([pt_text_before_vs, pt_text_after_ve])
    print(f"\n[B] PT text-only portion (vis 슬랩 제거): shape={pt_text_only.shape}")
    print(f"[B] HF rendered tokenized:                shape={hf_seq.shape}")
    if pt_text_only.shape == hf_seq.shape:
        same = bool((pt_text_only == hf_seq).all())
        print(f"[B] byte-identical: {same}")
        if not same:
            diff_idxs = np.where(pt_text_only != hf_seq)[0]
            print(f"    diff at {len(diff_idxs)} positions: {diff_idxs.tolist()[:10]}")
            for i in diff_idxs[:5]:
                print(f"      idx {i}: PT={int(pt_text_only[i])} HF={int(hf_seq[i])}")
    else:
        print(f"[B] shape mismatch — HF format different from PT render_qwenvl_prompt")
        print(f"    PT decoded: {tok.decode(pt_text_only.tolist())!r}")
        print(f"    HF decoded: {tok.decode(hf_seq.tolist())!r}")
except Exception as ex:
    print(f"[B] apply_chat_template failed: {ex}")
    print("    → fallback to A (direct ValidationDataset call) for runtime")

# ---- save A fixture ----
os.makedirs("out", exist_ok=True)
np.save("out/stage9_pt_t2v_seq_real.npy", pt_seq)

meta = {
    "doctrine_source": "PT ValidationDataset.t2v_sample(0) text_template=True production",
    "user_prompt": USER_PROMPT,
    "video_TpxHpxWpx": [T_VIDEO, H_PIX, W_PIX],
    "video_grid_thw": [t_lat, h_lat, w_lat],
    "num_vid_tokens": num_vid_tokens,
    "L_sequence": L,
    "split_lens": split_lens,
    "attn_modes": attn_modes,
    "sample_modality_unique": sorted(set(np.array(sample_modality).tolist())),
    "packed_vae_token_indexes_len": len(packed_vae_token_indexes) if hasattr(packed_vae_token_indexes, "__len__") else None,
    "text_template": True,
    "apply_qwen_2_5_vl_pos_emb": True,
}
with open("out/stage9_pt_t2v_seq_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"\n[save] out/stage9_pt_t2v_seq_real.npy  shape={pt_seq.shape}")
print(f"       out/stage9_pt_t2v_seq_meta.json")
