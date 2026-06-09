"""STAGE 11 step 2 — independent x2t-VIDEO position check vs PT (the real truth).

Closes the (c)-video position-reuse BLINDNESS the adversarial review found: the
old (c)-video harness fed BOTH sides our build_positions_for_layout output
(pos_pt = from_numpy(pos_mlx)), so it could not see a position-convention error
— and there WAS one (x2t-video used a UNIT temporal step; PT uses
arange(t)*second_per_grid_t*tokens_per_second = step 2).

Here PT derives its OWN positions via PT Lance get_rope_index (the same call
STAGE 9 uses), and we byte-diff against our x2t-video positions (now built with
temporal_scale=VIDEO_TEMPORAL_SCALE=2).  LIGHT — no model/ViT weights;
get_rope_index is pure index math.

Expected: byte-identical AFTER the fix.  With the OLD unit-step code this would
byte-diff on the 16 temporal mRoPE channels of every visual token.
"""
from __future__ import annotations

import os
import sys
import importlib

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("refs/Lance"))

# transformers >=5.9 flash-probe neutraliser (must precede PT modeling import).
import transformers.utils.import_utils as _iu
import transformers.utils as _tu
import transformers.modeling_flash_attention_utils as _mfa
def _false(*_a, **_k):
    return False
for _m in (_iu, _tu, _mfa):
    for _fn in ("is_flash_attn_2_available", "is_flash_attn_3_available",
                "is_flash_attn_4_available", "flash_attn_supports_top_left_mask"):
        if hasattr(_m, _fn):
            setattr(_m, _fn, _false)

# qwen2_navit does `from flash_attn import flash_attn_varlen_func` at import — get_rope_index
# never calls it (pure index math), so a stub module is enough to satisfy the import.
import types as _types
import importlib.machinery as _imach
if "flash_attn" not in sys.modules:
    _fa = _types.ModuleType("flash_attn")
    _fa.__spec__ = _imach.ModuleSpec("flash_attn", loader=None)
    _fa.flash_attn_varlen_func = lambda *a, **k: None
    sys.modules["flash_attn"] = _fa

import numpy as np
import torch
import mlx.core as mx
from transformers import AutoTokenizer

qwen2_navit = importlib.import_module("modeling.lance.qwen2_navit")
PtQwen2ForCausalLM = qwen2_navit.Qwen2ForCausalLM

from lance_mlx.pipelines.x2t import (
    preprocess_video, VIDEO_TEMPORAL_SCALE,
    IM_START_ID, IM_END_ID, VIS_START_ID, VIS_END_ID, VIDEO_PAD_ID,
    SPATIAL_MERGE_SIZE,
)
from lance_mlx.rope import VisionSpec, build_positions_for_layout

FRAMES = "out/stage11_assets/vqa01_frames.npy"
MAX_PIX = 14 * 14 * 12 * 12
N_FRAMES = 8
QUESTION = "What is happening in this video?"


class _MockVisionConfig(dict):
    pass


class _MockConfig:
    def __init__(self):
        self.image_token_id = 151655
        self.video_token_id = 151656
        self.vision_start_token_id = 151652
        self.vision_config = _MockVisionConfig({
            "spatial_merge_size": 2, "tokens_per_second": 2, "temporal_patch_size": 2,
        })


class _MockSelf:
    def __init__(self):
        self.config = _MockConfig()


def main() -> None:
    tok = AutoTokenizer.from_pretrained("checkpoints/Lance-3B-MLX", use_fast=True)
    clip = np.load(FRAMES)[:N_FRAMES]
    _, (T_g, H_g, W_g) = preprocess_video(clip, max_pixels=MAX_PIX)
    h_lat, w_lat = H_g // SPATIAL_MERGE_SIZE, W_g // SPATIAL_MERGE_SIZE
    n_vis = T_g * h_lat * w_lat

    # Same chat sequence as stage11_x2t_video_compare (VIDEO_PAD_ID placeholders).
    sys_ids = tok("You are a helpful assistant.", add_special_tokens=False)["input_ids"]
    q_ids = tok(QUESTION, add_special_tokens=False)["input_ids"]
    nl = tok("\n", add_special_tokens=False)["input_ids"]
    s = tok("system", add_special_tokens=False)["input_ids"]
    u = tok("user", add_special_tokens=False)["input_ids"]
    a = tok("assistant", add_special_tokens=False)["input_ids"]
    seq = ([IM_START_ID] + s + nl + sys_ids + [IM_END_ID] + nl
           + [IM_START_ID] + u + nl
           + [VIS_START_ID] + [VIDEO_PAD_ID] * n_vis + [VIS_END_ID]
           + q_ids + [IM_END_ID] + nl + [IM_START_ID] + a + nl)
    L = len(seq)
    vis_start = seq.index(VIS_START_ID) + 1
    print(f"[seq] L={L}  vis_start={vis_start}  n_vis={n_vis}  grid=(T={T_g},H={H_g},W={W_g}) "
          f"llm=({T_g},{h_lat},{w_lat})  temporal_scale={VIDEO_TEMPORAL_SCALE}")

    # OURS — exactly the x2t_video code path.
    pos_ours = np.asarray(build_positions_for_layout(
        L, [VisionSpec(start=vis_start - 1, length=n_vis, t=T_g, h=h_lat, w=w_lat,
                       temporal_scale=VIDEO_TEMPORAL_SCALE)]))            # (3,1,L)

    # PT — its OWN get_rope_index (grid in PATCH units; video branch via video_token_id).
    input_ids = torch.tensor([seq], dtype=torch.long)
    video_grid_thw = torch.tensor([[T_g, H_g, W_g]])
    pos_pt, _ = PtQwen2ForCausalLM.get_rope_index(
        _MockSelf(), input_ids=input_ids,
        image_grid_thw=video_grid_thw, video_grid_thw=video_grid_thw,
        second_per_grid_ts=torch.tensor([1.0]), attention_mask=None)
    pos_pt = pos_pt.numpy()                                              # (3,1,L)

    byte_id = np.array_equal(pos_ours, pos_pt)
    # show the temporal row of the visual block (the bug locus)
    vt_ours = pos_ours[0, 0, vis_start: vis_start + n_vis].reshape(T_g, h_lat, w_lat)[:, 0, 0]
    vt_pt = pos_pt[0, 0, vis_start: vis_start + n_vis].reshape(T_g, h_lat, w_lat)[:, 0, 0]
    print(f"[visual temporal per-frame] ours={vt_ours.tolist()}  PT={vt_pt.tolist()}")
    print(f"[first text-after-vision pos] ours={int(pos_ours[0,0,vis_start+n_vis+1])}  "
          f"PT={int(pos_pt[0,0,vis_start+n_vis+1])}")
    if not byte_id:
        diff = (pos_ours != pos_pt).any(axis=(0, 1))
        idxs = np.where(diff)[0]
        print(f"[DIFF] {len(idxs)} positions differ; first idxs={idxs[:12].tolist()}")
        for ax, name in enumerate("thw"):
            d = int((pos_ours[ax] != pos_pt[ax]).sum())
            print(f"   axis {name}: {d} differing entries")
    print("=" * 60)
    print("GATE step 2 (positions):", "PASS — x2t-video positions byte-identical to PT get_rope_index"
          if byte_id else "FAIL — positions diverge from PT (localize above)")
    sys.exit(0 if byte_id else 1)


if __name__ == "__main__":
    main()
