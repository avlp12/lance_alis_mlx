"""STAGE 9 §0 A.2 단계 1 — PT validation_gen 의 진짜 t2v sequence 로 정답지.

이전 v1 (`stage9_pt_video_dit_smoke.py`) 의 manual sequence/positions 가 PT
validation_dataset.t2v_sample 의 진짜 패턴과 다름 확인됨. 우리 manual 해석
없이 PT 코드 그대로 사용:

  sequence (text_template=False):
    text_ids = [bos] + tokenize(prompt) + [eos]
             + [start_of_image] + [IMG]*N + [end_of_image]

  positions (apply_qwen_2_5_vl_pos_emb=False):
    text:  range(0, text_split_len)
    video: [text_split_len] * video_split_len    # ★ CONSTANT (1D)

  attn_modes = ["causal", "noise"]
  sample_modality: text=0, video (vis_start/IMG/vis_end)=1

→ MLX Qwen2RotaryEmbedding 이 (3, 1, L) broadcast 시 3축 모두 같은 값 = 1D RoPE
   동등. 단계 1 의 minimal smoke.

PRNG: numpy.random.default_rng(0) 단일 (Lesson 9). PT validation_gen 의
torch.randn 우회 — 외부 numpy noise 주입.

v_t intercept: validation_gen 의 step 0 forward 끝에서 v_t (n_noise, 48) 추출.

manual v1 fixture: `out/audit_manual_v_t/` 에 audit trail 로 보존.
새 fixture: `out/stage9_pt_video_*.npy` (덮어쓰기).
"""
from __future__ import annotations

import json
import os
import sys
import importlib

# STAGE 9+ PT smoke 공용 환경 셋업 — Lesson E containment.
# (이전 v2 에는 env shim 코드가 inline 으로 들어있었음. STAGE 9 §1 단계 3 에서
#  공용 헬퍼로 분리 → 미래 PT smoke 가 같은 env + Lesson E contract 사용.)
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("refs/Lance"))
from tools._pt_smoke_common import install_pt_smoke_env, pt_layer_mask
install_pt_smoke_env()


# ---------- regular imports -------------------------------------------------
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from lance_mlx.backbone import LanceTextConfig

qwen2_navit = importlib.import_module("modeling.lance.qwen2_navit")
Qwen2MoTDecoderLayer = qwen2_navit.Qwen2MoTDecoderLayer
from modeling.qwen2.configuration_qwen2 import Qwen2Config
from modeling.qwen2.modeling_qwen2 import Qwen2RMSNorm
from data.data_utils import create_sparse_mask
from data.common import shift_position_ids
from torch.nn.attention.flex_attention import create_block_mask
import torch.nn.attention.flex_attention as _fa_mod
qwen2_navit.flex_attention = _fa_mod.flex_attention


# ---------- constants -------------------------------------------------------
PT_WEIGHTS_IMG = "checkpoints/Lance/Lance_3B/model.safetensors"
PT_WEIGHTS_VID_SUP = "checkpoints/Lance/Lance_3B_Video/model_supplement.safetensors"

# Production config (see `inference_lance.sh:122-123`)
MAX_NUM_LATENT_FRAMES = 31           # = 121 // 4 + 1
MAX_LATENT_SIZE = 64
LATENT_PATCH = (1, 1, 1)
VAE_DOWN_SPATIAL = 16
VAE_DOWN_TEMPORAL = 4
LATENT_CHANNEL = 48
PATCH_LATENT_DIM = LATENT_PATCH[0] * LATENT_PATCH[1] * LATENT_PATCH[2] * LATENT_CHANNEL

# Small smoke video
T_VIDEO = 5
H_PIX = W_PIX = 128
USER_PROMPT = "A red panda riding a wave at sunset."

# Special tokens (from refs/Lance/data/data_utils.py:add_special_tokens)
BOS_TOKEN = "<|im_start|>"
EOS_TOKEN = "<|im_end|>"
START_OF_IMAGE_TOKEN = "<|vision_start|>"
END_OF_IMAGE_TOKEN = "<|vision_end|>"
IMG_TOKEN_ID = 151655   # <|image_pad|> (image_token_id)

NUMPY_SEED = 0


# ---------- PT model (same as v1, latent_pos_embed for 31 frames) -----------
class PtLanceVideoT2V:
    def __init__(self):
        self.cfg = LanceTextConfig()
        self.q_cfg = Qwen2Config(
            hidden_size=self.cfg.hidden_size,
            intermediate_size=self.cfg.intermediate_size,
            num_hidden_layers=self.cfg.num_hidden_layers,
            num_attention_heads=self.cfg.num_attention_heads,
            num_key_value_heads=self.cfg.num_key_value_heads,
            vocab_size=self.cfg.vocab_size,
            rms_norm_eps=self.cfg.rms_norm_eps,
            rope_theta=self.cfg.rope_theta,
            max_position_embeddings=self.cfg.max_position_embeddings,
            rope_scaling=self.cfg.rope_scaling,
            tie_word_embeddings=self.cfg.tie_word_embeddings,
        )
        self.q_cfg.qk_norm = True
        self.q_cfg.qk_norm_und = True
        self.q_cfg.qk_norm_gen = True
        self.q_cfg.layer_module = "Qwen2MoTDecoderLayer"
        self.q_cfg.freeze_und = False

        self.embed_tokens = torch.nn.Embedding(self.cfg.vocab_size, self.cfg.hidden_size)
        self.layers = torch.nn.ModuleList([
            Qwen2MoTDecoderLayer(self.q_cfg, layer_idx=i)
            for i in range(self.cfg.num_hidden_layers)
        ])
        self.vae2llm = torch.nn.Linear(LATENT_CHANNEL, self.cfg.hidden_size, bias=True)
        self.llm2vae = torch.nn.Linear(self.cfg.hidden_size, LATENT_CHANNEL, bias=True)
        self.time_fc1 = torch.nn.Linear(256, self.cfg.hidden_size, bias=True)
        self.time_fc2 = torch.nn.Linear(self.cfg.hidden_size, self.cfg.hidden_size, bias=True)
        self.latent_pos_embed = torch.nn.Embedding(
            MAX_NUM_LATENT_FRAMES * MAX_LATENT_SIZE * MAX_LATENT_SIZE,
            self.cfg.hidden_size,
        )
        self.final_norm = Qwen2RMSNorm(self.cfg.hidden_size, eps=self.cfg.rms_norm_eps)
        self.norm_moe_gen = Qwen2RMSNorm(self.cfg.hidden_size, eps=self.cfg.rms_norm_eps)

    def to_bf16(self):
        for m in (self.embed_tokens, self.vae2llm, self.llm2vae, self.time_fc1,
                  self.time_fc2, self.latent_pos_embed, self.final_norm, self.norm_moe_gen):
            m.to(torch.bfloat16)
        for L in self.layers:
            L.to(torch.bfloat16)

    def load_pt(self):
        d = {}
        with safe_open(PT_WEIGHTS_IMG, framework="pt", device="cpu") as f:
            for k in f.keys():
                d[k] = f.get_tensor(k).to(torch.bfloat16)
        with safe_open(PT_WEIGHTS_VID_SUP, framework="pt", device="cpu") as f:
            for k in f.keys():
                d[k] = f.get_tensor(k).to(torch.bfloat16)

        self.embed_tokens.weight.data = d["language_model.model.embed_tokens.weight"].clone()
        for i, L in enumerate(self.layers):
            prefix = f"language_model.model.layers.{i}."
            state = {k[len(prefix):]: v for k, v in d.items() if k.startswith(prefix)}
            L.load_state_dict(state, strict=True)
        self.final_norm.weight.data = d["language_model.model.norm.weight"].clone()
        self.norm_moe_gen.weight.data = d["language_model.model.norm_moe_gen.weight"].clone()
        self.vae2llm.weight.data = d["vae2llm.weight"].clone()
        self.vae2llm.bias.data   = d["vae2llm.bias"].clone()
        self.llm2vae.weight.data = d["llm2vae.weight"].clone()
        self.llm2vae.bias.data   = d["llm2vae.bias"].clone()
        self.time_fc1.weight.data = d["time_embedder.mlp.0.weight"].clone()
        self.time_fc1.bias.data   = d["time_embedder.mlp.0.bias"].clone()
        self.time_fc2.weight.data = d["time_embedder.mlp.2.weight"].clone()
        self.time_fc2.bias.data   = d["time_embedder.mlp.2.bias"].clone()
        self.latent_pos_embed.weight.data = d["latent_pos_embed.pos_embed"].clone()

    def time_embed(self, t: torch.Tensor) -> torch.Tensor:
        half = 256 // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(10000.0))
            * torch.arange(0, half, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(torch.bfloat16)
        h = self.time_fc1(emb)
        h = torch.nn.functional.silu(h)
        return self.time_fc2(h)

    def mrope_cos_sin(self, position_ids_3: torch.Tensor):
        head_dim = self.cfg.head_dim
        L = position_ids_3.shape[-1]
        base = self.cfg.rope_theta
        ms = self.cfg.rope_scaling["mrope_section"]
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        inv = inv_freq[None, None, :, None].expand(3, L, -1, 1)
        pos = position_ids_3.float()[:, :, None, :]
        freqs = (inv @ pos).transpose(2, 3)
        s_t, s_h, s_w = ms
        t_p = freqs[0, 0, :, :s_t]
        h_p = freqs[1, 0, :, s_t:s_t + s_h]
        w_p = freqs[2, 0, :, s_t + s_h:s_t + s_h + s_w]
        f = torch.cat([t_p, h_p, w_p], dim=-1)
        emb = torch.cat([f, f], dim=-1)
        return emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)


# ---------- sequence build — PT validation_dataset.t2v_sample 코드 그대로 ---
def build_t2v_sequence_pt(tokenizer, user_prompt: str, t_lat: int, h_lat: int, w_lat: int) -> dict:
    """PT `validation_dataset.py:t2v_sample` 의 *text_template=False* 분기 그대로.

    `apply_qwen_2_5_vl_pos_emb=False` 케이스: validation_dataset 가 만든
    `packed_position_ids` 그대로 사용 (text=range, video=constant).

    우리 manual interpretation 없음 — PT 코드 line-for-line.
    """
    new_token_ids = {
        "bos_token_id": tokenizer.convert_tokens_to_ids("<|im_start|>"),
        "eos_token_id": tokenizer.convert_tokens_to_ids("<|im_end|>"),
        "start_of_image": tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        "end_of_image": tokenizer.convert_tokens_to_ids("<|vision_end|>"),
    }
    image_token_id = IMG_TOKEN_ID

    # PT validation_dataset.py:805-833 verbatim (text_template=False 분기)
    packed_text_indexes, packed_position_ids, sample_modality = [], [], []

    text_ids = tokenizer.encode(user_prompt)
    text_ids = [new_token_ids["bos_token_id"]] + text_ids + [new_token_ids["eos_token_id"]]
    text_split_len = len(text_ids)
    packed_text_indexes.extend(range(0, text_split_len))
    packed_position_ids.extend(range(0, text_split_len))
    sample_modality.extend([0] * text_split_len)        # modality_map['text'] = 0

    num_vid_tokens = t_lat * h_lat * w_lat

    text_ids.append(new_token_ids["start_of_image"])
    packed_text_indexes.append(text_split_len)
    packed_vae_token_indexes = list(range(len(text_ids), len(text_ids) + num_vid_tokens))
    text_ids.extend([image_token_id] * num_vid_tokens)
    text_ids.append(new_token_ids["end_of_image"])
    packed_text_indexes.append(len(text_ids) - 1)
    video_split_len = num_vid_tokens + 2

    packed_position_ids.extend([text_split_len] * video_split_len)   # ★ CONSTANT
    sample_modality.extend([1] * video_split_len)        # modality_map['noise'] = 1

    L = text_split_len + video_split_len
    split_lens = [text_split_len, video_split_len]
    attn_modes = ["causal", "noise"]
    sample_task = [0] * L                                # sample_task_map['t2v'] = 0

    # validation_gen 에서 mse_loss_indexes = packed_vae_token_indexes
    mse_loss_indexes = list(packed_vae_token_indexes)
    # current_vae_mse_indexes_local_in_vae = arange(num_vid) (mse 와 vae 동일하면)
    vae_mse_indexes_local_in_vae = list(range(num_vid_tokens))

    return {
        "ids": text_ids,
        "L": L,
        "text_split_len": text_split_len,
        "video_split_len": video_split_len,
        "noise_span": (text_split_len + 1, text_split_len + 1 + num_vid_tokens),  # IMG only
        "packed_text_indexes": packed_text_indexes,
        "packed_vae_token_indexes": packed_vae_token_indexes,
        "vae_mse_indexes_local_in_vae": vae_mse_indexes_local_in_vae,
        "packed_position_ids": packed_position_ids,
        "sample_modality": sample_modality,
        "sample_task": sample_task,
        "split_lens": split_lens,
        "attn_modes": attn_modes,
        "num_vid_tokens": num_vid_tokens,
    }


def build_mask_pt(layout: dict, num_heads: int):
    attn_modes_ = ["full" if m in ("full_noise", "full_noise_target") else m
                   for m in layout["attn_modes"]]
    predicate = create_sparse_mask(
        document_lens=[layout["L"]],
        split_lens=layout["split_lens"],
        attn_modes=attn_modes_,
        device=torch.device("cpu"),
    )
    block_mask = create_block_mask(
        predicate, B=1, H=num_heads, Q_LEN=layout["L"], KV_LEN=layout["L"],
        device=torch.device("cpu"), BLOCK_SIZE=128, _compile=False,
    )
    q = torch.arange(layout["L"])[:, None]
    k = torch.arange(layout["L"])[None, :]
    b = torch.tensor(0); h = torch.tensor(0)
    dense_bool = predicate(b=b, h=h, q_idx=q, kv_idx=k)
    return block_mask, dense_bool


def vae_position_indices_video(t_lat: int, h_lat: int, w_lat: int) -> np.ndarray:
    """get_flattened_position_ids_extrapolate_video(t,h,w, max_latent_size=64)."""
    coords_t = np.arange(t_lat, dtype=np.int64)
    coords_h = np.arange(h_lat, dtype=np.int64)
    coords_w = np.arange(w_lat, dtype=np.int64)
    M = MAX_LATENT_SIZE
    ids = (
        coords_t[:, None, None] * M * M
        + coords_h[None, :, None] * M
        + coords_w[None, None, :]
    ).flatten()
    return ids


# ---------- PT forward (step 0, validation_gen 의 line 656-685 그대로) -----
def pt_forward_first_step(pt: PtLanceVideoT2V, layout: dict, x_t_pt: torch.Tensor,
                          t_scalar: float, vae_pos_ids_pt: torch.Tensor,
                          pos_ids_pt: torch.Tensor, attn_dense_bool: torch.Tensor):
    L = layout["L"]
    ns, ne = layout["noise_span"]
    n_noise = ne - ns

    # validation_gen line 541-552 — text embedding into current_sequence
    ids = torch.tensor(layout["ids"], dtype=torch.long).unsqueeze(0)
    text_embed_full = pt.embed_tokens(ids)            # (1, L, D) bf16
    current_sequence = torch.zeros((1, L, pt.cfg.hidden_size), dtype=torch.bfloat16)
    text_idx = torch.tensor(layout["packed_text_indexes"], dtype=torch.long)
    current_sequence[:, text_idx, :] = text_embed_full[:, text_idx, :]

    # validation_gen line 644-660 — timestep + vae_embed (only at IMG positions)
    # timestep = zeros(L); timestep[mse_idx] = t_scalar  (only video IMG tokens)
    # Then vae_embed = vae2llm(x_t) + time_embedder(timestep[mse_idx])
    #                + latent_pos_embed(vae_position_ids)
    timestep_per = torch.full((n_noise,), t_scalar, dtype=torch.float32)
    vae_embed = (pt.vae2llm(x_t_pt.to(torch.bfloat16))
                 + pt.time_embed(timestep_per)
                 + pt.latent_pos_embed(vae_pos_ids_pt))
    # validation_gen line 663
    vae_token_idx = torch.tensor(layout["packed_vae_token_indexes"], dtype=torch.long)
    current_sequence[:, vae_token_idx, :] = vae_embed.unsqueeze(0)

    # validation_gen line 666-674 — packed_und/gen_token_indexes
    packed_gen_idx = vae_token_idx
    all_idx = torch.arange(L, dtype=torch.long)
    gen_mask_bool = torch.zeros(L, dtype=torch.bool)
    gen_mask_bool[packed_gen_idx] = True
    packed_und_idx = all_idx[~gen_mask_bool]

    # ★ Lesson E containment via `pt_layer_mask` — bool dense only.  Helper
    #   asserts dtype=bool and routes around the flex_attention SDPA patch's
    #   `dense.to(torch.bool)` polarity-inversion trap. See _pt_smoke_common.py.
    layer_mask = pt_layer_mask(attn_dense_bool)

    cos, sin = pt.mrope_cos_sin(pos_ids_pt)
    h = current_sequence[0]
    sample_lens = [L]
    for Lyr in pt.layers:
        h = Lyr(
            packed_sequence=h, sample_lens=sample_lens,
            attention_mask=layer_mask,
            packed_position_embeddings=(cos, sin),
            packed_und_token_indexes=packed_und_idx,
            packed_gen_token_indexes=packed_gen_idx,
            mode_forward="validation",
        )
    h_und = pt.final_norm(h)
    h_gen = pt.norm_moe_gen(h)
    out = torch.zeros_like(h)
    out[packed_und_idx] = h_und[packed_und_idx]
    out[packed_gen_idx] = h_gen[packed_gen_idx]
    # validation_gen line 685
    v_t = pt.llm2vae(out[packed_gen_idx].to(torch.bfloat16))
    return v_t.to(torch.float32)


def main():
    os.makedirs("out", exist_ok=True)
    print("=" * 72)
    print("STAGE 9 §0 A.2 단계 1 — PT validation_gen 진짜 t2v 정답지")
    print("=" * 72)

    # ---- latent shape (PT t2v_sample 의 t,h,w 계산 그대로) ----
    t_lat = (T_VIDEO - 1) // VAE_DOWN_TEMPORAL + 1
    h_lat = H_PIX // VAE_DOWN_SPATIAL
    w_lat = W_PIX // VAE_DOWN_SPATIAL
    n_noise = t_lat * h_lat * w_lat
    grid_thw = [t_lat, h_lat, w_lat]
    print(f"[shape] T={T_VIDEO} px=({H_PIX},{W_PIX}) → latent t={t_lat} h={h_lat} w={w_lat}")
    print(f"        n_noise={n_noise}  patch_latent_dim={PATCH_LATENT_DIM}")

    # ---- noise (numpy seed=0, Lesson 9) ----
    rng = np.random.default_rng(NUMPY_SEED)
    x_t_init = rng.standard_normal((n_noise, PATCH_LATENT_DIM), dtype=np.float32)
    print(f"[input] x_t: numpy seed={NUMPY_SEED}  shape={x_t_init.shape} "
          f"mean={x_t_init.mean():+.4f} std={x_t_init.std():.4f}")

    # ---- tokenizer ----
    tok = AutoTokenizer.from_pretrained("checkpoints/Lance-3B-MLX", use_fast=True)

    # ---- sequence via PT validation_dataset 패턴 (manual 없음) ----
    layout = build_t2v_sequence_pt(tok, USER_PROMPT, t_lat, h_lat, w_lat)
    print(f"[seq]   L={layout['L']}  noise_span={layout['noise_span']}")
    print(f"        text_split_len={layout['text_split_len']}  video_split_len={layout['video_split_len']}")
    print(f"        split_lens={layout['split_lens']}  attn_modes={layout['attn_modes']}")
    print(f"        prompt={USER_PROMPT!r}")

    # ---- positions (apply_qwen_2_5_vl_pos_emb=False — PT 의 packed_position_ids 그대로) ----
    # PT validation_gen 이 받는 current_pos_ids 형식: apply_qwen_2_5_vl_pos_emb=False 면
    # `val_packed_position_ids` 가 1D (L,). PT Qwen2MoTDecoderLayer 가 받는 형식 — STAGE 7 와
    # 마찬가지로 (3, 1, L) 호출을 하므로 1D 를 broadcast.
    pos_1d = np.array(layout["packed_position_ids"], dtype=np.int64)
    pos_ids_np = np.broadcast_to(pos_1d[None, None, :], (3, 1, layout["L"])).copy()
    pos_ids_pt = torch.from_numpy(pos_ids_np).contiguous()
    print(f"[pos] (apply_qwen_2_5_vl_pos_emb=False) shape={tuple(pos_ids_pt.shape)} "
          f"text head=[{pos_1d[0]}, {pos_1d[1]}, {pos_1d[2]}], "
          f"video start={pos_1d[layout['text_split_len']]}, "
          f"video tail={pos_1d[-1]}")

    cfg_tmp = PtLanceVideoT2V()
    num_heads = cfg_tmp.cfg.num_attention_heads
    _, attn_dense_bool = build_mask_pt(layout, num_heads=num_heads)
    attn_dense_bool = attn_dense_bool.contiguous()
    print(f"[mask] dense shape={tuple(attn_dense_bool.shape)} "
          f"non-zero ratio={float(attn_dense_bool.float().mean()):.4f}")

    vae_pos_ids_np = vae_position_indices_video(t_lat, h_lat, w_lat)
    vae_pos_ids_pt = torch.from_numpy(vae_pos_ids_np)
    print(f"[lat_pos] vae_pos_ids: shape={vae_pos_ids_np.shape} "
          f"range=[{vae_pos_ids_np.min()}, {vae_pos_ids_np.max()}]")

    # ---- build + load PT ----
    print("\n[build+load] PtLanceVideoT2V (image backbone + video supplement bf16) ...")
    pt = PtLanceVideoT2V()
    pt.load_pt()
    pt.to_bf16()
    print(f"[load] latent_pos_embed.weight.shape = {tuple(pt.latent_pos_embed.weight.shape)}")

    # ---- forward (validation_gen step 0 그대로) ----
    t_scalar = 1.0
    x_t_pt = torch.from_numpy(x_t_init)
    print(f"\n[forward] PT validation_gen step 0 (timestep={t_scalar}) ...")
    with torch.no_grad():
        v_t = pt_forward_first_step(
            pt, layout, x_t_pt, t_scalar,
            vae_pos_ids_pt, pos_ids_pt, attn_dense_bool,
        )
    v_t_np = v_t.cpu().numpy()
    print(f"[forward] v_t: shape={v_t_np.shape} "
          f"||v_t||={float(np.linalg.norm(v_t_np)):.3f} "
          f"mean={v_t_np.mean():+.4f} std={v_t_np.std():.4f}")
    print(f"          range=[{v_t_np.min():+.3f}, {v_t_np.max():+.3f}]")
    assert not np.isnan(v_t_np).any(), "v_t has NaN"

    # ---- save fixtures ----
    print("\n[save] new fixtures (manual v1 is in out/audit_manual_v_t/) →")
    fixtures = {
        "stage9_pt_video_x_t_init.npy"     : x_t_init,
        "stage9_pt_video_text_ids.npy"     : np.array(layout["ids"], dtype=np.int64),
        "stage9_pt_video_vae_pos_ids.npy"  : vae_pos_ids_np,
        "stage9_pt_video_current_pos_ids.npy": pos_ids_pt.numpy(),
        "stage9_pt_video_attn_mask.npy"    : attn_dense_bool.numpy(),
        "stage9_pt_video_v_t_step0.npy"    : v_t_np,
    }
    for name, arr in fixtures.items():
        np.save(f"out/{name}", arr)
        print(f"  out/{name}  shape={arr.shape}  dtype={arr.dtype}")

    meta = {
        "doctrine_source": "original ByteDance bytedance-research/Lance + PT validation_dataset.t2v_sample (text_template=False)",
        "validation_gen_path": "PT validation_gen step 0 forward (lance.py:656-685)",
        "manual_v1_audit_location": "out/audit_manual_v_t/",
        "text_template": False,
        "apply_qwen_2_5_vl_pos_emb": False,
        "prng": "numpy.random.default_rng",
        "prng_seed": NUMPY_SEED,
        "user_prompt": USER_PROMPT,
        "video_grid_thw": grid_thw,
        "video_size_TpxHpxWpx": [T_VIDEO, H_PIX, W_PIX],
        "vae_down_spatial": VAE_DOWN_SPATIAL,
        "vae_down_temporal": VAE_DOWN_TEMPORAL,
        "latent_patch_size": list(LATENT_PATCH),
        "max_num_latent_frames": MAX_NUM_LATENT_FRAMES,
        "max_latent_size": MAX_LATENT_SIZE,
        "latent_channel": LATENT_CHANNEL,
        "patch_latent_dim": PATCH_LATENT_DIM,
        "n_noise": n_noise,
        "L_sequence": layout["L"],
        "text_split_len": layout["text_split_len"],
        "video_split_len": layout["video_split_len"],
        "noise_span": list(layout["noise_span"]),
        "split_lens": layout["split_lens"],
        "attn_modes": layout["attn_modes"],
        "timestep_first_step": t_scalar,
        "v_t_norm": float(np.linalg.norm(v_t_np)),
        "v_t_stats": {
            "mean": float(v_t_np.mean()),
            "std": float(v_t_np.std()),
            "min": float(v_t_np.min()),
            "max": float(v_t_np.max()),
        },
    }
    with open("out/stage9_pt_video_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("  out/stage9_pt_video_meta.json")
    print("\n[OK] A.2 단계 1 — PT validation_gen 진짜 t2v 정답지 확보.")
    print("     다음: MLX byte-diff harness 재실행 → cos ≥ 0.999 게이트.")


if __name__ == "__main__":
    main()
