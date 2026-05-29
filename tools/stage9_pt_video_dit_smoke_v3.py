"""STAGE 9 §1 단계 2 — v3 production PT smoke (full + uncond v_t).

Production: text_template=True + apply_qwen_2_5_vl_pos_emb=True.

PT path:
  - Sequence: ValidationDataset.t2v_sample(0) (단계 4-3 코드 차용)
  - Positions: PT get_rope_index 직접 호출 (full + uncond 별도)
  - Mask: build_lance_attention_mask (PT predicate byte-identical)
  - Full forward: validation_gen line 656-685 그대로
  - Uncond forward: line 689-690 그대로 (uncond_sequence = current_sequence[uncond_mask])

★ Lesson E containment via pt_layer_mask — full + uncond 양쪽 모두.
★ PRNG: numpy seed=0 (Lesson 9), x_t 외부 주입.

Output fixtures (production case):
  - out/stage9_pt_video_v_full_step0.npy   (n_video, 48)
  - out/stage9_pt_video_v_unc_step0.npy    (n_video, 48)
  - out/stage9_pt_video_x_t_init_prod.npy  (n_video, 48)  ← MLX 측 입력으로 사용
"""
from __future__ import annotations

import json
import os
import sys
import importlib

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("refs/Lance"))
from tools._pt_smoke_common import install_pt_smoke_env, pt_layer_mask
install_pt_smoke_env()

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from lance_mlx.backbone import LanceTextConfig
from lance_mlx.pipelines.t2v import (
    build_t2v_positions, vae_latent_position_indices,
    MAX_NUM_LATENT_FRAMES, MAX_LATENT_SIZE, LATENT_CHANNEL, SPATIAL_MERGE_SIZE,
    TOKENS_PER_SECOND, START_OF_IMAGE, VIDEO_TOKEN_ID,
)
from lance_mlx.pipelines._t2v_seq import build_t2v_sequence_pt
from lance_mlx.attn_mask import build_lance_attention_mask

qwen2_navit = importlib.import_module("modeling.lance.qwen2_navit")
Qwen2MoTDecoderLayer = qwen2_navit.Qwen2MoTDecoderLayer
PtQwen2ForCausalLM = qwen2_navit.Qwen2ForCausalLM
from modeling.qwen2.configuration_qwen2 import Qwen2Config
from modeling.qwen2.modeling_qwen2 import Qwen2RMSNorm
from data.data_utils import create_sparse_mask
from data.common import shift_position_ids
import torch.nn.attention.flex_attention as _fa_mod
qwen2_navit.flex_attention = _fa_mod.flex_attention


# Constants
PT_WEIGHTS_IMG = "checkpoints/Lance/Lance_3B/model.safetensors"
PT_WEIGHTS_VID_SUP = "checkpoints/Lance/Lance_3B_Video/model_supplement.safetensors"
USER_PROMPT = "A red panda riding a wave at sunset."
T_VIDEO, H_PIX, W_PIX = 5, 128, 128
VAE_DOWN_T, VAE_DOWN_S = 4, 16
NUMPY_SEED = 0


# ---- PT model (v2 클래스 그대로) ----
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
        self.q_cfg.qk_norm = True; self.q_cfg.qk_norm_und = True
        self.q_cfg.qk_norm_gen = True; self.q_cfg.layer_module = "Qwen2MoTDecoderLayer"
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
        self.vae2llm.bias.data = d["vae2llm.bias"].clone()
        self.llm2vae.weight.data = d["llm2vae.weight"].clone()
        self.llm2vae.bias.data = d["llm2vae.bias"].clone()
        self.time_fc1.weight.data = d["time_embedder.mlp.0.weight"].clone()
        self.time_fc1.bias.data = d["time_embedder.mlp.0.bias"].clone()
        self.time_fc2.weight.data = d["time_embedder.mlp.2.weight"].clone()
        self.time_fc2.bias.data = d["time_embedder.mlp.2.bias"].clone()
        self.latent_pos_embed.weight.data = d["latent_pos_embed.pos_embed"].clone()

    def time_embed(self, t):
        half = 256 // 2
        freqs = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(0, half, dtype=torch.float32) / half)
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(torch.bfloat16)
        h = self.time_fc1(emb)
        h = torch.nn.functional.silu(h)
        return self.time_fc2(h)


# ---- forward helpers ----
def _pt_forward_one(pt, input_ids, position_ids, attn_dense_bool, vae_indices,
                    x_t, t_scalar, vae_pos_ids):
    """One PT forward → v at vae_indices (n_video, 48) f32.

    Mirrors validation_gen step 0 (line 656-685) for full, or uncond_forward
    (line 854-872) for uncond.  Same logic — just different inputs.
    """
    L = int(input_ids.shape[-1])
    n_video = int(vae_indices.shape[0])

    # text embed → fill at text positions; vae_embed → fill at vae positions
    text_embed_full = pt.embed_tokens(input_ids.unsqueeze(0))   # (1, L, D)
    current_sequence = torch.zeros((1, L, pt.cfg.hidden_size), dtype=torch.bfloat16)
    # text positions = everything except vae_indices
    all_idx = torch.arange(L, dtype=torch.long)
    vae_idx_t = torch.from_numpy(vae_indices).long()
    gen_mask_bool = torch.zeros(L, dtype=torch.bool)
    gen_mask_bool[vae_idx_t] = True
    text_idx_t = all_idx[~gen_mask_bool]
    current_sequence[:, text_idx_t, :] = text_embed_full[:, text_idx_t, :]

    # vae_embed
    timestep_per = torch.full((n_video,), t_scalar, dtype=torch.float32)
    vae_embed = (pt.vae2llm(x_t.to(torch.bfloat16))
                 + pt.time_embed(timestep_per)
                 + pt.latent_pos_embed(vae_pos_ids))
    current_sequence[:, vae_idx_t, :] = vae_embed.unsqueeze(0)

    # packed_und/gen
    packed_gen_idx = vae_idx_t
    packed_und_idx = all_idx[~gen_mask_bool]

    # ★ Lesson E containment: bool dense only.
    layer_mask = pt_layer_mask(attn_dense_bool)

    # mrope cos/sin
    head_dim = pt.cfg.head_dim
    L_pos = position_ids.shape[-1]
    base = pt.cfg.rope_theta
    ms = pt.cfg.rope_scaling["mrope_section"]
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    inv = inv_freq[None, None, :, None].expand(3, L_pos, -1, 1)
    pos = position_ids.float()[:, :, None, :]
    freqs = (inv @ pos).transpose(2, 3)
    s_t, s_h, s_w = ms
    t_p = freqs[0, 0, :, :s_t]
    h_p = freqs[1, 0, :, s_t:s_t+s_h]
    w_p = freqs[2, 0, :, s_t+s_h:s_t+s_h+s_w]
    f_ = torch.cat([t_p, h_p, w_p], dim=-1)
    emb = torch.cat([f_, f_], dim=-1)
    cos_ = emb.cos().to(torch.bfloat16)
    sin_ = emb.sin().to(torch.bfloat16)

    h = current_sequence[0]
    sample_lens = [L]
    for Lyr in pt.layers:
        h = Lyr(
            packed_sequence=h, sample_lens=sample_lens,
            attention_mask=layer_mask,
            packed_position_embeddings=(cos_, sin_),
            packed_und_token_indexes=packed_und_idx,
            packed_gen_token_indexes=packed_gen_idx,
            mode_forward="validation",
        )
    h_und = pt.final_norm(h)
    h_gen = pt.norm_moe_gen(h)
    out = torch.zeros_like(h)
    out[packed_und_idx] = h_und[packed_und_idx]
    out[packed_gen_idx] = h_gen[packed_gen_idx]
    v = pt.llm2vae(out[packed_gen_idx].to(torch.bfloat16))
    return v.to(torch.float32)


def main():
    os.makedirs("out", exist_ok=True)
    print("=" * 72)
    print("STAGE 9 §1 단계 2 — v3 production PT smoke (full + uncond v_t)")
    print("=" * 72)

    # ---- shapes ----
    t_lat = (T_VIDEO - 1) // VAE_DOWN_T + 1
    h_lat = H_PIX // VAE_DOWN_S
    w_lat = W_PIX // VAE_DOWN_S
    n_video = t_lat * h_lat * w_lat
    print(f"[shape] T={T_VIDEO} px=({H_PIX},{W_PIX}) → latent t={t_lat} h={h_lat} w={w_lat}  n_video={n_video}")

    # ---- noise (numpy seed=0) ----
    rng = np.random.default_rng(NUMPY_SEED)
    x_t_np = rng.standard_normal((n_video, LATENT_CHANNEL), dtype=np.float32)
    x_t = torch.from_numpy(x_t_np)
    print(f"[input] x_t shape={x_t_np.shape}  std={x_t_np.std():.4f}")

    # ---- tokenizer + sequence (production, text_template=True) ----
    tok = AutoTokenizer.from_pretrained("checkpoints/Lance-3B-MLX", use_fast=True)
    seq = build_t2v_sequence_pt(USER_PROMPT, tok,
                                num_frames=T_VIDEO, H=H_PIX, W=W_PIX,
                                vae_down_t=VAE_DOWN_T, vae_down_s=VAE_DOWN_S)
    input_ids_np = seq["input_ids"].astype(np.int64)
    L = seq["L"]
    modality = seq["sample_modality"]
    vae_token_indices = seq["packed_vae_token_indexes"].astype(np.int64)
    split_lens = seq["split_lens"]
    attn_modes = seq["attn_modes"]
    vs_idx = int(np.where(input_ids_np == START_OF_IMAGE)[0][0])
    print(f"[seq] L={L}  text_split_len={vs_idx}  split_lens={split_lens}  attn_modes={attn_modes}")

    # ---- positions (full + uncond) ----
    full_pos_np = build_t2v_positions(vs_idx, t_lat, h_lat, w_lat, L, second_per_grid_t=1.0)
    full_pos_pt = torch.from_numpy(full_pos_np.astype(np.int64))

    uncond_mask = modality != 0
    uncond_input_ids_np = input_ids_np[uncond_mask]
    uncond_L = int(uncond_mask.sum())
    u_vs_idx = int(np.where(uncond_input_ids_np == START_OF_IMAGE)[0][0])
    uncond_pos_np = build_t2v_positions(u_vs_idx, t_lat, h_lat, w_lat, uncond_L, second_per_grid_t=1.0)
    uncond_pos_pt = torch.from_numpy(uncond_pos_np.astype(np.int64))

    # uncond split-level filter
    uncond_split_lens, uncond_attn_modes = [], []
    cursor = 0
    for sl, am in zip(split_lens, attn_modes):
        sub_mod = modality[cursor:cursor + sl]
        keep = int((sub_mod != 0).sum())
        cursor += sl
        if keep == 0:
            continue
        uncond_split_lens.append(keep)
        uncond_attn_modes.append(am)
    uncond_vae_token_indices = np.where(uncond_input_ids_np == VIDEO_TOKEN_ID)[0].astype(np.int64)
    print(f"[uncond] L={uncond_L}  split_lens={uncond_split_lens}  attn_modes={uncond_attn_modes}")

    # ---- attn_mask (full + uncond) — bool dense via PT create_sparse_mask ----
    def _pt_dense_mask(L_, sl_, am_):
        attn_modes_ = ["full" if m in ("full_noise","full_noise_target") else m for m in am_]
        predicate = create_sparse_mask(document_lens=[L_], split_lens=sl_,
                                       attn_modes=attn_modes_, device=torch.device("cpu"))
        q = torch.arange(L_)[:, None]; k = torch.arange(L_)[None, :]
        b = torch.tensor(0); h = torch.tensor(0)
        return predicate(b=b, h=h, q_idx=q, kv_idx=k).contiguous()

    full_dense = _pt_dense_mask(L, split_lens, attn_modes)
    uncond_dense = _pt_dense_mask(uncond_L, uncond_split_lens, uncond_attn_modes)
    print(f"[mask] full: {tuple(full_dense.shape)} True={int(full_dense.sum())}")
    print(f"[mask] uncond: {tuple(uncond_dense.shape)} True={int(uncond_dense.sum())}")

    # ---- VAE latent position ids ----
    vae_pos_ids_np = vae_latent_position_indices(t_lat, h_lat, w_lat)
    vae_pos_ids_pt = torch.from_numpy(vae_pos_ids_np.astype(np.int64))

    # ---- build + load PT model (bf16) ----
    print("\n[build+load] PtLanceVideoT2V (image backbone + video supplement bf16) ...")
    pt = PtLanceVideoT2V()
    pt.load_pt()
    pt.to_bf16()
    print(f"[load] latent_pos_embed.shape = {tuple(pt.latent_pos_embed.weight.shape)}")

    # ---- full forward ----
    t_scalar = 1.0
    print(f"\n[forward FULL] PT step 0 (t={t_scalar}) ...")
    with torch.no_grad():
        v_full = _pt_forward_one(
            pt, torch.from_numpy(input_ids_np), full_pos_pt, full_dense,
            vae_token_indices, x_t, t_scalar, vae_pos_ids_pt,
        )
    v_full_np = v_full.cpu().numpy()
    print(f"  v_full: shape={v_full_np.shape}  ||v||={np.linalg.norm(v_full_np):.3f}  "
          f"mean={v_full_np.mean():+.4f}  std={v_full_np.std():.4f}")

    # ---- uncond forward ----
    print(f"\n[forward UNCOND] PT step 0 — uncond_sequence = current_sequence[uncond_mask]")
    print(f"                 uncond_input_ids shape={uncond_input_ids_np.shape}")
    with torch.no_grad():
        v_unc = _pt_forward_one(
            pt, torch.from_numpy(uncond_input_ids_np), uncond_pos_pt, uncond_dense,
            uncond_vae_token_indices, x_t, t_scalar, vae_pos_ids_pt,
        )
    v_unc_np = v_unc.cpu().numpy()
    print(f"  v_unc: shape={v_unc_np.shape}  ||v||={np.linalg.norm(v_unc_np):.3f}  "
          f"mean={v_unc_np.mean():+.4f}  std={v_unc_np.std():.4f}")

    # ---- CFG blend (production: cfg_text_scale=4.0, global renorm) ----
    cfg_text_scale = 4.0
    v_blend = v_unc + cfg_text_scale * (v_full - v_unc)
    norm_v_full = np.linalg.norm(v_full_np)
    norm_v_blend = np.linalg.norm(v_blend.numpy())
    scale = float(min(norm_v_full / (norm_v_blend + 1e-8), 1.0))
    scale = max(scale, 0.0)  # cfg_renorm_min=0
    v_final = (v_blend * scale).numpy()
    print(f"\n[blend] cfg=4.0, global renorm min=0:")
    print(f"  v_blend (pre-renorm): ||v||={norm_v_blend:.3f}")
    print(f"  renorm scale: {scale:.4f}  (= clamp(||v_full||/||v_blend||, 0, 1))")
    print(f"  v_final:  ||v||={np.linalg.norm(v_final):.3f}  std={v_final.std():.4f}")
    assert not np.isnan(v_final).any(), "v_final has NaN"

    # ---- save ----
    np.save("out/stage9_pt_video_v_full_step0.npy", v_full_np)
    np.save("out/stage9_pt_video_v_unc_step0.npy", v_unc_np)
    np.save("out/stage9_pt_video_v_blend_step0.npy", v_final)
    np.save("out/stage9_pt_video_x_t_init_prod.npy", x_t_np)
    meta = {
        "doctrine": "v3 production: text_template=True + apply_qwen=True, PT direct call",
        "prng": "numpy default_rng",
        "prng_seed": NUMPY_SEED,
        "user_prompt": USER_PROMPT,
        "video_TpxHpxWpx": [T_VIDEO, H_PIX, W_PIX],
        "video_grid_thw": [t_lat, h_lat, w_lat],
        "L_full": L, "L_uncond": uncond_L,
        "vae_token_indices_len": int(vae_token_indices.shape[0]),
        "cfg_text_scale": cfg_text_scale,
        "cfg_renorm_min": 0.0,
        "cfg_renorm_type": "global",
        "timestep_first_step": t_scalar,
        "v_full_norm": float(np.linalg.norm(v_full_np)),
        "v_unc_norm":  float(np.linalg.norm(v_unc_np)),
        "v_blend_norm_post_renorm": float(np.linalg.norm(v_final)),
        "renorm_scale": scale,
    }
    with open("out/stage9_pt_video_v3_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[save] v_full / v_unc / v_blend / x_t_init  +  meta")
    print(f"[OK] v3 production 정답지 확보. 다음: 단계 5 single-step byte-diff.")


if __name__ == "__main__":
    main()
