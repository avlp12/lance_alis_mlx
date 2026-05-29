"""STAGE 9 §0 — PT 정답지 (first-step velocity) 확보.

원본 ByteDance Lance_3B_Video PT weight 로 첫 step velocity v_t 계산 후
out/stage9_pt_video_*.npy + meta.json 으로 저장. 같은 입력 + meta 정보로
MLX 측이 재현 가능.

Doctrine (STAGE 1~8 통일):
  - source-of-truth: 원본 ByteDance PT (`checkpoints/Lance/Lance_3B/model.safetensors`
    + `checkpoints/Lance/Lance_3B_Video/model_supplement.safetensors`).
    RockTalk Lance-3B-Video-MLX 는 변환 대조용으로 별도 보존.
  - PRNG: numpy seed=0 통일 (STAGE 6 교훈 9). mx.random 안 섞음.

Inputs (소형):
  - T_video=5 frames, H=W=128 px
  - latent: t=2, h=w=8 → n_noise = t·h·w = 128
  - patch_latent_dim = 1·1·1·48 = 48 (video config latent_patch_size=[1,1,1])
  - prompt: "A red panda riding a wave at sunset."

Outputs in out/:
  - stage9_pt_video_x_t_init.npy     (128, 48)  initial noise
  - stage9_pt_video_text_ids.npy     (L,)       full token sequence
  - stage9_pt_video_vae_pos_ids.npy  (128,)     latent position indices
  - stage9_pt_video_current_pos_ids.npy (3, 1, L)  mRoPE positions
  - stage9_pt_video_attn_mask.npy    (L, L) bool  attention mask
  - stage9_pt_video_v_t_step0.npy    (128, 48)  first-step velocity (the gate)
  - stage9_pt_video_meta.json        PRNG/seed/grid_thw/layout meta
"""
from __future__ import annotations

import json
import os
import sys
import types
import importlib
import importlib.machinery


# ---------- PT environment shim (mirror stage7_ti2i_compare.py) -------------
def _install_flash_attn_stub() -> None:
    import torch
    import torch.nn.functional as F

    def _shim(q, k, v, cu_seqlens_q, cu_seqlens_k,
              max_seqlen_q, max_seqlen_k, causal=True, **_kw):
        if cu_seqlens_q.numel() != 2 or cu_seqlens_k.numel() != 2:
            raise NotImplementedError("flash_attn shim handles single-sequence only")
        n_heads = q.shape[1]; n_kv = k.shape[1]
        if n_kv < n_heads:
            rep = n_heads // n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        q4 = q.transpose(0, 1).unsqueeze(0)
        k4 = k.transpose(0, 1).unsqueeze(0)
        v4 = v.transpose(0, 1).unsqueeze(0)
        out = F.scaled_dot_product_attention(q4, k4, v4, is_causal=bool(causal))
        return out.squeeze(0).transpose(0, 1).contiguous()

    mock = types.ModuleType("flash_attn")
    mock.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
    mock.flash_attn_varlen_func = _shim
    sys.modules["flash_attn"] = mock

    import transformers.utils.import_utils as _imp_utils
    for fn in ("is_flash_attn_2_available", "is_flash_attn_3_available",
               "is_flash_attn_4_available"):
        setattr(_imp_utils, fn, lambda: False)
    import transformers.utils as _utils
    for fn in ("is_flash_attn_2_available", "is_flash_attn_3_available",
               "is_flash_attn_4_available"):
        setattr(_utils, fn, lambda: False)


def _install_modeling_lance_stub() -> None:
    lance_dir = os.path.abspath("refs/Lance/modeling/lance")
    pkg = types.ModuleType("modeling.lance")
    pkg.__path__ = [lance_dir]
    sys.modules["modeling.lance"] = pkg


def _install_flex_attention_sdpa_patch() -> None:
    import torch
    import torch.nn.functional as F
    import torch.nn.attention.flex_attention as _fa
    from torch.nn.attention.flex_attention import BlockMask

    def _dense_from_block_mask(bm: BlockMask, L: int) -> torch.Tensor:
        q = torch.arange(L)[:, None]; k = torch.arange(L)[None, :]
        b = torch.tensor(0); h = torch.tensor(0)
        return bm.mask_mod(b, h, q, k)

    def patched_flex_attention(query, key, value, block_mask, enable_gqa=True,
                                return_lse=False, kernel_options=None, **kw):
        assert query.dim() == 4
        n_h = query.shape[1]; n_kv = key.shape[1]
        q4, k4, v4 = query, key, value
        if enable_gqa and n_kv < n_h:
            rep = n_h // n_kv
            k4 = k4.repeat_interleave(rep, dim=1)
            v4 = v4.repeat_interleave(rep, dim=1)
        L_q = query.shape[2]
        if isinstance(block_mask, BlockMask):
            dense = _dense_from_block_mask(block_mask, L_q)
        else:
            dense = block_mask
        if dense.dtype != torch.bool:
            dense = dense.to(torch.bool)
        # STAGE 7 §3 Lesson E: pass bf16 0/-inf additive mask directly to PT-side.
        # The bool dense conversion would invert -inf (truthy) ↔ 0 (falsy).
        add = torch.zeros(dense.shape, dtype=q4.dtype, device=q4.device)
        add.masked_fill_(~dense, float("-inf"))
        attn_mask = add[None, None, :, :]
        return F.scaled_dot_product_attention(q4, k4, v4, attn_mask=attn_mask)

    _fa.flex_attention = patched_flex_attention


_install_flash_attn_stub()
sys.path.insert(0, os.path.abspath("refs/Lance"))
_install_modeling_lance_stub()
_install_flex_attention_sdpa_patch()


# ---------- regular imports (after shim) ------------------------------------
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from lance_mlx.backbone import LanceTextConfig
from lance_mlx.pipelines.x2t import (
    IM_START_ID, IM_END_ID, VIS_START_ID, VIS_END_ID, IMG_TOKEN_ID,
)

# PT bits
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
# Original ByteDance PT (NOT RockTalk — RockTalk is for conversion-cross-check only).
PT_WEIGHTS_IMG = "checkpoints/Lance/Lance_3B/model.safetensors"
PT_WEIGHTS_VID_SUP = "checkpoints/Lance/Lance_3B_Video/model_supplement.safetensors"

# Production config — confirmed in refs/Lance/inference_lance.sh:122-123
#   --max_num_frames 121  --max_latent_size 64
# → max_num_latent_frames = 121 // 4 + 1 = 31
MAX_NUM_LATENT_FRAMES = 31
MAX_LATENT_SIZE = 64
LATENT_PATCH = (1, 1, 1)             # video config latent_patch_size
VAE_DOWN_SPATIAL = 16                # vae_downsample_spatial × latent_patch_size[-1] = 16·1 = 16
VAE_DOWN_TEMPORAL = 4
LATENT_CHANNEL = 48                  # vae z_dim
PATCH_LATENT_DIM = LATENT_PATCH[0] * LATENT_PATCH[1] * LATENT_PATCH[2] * LATENT_CHANNEL  # = 48

# Smoke input — small enough for one PT forward on CPU
T_VIDEO = 5                          # frames
H_PIX = W_PIX = 128
USER_PROMPT = "A red panda riding a wave at sunset."

# T2V system prompt — refs/Lance/data/common.py:30-31
T2V_SYSTEM_PROMPT = ("Describe the video by detailing the color, quantity, "
                    "visible text, shape, size, texture, spatial relationships "
                    "and motion/camera movements of the objects and background:")

# Numpy PRNG (STAGE 6 lesson 9 — single source of randomness for fixture)
NUMPY_SEED = 0


# ---------- PT model -------------------------------------------------------
class PtLanceVideoT2V:
    """PT-side Lance LLM + adapters with t2v-specific latent_pos_embed (31·64²)."""

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
        # video latent_pos_embed: 31·64·64 = 126976 (image is 1·64·64 = 4096)
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
        """Merge image backbone + video supplement → bf16 PT model."""
        d = {}
        with safe_open(PT_WEIGHTS_IMG, framework="pt", device="cpu") as f:
            for k in f.keys():
                d[k] = f.get_tensor(k).to(torch.bfloat16)
        with safe_open(PT_WEIGHTS_VID_SUP, framework="pt", device="cpu") as f:
            for k in f.keys():
                # Supplement overrides latent_pos_embed.pos_embed (4096→126976)
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
        # video supplement provides the (126976, 2048) table
        self.latent_pos_embed.weight.data = d["latent_pos_embed.pos_embed"].clone()
        assert self.latent_pos_embed.weight.shape == (
            MAX_NUM_LATENT_FRAMES * MAX_LATENT_SIZE * MAX_LATENT_SIZE,
            self.cfg.hidden_size,
        ), f"unexpected latent_pos_embed shape {self.latent_pos_embed.weight.shape}"

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


# ---------- sequence construction (t2v: text + noise only) ------------------
def build_t2v_sequence(tokenizer, user_prompt: str, n_noise: int) -> dict:
    """text + noise. No ViT, no VAE cond.

    Layout:
      <im_start>system\n[T2V_SYSTEM_PROMPT]<im_end>\n
      <im_start>user\n[user_prompt]<im_end>\n
      <im_start>assistant\n
      <vis_start>[IMG_TOKEN × n_noise]<vis_end>
    """
    sys_ids   = tokenizer(T2V_SYSTEM_PROMPT, add_special_tokens=False)["input_ids"]
    user_ids  = tokenizer(user_prompt,        add_special_tokens=False)["input_ids"]
    newline   = tokenizer("\n",               add_special_tokens=False)["input_ids"]
    sys_lbl   = tokenizer("system",           add_special_tokens=False)["input_ids"]
    usr_lbl   = tokenizer("user",             add_special_tokens=False)["input_ids"]
    asst_lbl  = tokenizer("assistant",        add_special_tokens=False)["input_ids"]

    sys_section  = [IM_START_ID] + sys_lbl + newline + sys_ids  + [IM_END_ID] + newline
    user_section = [IM_START_ID] + usr_lbl + newline + user_ids + [IM_END_ID] + newline
    asst_open    = [IM_START_ID] + asst_lbl + newline
    noise_slab   = [VIS_START_ID] + [IMG_TOKEN_ID] * n_noise + [VIS_END_ID]

    full = sys_section + user_section + asst_open + noise_slab
    L = len(full)
    noise_start = len(sys_section) + len(user_section) + len(asst_open) + 1   # one past <vis_start>

    # split_lens / attn_modes:
    #   pre-noise causal slab + noise slab (vis_start + N + vis_end)
    sl_pre_noise = noise_start - 1
    sl_noise     = n_noise + 2
    split_lens   = [sl_pre_noise, sl_noise]
    attn_modes   = ["causal", "noise"]
    assert sum(split_lens) == L, f"split sum {sum(split_lens)} != L {L}"

    # modality (pro_type=10 shifting): 0=text default, 1=noise
    modality = [0] * L
    for i in range(noise_start, noise_start + n_noise):
        modality[i] = 1

    return {
        "ids": full,
        "L": L,
        "noise_span": (noise_start, noise_start + n_noise),
        "split_lens": split_lens,
        "attn_modes": attn_modes,
        "modality": modality,
    }


def build_positions_pt(layout: dict, t_lat: int, h_lat: int, w_lat: int):
    """mRoPE positions (3, 1, L) for t2v: text running counter + noise 3D coords."""
    L = layout["L"]
    t_pos = np.zeros(L, dtype=np.int64)
    h_pos = np.zeros(L, dtype=np.int64)
    w_pos = np.zeros(L, dtype=np.int64)

    counter = 0
    i = 0
    while i < L:
        if i == layout["noise_span"][0]:
            n = layout["noise_span"][1] - layout["noise_span"][0]
            for ti in range(t_lat):
                for hi in range(h_lat):
                    for wi in range(w_lat):
                        idx = layout["noise_span"][0] + ti * h_lat * w_lat + hi * w_lat + wi
                        t_pos[idx] = counter + ti
                        h_pos[idx] = counter + hi
                        w_pos[idx] = counter + wi
            counter += max(t_lat, h_lat, w_lat)
            i += n
        else:
            t_pos[i] = h_pos[i] = w_pos[i] = counter
            counter += 1
            i += 1

    pos_ids = np.stack([t_pos, h_pos, w_pos], axis=0)[:, None, :]   # (3, 1, L)
    modality_t = torch.tensor(layout["modality"], dtype=torch.long)
    pos_t = torch.from_numpy(pos_ids).contiguous()
    shifted = shift_position_ids(
        pos_t,
        pos_shift=1000,
        attn_modes=layout["attn_modes"],
        split_lens=layout["split_lens"],
        shift_attn_mode=["full_noise", "full"],
        pro_type=10,
        i_sample_task=torch.tensor([0] * L),    # t2v task code (TASK_T2V=0 in lance.py)
        i_sample_modality=modality_t,
    )
    return shifted


def build_mask_pt(layout: dict, num_heads: int):
    """Build (BlockMask, dense L×L bool) for PT flex_attention path."""
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
    """`get_flattened_position_ids_extrapolate_video(t,h,w, max_latent_size=64)`:
       flat = t·max² + h·max + w.  refs/Lance/data/data_utils.py:58."""
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


# ---------- PT forward (one step) -------------------------------------------
def pt_forward_first_step(pt: PtLanceVideoT2V, layout: dict, x_t_pt: torch.Tensor,
                          t_scalar: float, vae_pos_ids_pt: torch.Tensor,
                          pos_ids_pt: torch.Tensor, attn_dense_bool: torch.Tensor,
                          intermediate_dump: dict = None):
    """Returns v_t at noise span (n_noise, 48) in f32.

    If intermediate_dump is a dict, populates with debugging tensors:
        text_embed, vae2llm_out, time_embed_out, latent_pos_out, vae_embed,
        embed_pre_transformer, hidden_layer0
    """
    L = layout["L"]
    ids = torch.tensor(layout["ids"], dtype=torch.long).unsqueeze(0)
    text_embed = pt.embed_tokens(ids)            # (1, L, D) bf16

    # noise slab: vae_embed = vae2llm(x_t) + time_embedder(t) + latent_pos_embed(vae_pos)
    ns, ne = layout["noise_span"]
    n_noise = ne - ns
    timestep_per = torch.full((n_noise,), t_scalar, dtype=torch.float32)
    vae2llm_out = pt.vae2llm(x_t_pt.to(torch.bfloat16))
    time_embed_out = pt.time_embed(timestep_per)
    latent_pos_out = pt.latent_pos_embed(vae_pos_ids_pt)
    vae_embed = vae2llm_out + time_embed_out + latent_pos_out

    embed = text_embed.clone()
    embed[:, ns:ne, :] = vae_embed.unsqueeze(0)
    if intermediate_dump is not None:
        intermediate_dump["text_embed"]    = text_embed[0].detach().to(torch.float32).cpu().numpy()
        intermediate_dump["vae2llm_out"]   = vae2llm_out.detach().to(torch.float32).cpu().numpy()
        intermediate_dump["time_embed_out"]= time_embed_out.detach().to(torch.float32).cpu().numpy()
        intermediate_dump["latent_pos_out"]= latent_pos_out.detach().to(torch.float32).cpu().numpy()
        intermediate_dump["vae_embed"]     = vae_embed.detach().to(torch.float32).cpu().numpy()
        intermediate_dump["embed_pre_transformer"] = embed[0].detach().to(torch.float32).cpu().numpy()

    # packed_und = non-noise tokens; packed_gen = noise tokens (no VAE cond in t2v)
    all_idx = torch.arange(L, dtype=torch.long)
    gen_mask = torch.zeros(L, dtype=torch.bool)
    gen_mask[ns:ne] = True
    packed_gen_idx = all_idx[gen_mask]
    packed_und_idx = all_idx[~gen_mask]

    # Build attn mask in bf16 0/-inf form for PT-side flex_attention shim.
    # (Lesson E: bool conversion would invert. Pass additive.)
    attn_add = torch.zeros((L, L), dtype=torch.bfloat16)
    attn_add.masked_fill_(~attn_dense_bool, float("-inf"))

    cos, sin = pt.mrope_cos_sin(pos_ids_pt)
    h = embed[0]
    sample_lens = [L]
    for li, Lyr in enumerate(pt.layers):
        h = Lyr(
            packed_sequence=h, sample_lens=sample_lens,
            attention_mask=attn_add,
            packed_position_embeddings=(cos, sin),
            packed_und_token_indexes=packed_und_idx,
            packed_gen_token_indexes=packed_gen_idx,
            mode_forward="validation",
        )
        if intermediate_dump is not None and li == 0:
            intermediate_dump["hidden_layer0"] = h.detach().to(torch.float32).cpu().numpy()
    h_und = pt.final_norm(h)
    h_gen = pt.norm_moe_gen(h)
    out = torch.zeros_like(h)
    out[packed_und_idx] = h_und[packed_und_idx]
    out[packed_gen_idx] = h_gen[packed_gen_idx]
    v = pt.llm2vae(out[ns:ne].to(torch.bfloat16))
    return v.to(torch.float32)


# ---------- main ------------------------------------------------------------
def main():
    os.makedirs("out", exist_ok=True)
    print("=" * 70)
    print("STAGE 9 §0 — PT first-step velocity fixture (original ByteDance PT)")
    print("=" * 70)

    # ---- derive latent shape ----
    t_lat = (T_VIDEO - 1) // VAE_DOWN_TEMPORAL + 1   # (5-1)/4+1 = 2
    h_lat = H_PIX // VAE_DOWN_SPATIAL                # 128/16 = 8
    w_lat = W_PIX // VAE_DOWN_SPATIAL                # 128/16 = 8
    n_noise = t_lat * h_lat * w_lat                  # 128
    grid_thw = [t_lat, h_lat, w_lat]
    print(f"[shape] T={T_VIDEO} px=({H_PIX},{W_PIX}) → latent t={t_lat} h={h_lat} w={w_lat}")
    print(f"        n_noise={n_noise}  patch_latent_dim={PATCH_LATENT_DIM}")

    # ---- synthetic noise (numpy seed=0, NO mx.random — Lesson 9) ----
    rng = np.random.default_rng(NUMPY_SEED)
    x_t_init = rng.standard_normal((n_noise, PATCH_LATENT_DIM), dtype=np.float32)
    print(f"[input] x_t: shape={x_t_init.shape} stats: "
          f"mean={x_t_init.mean():+.4f} std={x_t_init.std():.4f}")

    # ---- tokenizer + sequence ----
    print(f"[tokenizer] loading from checkpoints/Lance-3B-MLX/ (shared tokenizer.json) ...")
    tok = AutoTokenizer.from_pretrained("checkpoints/Lance-3B-MLX", use_fast=True)
    layout = build_t2v_sequence(tok, USER_PROMPT, n_noise)
    print(f"[seq]   L={layout['L']}  noise_span={layout['noise_span']}")
    print(f"        split_lens={layout['split_lens']}  attn_modes={layout['attn_modes']}")
    print(f"        prompt={USER_PROMPT!r}")

    # ---- positions + mask ----
    pos_ids_pt = build_positions_pt(layout, t_lat=t_lat, h_lat=h_lat, w_lat=w_lat)
    print(f"[pos] shape={tuple(pos_ids_pt.shape)} "
          f"noise_start_pos={pos_ids_pt[:, 0, layout['noise_span'][0]].tolist()}")

    cfg_tmp = PtLanceVideoT2V()
    num_heads = cfg_tmp.cfg.num_attention_heads
    _, attn_dense_bool = build_mask_pt(layout, num_heads=num_heads)
    attn_dense_bool = attn_dense_bool.contiguous()
    print(f"[mask] dense shape={tuple(attn_dense_bool.shape)} "
          f"non-zero ratio={float(attn_dense_bool.float().mean()):.4f}")

    # ---- VAE latent position indices (flat 3D index in the 31·64² table) ----
    vae_pos_ids_np = vae_position_indices_video(t_lat, h_lat, w_lat)
    vae_pos_ids_pt = torch.from_numpy(vae_pos_ids_np)
    print(f"[lat_pos] vae_pos_ids: shape={vae_pos_ids_np.shape} "
          f"range=[{vae_pos_ids_np.min()}, {vae_pos_ids_np.max()}]")
    assert vae_pos_ids_np.max() < MAX_NUM_LATENT_FRAMES * MAX_LATENT_SIZE ** 2, \
        f"pos_id {vae_pos_ids_np.max()} >= table size {MAX_NUM_LATENT_FRAMES * MAX_LATENT_SIZE ** 2}"

    # ---- build + load PT model (~50s) ----
    print("[build] PT model (PtLanceVideoT2V) ...")
    pt = PtLanceVideoT2V()
    print("[load] image backbone + video supplement (bf16) ...")
    pt.load_pt()
    pt.to_bf16()
    print("[load] done.  latent_pos_embed.weight.shape =",
          tuple(pt.latent_pos_embed.weight.shape))

    # ---- one PT forward (first step: timestep = 1.0) ----
    t_scalar = 1.0
    x_t_pt = torch.from_numpy(x_t_init)
    print(f"\n[forward] PT first-step velocity (timestep={t_scalar}) ...")
    intermediates = {}
    with torch.no_grad():
        v_t = pt_forward_first_step(
            pt, layout, x_t_pt, t_scalar,
            vae_pos_ids_pt, pos_ids_pt, attn_dense_bool,
            intermediate_dump=intermediates,
        )
    v_t_np = v_t.cpu().numpy()
    print(f"[forward] v_t: shape={v_t_np.shape} "
          f"||v_t||={float(np.linalg.norm(v_t_np)):.3f} "
          f"mean={v_t_np.mean():+.4f} std={v_t_np.std():.4f}")
    print(f"          range=[{v_t_np.min():+.3f}, {v_t_np.max():+.3f}]")
    assert not np.isnan(v_t_np).any(), "v_t has NaN"

    # ---- save fixtures (numpy npy + meta json) ----
    print("\n[save] fixtures →")
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

    # Diagnostic intermediates (for §0 gate failure debugging)
    for name, arr in intermediates.items():
        np.save(f"out/stage9_pt_video_diag_{name}.npy", arr)
        print(f"  out/stage9_pt_video_diag_{name}.npy  shape={arr.shape}  dtype={arr.dtype}")

    meta = {
        "doctrine_source": ("original ByteDance bytedance-research/Lance, "
                            "merged: Lance_3B (image backbone) + Lance_3B_Video/model_supplement"),
        "prng": "numpy.random.default_rng",
        "prng_seed": NUMPY_SEED,
        "user_prompt": USER_PROMPT,
        "t2v_system_prompt": T2V_SYSTEM_PROMPT,
        "video_grid_thw": grid_thw,         # [t_lat, h_lat, w_lat]
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
        "noise_span": list(layout["noise_span"]),
        "split_lens": layout["split_lens"],
        "attn_modes": layout["attn_modes"],
        "timestep_first_step": t_scalar,
        "weight_files": {
            "image_backbone": PT_WEIGHTS_IMG,
            "video_supplement": PT_WEIGHTS_VID_SUP,
        },
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
    print("\n[OK] STAGE 9 §0 PT 정답지 확보. 다음: MLX 측 forward 작성 후 byte-diff 게이트.")


if __name__ == "__main__":
    main()
