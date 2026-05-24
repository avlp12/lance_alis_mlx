"""STAGE 7 §3 PT byte-diff — TI2I 3-forward cosine.

Diagnostic harness for `lance_mlx/pipelines/image_edit.py`.  Reproduces
PT's validation_gen logic for TI2I (lance.py:377+) and runs *one*
denoise step on both sides with identical cond image + instruction +
seed.  Captures the three CFG forwards:

    v_full       — text + ViT + VAE-cond present
    v_t_uncond   — text dropped (ViT + VAE-cond present)
    v_tv_uncond  — text + ViT dropped (VAE-cond only)

PT side uses *PT's* correct routing/positions/timestep logic (read from
lance.py).  MLX side uses our `image_edit.image_edit()` step-0 forward
verbatim.  Divergence localises the bug.

Gate: each cos ≥ 0.999.
"""
from __future__ import annotations

import os
import sys
import types
import importlib
import importlib.machinery


# ---------- PT environment shim (mirror stage6_pt_denoise_compare.py) -------
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


_install_flash_attn_stub()
sys.path.insert(0, os.path.abspath("refs/Lance"))
_install_modeling_lance_stub()


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
        add = torch.zeros(dense.shape, dtype=q4.dtype, device=q4.device)
        add.masked_fill_(~dense, float("-inf"))
        attn_mask = add[None, None, :, :]
        return F.scaled_dot_product_attention(q4, k4, v4, attn_mask=attn_mask)

    _fa.flex_attention = patched_flex_attention


_install_flex_attention_sdpa_patch()


# ---------- imports ----------
import numpy as np
import torch
from safetensors import safe_open

import mlx.core as mx
from PIL import Image
from transformers import AutoTokenizer

from lance_mlx.backbone import LanceLLM, LanceTextConfig, load_full_lance
from lance_mlx.vit import LanceViT, load_lance_vit
from lance_mlx.vae_wan22 import Wan2_2_VAE, Wan22VAEConfig
from lance_mlx.pipelines.x2t import (
    preprocess_image, IM_START_ID, IM_END_ID, VIS_START_ID, VIS_END_ID, IMG_TOKEN_ID,
    SPATIAL_MERGE_SIZE,
)
from lance_mlx.pipelines.image_edit import (
    EDIT_SYSTEM_PROMPT, _vae_preprocess, _latent_position_indices, Z_DIM,
    SPATIAL_DOWNSAMPLE,
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


PT_WEIGHTS  = "checkpoints/Lance/Lance_3B/model.safetensors"
MLX_WEIGHTS = "checkpoints/Lance-3B-MLX/model.safetensors"


# ---------- shared PT helpers ----------
def cos_mlx(a: mx.array, b: mx.array) -> float:
    a_np = np.asarray(a, dtype=np.float32).flatten()
    b_np = np.asarray(b, dtype=np.float32).flatten()
    return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np) + 1e-12))


def cos_pt_mlx(pt: torch.Tensor, mlx_: mx.array) -> float:
    a = pt.detach().to(torch.float32).cpu().numpy().flatten()
    b = np.asarray(mlx_, dtype=np.float32).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ---------- PT model ----------
class PtLanceTI2I:
    """PT-side Lance LLM + adapters with TI2I-correct forward."""

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
        self.vae2llm = torch.nn.Linear(48, self.cfg.hidden_size, bias=True)
        self.llm2vae = torch.nn.Linear(self.cfg.hidden_size, 48, bias=True)
        self.time_fc1 = torch.nn.Linear(256, self.cfg.hidden_size, bias=True)
        self.time_fc2 = torch.nn.Linear(self.cfg.hidden_size, self.cfg.hidden_size, bias=True)
        # max_latent_size = 64 (image variant) → 1*64*64 = 4096 entries
        self.latent_pos_embed = torch.nn.Embedding(1 * 64 * 64, self.cfg.hidden_size)
        self.final_norm = Qwen2RMSNorm(self.cfg.hidden_size, eps=self.cfg.rms_norm_eps)
        self.norm_moe_gen = Qwen2RMSNorm(self.cfg.hidden_size, eps=self.cfg.rms_norm_eps)

    def to_bf16(self):
        for m in (self.embed_tokens, self.vae2llm, self.llm2vae, self.time_fc1,
                  self.time_fc2, self.latent_pos_embed, self.final_norm, self.norm_moe_gen):
            m.to(torch.bfloat16)
        for L in self.layers: L.to(torch.bfloat16)

    def load_pt(self):
        with safe_open(PT_WEIGHTS, framework="pt", device="cpu") as f:
            d = {k: f.get_tensor(k).to(torch.bfloat16) for k in f.keys()}
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
        """Sinusoidal + MLP, bf16 throughout (matches PT precision contract)."""
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
        h_p = freqs[1, 0, :, s_t:s_t+s_h]
        w_p = freqs[2, 0, :, s_t+s_h:s_t+s_h+s_w]
        f = torch.cat([t_p, h_p, w_p], dim=-1)
        emb = torch.cat([f, f], dim=-1)
        return emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)


# ---------- sequence construction (shared by both sides) --------------------
def build_sequences(tokenizer, n_vit_tokens: int, n_cond: int, n_noise: int,
                    *, include_text: bool, include_vit: bool):
    """Returns dict with: ids list[int], slab spans, modality codes per token."""
    sys_ids = tokenizer(EDIT_SYSTEM_PROMPT, add_special_tokens=False)["input_ids"]
    inst_ids = tokenizer("Make it more vibrant and saturated.", add_special_tokens=False)["input_ids"]
    newline = tokenizer("\n", add_special_tokens=False)["input_ids"]
    sys_lbl  = tokenizer("system", add_special_tokens=False)["input_ids"]
    usr_lbl  = tokenizer("user", add_special_tokens=False)["input_ids"]
    asst_lbl = tokenizer("assistant", add_special_tokens=False)["input_ids"]

    sys_section = ([IM_START_ID] + sys_lbl + newline
                   + (sys_ids if include_text else [])
                   + [IM_END_ID] + newline)
    user_open = [IM_START_ID] + usr_lbl + newline
    vit_slab = ([VIS_START_ID] + [IMG_TOKEN_ID]*n_vit_tokens + [VIS_END_ID]
                if include_vit else [])
    inst_section = (inst_ids if include_text else []) + [IM_END_ID] + newline
    asst_open = [IM_START_ID] + asst_lbl + newline
    vae_slab   = [VIS_START_ID] + [IMG_TOKEN_ID]*n_cond + [VIS_END_ID]
    noise_slab = [VIS_START_ID] + [IMG_TOKEN_ID]*n_noise + [VIS_END_ID]
    full = (sys_section + user_open + vit_slab + inst_section
            + asst_open + vae_slab + noise_slab)
    L = len(full)

    # spans
    cursor = len(sys_section) + len(user_open)
    vit_start = (cursor + 1) if include_vit else -1
    cursor += len(vit_slab) + len(inst_section) + len(asst_open)
    vae_start   = cursor + 1
    cursor += len(vae_slab)
    noise_start = cursor + 1

    # split_lens + attn_modes — per PT validation_dataset.py:518-532.
    # Each vision slab (vis_start + placeholders + vis_end) is ONE entry:
    #   ViT slab → "full" (modality=4)
    #   VAE-cond slab → "full_noise" (modality=2)
    #   noise slab → "noise" (modality=1)
    # Text between slabs is "causal".
    split_lens = []
    attn_modes = []
    if include_vit:
        sl_pre_vit = vit_start - 1                  # text before vis_start(vit)
        split_lens.append(sl_pre_vit);    attn_modes.append("causal")
        sl_vit = n_vit_tokens + 2                   # vis_start + N + vis_end
        split_lens.append(sl_vit);        attn_modes.append("full")
        mid_start = vit_start + n_vit_tokens + 1    # one past vis_end(vit)
    else:
        mid_start = 0
    sl_mid = (vae_start - 1) - mid_start            # text up to vis_start(vae)
    split_lens.append(sl_mid);            attn_modes.append("causal")
    sl_vae = n_cond + 2
    split_lens.append(sl_vae);            attn_modes.append("full_noise")
    sl_noise = n_noise + 2
    split_lens.append(sl_noise);          attn_modes.append("noise")
    sl_tail = L - (noise_start + n_noise + 1)
    if sl_tail > 0:
        split_lens.append(sl_tail);       attn_modes.append("causal")
    assert sum(split_lens) == L, f"split_lens sum {sum(split_lens)} != L {L}"

    # Per-token modality (for shift_position_ids pro_type=10)
    modality = [0] * L  # default text
    if include_vit:
        for i in range(vit_start, vit_start + n_vit_tokens):
            modality[i] = 4   # ref_vit
    for i in range(vae_start, vae_start + n_cond):
        modality[i] = 2       # ref_source
    for i in range(noise_start, noise_start + n_noise):
        modality[i] = 1       # noise

    return {
        "ids": full, "L": L,
        "vit_span": (vit_start, vit_start + n_vit_tokens) if include_vit else None,
        "vae_span": (vae_start, vae_start + n_cond),
        "noise_span": (noise_start, noise_start + n_noise),
        "split_lens": split_lens,
        "attn_modes": attn_modes,
        "modality": modality,
    }


def build_positions_pt(layout: dict, T_g: int, H_g_m: int, W_g_m: int,
                       t_lat: int, h_lat: int, w_lat: int):
    """Build mRoPE (3, 1, L) positions per PT logic with shift_position_ids."""
    L = layout["L"]
    # Initialize per-axis positions naive (1D cumsum) — PT's get_rope_index
    # logic for non-vision tokens is just a running counter, with VIS spans
    # taking 3D coords; here we replicate the *outcome* directly.
    t_pos = np.zeros(L, dtype=np.int64)
    h_pos = np.zeros(L, dtype=np.int64)
    w_pos = np.zeros(L, dtype=np.int64)

    counter = 0
    i = 0
    while i < L:
        if layout["vit_span"] is not None and i == layout["vit_span"][0]:
            # ViT slab — 3D coords
            n = layout["vit_span"][1] - layout["vit_span"][0]
            for ti in range(T_g):
                for hi in range(H_g_m):
                    for wi in range(W_g_m):
                        idx = layout["vit_span"][0] + ti*H_g_m*W_g_m + hi*W_g_m + wi
                        t_pos[idx] = counter + ti
                        h_pos[idx] = counter + hi
                        w_pos[idx] = counter + wi
            counter += max(T_g, H_g_m, W_g_m)
            i += n
        elif i == layout["vae_span"][0]:
            n = layout["vae_span"][1] - layout["vae_span"][0]
            for ti in range(t_lat):
                for hi in range(h_lat):
                    for wi in range(w_lat):
                        idx = layout["vae_span"][0] + ti*h_lat*w_lat + hi*w_lat + wi
                        t_pos[idx] = counter + ti
                        h_pos[idx] = counter + hi
                        w_pos[idx] = counter + wi
            counter += max(t_lat, h_lat, w_lat)
            i += n
        elif i == layout["noise_span"][0]:
            n = layout["noise_span"][1] - layout["noise_span"][0]
            for ti in range(t_lat):
                for hi in range(h_lat):
                    for wi in range(w_lat):
                        idx = layout["noise_span"][0] + ti*h_lat*w_lat + hi*w_lat + wi
                        t_pos[idx] = counter + ti
                        h_pos[idx] = counter + hi
                        w_pos[idx] = counter + wi
            counter += max(t_lat, h_lat, w_lat)
            i += n
        else:
            t_pos[i] = h_pos[i] = w_pos[i] = counter
            counter += 1
            i += 1

    pos_ids = np.stack([t_pos, h_pos, w_pos], axis=0)  # (3, L)
    pos_ids = pos_ids[:, None, :]  # (3, 1, L)
    # shift_position_ids: PT pro_type=10
    modality_t = torch.tensor(layout["modality"], dtype=torch.long)
    pos_t = torch.from_numpy(pos_ids).contiguous()
    shifted = shift_position_ids(
        pos_t,
        pos_shift=1000,
        attn_modes=layout["attn_modes"],
        split_lens=layout["split_lens"],
        shift_attn_mode=["full_noise", "full"],
        pro_type=10,
        i_sample_task=torch.tensor([2]*L),  # edit task
        i_sample_modality=modality_t,
    )
    return shifted  # (3, 1, L) torch tensor


def build_mask_pt(layout: dict, num_heads: int):
    """Returns (BlockMask, dense_LL_bool)."""
    attn_modes_ = ["full" if m in ("full_noise","full_noise_target") else m
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


def pt_forward_v(pt: PtLanceTI2I, layout: dict, visual_und_pt: torch.Tensor,
                 cond_flat_pt: torch.Tensor, x_t_pt: torch.Tensor,
                 t_scalar: float, latent_pos_ids_pt: torch.Tensor,
                 pos_ids_pt: torch.Tensor, attn_mask_LL: torch.Tensor,
                 *, return_embed: bool = False) -> torch.Tensor:
    """One forward; returns v_t at noise span (N_noise, 48) in f32."""
    L = layout["L"]
    ids = torch.tensor(layout["ids"], dtype=torch.long).unsqueeze(0)
    text_embed = pt.embed_tokens(ids)  # (1, L, D) bf16

    # ViT slab (if present)
    embed = text_embed.clone()
    if layout["vit_span"] is not None:
        vs, ve = layout["vit_span"]
        embed[:, vs:ve, :] = visual_und_pt.to(torch.bfloat16).unsqueeze(0)

    # Combined latent embed: ALL latents get vae2llm + time_embedder + pos_embed.
    # PT: cond tokens timestep=0; noise tokens timestep=t_scalar.
    n_cond = layout["vae_span"][1] - layout["vae_span"][0]
    n_noise = layout["noise_span"][1] - layout["noise_span"][0]
    # x_combined: (n_cond + n_noise, 48); cond from cond_flat, noise from x_t
    x_combined = torch.cat([cond_flat_pt, x_t_pt], dim=0).to(torch.bfloat16)
    timestep_per = torch.zeros(n_cond + n_noise, dtype=torch.float32)
    timestep_per[n_cond:] = t_scalar
    pos_combined = torch.cat([latent_pos_ids_pt, latent_pos_ids_pt], dim=0)  # same indices
    vae_embed = (pt.vae2llm(x_combined)
                 + pt.time_embed(timestep_per)
                 + pt.latent_pos_embed(pos_combined))

    vs, ve = layout["vae_span"]
    embed[:, vs:ve, :] = vae_embed[:n_cond].unsqueeze(0)
    ns, ne = layout["noise_span"]
    embed[:, ns:ne, :] = vae_embed[n_cond:].unsqueeze(0)

    if return_embed:
        return embed

    # packed_und_token_indexes = text + ViT positions
    # packed_gen_token_indexes = ALL VAE positions (cond + noise)
    all_idx = torch.arange(L, dtype=torch.long)
    gen_mask = torch.zeros(L, dtype=torch.bool)
    gen_mask[vs:ve] = True
    gen_mask[ns:ne] = True
    packed_gen_idx = all_idx[gen_mask]
    packed_und_idx = all_idx[~gen_mask]

    cos, sin = pt.mrope_cos_sin(pos_ids_pt)
    h = embed[0]  # (L, D)
    sample_lens = [L]
    for Lyr in pt.layers:
        h = Lyr(
            packed_sequence=h, sample_lens=sample_lens,
            attention_mask=attn_mask_LL,
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
    v = pt.llm2vae(out[ns:ne].to(torch.bfloat16))
    return v.to(torch.float32)


# ---------- MLX forward (mirrors FIXED image_edit.py) -----------------------
def mlx_forward_v(model: LanceLLM, layout: dict, visual_und: mx.array,
                  cond_flat: mx.array, x_t: mx.array,
                  t_scalar: mx.array, latent_pos_ids: mx.array) -> mx.array:
    """Mirrors `image_edit._forward_v` (post-STAGE 7 §3 fix)."""
    from lance_mlx.attn_mask import build_lance_attention_mask
    from lance_mlx.rope import VisionSpec, build_positions_for_layout
    L = layout["L"]
    ids = mx.array([layout["ids"]], dtype=mx.int32)
    text_embed = model.language_model.model.embed_tokens(ids)

    embed = text_embed
    if layout["vit_span"] is not None:
        vs, ve = layout["vit_span"]
        embed = mx.concatenate([embed[:, :vs, :], visual_und[None, :, :], embed[:, ve:, :]], axis=1)

    # Fix A: cond gets time_embedder(0), noise gets time_embedder(t)
    t_zero = mx.zeros_like(t_scalar)
    vae_cond_embed = (model.vae2llm(cond_flat)
                      + model.time_embedder(t_zero)
                      + model.latent_pos_embed(latent_pos_ids))
    vs, ve = layout["vae_span"]
    embed = mx.concatenate([embed[:, :vs, :], vae_cond_embed[None, :, :], embed[:, ve:, :]], axis=1)
    noise_embed = (model.vae2llm(x_t)
                   + model.time_embedder(t_scalar)
                   + model.latent_pos_embed(latent_pos_ids))
    ns, ne = layout["noise_span"]
    embed = mx.concatenate([embed[:, :ns, :], noise_embed[None, :, :], embed[:, ne:, :]], axis=1)
    # Fix B: gen_mask covers BOTH cond and noise slabs
    cols = mx.arange(L)
    gen_mask = (((cols >= vs) & (cols < ve))
                | ((cols >= ns) & (cols < ne)))[None, :]

    # positions via build_positions_for_layout
    T_g_m, H_g_m, W_g_m = layout["_grid_thw_merged"]
    t_lat, h_lat, w_lat = layout["_lat_shape"]
    spans = []
    if layout["vit_span"] is not None:
        vs_, ve_ = layout["vit_span"]
        n_vit_tokens = ve_ - vs_
        spans.append(VisionSpec(start=vs_ - 1, length=n_vit_tokens, t=T_g_m, h=H_g_m, w=W_g_m))
    vs_, ve_ = layout["vae_span"]
    n_cond = ve_ - vs_
    spans.append(VisionSpec(start=vs_ - 1, length=n_cond, t=t_lat, h=h_lat, w=w_lat))
    ns_, ne_ = layout["noise_span"]
    n_noise = ne_ - ns_
    spans.append(VisionSpec(start=ns_ - 1, length=n_noise, t=t_lat, h=h_lat, w=w_lat))
    pos = build_positions_for_layout(L, spans)

    # FIX C: pro_type=10 shifts (ViT base=1000; noise pos ← cond pos)
    pos_np = np.asarray(pos)
    if layout["vit_span"] is not None:
        vit_s, vit_e = layout["vit_span"]
        shift = 1000 - int(pos_np[0, 0, vit_s])
        pos_np[0, :, vit_s:vit_e] += shift
        pos_np[1, :, vit_s:vit_e] += shift
        pos_np[2, :, vit_s:vit_e] += shift
    pos_np[:, :, ns:ne] = pos_np[:, :, vs:ve]
    pos = mx.array(pos_np)

    # FIX D: attention mask with PT split_lens / attn_modes.
    if layout["vit_span"] is not None:
        vit_s, vit_e = layout["vit_span"]
        sl_pre_vit = vit_s
        sl_vit     = vit_e - vit_s
        sl_mid     = vs - vit_e
    else:
        sl_pre_vit = 0
        sl_vit     = 0
        sl_mid     = vs
    sl_vae   = ve - vs
    sl_sep   = ns - ve
    sl_noise = ne - ns
    sl_tail  = L - ne
    split_lens, attn_modes = [], []
    if sl_pre_vit > 0:
        split_lens.append(sl_pre_vit); attn_modes.append("causal")
    if sl_vit > 0:
        split_lens.append(sl_vit);     attn_modes.append("full")
    split_lens.append(sl_mid);   attn_modes.append("causal")
    split_lens.append(sl_vae);   attn_modes.append("full")
    split_lens.append(sl_sep);   attn_modes.append("causal")
    split_lens.append(sl_noise); attn_modes.append("noise")
    split_lens.append(sl_tail);  attn_modes.append("causal")
    attn_mask = build_lance_attention_mask(seq_len=L, split_lens=split_lens,
                                            attn_modes=attn_modes)

    hidden = model.language_model.model(
        input_ids=None, position_ids=pos, inputs_embeds=embed,
        mask=attn_mask, gen_mask=gen_mask,
    )
    v = model.llm2vae(hidden[0, ns:ne, :])
    return v


def mlx_forward_v_shared(model: LanceLLM, layout: dict, visual_und: mx.array,
                          cond_flat: mx.array, x_t: mx.array,
                          t_scalar: mx.array, latent_pos_ids: mx.array,
                          pos_shared: mx.array,
                          attn_mask_shared: mx.array,
                          *, return_embed: bool = False) -> mx.array:
    """MLX forward using EXTERNAL positions + attention mask (for byte-diff)."""
    L = layout["L"]
    ids = mx.array([layout["ids"]], dtype=mx.int32)
    text_embed = model.language_model.model.embed_tokens(ids)
    embed = text_embed
    if layout["vit_span"] is not None:
        vs, ve = layout["vit_span"]
        embed = mx.concatenate([embed[:, :vs, :], visual_und[None, :, :], embed[:, ve:, :]], axis=1)
    t_zero = mx.zeros_like(t_scalar)
    vae_cond_embed = (model.vae2llm(cond_flat)
                      + model.time_embedder(t_zero)
                      + model.latent_pos_embed(latent_pos_ids))
    vs, ve = layout["vae_span"]
    embed = mx.concatenate([embed[:, :vs, :], vae_cond_embed[None, :, :], embed[:, ve:, :]], axis=1)
    noise_embed = (model.vae2llm(x_t)
                   + model.time_embedder(t_scalar)
                   + model.latent_pos_embed(latent_pos_ids))
    ns, ne = layout["noise_span"]
    embed = mx.concatenate([embed[:, :ns, :], noise_embed[None, :, :], embed[:, ne:, :]], axis=1)
    cols = mx.arange(L)
    gen_mask = (((cols >= vs) & (cols < ve))
                | ((cols >= ns) & (cols < ne)))[None, :]
    if return_embed:
        return embed
    hidden = model.language_model.model(
        input_ids=None, position_ids=pos_shared, inputs_embeds=embed,
        mask=attn_mask_shared, gen_mask=gen_mask,
    )
    v = model.llm2vae(hidden[0, ns:ne, :])
    return v


# ---------- main ----------
def main():
    print("=" * 70)
    print("STAGE 7 §3 PT byte-diff — TI2I 3-forward")
    print("=" * 70)

    # ---- shared inputs ----
    IMAGE = "out/test_synthetic.png"
    print(f"[in] cond image: {IMAGE}")
    print(f"[in] instruction: Make it more vibrant and saturated.")
    print(f"[in] seed: 0")

    SIZE = 256
    h_lat = w_lat = SIZE // SPATIAL_DOWNSAMPLE
    t_lat = 1
    n_cond = t_lat * h_lat * w_lat
    n_noise = n_cond
    print(f"[in] size={SIZE}² lat=({t_lat},{h_lat},{w_lat}) n_cond={n_cond} n_noise={n_noise}")

    # ---- MLX side: build models + visual_und + cond_latent ----
    print("\n[mlx] building models ...")
    model = LanceLLM(LanceTextConfig())
    load_full_lance(model, MLX_WEIGHTS)
    model.eval()
    vit = LanceViT(); load_lance_vit(vit, "checkpoints/Lance-3B-MLX/vit.safetensors"); vit.eval()
    vae = Wan2_2_VAE(Wan22VAEConfig())
    vae.load_weights(list(mx.load("checkpoints/Wan2.2-VAE-MLX/model.safetensors").items()), strict=True)
    mx.eval(vae.parameters()); vae.eval()

    vit_patches, (T_g, H_g, W_g) = preprocess_image(IMAGE)
    grid_thw = mx.array([[T_g, H_g, W_g]], dtype=mx.int32)
    visual_und = vit(vit_patches, grid_thw)
    n_vit = int(visual_und.shape[0])
    H_g_m = H_g // SPATIAL_MERGE_SIZE
    W_g_m = W_g // SPATIAL_MERGE_SIZE
    print(f"[mlx] visual_und: {visual_und.shape}  (T={T_g}, H/m={H_g_m}, W/m={W_g_m})")

    vae_in = _vae_preprocess(IMAGE, size=SIZE)
    cond_latent = vae.encode(vae_in)
    cond_flat = cond_latent.reshape(n_cond, Z_DIM)
    print(f"[mlx] cond_latent: {cond_latent.shape} -> flat {cond_flat.shape}")

    # noise (numpy PRNG, seed=0)
    rng = np.random.default_rng(0)
    x_t_np = rng.standard_normal((n_noise, Z_DIM)).astype("float32")
    x_t = mx.array(x_t_np)
    x_t_pt = torch.from_numpy(x_t_np.copy()).to(torch.float32)
    print(f"[in] x_t: {x_t.shape}  ||x_t||={float(np.linalg.norm(x_t_np)):.2f}")

    t_scalar_val = 1.0  # step 0 of flow
    t_scalar = mx.array([t_scalar_val], dtype=mx.float32)
    print(f"[in] t (step 0) = {t_scalar_val}")

    # PT-side tensors
    visual_und_pt = torch.from_numpy(np.asarray(visual_und, dtype=np.float32))
    cond_flat_pt = torch.from_numpy(np.asarray(cond_flat, dtype=np.float32))
    latent_pos_ids_mlx = _latent_position_indices(t_lat, h_lat, w_lat)
    latent_pos_ids_pt = torch.from_numpy(np.asarray(latent_pos_ids_mlx, dtype=np.int64))

    # ---- tokenizer + layouts ----
    tok = AutoTokenizer.from_pretrained("checkpoints/Lance-3B-MLX", trust_remote_code=True)

    def build_full_layout(*, include_text, include_vit):
        lay = build_sequences(tok, n_vit, n_cond, n_noise,
                              include_text=include_text, include_vit=include_vit)
        lay["_grid_thw_merged"] = (T_g, H_g_m, W_g_m)
        lay["_lat_shape"] = (t_lat, h_lat, w_lat)
        return lay

    layouts = {
        "v_full":      build_full_layout(include_text=True,  include_vit=True),
        "v_t_uncond":  build_full_layout(include_text=False, include_vit=True),
        "v_tv_uncond": build_full_layout(include_text=False, include_vit=False),
    }
    for k, lay in layouts.items():
        print(f"[layout {k}] L={lay['L']}  vit_span={lay['vit_span']} "
              f"vae_span={lay['vae_span']}  noise_span={lay['noise_span']}")

    # ---- PT side: build models ----
    print("\n[pt] building Lance LLM (36 bf16 layers + adapters) ...")
    pt = PtLanceTI2I()
    pt.load_pt()
    pt.to_bf16()
    print("[pt] loaded.")

    # ---- run all three forwards, both sides ----
    print("\n[run] computing 3 forwards on PT + MLX ...")
    print("[diag] positions/masks built ONCE per layout, identical for both sides.")
    results = {}
    for name, lay in layouts.items():
        print(f"\n--- {name}  (L={lay['L']}) ---")

        # --- positions: build once in MLX, then apply pro_type=10, share ---
        from lance_mlx.rope import VisionSpec, build_positions_for_layout
        spans = []
        if lay["vit_span"] is not None:
            vs_, ve_ = lay["vit_span"]
            spans.append(VisionSpec(start=vs_ - 1, length=ve_ - vs_,
                                     t=T_g, h=H_g_m, w=W_g_m))
        vs_, ve_ = lay["vae_span"]
        spans.append(VisionSpec(start=vs_ - 1, length=ve_ - vs_,
                                 t=t_lat, h=h_lat, w=w_lat))
        ns_, ne_ = lay["noise_span"]
        spans.append(VisionSpec(start=ns_ - 1, length=ne_ - ns_,
                                 t=t_lat, h=h_lat, w=w_lat))
        pos_mlx = build_positions_for_layout(lay["L"], spans)
        # Apply pro_type=10 shifts: ViT T axis only to 1000, noise ← cond
        pos_np = np.asarray(pos_mlx)
        if lay["vit_span"] is not None:
            vs_, ve_ = lay["vit_span"]
            shift = 1000 - int(pos_np[0, 0, vs_])
            pos_np[0, :, vs_:ve_] += shift   # T axis only per PT common.py:62
        vs_, ve_ = lay["vae_span"]
        ns_, ne_ = lay["noise_span"]
        pos_np[:, :, ns_:ne_] = pos_np[:, :, vs_:ve_]
        pos_mlx_shifted = mx.array(pos_np)
        pos_pt = torch.from_numpy(pos_np.copy()).contiguous()
        # Sanity: print first/last vit & noise position
        if lay["vit_span"] is not None:
            print(f"  pos[vit start]={pos_np[:, 0, lay['vit_span'][0]].tolist()} "
                  f"pos[vae start]={pos_np[:, 0, lay['vae_span'][0]].tolist()} "
                  f"pos[noise start]={pos_np[:, 0, lay['noise_span'][0]].tolist()}")

        # --- masks: build PT mask (bool), reuse on both sides ---
        _, dense_bool = build_mask_pt(lay, num_heads=pt.cfg.num_attention_heads)
        # PT-side: pass *bool* tensor (so flex_attention monkey-patch's
        # `dense.to(torch.bool)` is an identity, NOT a bf16→bool reinterpret
        # that would invert 0↔-inf into False↔True).
        attn_mask_LL_pt  = dense_bool
        attn_mask_LL_mx = mx.array(np.where(dense_bool.numpy(), 0.0, -np.inf).astype(np.float32))

        # --- Diag: embed cos (no layer forward) ---
        embed_mlx = mlx_forward_v_shared(model, lay, visual_und, cond_flat, x_t,
                                          t_scalar, latent_pos_ids_mlx,
                                          pos_shared=pos_mlx_shifted,
                                          attn_mask_shared=attn_mask_LL_mx,
                                          return_embed=True)
        mx.eval(embed_mlx)
        with torch.no_grad():
            embed_pt = pt_forward_v(pt, lay, visual_und_pt, cond_flat_pt, x_t_pt,
                                     t_scalar_val, latent_pos_ids_pt, pos_pt, attn_mask_LL_pt,
                                     return_embed=True)
        # Compute cos for whole embed, then cond slab, then noise slab
        emb_pt_full = embed_pt[0].to(torch.float32).cpu().numpy()
        emb_mx_full = np.asarray(embed_mlx[0], dtype=np.float32)
        c_embed_all = float(np.dot(emb_pt_full.flatten(), emb_mx_full.flatten()) /
                            (np.linalg.norm(emb_pt_full)*np.linalg.norm(emb_mx_full) + 1e-12))
        vs_, ve_ = lay["vae_span"]
        ns_, ne_ = lay["noise_span"]
        c_embed_vae = float(np.dot(emb_pt_full[vs_:ve_].flatten(), emb_mx_full[vs_:ve_].flatten()) /
                            (np.linalg.norm(emb_pt_full[vs_:ve_])*np.linalg.norm(emb_mx_full[vs_:ve_]) + 1e-12))
        c_embed_noise = float(np.dot(emb_pt_full[ns_:ne_].flatten(), emb_mx_full[ns_:ne_].flatten()) /
                              (np.linalg.norm(emb_pt_full[ns_:ne_])*np.linalg.norm(emb_mx_full[ns_:ne_]) + 1e-12))
        print(f"  embed cos: all={c_embed_all:.6f}  vae_slab={c_embed_vae:.6f}  noise_slab={c_embed_noise:.6f}")

        # --- MLX forward (using shared positions + mask) ---
        v_mlx = mlx_forward_v_shared(model, lay, visual_und, cond_flat, x_t,
                                      t_scalar, latent_pos_ids_mlx,
                                      pos_shared=pos_mlx_shifted,
                                      attn_mask_shared=attn_mask_LL_mx)
        mx.eval(v_mlx)

        # --- PT forward ---
        with torch.no_grad():
            v_pt = pt_forward_v(pt, lay, visual_und_pt, cond_flat_pt, x_t_pt,
                                 t_scalar_val, latent_pos_ids_pt, pos_pt, attn_mask_LL_pt)
        c = cos_pt_mlx(v_pt, v_mlx)
        print(f"  ||v_pt|| = {float(torch.norm(v_pt)):.3f}   "
              f"||v_mlx|| = {float(np.linalg.norm(np.asarray(v_mlx))):.3f}")
        print(f"  cos(PT, MLX) = {c:.6f}  {'PASS' if c >= 0.999 else 'FAIL'}")
        results[name] = c

    print("\n" + "=" * 70)
    print(f"SUMMARY  (gate: cos ≥ 0.999)")
    for k, c in results.items():
        print(f"  {k:14s} cos = {c:.6f}  {'PASS' if c >= 0.999 else 'FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
