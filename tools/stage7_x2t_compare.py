"""STAGE 7 §2 verification — PT direct-import X→T first-token logit cosine.

Doctrine: greedy AR — if first-token logits match, all subsequent tokens
match (deterministic argmax).  Compare logits at the assistant-prefix
position between PT (refs/Lance qwen2_navit Qwen2MoTDecoderLayer × 36 +
PT ViT + adapters) and our MLX (LanceLLM + LanceViT).

Gate: cos ≥ 0.999 at next-token logits over vocab=151936.
"""
from __future__ import annotations

import os
import sys
import types
import importlib
import importlib.machinery

import numpy as np
import torch
from safetensors import safe_open

# ---- PT shims (reuse STAGE 6 pattern) ----
def _install_shims():
    mock = types.ModuleType("flash_attn")
    mock.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)

    def _shim_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k,
                     max_seqlen_q, max_seqlen_k, causal=True, **_kw):
        import torch.nn.functional as F
        if q.dim() == 4:
            q4, k4, v4 = q, k, v
        elif q.dim() == 3:
            n_h = q.shape[1]; n_kv = k.shape[1]
            if n_kv < n_h:
                rep = n_h // n_kv
                k = k.repeat_interleave(rep, dim=1)
                v = v.repeat_interleave(rep, dim=1)
            q4 = q.transpose(0, 1).unsqueeze(0)
            k4 = k.transpose(0, 1).unsqueeze(0)
            v4 = v.transpose(0, 1).unsqueeze(0)
        else:
            raise ValueError(f"unexpected q shape {q.shape}")
        seqlens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
        if len(seqlens) == 1:
            attn_mask = None
        else:
            L = q4.shape[-2]
            mask = torch.zeros((L, L), dtype=torch.bool)
            cursor = 0
            for s in seqlens:
                mask[cursor:cursor+s, cursor:cursor+s] = True
                cursor += s
            attn_mask = torch.where(mask, 0.0, float("-inf")).to(q4.dtype)
        out = F.scaled_dot_product_attention(q4, k4, v4, attn_mask=attn_mask,
                                              is_causal=bool(causal))
        if q.dim() == 3:
            out = out.squeeze(0).transpose(0, 1).contiguous()
        return out

    mock.flash_attn_varlen_func = _shim_varlen
    rotary_mod = types.ModuleType("flash_attn.layers.rotary")
    rotary_mod.apply_rotary_emb = lambda *a, **kw: (_ for _ in ()).throw(
        NotImplementedError("apply_rotary_emb not wired"))
    layers_mod = types.ModuleType("flash_attn.layers")
    layers_mod.rotary = rotary_mod
    mock.layers = layers_mod
    sys.modules["flash_attn"] = mock
    sys.modules["flash_attn.layers"] = layers_mod
    sys.modules["flash_attn.layers.rotary"] = rotary_mod

    import transformers.utils.import_utils as _iu
    for fn in ("is_flash_attn_2_available", "is_flash_attn_3_available",
               "is_flash_attn_4_available"):
        setattr(_iu, fn, lambda: False)
    import transformers.utils as _utils
    for fn in ("is_flash_attn_2_available", "is_flash_attn_3_available",
               "is_flash_attn_4_available"):
        setattr(_utils, fn, lambda: False)

    # Stub modeling.lance pkg
    lance_dir = os.path.abspath("refs/Lance/modeling/lance")
    pkg = types.ModuleType("modeling.lance")
    pkg.__path__ = [lance_dir]
    sys.modules["modeling.lance"] = pkg

    # Patch flex_attention for the SDPA shim path (STAGE 6 pattern)
    import torch.nn.attention.flex_attention as _fa
    from torch.nn.attention.flex_attention import BlockMask

    def _dense_from_block_mask(bm, L):
        q = torch.arange(L)[:, None]; k = torch.arange(L)[None, :]
        b = torch.tensor(0); h = torch.tensor(0)
        return bm.mask_mod(b, h, q, k)

    def patched_flex_attention(query, key, value, block_mask, enable_gqa=True,
                                return_lse=False, kernel_options=None, **kw):
        import torch.nn.functional as F
        assert query.dim() == 4
        n_h = query.shape[1]; n_kv = key.shape[1]
        q4, k4, v4 = query, key, value
        if enable_gqa and n_kv < n_h:
            rep = n_h // n_kv
            k4 = k4.repeat_interleave(rep, dim=1)
            v4 = v4.repeat_interleave(rep, dim=1)
        L_q = q4.shape[2]
        if isinstance(block_mask, BlockMask):
            dense = _dense_from_block_mask(block_mask, L_q)
        else:
            dense = block_mask
        if dense.dtype != torch.bool:
            dense = dense.to(torch.bool)
        add = torch.zeros(dense.shape, dtype=q4.dtype, device=q4.device)
        add.masked_fill_(~dense, float("-inf"))
        return F.scaled_dot_product_attention(q4, k4, v4, attn_mask=add[None, None])

    _fa.flex_attention = patched_flex_attention


_install_shims()
sys.path.insert(0, os.path.abspath("refs/Lance"))

# Bind patched flex_attention into qwen2_navit
qwen2_navit = importlib.import_module("modeling.lance.qwen2_navit")
import torch.nn.attention.flex_attention as _fa_mod
qwen2_navit.flex_attention = _fa_mod.flex_attention

Qwen2MoTDecoderLayer = qwen2_navit.Qwen2MoTDecoderLayer
from modeling.qwen2.configuration_qwen2 import Qwen2Config
from modeling.vit.qwen2_5_vl_vit import Qwen2_5_VisionTransformerPretrainedModel
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
from data.data_utils import create_sparse_mask
from torch.nn.attention.flex_attention import create_block_mask

import mlx.core as mx
from transformers import AutoTokenizer

from lance_mlx.backbone import LanceLLM, LanceTextConfig, load_full_lance
from lance_mlx.vit import LanceViT, load_lance_vit
from lance_mlx.pipelines.x2t import (
    preprocess_image, IM_START_ID, IM_END_ID, VIS_START_ID, VIS_END_ID, IMG_TOKEN_ID,
)
from lance_mlx.rope import VisionSpec, build_positions_for_layout
from lance_mlx.attn_mask import build_lance_attention_mask
from tools.stage5_pt_compare import mlx_to_pt_state as _mlx_to_pt_vae   # unused but reuses path
from tools.stage7_vit_compare import mlx_to_pt_vit_state                # reuse


PT_WEIGHTS  = "checkpoints/Lance/Lance_3B/model.safetensors"
MLX_WEIGHTS = "checkpoints/Lance-3B-MLX/model.safetensors"
VIT_WEIGHTS = "checkpoints/Lance-3B-MLX/vit.safetensors"


def _cosine(a, b):
    af = np.asarray(a).astype(np.float64).flatten()
    bf = np.asarray(b).astype(np.float64).flatten()
    return float((af @ bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))


# ---- Build PT side: 36-layer LLM + adapters + ViT, all bf16 ----
def build_pt_x2t_model():
    cfg = LanceTextConfig()
    q_cfg = Qwen2Config(
        hidden_size=cfg.hidden_size, intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers, num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads, vocab_size=cfg.vocab_size,
        rms_norm_eps=cfg.rms_norm_eps, rope_theta=cfg.rope_theta,
        max_position_embeddings=cfg.max_position_embeddings,
        rope_scaling=cfg.rope_scaling, tie_word_embeddings=cfg.tie_word_embeddings,
    )
    q_cfg.qk_norm = True
    q_cfg.qk_norm_und = True
    q_cfg.qk_norm_gen = True
    q_cfg.layer_module = "Qwen2MoTDecoderLayer"
    q_cfg.freeze_und = False

    embed = torch.nn.Embedding(cfg.vocab_size, cfg.hidden_size)
    layers = torch.nn.ModuleList(
        [Qwen2MoTDecoderLayer(q_cfg, layer_idx=i) for i in range(cfg.num_hidden_layers)]
    )
    from modeling.qwen2.modeling_qwen2 import Qwen2RMSNorm
    final_norm = Qwen2RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    lm_head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    # Load weights
    with safe_open(PT_WEIGHTS, framework="pt", device="cpu") as f:
        d = {k: f.get_tensor(k).to(torch.bfloat16) for k in f.keys()}
    embed.weight.data = d["language_model.model.embed_tokens.weight"].clone()
    lm_head.weight.data = d["language_model.lm_head.weight"].clone()
    for i, L in enumerate(layers):
        prefix = f"language_model.model.layers.{i}."
        state = {k[len(prefix):]: v for k, v in d.items() if k.startswith(prefix)}
        L.load_state_dict(state, strict=True)
    final_norm.weight.data = d["language_model.model.norm.weight"].clone()

    # Cast modules to bf16
    embed = embed.to(torch.bfloat16); embed.eval()
    layers = layers.to(torch.bfloat16); layers.eval()
    final_norm = final_norm.to(torch.bfloat16); final_norm.eval()
    lm_head = lm_head.to(torch.bfloat16); lm_head.eval()

    # PT ViT
    vit_cfg = Qwen2_5_VLVisionConfig(
        depth=32, hidden_size=1280, intermediate_size=3420, num_heads=16,
        in_channels=3, patch_size=14, spatial_patch_size=14,
        spatial_merge_size=2, temporal_patch_size=2, window_size=112,
        layer_norm_eps=1e-6, tokens_per_second=2,
        out_hidden_size=2048, fullatt_block_indexes=[7, 15, 23, 31],
        hidden_act="silu",
    )
    pt_vit = Qwen2_5_VisionTransformerPretrainedModel(vit_cfg)
    vit_w = mx.load(VIT_WEIGHTS)
    pt_vit_state = mlx_to_pt_vit_state(vit_w)
    pt_vit.load_state_dict(pt_vit_state, strict=False)
    pt_vit.eval()

    return embed, layers, final_norm, lm_head, pt_vit


def mrope_cos_sin(position_ids_3, head_dim, base, mrope_section):
    L = position_ids_3.shape[-1]
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    inv_exp = inv[None, None, :, None].expand(3, L, -1, 1)
    pos_exp = position_ids_3.float()[:, :, None, :]
    freqs = (inv_exp @ pos_exp).transpose(2, 3)
    s_t, s_h, s_w = mrope_section
    f = torch.cat([freqs[0, 0, :, :s_t],
                   freqs[1, 0, :, s_t:s_t+s_h],
                   freqs[2, 0, :, s_t+s_h:s_t+s_h+s_w]], dim=-1)
    emb = torch.cat([f, f], dim=-1)
    return emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)


def main() -> None:
    QUESTION = "Describe this image briefly."
    IMAGE = "out/test_synthetic.png"

    tok = AutoTokenizer.from_pretrained("checkpoints/Lance-3B-MLX", trust_remote_code=True)

    # ---- Image preprocess (shared by both sides) ----
    patches_mlx, (T_g, H_g, W_g) = preprocess_image(IMAGE)
    patches_np = np.asarray(patches_mlx)
    patches_pt = torch.from_numpy(patches_np)
    grid_thw_np = np.array([[T_g, H_g, W_g]], dtype=np.int64)
    grid_mlx = mx.array(grid_thw_np.astype(np.int32))
    grid_pt = torch.from_numpy(grid_thw_np)

    # ---- ViT forward (both sides) ----
    print("[build] LanceLLM + LanceViT (MLX) ...")
    mlx_model = LanceLLM(LanceTextConfig())
    load_full_lance(mlx_model, MLX_WEIGHTS)
    mlx_model.eval()
    mlx_vit = LanceViT()
    load_lance_vit(mlx_vit, VIT_WEIGHTS)
    mlx_vit.eval()
    visual_mlx = mlx_vit(patches_mlx, grid_mlx)
    n_vis = int(visual_mlx.shape[0])
    h_lat = H_g // 2
    w_lat = W_g // 2

    print("[build] PT (full LLM + ViT) ...")
    pt_embed, pt_layers, pt_final_norm, pt_lm_head, pt_vit = build_pt_x2t_model()
    # PT ViT runs in fp32 (we tested it that way at §1).  Keep fp32 for ViT,
    # then cast to bf16 when feeding into the LLM (PT LLM is bf16).
    with torch.no_grad():
        visual_pt = pt_vit(hidden_states=patches_pt, grid_thw=grid_pt)
    visual_pt_bf16 = visual_pt.to(torch.bfloat16)

    # Sanity: ViT outputs already byte-checked at §1; quick reconfirm.
    cos_vit = _cosine(visual_pt.cpu().numpy(), np.asarray(visual_mlx))
    print(f"[ViT] cos(PT, MLX) = {cos_vit:.6f}")

    # ---- Build chat sequence (identical both sides) ----
    sys_ids = tok("You are a helpful assistant.", add_special_tokens=False)["input_ids"]
    q_ids   = tok(QUESTION, add_special_tokens=False)["input_ids"]
    newline = tok("\n", add_special_tokens=False)["input_ids"]
    sys_lbl = tok("system", add_special_tokens=False)["input_ids"]
    usr_lbl = tok("user",   add_special_tokens=False)["input_ids"]
    asst_lbl = tok("assistant", add_special_tokens=False)["input_ids"]

    seq = (
        [IM_START_ID] + sys_lbl + newline + sys_ids + [IM_END_ID] + newline
        + [IM_START_ID] + usr_lbl + newline
        + [VIS_START_ID] + [IMG_TOKEN_ID] * n_vis + [VIS_END_ID]
        + q_ids + [IM_END_ID] + newline
        + [IM_START_ID] + asst_lbl + newline
    )
    L = len(seq)
    vis_start = seq.index(VIS_START_ID) + 1
    vis_end = vis_start + n_vis
    print(f"[seq] L={L}  vis=[{vis_start},{vis_end})  N_vis={n_vis}")

    # ---- positions + mask (shared) ----
    pos_mlx = build_positions_for_layout(
        L, [VisionSpec(start=vis_start - 1, length=n_vis, t=T_g, h=h_lat, w=w_lat)]
    )
    pos_pt = torch.from_numpy(np.asarray(pos_mlx).astype(np.int64))

    # X→T uses pure causal (no GEN slab); mask via single causal split
    block_mask_pt = create_block_mask(
        create_sparse_mask([L], [L], ["causal"], torch.device("cpu")),
        B=1, H=16, Q_LEN=L, KV_LEN=L, device=torch.device("cpu"),
        BLOCK_SIZE=128, _compile=False,
    )
    attn_mask_mlx = build_lance_attention_mask(seq_len=L, split_lens=[L], attn_modes=["causal"])

    # ---- MLX forward ----
    ids_mlx = mx.array([seq], dtype=mx.int32)
    text_embed_mlx = mlx_model.language_model.model.embed_tokens(ids_mlx)
    embed_mlx = mx.concatenate([
        text_embed_mlx[:, :vis_start, :],
        visual_mlx[None, :, :],
        text_embed_mlx[:, vis_end:, :],
    ], axis=1)
    hidden_mlx = mlx_model.language_model.model(
        input_ids=None, position_ids=pos_mlx, inputs_embeds=embed_mlx,
        mask=attn_mask_mlx, gen_mask=None,
    )
    logits_mlx = mlx_model.language_model.lm_head(hidden_mlx[0, -1:, :])    # (1, V)

    # ---- PT forward ----
    ids_pt = torch.tensor([seq], dtype=torch.long)
    text_embed_pt = pt_embed(ids_pt).to(torch.bfloat16)
    embed_pt = torch.cat([
        text_embed_pt[:, :vis_start, :],
        visual_pt_bf16.unsqueeze(0),
        text_embed_pt[:, vis_end:, :],
    ], dim=1)
    packed_seq = embed_pt[0]                               # (L, D) for packed forward
    sample_lens = [L]
    cos_pt, sin_pt = mrope_cos_sin(pos_pt, head_dim=128,
                                    base=1_000_000.0, mrope_section=[16, 24, 24])
    all_idx = torch.arange(L, dtype=torch.long)
    h = packed_seq
    with torch.no_grad():
        for L_layer in pt_layers:
            h = L_layer(
                packed_sequence=h, sample_lens=sample_lens,
                attention_mask=block_mask_pt,
                packed_position_embeddings=(cos_pt, sin_pt),
                packed_und_token_indexes=all_idx,
                packed_gen_token_indexes=torch.empty(0, dtype=torch.long),
                mode_forward="validation",
            )
        h = pt_final_norm(h)
        logits_pt = pt_lm_head(h[-1:])                                      # (1, V)

    # ---- compare logits ----
    pt_arr = logits_pt.to(torch.float32).cpu().numpy()
    mlx_arr = np.asarray(logits_mlx)
    cos = _cosine(pt_arr, mlx_arr)
    max_abs = float(np.abs(pt_arr - mlx_arr).max())
    pt_top = int(np.argmax(pt_arr[0]))
    mlx_top = int(np.argmax(mlx_arr[0]))
    print()
    print(f"first-token logits: cos = {cos:.6f}   max|Δ| = {max_abs:.3e}")
    print(f"PT argmax  = {pt_top:>6d}  ('{tok.decode([pt_top])}')")
    print(f"MLX argmax = {mlx_top:>6d}  ('{tok.decode([mlx_top])}')")
    print(f"Gate ≥ 0.999: {'PASS' if cos >= 0.999 else 'FAIL'}")
    print(f"Top-1 agreement: {'YES' if pt_top == mlx_top else 'NO'}")


if __name__ == "__main__":
    main()
