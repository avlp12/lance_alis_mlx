"""STAGE 7 §1 verification — original PT ViT direct import vs MLX.

Same doctrine as STAGE 6 §0:
  - import `refs/Lance/modeling/vit/qwen2_5_vl_vit.py` directly
  - load checkpoint into PT (after layout transpose for conv weights)
  - forward identical input through PT + our MLX ViT
  - cosine on output (LLM-ready visual tokens, 2048-dim)

Gate: cos ≥ 0.999 against original PT.  Image-track precedent (STAGE 6
§0): bf16 PT vs f32 MLX gives ~0.99994 cos, *not* 1.0, because we cross
the precision boundary — that's the *desirable* sign.
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

import mlx.core as mx
from lance_mlx.vit import LanceViT, load_lance_vit


# ---- PT shims (mirror STAGE 6 setup) -----------------------------------
def _install_flash_attn_stub() -> None:
    mock = types.ModuleType("flash_attn")
    mock.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)

    def _shim_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k,
                     max_seqlen_q, max_seqlen_k, causal=True, **_kw):
        # Single-sequence packing only (cu_seqlens=[0, L]).  ViT path uses
        # multiple cu_seqlens (one per image in batch) — block-diagonal mask.
        # For our test (single image) cu_seqlens=[0, L] — same as STAGE 6.
        import torch.nn.functional as F
        assert q.dim() == 3, f"expected (L, n_heads, head_dim), got {q.shape}"
        # q/k/v: (L, n_heads, head_dim) — PT ViT packed.
        L = q.shape[0]
        n_h = q.shape[1]
        n_kv = k.shape[1]
        if n_kv < n_h:
            rep = n_h // n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        # Build block-diagonal mask from cu_seqlens
        seqlens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
        # Causal is False for ViT — bidirectional.
        if len(seqlens) == 1:
            attn_mask = None
        else:
            # Block-diagonal: token i only attends to its own sample.
            mask = torch.zeros((L, L), dtype=torch.bool)
            cursor = 0
            for s in seqlens:
                mask[cursor:cursor+s, cursor:cursor+s] = True
                cursor += s
            attn_mask = torch.where(mask, 0.0, float("-inf")).to(q.dtype)
        q4 = q.transpose(0, 1).unsqueeze(0)   # (1, n_h, L, d)
        k4 = k.transpose(0, 1).unsqueeze(0)
        v4 = v.transpose(0, 1).unsqueeze(0)
        out = F.scaled_dot_product_attention(q4, k4, v4, attn_mask=attn_mask,
                                              is_causal=bool(causal))
        return out.squeeze(0).transpose(0, 1).contiguous()

    def _shim_rotary(q, k, cos, sin):
        # ViT uses rotary on q/k packed.  Standard rotary embed.
        from flash_attn.layers.rotary import apply_rotary_emb  # dummy ref
        raise NotImplementedError("apply_rotary_emb shim not wired — PT ViT will use sdpa path")

    mock.flash_attn_varlen_func = _shim_varlen
    # apply_rotary_emb sub-module
    rotary_mod = types.ModuleType("flash_attn.layers.rotary")
    rotary_mod.apply_rotary_emb = lambda *a, **kw: (_ for _ in ()).throw(
        NotImplementedError("apply_rotary_emb not wired"))
    layers_mod = types.ModuleType("flash_attn.layers")
    layers_mod.rotary = rotary_mod
    mock.layers = layers_mod
    sys.modules["flash_attn"] = mock
    sys.modules["flash_attn.layers"] = layers_mod
    sys.modules["flash_attn.layers.rotary"] = rotary_mod

    # Block transformers' flash_attn probe.
    import transformers.utils.import_utils as _imp_utils
    for fn in ("is_flash_attn_2_available", "is_flash_attn_3_available",
               "is_flash_attn_4_available"):
        setattr(_imp_utils, fn, lambda: False)
    import transformers.utils as _utils
    for fn in ("is_flash_attn_2_available", "is_flash_attn_3_available",
               "is_flash_attn_4_available"):
        setattr(_utils, fn, lambda: False)


_install_flash_attn_stub()
sys.path.insert(0, os.path.abspath("refs/Lance"))


# ---- PT ViT import (after shims) ---------------------------------------
from modeling.vit.qwen2_5_vl_vit import Qwen2_5_VisionTransformerPretrainedModel
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig


# ---- MLX → PT layout transpose (conv weight) ---------------------------
def mlx_to_pt_vit_state(mlx_w: dict) -> dict:
    """`patch_embed.proj.weight` (1280, 2, 14, 14, 3) — MLX layout
    (O, T, H, W, I) — PT wants (O, I, T, H, W)."""
    out = {}
    for k, v in mlx_w.items():
        arr = np.asarray(v)
        # Strip `vision_tower.` prefix for PT module (PT class is the
        # ViT itself, no outer wrapper).
        pt_k = k[len("vision_tower."):] if k.startswith("vision_tower.") else k
        if "patch_embed.proj.weight" in pt_k and arr.ndim == 5:
            arr = np.transpose(arr, (0, 4, 1, 2, 3))
        out[pt_k] = torch.from_numpy(arr.copy()).float()
    return out


# ---- main --------------------------------------------------------------
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64).flatten()
    bf = b.astype(np.float64).flatten()
    return float((af @ bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))


def main() -> None:
    mlx_w = mx.load("checkpoints/Lance-3B-MLX/vit.safetensors")
    print(f"[load] vit.safetensors: {len(mlx_w)} tensors")

    # Build PT ViT
    pt_cfg = Qwen2_5_VLVisionConfig(
        depth=32, hidden_size=1280, intermediate_size=3420, num_heads=16,
        in_channels=3, patch_size=14, spatial_patch_size=14,
        spatial_merge_size=2, temporal_patch_size=2, window_size=112,
        layer_norm_eps=1e-6, tokens_per_second=2,
        out_hidden_size=2048, fullatt_block_indexes=[7, 15, 23, 31],
        hidden_act="silu",
    )
    pt_vit = Qwen2_5_VisionTransformerPretrainedModel(pt_cfg)
    pt_vit.eval()
    pt_state = mlx_to_pt_vit_state(mlx_w)
    miss, unexp = pt_vit.load_state_dict(pt_state, strict=False)
    if miss:
        print(f"[load] PT missing: {miss[:5]}... ({len(miss)})")
    if unexp:
        print(f"[load] PT unexpected: {unexp[:5]}... ({len(unexp)})")

    # Build MLX ViT
    mlx_vit = LanceViT()
    load_lance_vit(mlx_vit, "checkpoints/Lance-3B-MLX/vit.safetensors")
    mlx_vit.eval()

    # Synthetic input: 1 image, 8×8 patch grid, T=1.  Same input both sides.
    # Per Qwen2.5-VL: input is flat patches (N, C·t_p·p·p) where
    # patch_dim = 3 channels × temporal_patch_size=2 × patch_size=14² = 1176.
    T, H, W = 1, 8, 8
    N = T * H * W
    PATCH_DIM = 3 * 2 * 14 * 14
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((N, PATCH_DIM)).astype(np.float32) * 0.5
    grid_thw = np.array([[T, H, W]], dtype=np.int64)
    x_pt = torch.from_numpy(x_np)
    grid_pt = torch.from_numpy(grid_thw)
    x_mlx = mx.array(x_np)
    grid_mlx = mx.array(grid_thw.astype(np.int32))

    with torch.no_grad():
        out_pt = pt_vit(hidden_states=x_pt, grid_thw=grid_pt).cpu().numpy()
    out_mlx = np.asarray(mlx_vit(x_mlx, grid_mlx))
    cos = _cosine(out_pt, out_mlx)
    max_abs = float(np.abs(out_pt - out_mlx).max())
    rel = float(np.linalg.norm(out_pt - out_mlx) / (np.linalg.norm(out_pt) + 1e-12))
    print()
    print(f"PT  out shape={out_pt.shape}  range=[{out_pt.min():+.3f},{out_pt.max():+.3f}]")
    print(f"MLX out shape={out_mlx.shape}  range=[{out_mlx.min():+.3f},{out_mlx.max():+.3f}]")
    print(f"cos = {cos:.6f}   max|Δ| = {max_abs:.3e}   rel_L2 = {rel:.3e}")
    print(f"Gate ≥ 0.999: {'PASS' if cos >= 0.999 else 'FAIL'}")


if __name__ == "__main__":
    main()
