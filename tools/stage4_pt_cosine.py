"""STAGE 4 Tier 3 + settle [STAGE 2 → STAGE 4] backlog: cosine vs PT Lance.

Builds a *minimal* PyTorch reimplementation of Lance's `Qwen2MoTDecoderLayer`
(forward_inference path, "und" + "gen" routing — no flex_attention, no
flash_attn) that we can load PT-Lance weights into and run on CPU.
Output: per-layer hidden state.  Compared to our MLX LanceLLM running
the same input and weights → cosine ≥ 0.999 at layer 0 (and layers 12, 24
for spread) settles the workorder's "PyTorch와 hidden state cosine sim
≥ 0.999 at 3 layers".

The PT side here is a *fresh, clean* reimplementation by us — it doesn't
import refs/Lance/modeling because that pulls flash_attn and a project
package layout we'd have to shim.  Instead we translate
`PackedAttentionMoT.forward_inference` (mode="und"/"gen") + the layer
wiring at refs/Lance/modeling/lance/qwen2_navit.py:575-740 into clean PT.

What this verifies, and what it does NOT
----------------------------------------
This script runs *both* the PT reference and the MLX forward in
float32 throughout, with no bf16 anywhere.  That tests **algorithmic
shape parity**: same projection/norm/attention/MLP ordering, same
qk_norm axis, same routing decisions per token.

PT Lance in production uses bf16 mixed precision in places
(refs/Lance/modeling/lance/qwen2_navit.py:418-424 upcasts q/k to fp32
for the moe_gen norm before casting back to bf16 for sdpa).  We do NOT
exercise that bf16 path here — covering it would require running both
sides in bf16 with matching upcast/downcast points, which the MLX side
doesn't currently do (we run fp32 end-to-end at STAGE 4).  bf16-fidelity
parity is logged as a follow-up; for the current workorder criterion
("PyTorch와 hidden state cosine sim ≥ 0.999 at 3 layers"), the fp32
algorithmic check is the right granularity.
"""
from __future__ import annotations

import time

import mlx.core as mx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from safetensors import safe_open

from lance_mlx.backbone import LanceLLM, LanceTextConfig
from lance_mlx.rope import build_positions_for_layout, VisionSpec


# --------------------------------------------------------------------------
# Minimal PT MoT decoder layer (forward_inference equivalent)
# --------------------------------------------------------------------------
class PtRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # Match transformers' Qwen2RMSNorm: float() upcast inside, then back.
        input_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(input_dtype)


class PtQwen2MLP(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj   = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _make_mrope_cos_sin(position_ids_3, head_dim, base,
                        mrope_section):
    """position_ids_3: (3, B, L) int.  Returns cos, sin of shape (B, L, head_dim)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    # (3, 1, dim/2, 1)
    inv_exp = inv_freq[None, None, :, None].expand(3, position_ids_3.shape[1],
                                                    inv_freq.shape[0], 1).contiguous()
    # (3, B, 1, L)
    pos_exp = position_ids_3[:, :, None, :].float()
    freqs = inv_exp @ pos_exp                # (3, B, dim/2, L)
    freqs = freqs.transpose(2, 3)            # (3, B, L, dim/2)
    # Apply mrope packing
    s_t, s_h, s_w = mrope_section
    t = freqs[0, ..., :s_t]
    h = freqs[1, ..., s_t:s_t+s_h]
    w = freqs[2, ..., s_t+s_h:s_t+s_h+s_w]
    freqs = torch.cat([t, h, w], dim=-1)     # (B, L, dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (B, L, dim)
    return emb.cos(), emb.sin()


class PtMotAttention(nn.Module):
    def __init__(self, hidden, n_heads, n_kv_heads, head_dim, eps,
                 rope_base, mrope_section):
        super().__init__()
        self.hidden = hidden
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.rope_base = rope_base
        self.mrope_section = mrope_section

        Hq = n_heads * head_dim
        Hkv = n_kv_heads * head_dim
        self.q_proj = nn.Linear(hidden, Hq,  bias=True)
        self.k_proj = nn.Linear(hidden, Hkv, bias=True)
        self.v_proj = nn.Linear(hidden, Hkv, bias=True)
        self.o_proj = nn.Linear(Hq, hidden, bias=False)
        self.q_norm = PtRMSNorm(head_dim, eps)
        self.k_norm = PtRMSNorm(head_dim, eps)

        self.q_proj_moe_gen = nn.Linear(hidden, Hq,  bias=True)
        self.k_proj_moe_gen = nn.Linear(hidden, Hkv, bias=True)
        self.v_proj_moe_gen = nn.Linear(hidden, Hkv, bias=True)
        self.o_proj_moe_gen = nn.Linear(Hq, hidden, bias=False)
        self.q_norm_moe_gen = PtRMSNorm(head_dim, eps)
        self.k_norm_moe_gen = PtRMSNorm(head_dim, eps)

    def forward(self, x, position_ids_3, gen_mask):
        """x: (B, L, D); position_ids_3: (3, B, L); gen_mask: (B, L) bool."""
        B, L, _ = x.shape

        # Both branches on full sequence (same strategy as our MLX), then where.
        def _qkv(qp, kp, vp, qn, kn):
            q = qp(x).reshape(B, L, self.n_heads, self.head_dim)
            k = kp(x).reshape(B, L, self.n_kv_heads, self.head_dim)
            v = vp(x).reshape(B, L, self.n_kv_heads, self.head_dim)
            q = qn(q); k = kn(k)
            return q, k, v

        qc, kc, vc = _qkv(self.q_proj, self.k_proj, self.v_proj,
                          self.q_norm, self.k_norm)
        qg, kg, vg = _qkv(self.q_proj_moe_gen, self.k_proj_moe_gen,
                          self.v_proj_moe_gen, self.q_norm_moe_gen, self.k_norm_moe_gen)
        m = gen_mask[:, :, None, None]
        q = torch.where(m, qg, qc)
        k = torch.where(m, kg, kc)
        v = torch.where(m, vg, vc)

        # mRoPE
        cos, sin = _make_mrope_cos_sin(position_ids_3, self.head_dim,
                                       self.rope_base, self.mrope_section)
        cos = cos[:, None, :, :]            # (B, 1, L, D)
        sin = sin[:, None, :, :]

        q = q.transpose(1, 2)               # (B, H, L, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        # GQA via expand: repeat k/v across kv head groups.
        rep = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

        attn = F.scaled_dot_product_attention(q, k, v, scale=self.scale, is_causal=True)
        attn = attn.transpose(1, 2).reshape(B, L, -1)

        out_c = self.o_proj(attn)
        out_g = self.o_proj_moe_gen(attn)
        return torch.where(gen_mask[:, :, None], out_g, out_c)


class PtMotLayer(nn.Module):
    def __init__(self, hidden, intermediate, n_heads, n_kv_heads, head_dim, eps,
                 rope_base, mrope_section):
        super().__init__()
        self.self_attn = PtMotAttention(hidden, n_heads, n_kv_heads, head_dim, eps,
                                         rope_base, mrope_section)
        self.mlp          = PtQwen2MLP(hidden, intermediate)
        self.mlp_moe_gen  = PtQwen2MLP(hidden, intermediate)
        self.input_layernorm                  = PtRMSNorm(hidden, eps)
        self.input_layernorm_moe_gen          = PtRMSNorm(hidden, eps)
        self.post_attention_layernorm         = PtRMSNorm(hidden, eps)
        self.post_attention_layernorm_moe_gen = PtRMSNorm(hidden, eps)

    def forward(self, x, position_ids_3, gen_mask):
        # input_layernorm split
        n_c = self.input_layernorm(x)
        n_g = self.input_layernorm_moe_gen(x)
        n = torch.where(gen_mask[:, :, None], n_g, n_c)
        h = x + self.self_attn(n, position_ids_3, gen_mask)
        # post_attention_layernorm + MLP split
        pa_c = self.post_attention_layernorm(h)
        pa_g = self.post_attention_layernorm_moe_gen(h)
        pa = torch.where(gen_mask[:, :, None], pa_g, pa_c)
        mc = self.mlp(pa)
        mg = self.mlp_moe_gen(pa)
        m = torch.where(gen_mask[:, :, None], mg, mc)
        return h + m


# --------------------------------------------------------------------------
# Loader: pull PT Lance layer-i weights and embed_tokens into the PT shell
# --------------------------------------------------------------------------
def load_pt_layer(layer_idx: int, pt_safetensors_path: str,
                  hidden=2048, intermediate=11008,
                  n_heads=16, n_kv_heads=2, head_dim=128, eps=1e-6,
                  rope_base=1_000_000.0, mrope_section=(16, 24, 24)):
    layer = PtMotLayer(hidden, intermediate, n_heads, n_kv_heads, head_dim, eps,
                       rope_base, mrope_section)
    prefix = f"language_model.model.layers.{layer_idx}."

    state = {}
    with safe_open(pt_safetensors_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith(prefix):
                # strip prefix
                sub = k[len(prefix):]
                state[sub] = f.get_tensor(k).to(torch.float32)
    # Map to module names — PT Lance uses identical naming to our MLX, so direct.
    missing, unexpected = layer.load_state_dict(state, strict=True)
    return layer


def main() -> None:
    PT_WEIGHTS  = "checkpoints/Lance/Lance_3B/model.safetensors"
    MLX_WEIGHTS = "checkpoints/Lance-3B-MLX/model.safetensors"

    # ---- Build synthetic input ----
    torch.manual_seed(0)
    B, L = 1, 48
    GEN_START, GEN_END = 24, 40           # 16-token GEN slab in middle
    HIDDEN = 2048

    # Random hidden state (this is the *input* to layer i — we'll inject
    # the same tensor into both PT and MLX layers, removing embed_tokens
    # from the comparison so we test the *layer itself* in isolation).
    x_torch = torch.randn(B, L, HIDDEN, dtype=torch.float32) * 0.1
    x_mlx = mx.array(x_torch.numpy(), dtype=mx.float32)

    # Position ids — pin int32 explicitly on both sides for dtype hygiene.
    pos = torch.arange(L, dtype=torch.int32)[None, None, :].expand(3, 1, L).contiguous()
    pos_mlx = mx.array(pos.numpy())

    # GEN mask
    gen_torch = torch.zeros(B, L, dtype=torch.bool)
    gen_torch[:, GEN_START:GEN_END] = True
    gen_mlx = mx.array(gen_torch.numpy(), dtype=mx.bool_)

    print(f"[setup] B={B} L={L}  GEN slab=[{GEN_START},{GEN_END})  HIDDEN={HIDDEN}")

    # ---- Build MLX layer (just instantiate the full LanceLLM and call layer 0 directly) ----
    cfg = LanceTextConfig()
    mlx_model = LanceLLM(cfg)
    from lance_mlx.backbone import load_text_backbone
    load_text_backbone(mlx_model, MLX_WEIGHTS)
    mlx_model.eval()

    # ---- Run cosine for layers [0, 12, 24] ----
    cosines = []
    for layer_idx in (0, 12, 24):
        print(f"\n--- layer {layer_idx} ---")
        t0 = time.time()
        pt_layer = load_pt_layer(layer_idx, PT_WEIGHTS, hidden=cfg.hidden_size,
                                  intermediate=cfg.intermediate_size,
                                  n_heads=cfg.num_attention_heads,
                                  n_kv_heads=cfg.num_key_value_heads,
                                  head_dim=cfg.head_dim, eps=cfg.rms_norm_eps,
                                  rope_base=cfg.rope_theta,
                                  mrope_section=tuple(cfg.rope_scaling["mrope_section"]))
        pt_layer.eval()
        print(f"  [load PT] {time.time()-t0:.1f}s")

        with torch.no_grad():
            pt_out = pt_layer(x_torch, pos, gen_torch)             # (B, L, D)

        # MLX: call only layer i directly.
        mlx_layer = mlx_model.language_model.model.layers[layer_idx]
        mlx_out = mlx_layer(x_mlx, pos_mlx, mask="causal", gen_mask=gen_mlx)
        mx.eval(mlx_out)

        # Compare numpy
        pt_np = pt_out.cpu().numpy()
        ours_np = np.asarray(mlx_out)
        max_abs = float(np.abs(pt_np - ours_np).max())
        af = pt_np.flatten().astype(np.float64)
        bf = ours_np.flatten().astype(np.float64)
        cos = float((af @ bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))
        # Relative L2 — picks up energy drift cos can mask.
        rel_l2 = float(np.linalg.norm(af - bf) / (np.linalg.norm(af) + 1e-12))
        print(f"  cos = {cos:.6f}   max|Δ| = {max_abs:.3e}   rel_L2 = {rel_l2:.3e}")
        cosines.append(cos)

    print(f"\n[result] min cos across 3 layers = {min(cosines):.6f}  "
          f"(criterion ≥ 0.999 → {'PASS' if min(cosines) >= 0.999 else 'FAIL'})")


if __name__ == "__main__":
    main()
