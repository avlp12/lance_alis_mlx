"""STAGE 6 entry / STAGE 4 re-verification.

Direct-import-of-original-PT version of `tools/stage4_pt_cosine.py`.  The
STAGE 4 comparison used a clean hand-reimplementation of PT
`Qwen2MoTDecoderLayer.forward_inference` — STAGE 5 showed (via the
patchify / AvgDown3D / avg_shortcut bugs) that an algorithm
misunderstanding shared between *our* MLX and *our* PT can mask itself
as cos=1.0.

This script removes that risk: we `sys.path.insert(0, "refs/Lance")`
and `import modeling.lance.qwen2_navit` *directly* — the PT side is
the original ByteDance code, byte-for-byte.

What we work around to make it importable on Mac:
  - `flash_attn` is CUDA-only; we register a stub `flash_attn` module
    in `sys.modules` whose `flash_attn_varlen_func` raises if invoked.
    PT Lance only enters the flash path under `forward_train` /
    flex_attention; our `forward_inference` route avoids it.
  - `transformers 5.9` probes `flash_attn` via
    `is_flash_attn_{2,3,4}_available`.  We monkey-patch those to
    return False *before* any model imports run.
  - `refs/Lance/modeling/lance/__init__.py` triggers a chain into
    `lance.py` → `imageio`, `common.val.utils`, etc.  We pre-register a
    stub `modeling.lance` package (with `__path__` pointing at the
    refs dir) so Python doesn't run that __init__ and we can still
    import `modeling.lance.qwen2_navit` directly.

Verification target (workorder + STAGE 4 retro):
  layer 0 / 12 / 24 / 35  cos ≥ 0.999  vs ORIGINAL PT (not our reimpl)
"""
from __future__ import annotations

import os
import sys
import time
import types
import importlib
import importlib.machinery


# -----------------------------------------------------------------------------
# Step 1 — stub flash_attn + neutralize transformers' flash_attn probe.
# Must run BEFORE any model import touches transformers.
# -----------------------------------------------------------------------------
def _install_flash_attn_stub() -> None:
    """Provide a CPU-friendly drop-in for `flash_attn_varlen_func` using
    stock `torch.nn.functional.scaled_dot_product_attention`.  Supports the
    single-sequence case we exercise here (`cu_seqlens_q = [0, L]`,
    `cu_seqlens_k = [0, L]`).  Multi-sample batched packing would need a
    block-diagonal mask; raise for that.
    """
    import torch
    import torch.nn.functional as F

    def _shim(q, k, v, cu_seqlens_q, cu_seqlens_k,
              max_seqlen_q, max_seqlen_k, causal=True, **_kw):
        # q: (total_q, n_heads, head_dim)
        # k/v: (total_kv, n_kv_heads, head_dim)
        # For our single-sequence test, cu_seqlens_q == [0, L], cu_seqlens_k == [0, L]
        if cu_seqlens_q.numel() != 2 or cu_seqlens_k.numel() != 2:
            raise NotImplementedError(
                "flash_attn shim only handles single-sequence packing "
                f"(got cu_q={cu_seqlens_q.tolist()}, cu_k={cu_seqlens_k.tolist()})"
            )
        # GQA: repeat k/v across heads if n_kv < n_heads
        n_heads = q.shape[1]
        n_kv = k.shape[1]
        if n_kv < n_heads:
            rep = n_heads // n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        # SDPA wants (B, n_heads, L, head_dim).  We have (L, n_heads, head_dim).
        q4 = q.transpose(0, 1).unsqueeze(0)
        k4 = k.transpose(0, 1).unsqueeze(0)
        v4 = v.transpose(0, 1).unsqueeze(0)
        # PT Lance default scale = 1/sqrt(head_dim); SDPA uses same default.
        out = F.scaled_dot_product_attention(q4, k4, v4, is_causal=bool(causal))
        # Back to (L, n_heads, head_dim)
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
    """Pre-register `modeling.lance` package without triggering its __init__."""
    lance_dir = os.path.abspath("refs/Lance/modeling/lance")
    pkg = types.ModuleType("modeling.lance")
    pkg.__path__ = [lance_dir]
    sys.modules["modeling.lance"] = pkg


_install_flash_attn_stub()
sys.path.insert(0, os.path.abspath("refs/Lance"))
_install_modeling_lance_stub()


# -----------------------------------------------------------------------------
# Step 2 — import ORIGINAL PT modules
# -----------------------------------------------------------------------------
qwen2_navit = importlib.import_module("modeling.lance.qwen2_navit")
Qwen2MoTDecoderLayer = qwen2_navit.Qwen2MoTDecoderLayer
PackedAttentionMoT   = qwen2_navit.PackedAttentionMoT
from modeling.qwen2.configuration_qwen2 import Qwen2Config

import numpy as np
import torch
from safetensors import safe_open

import mlx.core as mx

from lance_mlx.backbone import LanceLLM, LanceTextConfig, load_text_backbone


PT_WEIGHTS  = "checkpoints/Lance/Lance_3B/model.safetensors"
MLX_WEIGHTS = "checkpoints/Lance-3B-MLX/model.safetensors"


# -----------------------------------------------------------------------------
# Build a PT Qwen2MoTDecoderLayer with Lance config and load layer i weights.
# -----------------------------------------------------------------------------
def make_pt_layer(layer_idx: int, cfg: LanceTextConfig) -> Qwen2MoTDecoderLayer:
    # Construct a Qwen2Config matching Lance.  We need the moe + qk_norm flags
    # the real Lance training used.
    q_cfg = Qwen2Config(
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        vocab_size=cfg.vocab_size,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta,
        max_position_embeddings=cfg.max_position_embeddings,
        rope_scaling=cfg.rope_scaling,
        tie_word_embeddings=cfg.tie_word_embeddings,
    )
    # Lance modifications: qk_norm + MoE-gen siblings.
    q_cfg.qk_norm = True
    q_cfg.qk_norm_und = True
    q_cfg.qk_norm_gen = True
    q_cfg.layer_module = "Qwen2MoTDecoderLayer"
    q_cfg.freeze_und = False

    layer = Qwen2MoTDecoderLayer(q_cfg, layer_idx=layer_idx)
    layer.eval()

    # Load PT Lance_3B weights for this layer.  PT's mode="gen" path runs
    # the whole layer in bf16 except for an explicit q/k upcast inside
    # qk_norm (qwen2_navit.py:418, 422).  We cast the entire layer to bf16
    # so the input_layernorm / mlp / etc dtype-match; the qk_norm `.to(f32)`
    # callsites still upcast internally, so numerical behaviour matches
    # PT's documented mixed-precision path.
    prefix = f"language_model.model.layers.{layer_idx}."
    state = {}
    with safe_open(PT_WEIGHTS, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith(prefix):
                state[k[len(prefix):]] = f.get_tensor(k).to(torch.bfloat16)
    layer = layer.to(torch.bfloat16)
    missing, unexpected = layer.load_state_dict(state, strict=False)
    if unexpected:
        # We strip prefix; if PT lance layer has extra keys we don't know about,
        # that's a config flag we missed.
        print(f"[load] WARN unexpected keys for layer {layer_idx}: {unexpected[:3]}...")
    if missing:
        print(f"[load] WARN missing keys for layer {layer_idx}: {missing[:5]}...")
    return layer


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64).flatten()
    bf = b.astype(np.float64).flatten()
    return float((af @ bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))


# -----------------------------------------------------------------------------
# Main comparison harness
# -----------------------------------------------------------------------------
def main() -> None:
    cfg = LanceTextConfig()
    print(f"[setup] LanceTextConfig: {cfg.num_hidden_layers} layers, "
          f"hidden={cfg.hidden_size}, heads={cfg.num_attention_heads}, "
          f"kv={cfg.num_key_value_heads}, head_dim={cfg.head_dim}")

    # ---- input (matches STAGE 4 Tier 3 synthetic) ----
    torch.manual_seed(0)
    B, L = 1, 48
    GEN_START, GEN_END = 24, 40
    x_torch = torch.randn(B, L, cfg.hidden_size, dtype=torch.float32) * 0.1
    x_mlx = mx.array(x_torch.numpy(), dtype=mx.float32)

    # ---- position ids (text-only: arange × 3) ----
    # PT Lance expects (3, B, L) for mrope.  Build via Qwen2_5_VLRotaryEmbedding.
    pos = torch.arange(L, dtype=torch.int64)[None, None, :].expand(3, B, L).contiguous()
    pos_mlx = mx.array(pos.numpy().astype(np.int32))

    # Hand-build (cos, sin) matching Qwen2.5-VL's mrope formula.  Avoids
    # the transformers `ROPE_INIT_FUNCTIONS["default"]` lookup which 5.9
    # has refactored away.  Same formula as our MLX `Qwen2RotaryEmbedding`
    # (verified at STAGE 2 cos=1.0 vs mlx-vlm) — so cos/sin produced here
    # is the *same* PT-compatible tensor pair, just constructed inline.
    head_dim = cfg.head_dim
    base = cfg.rope_theta
    mrope_section = cfg.rope_scaling["mrope_section"]
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    inv_freq_exp = inv_freq[None, None, :, None].expand(3, L, -1, 1)           # (3, L, hd/2, 1)
    pos_exp = pos.float()[:, :, None, :]                                         # (3, B, 1, L)
    freqs = (inv_freq_exp @ pos_exp).transpose(2, 3)                             # (3, B, L, hd/2)
    s_t, s_h, s_w = mrope_section
    t_part = freqs[0, ..., :s_t]
    h_part = freqs[1, ..., s_t:s_t+s_h]
    w_part = freqs[2, ..., s_t+s_h:s_t+s_h+s_w]
    freqs_mrope = torch.cat([t_part, h_part, w_part], dim=-1)                    # (B, L, hd/2)
    emb = torch.cat([freqs_mrope, freqs_mrope], dim=-1)                          # (B, L, hd)
    cos_pt = emb.cos().to(x_torch.dtype)
    sin_pt = emb.sin().to(x_torch.dtype)

    # ---- gen / und routing — PT packed indices ----
    all_idx = torch.arange(L, dtype=torch.long)
    text_idx = torch.cat([all_idx[:GEN_START], all_idx[GEN_END:]])
    vae_idx  = all_idx[GEN_START:GEN_END]
    print(f"[setup] L={L}  text_idx={text_idx.numel()}  vae_idx={vae_idx.numel()} "
          f"(GEN [{GEN_START},{GEN_END}))")

    # MLX gen_mask
    gen_mlx = mx.zeros((B, L), dtype=mx.bool_)
    cols = mx.arange(L)
    gen_mlx = (cols >= GEN_START) & (cols < GEN_END)
    gen_mlx = gen_mlx[None, :]

    # ---- build MLX model, load weights ----
    mlx_model = LanceLLM(cfg)
    t0 = time.time()
    load_text_backbone(mlx_model, MLX_WEIGHTS)
    mlx_model.eval()
    print(f"[load] MLX strict-load OK in {time.time()-t0:.1f}s")

    # ---- per-layer cosine ----
    print()
    print(f"{'layer':>6s}  {'cos':>10s}  {'max|Δ|':>10s}  {'rel_L2':>10s}")
    print("-" * 50)
    target_layers = [0, 12, 24, 35]
    cosines = []
    for li in target_layers:
        pt_layer = make_pt_layer(li, cfg)

        # PT forward — flatten B and L, run forward_inference with mode="gen".
        # packed_query_sequence shape (L, D); packed_query_indexes is the
        # in-sample index (just arange L for single sample).
        with torch.no_grad():
            packed_seq = x_torch[0].to(torch.bfloat16)  # (L, D), match PT's internal cast
            # PT forward_inference signature:
            packed_query_indexes = all_idx.clone()
            packed_query_position_embeddings = (
                cos_pt[0].to(torch.bfloat16),
                sin_pt[0].to(torch.bfloat16),
            )
            pt_out, _ = pt_layer.forward_inference(
                packed_query_sequence=packed_seq,
                query_lens=torch.tensor([L]),
                packed_query_position_embeddings=packed_query_position_embeddings,
                packed_query_indexes=packed_query_indexes,
                past_key_values=None,
                key_values_lens=None,
                packed_key_value_indexes=packed_query_indexes,
                update_past_key_values=False,
                is_causal=True,
                mode="gen",
                packed_vae_token_indexes=vae_idx,
                packed_text_indexes=text_idx,
            )
        pt_np = pt_out.to(torch.float32).cpu().numpy()  # (L, D), upcast for cosine vs f32 MLX

        # MLX forward — through full LanceLLM.layers[li]
        mlx_layer = mlx_model.language_model.model.layers[li]
        mlx_out = mlx_layer(x_mlx, pos_mlx, mask="causal", gen_mask=gen_mlx)
        mx.eval(mlx_out)
        ours_np = np.asarray(mlx_out)[0]                # squeeze batch

        max_abs = float(np.abs(pt_np - ours_np).max())
        cos = _cosine(pt_np, ours_np)
        rel = float(np.linalg.norm(pt_np - ours_np) / (np.linalg.norm(pt_np) + 1e-12))
        cosines.append(cos)
        print(f"{li:>6d}  {cos:>10.6f}  {max_abs:>10.3e}  {rel:>10.3e}")

    print()
    print(f"min cos = {min(cosines):.6f}  "
          f"(criterion ≥ 0.999 → {'PASS' if min(cosines) >= 0.999 else 'FAIL'})")


if __name__ == "__main__":
    main()
