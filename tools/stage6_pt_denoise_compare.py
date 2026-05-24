"""STAGE 6 PT side-by-side denoise compare.

Per user directive: mask + position come from *original* PT functions,
denoising loop is ours (run in BOTH PT and MLX).  The variable being
tested is the denoising loop + backbone end-to-end; mask/position are
ground truth from refs/Lance.

  - mask     : `refs/Lance/data/data_utils.create_sparse_mask`  (predicate)
               evaluated on (L, L) grid → bool tensor → MLX additive mask.
  - position : `build_positions_for_layout` (our STAGE 3 helper, verified
               byte-identical to mlx-vlm on text+single-image cases).
               No `shift_position_ids` because for T2I all modalities
               are {0=text, 1=noise} → `pro_type=10` is a *no-op*
               (it only fires on modality ∈ {2, 3, 4} which are
               TI2I/refedit-only).

Run 30 steps, same seed/prompt on PT side and MLX side, step-by-step
cosine of x_t.  Gate: step cos ≥ 0.995.
"""
from __future__ import annotations

import os
import sys
import time
import types
import importlib
import importlib.machinery


# ----- PT environment shim (mirror stage6_pt_backbone_compare.py) -----------
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


# Monkey-patch `flex_attention` → SDPA with dense additive mask.
# PT `PackedAttentionMoT.forward_train` (qwen2_navit.py:357) calls flex_attention
# with `block_mask=<BlockMask>` and `enable_gqa=True`.  flex_attention pulls in
# torch.compile / inductor which on CPU/Mac balks; SDPA does the same math
# (different backend, identical algorithm) given the *same* (L, L) mask.
def _install_flex_attention_sdpa_patch() -> None:
    import torch
    import torch.nn.functional as F
    import torch.nn.attention.flex_attention as _fa
    from torch.nn.attention.flex_attention import BlockMask

    def _dense_from_block_mask(bm: BlockMask, L: int) -> torch.Tensor:
        # `bm.to_dense()` returns *block-level* (B, H, n_q_blocks, n_kv_blocks)
        # — NOT token-level.  We need (L, L) token mask.  Evaluate the
        # `mask_mod` predicate directly on the full grid.
        q = torch.arange(L)[:, None]; k = torch.arange(L)[None, :]
        b = torch.tensor(0); h = torch.tensor(0)
        return bm.mask_mod(b, h, q, k)

    def patched_flex_attention(query, key, value, block_mask, enable_gqa=True,
                                return_lse=False, kernel_options=None, **kw):
        # PT call site (qwen2_navit.py:357-363):
        #   query/key/value shape (1, n_heads, L_padded, head_dim) — already 4D.
        assert query.dim() == 4, f"expected 4D q, got {query.shape}"
        n_h = query.shape[1]
        n_kv = key.shape[1]
        q4, k4, v4 = query, key, value
        if enable_gqa and n_kv < n_h:
            rep = n_h // n_kv
            k4 = k4.repeat_interleave(rep, dim=1)
            v4 = v4.repeat_interleave(rep, dim=1)
        L_q = query.shape[2]
        if isinstance(block_mask, BlockMask):
            dense = _dense_from_block_mask(block_mask, L_q)   # (L, L)
        else:
            dense = block_mask
        if dense.dtype != torch.bool:
            dense = dense.to(torch.bool)
        # Convert bool → additive (0 / -inf), broadcast to (1, 1, L, L)
        add = torch.zeros(dense.shape, dtype=q4.dtype, device=q4.device)
        add.masked_fill_(~dense, float("-inf"))
        attn_mask = add[None, None, :, :]
        return F.scaled_dot_product_attention(q4, k4, v4, attn_mask=attn_mask)
        # returns (1, n_h, L, d) — matches PT downstream code

    _fa.flex_attention = patched_flex_attention
    # Also patch the qwen2_navit-imported binding
    sys.modules.setdefault("__patched_fa__", patched_flex_attention)


_install_flex_attention_sdpa_patch()


# ----- imports -----
import numpy as np
import torch
from safetensors import safe_open

import mlx.core as mx
from lance_mlx.backbone import LanceLLM, LanceTextConfig, load_full_lance
from lance_mlx.rope import VisionSpec, build_positions_for_layout
from lance_mlx.scheduler import make_schedule, cfg_velocity, euler_step

# PT bits (after shims are installed)
qwen2_navit = importlib.import_module("modeling.lance.qwen2_navit")
Qwen2MoTDecoderLayer = qwen2_navit.Qwen2MoTDecoderLayer
from modeling.qwen2.configuration_qwen2 import Qwen2Config
from data.data_utils import create_sparse_mask
from torch.nn.attention.flex_attention import create_block_mask
import torch.nn.attention.flex_attention as _fa_mod
# Re-bind qwen2_navit's local `flex_attention` to our patched version.
qwen2_navit.flex_attention = _fa_mod.flex_attention

from transformers import AutoTokenizer


# Lance special token ids
IM_START, IM_END = 151644, 151645
VIS_START, VIS_END = 151652, 151653
IMG_TOKEN = 151655


PT_WEIGHTS  = "checkpoints/Lance/Lance_3B/model.safetensors"
MLX_WEIGHTS = "checkpoints/Lance-3B-MLX/model.safetensors"


# ----- PT mask builder via create_sparse_mask predicate ----------------------
def build_pt_mask(seq_len: int, split_lens: list[int], attn_modes: list[str],
                  num_heads: int = 16):
    """Call PT `create_sparse_mask` and *also* `create_block_mask` (for PT
    forward_train) — returns (block_mask_for_PT, dense_LL_bool_for_MLX).
    Maps `full_noise`/`full_noise_target` → `full` per PT lance.py:152.
    """
    attn_modes_ = ["full" if m in ("full_noise", "full_noise_target") else m
                   for m in attn_modes]
    predicate = create_sparse_mask(
        document_lens=[seq_len],
        split_lens=split_lens,
        attn_modes=attn_modes_,
        device=torch.device("cpu"),
    )
    # PT BlockMask (consumed by flex_attention inside PackedAttention.forward_train)
    block_mask = create_block_mask(
        predicate, B=1, H=num_heads, Q_LEN=seq_len, KV_LEN=seq_len,
        device=torch.device("cpu"), BLOCK_SIZE=128, _compile=False,
    )
    # Dense (L, L) bool — for MLX sdpa.  Evaluate predicate on the grid.
    q = torch.arange(seq_len)[:, None]
    k = torch.arange(seq_len)[None, :]
    b = torch.tensor(0); h = torch.tensor(0)
    dense_bool = predicate(b=b, h=h, q_idx=q, kv_idx=k)
    return block_mask, dense_bool


def pt_mask_to_mlx_additive(mask_bool: torch.Tensor) -> mx.array:
    """Convert (L, L) bool mask → (L, L) additive mask for sdpa:
    0.0 where True (attend), -inf where False (block)."""
    L = mask_bool.shape[0]
    add = torch.zeros((L, L), dtype=torch.float32)
    add.masked_fill_(~mask_bool, float("-inf"))
    return mx.array(add.numpy())


# ----- sequence layout ------------------------------------------------------
def build_layout(prompt_ids: list[int], n_latent: int):
    cond_ids = ([IM_START] + list(prompt_ids) + [IM_END]
                + [VIS_START] + [IMG_TOKEN] * n_latent + [VIS_END])
    uncond_ids = ([IM_START, IM_END]
                  + [VIS_START] + [IMG_TOKEN] * n_latent + [VIS_END])

    # Split: text prefix (causal) | latent slab (noise) | trailing VIS_END (causal)
    cond_prefix_len = 1 + len(prompt_ids) + 1 + 1     # IM_START + prompt + IM_END + VIS_START
    cond_split_lens = [cond_prefix_len, n_latent, 1]   # 1 = VIS_END
    cond_attn_modes = ["causal", "noise", "causal"]
    cond_gen_span = (cond_prefix_len, cond_prefix_len + n_latent)

    uncond_prefix_len = 2 + 1                          # IM_START + IM_END + VIS_START
    uncond_split_lens = [uncond_prefix_len, n_latent, 1]
    uncond_attn_modes = ["causal", "noise", "causal"]
    uncond_gen_span = (uncond_prefix_len, uncond_prefix_len + n_latent)

    return {
        "cond_ids": cond_ids,
        "uncond_ids": uncond_ids,
        "cond_split_lens": cond_split_lens, "cond_attn_modes": cond_attn_modes,
        "uncond_split_lens": uncond_split_lens, "uncond_attn_modes": uncond_attn_modes,
        "cond_gen_span": cond_gen_span,
        "uncond_gen_span": uncond_gen_span,
    }


# ----- PT model assembly (one shared backbone, full 36 layers) --------------
class PtLanceModel:
    """Minimal PT Lance backbone: embed_tokens + 36 Qwen2MoTDecoderLayer +
    final norm + adapters (vae2llm, llm2vae, time_embedder, latent_pos_embed).

    Skips lm_head (we only need hidden states for v_t).
    """
    def __init__(self):
        self.cfg = LanceTextConfig()
        # Construct one Qwen2Config we can reuse across all 36 layers.
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
        self.layers = torch.nn.ModuleList(
            [Qwen2MoTDecoderLayer(self.q_cfg, layer_idx=i)
             for i in range(self.cfg.num_hidden_layers)]
        )
        self.final_norm = torch.nn.Module()                  # placeholder
        # Adapters
        self.vae2llm = torch.nn.Linear(48, self.cfg.hidden_size, bias=True)
        self.llm2vae = torch.nn.Linear(self.cfg.hidden_size, 48, bias=True)
        # TimestepEmbedder: fc1 + SiLU + fc2 (PT impl)
        self.time_fc1 = torch.nn.Linear(256, self.cfg.hidden_size, bias=True)
        self.time_fc2 = torch.nn.Linear(self.cfg.hidden_size, self.cfg.hidden_size, bias=True)
        # PositionEmbedding3D: just a (4096, 2048) buffer
        self.latent_pos_embed = torch.nn.Embedding(4 * 32 * 32, self.cfg.hidden_size)
        # Final norm — RMSNorm
        self.final_norm = self._make_rms(self.cfg.hidden_size, eps=self.cfg.rms_norm_eps)
        self.norm_moe_gen = self._make_rms(self.cfg.hidden_size, eps=self.cfg.rms_norm_eps)

    def _make_rms(self, dim, eps):
        from modeling.qwen2.modeling_qwen2 import Qwen2RMSNorm
        return Qwen2RMSNorm(dim, eps=eps)

    def to_bf16(self):
        self.embed_tokens = self.embed_tokens.to(torch.bfloat16)
        for L in self.layers: L.to(torch.bfloat16)
        self.vae2llm = self.vae2llm.to(torch.bfloat16)
        self.llm2vae = self.llm2vae.to(torch.bfloat16)
        self.time_fc1 = self.time_fc1.to(torch.bfloat16)
        self.time_fc2 = self.time_fc2.to(torch.bfloat16)
        self.latent_pos_embed = self.latent_pos_embed.to(torch.bfloat16)
        self.final_norm = self.final_norm.to(torch.bfloat16)
        self.norm_moe_gen = self.norm_moe_gen.to(torch.bfloat16)

    def load_from_pt_checkpoint(self):
        """Load Lance_3B PT weights into the matching modules.  Cast to bf16
        per Lance's mode='gen' precision contract."""
        with safe_open(PT_WEIGHTS, framework="pt", device="cpu") as f:
            d = {k: f.get_tensor(k).to(torch.bfloat16) for k in f.keys()}

        # embed_tokens + (tied) lm_head (we don't use lm_head)
        self.embed_tokens.weight.data = d["language_model.model.embed_tokens.weight"].clone()
        # 36 layers
        for i, L in enumerate(self.layers):
            prefix = f"language_model.model.layers.{i}."
            state = {k[len(prefix):]: v for k, v in d.items() if k.startswith(prefix)}
            L.load_state_dict(state, strict=True)
        # final norms
        self.final_norm.weight.data = d["language_model.model.norm.weight"].clone()
        self.norm_moe_gen.weight.data = d["language_model.model.norm_moe_gen.weight"].clone()
        # adapters
        self.vae2llm.weight.data = d["vae2llm.weight"].clone()
        self.vae2llm.bias.data   = d["vae2llm.bias"].clone()
        self.llm2vae.weight.data = d["llm2vae.weight"].clone()
        self.llm2vae.bias.data   = d["llm2vae.bias"].clone()
        # time_embedder: PT checkpoint uses nn.Sequential indices `.mlp.0` / `.mlp.2`.
        # (STAGE 1 renamed to fc1/fc2 for the MLX checkpoint; here we read PT directly.)
        self.time_fc1.weight.data = d["time_embedder.mlp.0.weight"].clone()
        self.time_fc1.bias.data   = d["time_embedder.mlp.0.bias"].clone()
        self.time_fc2.weight.data = d["time_embedder.mlp.2.weight"].clone()
        self.time_fc2.bias.data   = d["time_embedder.mlp.2.bias"].clone()
        # latent_pos_embed (4096, 2048)
        self.latent_pos_embed.weight.data = d["latent_pos_embed.pos_embed"].clone()

    def _time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half = 256 // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(10000.0)) * torch.arange(0, half, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(torch.bfloat16)
        # (1, 256) → fc1 → silu → fc2 → (1, hidden) — entire path bf16 to match weights
        h = self.time_fc1(emb)
        h = torch.nn.functional.silu(h)
        return self.time_fc2(h)

    def forward_to_v(self, input_ids: torch.Tensor, x_t: torch.Tensor,
                     t_scalar: torch.Tensor, latent_pos_ids: torch.Tensor,
                     gen_span: tuple[int, int],
                     position_ids_3: torch.Tensor,
                     attention_mask_LL: torch.Tensor,
                     packed_text_indexes: torch.Tensor,
                     packed_vae_token_indexes: torch.Tensor) -> torch.Tensor:
        """One full forward.  Returns v_t at the GEN slab: (N, 48)."""
        gs, ge = gen_span
        # text embed (1, L, D) bf16
        text_embed = self.embed_tokens(input_ids)   # (1, L, D) bf16
        # GEN slab embed = vae2llm(x_t) + time_embed + pos_embed
        x_t_bf = x_t.to(torch.bfloat16)
        slab = self.vae2llm(x_t_bf) + self._time_embedding(t_scalar) + self.latent_pos_embed(latent_pos_ids)
        # splice
        embed = torch.cat([text_embed[:, :gs], slab[None, :, :], text_embed[:, ge:]], dim=1)
        # Forward through 36 layers in "validation" (forward_train) mode with mode_forward="validation".
        # This is the mode that uses the (L, L) attention_mask + packed_*_token_indexes routing.
        packed_seq = embed[0]                       # (L, D)
        sample_lens = [embed.shape[1]]
        # Per Lance: forward_train with packed_und/gen indexes routes per-token via norm/proj/mlp.
        # We need packed_position_embeddings (cos, sin) — compute from position_ids_3.
        cos, sin = self._mrope_cos_sin(position_ids_3)
        h = packed_seq
        for L in self.layers:
            h = L(
                packed_sequence=h,
                sample_lens=sample_lens,
                attention_mask=attention_mask_LL,           # (L, L) additive
                packed_position_embeddings=(cos, sin),
                packed_und_token_indexes=packed_text_indexes,
                packed_gen_token_indexes=packed_vae_token_indexes,
                mode_forward="validation",
            )
        # Final norm — Lance pairs it (norm vs norm_moe_gen) by token.
        L_tot = h.shape[0]
        h_und = self.final_norm(h)
        h_gen = self.norm_moe_gen(h)
        out = torch.zeros_like(h)
        out[packed_text_indexes] = h_und[packed_text_indexes]
        out[packed_vae_token_indexes] = h_gen[packed_vae_token_indexes]
        # extract GEN slab + project to v
        v = self.llm2vae(out[gs:ge].to(torch.bfloat16))
        return v.to(torch.float32)

    def _mrope_cos_sin(self, position_ids_3: torch.Tensor):
        """Manual mRoPE cos/sin matching Qwen2.5-VL's `Qwen2_5_VLRotaryEmbedding`.

        PT `PackedAttention.forward_train` calls `apply_rotary_pos_emb` with
        `unsqueeze_dim=1` on (L, num_heads, head_dim) q, so cos/sin must be
        flat `(L, head_dim)` (no batch dim).
        """
        head_dim = self.cfg.head_dim
        L = position_ids_3.shape[-1]
        base = self.cfg.rope_theta
        ms = self.cfg.rope_scaling["mrope_section"]
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        inv = inv_freq[None, None, :, None].expand(3, L, -1, 1)
        pos = position_ids_3.float()[:, :, None, :]
        freqs = (inv @ pos).transpose(2, 3)                  # (3, B=1, L, hd/2)
        s_t, s_h, s_w = ms
        t_p = freqs[0, 0, :, :s_t]                            # (L, s_t)
        h_p = freqs[1, 0, :, s_t:s_t+s_h]
        w_p = freqs[2, 0, :, s_t+s_h:s_t+s_h+s_w]
        f = torch.cat([t_p, h_p, w_p], dim=-1)               # (L, hd/2)
        emb = torch.cat([f, f], dim=-1)                       # (L, hd)
        return emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)


# ----- MLX forward (mirrors our t2i.py per-step compute) --------------------
def mlx_forward_to_v(model: LanceLLM, input_ids: mx.array, x_t: mx.array,
                    t_scalar: mx.array, latent_pos_ids: mx.array,
                    gen_span: tuple[int, int],
                    position_ids_3: mx.array,
                    attention_mask_LL: mx.array,
                    gen_mask: mx.array) -> mx.array:
    text_embed = model.language_model.model.embed_tokens(input_ids)  # (1, L, D)
    vae_proj = model.vae2llm(x_t)                  # (N, D)
    t_emb    = model.time_embedder(t_scalar)        # (1, D)
    pos_emb  = model.latent_pos_embed(latent_pos_ids)  # (N, D)
    slab = vae_proj + t_emb + pos_emb
    gs, ge = gen_span
    embed = mx.concatenate([
        text_embed[:, :gs, :],
        slab[None, :, :],
        text_embed[:, ge:, :],
    ], axis=1)
    hidden = model.language_model.model(
        input_ids=None, position_ids=position_ids_3,
        inputs_embeds=embed, mask=attention_mask_LL, gen_mask=gen_mask,
    )
    v = model.llm2vae(hidden[0, gs:ge, :])
    return v


# ----- main side-by-side compare -------------------------------------------
def main() -> None:
    PROMPT = "a photo of a sunset over mountains"
    SEED, NUM_STEPS, SHIFT, CFG = 0, 30, 3.5, 4.0
    HEIGHT, WIDTH = 512, 512
    SPATIAL_DS, Z_DIM = 16, 48
    h_lat, w_lat, t_lat = HEIGHT // SPATIAL_DS, WIDTH // SPATIAL_DS, 1
    N = t_lat * h_lat * w_lat

    tok = AutoTokenizer.from_pretrained("checkpoints/Lance-3B-MLX", trust_remote_code=True)
    prompt_ids = tok(PROMPT, return_tensors=None, add_special_tokens=False)["input_ids"]
    layout = build_layout(prompt_ids, N)
    L_cond = len(layout["cond_ids"]);  L_unc = len(layout["uncond_ids"])
    print(f"[setup] L_cond={L_cond}  L_unc={L_unc}  N_latent={N}")

    # ---- mask: PT-derived BlockMask (for PT) + dense (L, L) (for MLX) ----
    cond_block_mask, cond_dense_bool   = build_pt_mask(L_cond, layout["cond_split_lens"], layout["cond_attn_modes"])
    unc_block_mask,  unc_dense_bool    = build_pt_mask(L_unc,  layout["uncond_split_lens"], layout["uncond_attn_modes"])
    cond_mask_mlx = pt_mask_to_mlx_additive(cond_dense_bool)
    uncond_mask_mlx = pt_mask_to_mlx_additive(unc_dense_bool)
    print(f"[mask] cond attend density: {cond_dense_bool.float().mean().item():.4f}  "
          f"uncond attend density: {unc_dense_bool.float().mean().item():.4f}")

    # ---- position: STAGE 3 helper (no shift — verified no-op for T2I modality {0,1}) ----
    cond_gs, cond_ge = layout["cond_gen_span"]
    unc_gs, unc_ge   = layout["uncond_gen_span"]
    span_c = VisionSpec(start=cond_gs - 1, length=N, t=t_lat, h=h_lat, w=w_lat)
    span_u = VisionSpec(start=unc_gs - 1,  length=N, t=t_lat, h=h_lat, w=w_lat)
    cond_pos_mlx = build_positions_for_layout(L_cond, [span_c])
    unc_pos_mlx  = build_positions_for_layout(L_unc,  [span_u])
    cond_pos_pt  = torch.from_numpy(np.asarray(cond_pos_mlx))
    unc_pos_pt   = torch.from_numpy(np.asarray(unc_pos_mlx))

    # ---- latent slab table-lookup indices ----
    # max_latent_size=64 per Lance-3B-MLX config (image variant).
    MAX_LATENT_SIZE = 64
    t_idx = mx.arange(t_lat).reshape(t_lat, 1, 1)
    h_idx = mx.arange(h_lat).reshape(1, h_lat, 1)
    w_idx = mx.arange(w_lat).reshape(1, 1, w_lat)
    flat_mlx = (t_idx * MAX_LATENT_SIZE * MAX_LATENT_SIZE
                + h_idx * MAX_LATENT_SIZE + w_idx)
    latent_pos_ids_mlx = mx.broadcast_to(flat_mlx, (t_lat, h_lat, w_lat)).flatten()
    latent_pos_ids_pt  = torch.from_numpy(np.asarray(latent_pos_ids_mlx))

    # ---- gen masks for MLX routing ----
    def _gen_mask(L, span):
        cols = mx.arange(L)
        return ((cols >= span[0]) & (cols < span[1]))[None, :]
    cond_gen_mask_mlx = _gen_mask(L_cond, (cond_gs, cond_ge))
    unc_gen_mask_mlx  = _gen_mask(L_unc,  (unc_gs, unc_ge))
    # PT side: packed indexes
    def _idx_pair(L, gs, ge):
        all_idx = torch.arange(L, dtype=torch.long)
        text = torch.cat([all_idx[:gs], all_idx[ge:]])
        vae  = all_idx[gs:ge]
        return text, vae
    cond_text_idx, cond_vae_idx = _idx_pair(L_cond, cond_gs, cond_ge)
    unc_text_idx,  unc_vae_idx  = _idx_pair(L_unc,  unc_gs, unc_ge)

    # ---- noise init (single shared numpy array) ----
    rng = np.random.default_rng(SEED)
    x_t_np = rng.standard_normal((N, Z_DIM)).astype(np.float32)
    x_t_pt  = torch.from_numpy(x_t_np.copy())
    x_t_mlx = mx.array(x_t_np.copy())

    # ---- schedule (same) ----
    sch = make_schedule(num_steps=NUM_STEPS, timestep_shift=SHIFT)
    timesteps_np = np.asarray(sch.timesteps)
    dts_np       = np.asarray(sch.dts)

    # ---- build models ----
    print("[build] MLX LanceLLM ...")
    mlx_model = LanceLLM(LanceTextConfig())
    load_full_lance(mlx_model, MLX_WEIGHTS)
    mlx_model.eval()

    print("[build] PT Lance backbone (36 layers + adapters, bf16) ...")
    pt_model = PtLanceModel()
    pt_model.to_bf16()
    t0 = time.time()
    pt_model.load_from_pt_checkpoint()
    for L in pt_model.layers: L.eval()
    print(f"[load] PT weights in {time.time()-t0:.1f}s")

    cond_ids_mlx = mx.array([layout["cond_ids"]],   dtype=mx.int32)
    unc_ids_mlx  = mx.array([layout["uncond_ids"]], dtype=mx.int32)
    cond_ids_pt  = torch.from_numpy(np.asarray(cond_ids_mlx))
    unc_ids_pt   = torch.from_numpy(np.asarray(unc_ids_mlx))

    # ---- side-by-side denoise ----
    def _cos(a_np, b_np):
        af = a_np.astype(np.float64).flatten()
        bf = b_np.astype(np.float64).flatten()
        return float((af @ bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))

    print()
    print(f"{'step':>4s}  {'t':>7s}  {'cos':>10s}  {'max|Δ|':>10s}  {'rel_L2':>10s}")
    print("-" * 50)
    fail_step = None
    for i in range(NUM_STEPS):
        t = float(timesteps_np[i])
        dt = float(dts_np[i])
        t_scalar_pt  = torch.tensor([t], dtype=torch.float32)
        t_scalar_mlx = mx.array([t], dtype=mx.float32)

        # ---- PT forward (cond + uncond) ----
        with torch.no_grad():
            v_cond_pt = pt_model.forward_to_v(
                cond_ids_pt, x_t_pt, t_scalar_pt, latent_pos_ids_pt,
                (cond_gs, cond_ge), cond_pos_pt, cond_block_mask,
                cond_text_idx, cond_vae_idx,
            )
            v_unc_pt = pt_model.forward_to_v(
                unc_ids_pt, x_t_pt, t_scalar_pt, latent_pos_ids_pt,
                (unc_gs, unc_ge), unc_pos_pt, unc_block_mask,
                unc_text_idx, unc_vae_idx,
            )
        # CFG + Lance norm rescale (numpy)
        v_c = v_cond_pt.numpy(); v_u = v_unc_pt.numpy()
        v_blend = v_u + CFG * (v_c - v_u)
        n_c = np.linalg.norm(v_c); n_b = np.linalg.norm(v_blend)
        scale_pt = float(np.clip(n_c / (n_b + 1e-8), 0.0, 1.0))
        v_pt = v_blend * scale_pt
        x_t_pt = x_t_pt - torch.from_numpy(v_pt) * dt

        # ---- MLX forward (cond + uncond) ----
        v_cond_mlx = mlx_forward_to_v(
            mlx_model, cond_ids_mlx, x_t_mlx, t_scalar_mlx, latent_pos_ids_mlx,
            (cond_gs, cond_ge), cond_pos_mlx, cond_mask_mlx, cond_gen_mask_mlx,
        )
        v_unc_mlx = mlx_forward_to_v(
            mlx_model, unc_ids_mlx, x_t_mlx, t_scalar_mlx, latent_pos_ids_mlx,
            (unc_gs, unc_ge), unc_pos_mlx, uncond_mask_mlx, unc_gen_mask_mlx,
        )
        v_t_mlx = cfg_velocity(v_cond_mlx, v_unc_mlx, scale=CFG)
        x_t_mlx = euler_step(x_t_mlx, v_t_mlx, dt)
        mx.eval(x_t_mlx)

        pt_arr = x_t_pt.numpy()
        mlx_arr = np.asarray(x_t_mlx)
        c   = _cos(pt_arr, mlx_arr)
        mxd = float(np.abs(pt_arr - mlx_arr).max())
        rl2 = float(np.linalg.norm(pt_arr - mlx_arr) / (np.linalg.norm(pt_arr) + 1e-12))
        flag = "" if c >= 0.995 else "  ✗"
        print(f"{i+1:>4d}  {t:>7.4f}  {c:>10.6f}  {mxd:>10.3e}  {rl2:>10.3e}{flag}")
        if c < 0.995 and fail_step is None:
            fail_step = i + 1

    print()
    if fail_step is None:
        print(f"latent step-cos: PASS — all {NUM_STEPS} steps cos ≥ 0.995")
    else:
        print(f"latent step-cos: FAIL — first divergence at step {fail_step}")

    # ===== latent → image cos (the missing leg) ====================
    # Reshape both final latents and run VAE decode on each side.
    print("\n=== latent → image decode path ===")

    # Reshape (N, 48) → (1, T, H, W, 48), NTHWC.
    x_t_pt_np = x_t_pt.numpy()
    x_t_mlx_np = np.asarray(x_t_mlx)
    cos_flat = _cos(x_t_pt_np, x_t_mlx_np)
    print(f"final flat latent     cos={cos_flat:.6f}  shape={x_t_pt_np.shape}")

    pt_lat = x_t_pt_np.reshape(1, t_lat, h_lat, w_lat, Z_DIM)
    mlx_lat = x_t_mlx_np.reshape(1, t_lat, h_lat, w_lat, Z_DIM)
    cos_resh = _cos(pt_lat, mlx_lat)
    print(f"reshape→NTHWC latent  cos={cos_resh:.6f}  shape={pt_lat.shape}")

    # Load PT WanVAE_ (refs/Lance) — used by Lance inference.  Convert MLX
    # vae weights to PT layout (re-using STAGE 5 logic).
    import importlib as _il
    pt_vae_mod = _il.import_module("modeling.vae.wan.vae2_2")
    PT_WanVAE = pt_vae_mod.WanVAE_
    sys.path.insert(0, ".")  # so we can reuse stage5_pt_compare.mlx_to_pt_state
    stage5 = _il.import_module("tools.stage5_pt_compare")

    pt_vae = PT_WanVAE(dim=160, dec_dim=256, z_dim=48,
                       dim_mult=[1, 2, 4, 4],
                       temperal_downsample=[False, True, True])
    pt_vae.eval()
    pt_vae_state = stage5.mlx_to_pt_state(
        mx.load("checkpoints/Wan2.2-VAE-MLX/model.safetensors")
    )
    pt_vae.load_state_dict(pt_vae_state, strict=False)

    # PT decode expects (B, C, T, H, W).  Convert.
    pt_lat_nctw = np.transpose(pt_lat, (0, 4, 1, 2, 3))           # (1, 48, T, H, W)
    pt_lat_torch = torch.from_numpy(pt_lat_nctw).float()
    with torch.no_grad():
        pt_image_torch = pt_vae.decode(pt_lat_torch, scale=[0, 1])
    pt_image_np = pt_image_torch.cpu().numpy()                    # (1, 3, T_pix, H_pix, W_pix)
    # Convert to NTHWC for comparison: (1, T_pix, H_pix, W_pix, 3)
    pt_image_nthwc = np.transpose(pt_image_np, (0, 2, 3, 4, 1))

    # MLX side
    from lance_mlx.vae_wan22 import Wan2_2_VAE, Wan22VAEConfig
    mlx_vae = Wan2_2_VAE(Wan22VAEConfig())
    mlx_vae.load_weights(list(mx.load("checkpoints/Wan2.2-VAE-MLX/model.safetensors").items()),
                          strict=True)
    mx.eval(mlx_vae.parameters()); mlx_vae.eval()
    mlx_lat_arr = mx.array(mlx_lat)
    mlx_image = mlx_vae.decode(mlx_lat_arr)                       # (1, T_pix, H_pix, W_pix, 3)
    mx.eval(mlx_image)
    mlx_image_np = np.asarray(mlx_image)

    cos_image = _cos(pt_image_nthwc, mlx_image_np)
    max_image = float(np.abs(pt_image_nthwc - mlx_image_np).max())
    print(f"VAE decode output      cos={cos_image:.6f}  max|Δ|={max_image:.4f}  shape={mlx_image_np.shape}")
    print(f"PT image   range=[{pt_image_nthwc.min():+.3f}, {pt_image_nthwc.max():+.3f}]")
    print(f"MLX image  range=[{mlx_image_np.min():+.3f}, {mlx_image_np.max():+.3f}]")

    # Save PNG side-by-side
    from PIL import Image
    import os
    os.makedirs("out", exist_ok=True)
    def _to_uint8(arr):
        return (np.clip(arr[0, 0] * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8)
    pt_img_u8 = _to_uint8(pt_image_nthwc)
    mlx_img_u8 = _to_uint8(mlx_image_np)
    Image.fromarray(pt_img_u8).save("out/stage6_pt_final.png")
    Image.fromarray(mlx_img_u8).save("out/stage6_mlx_final.png")
    # Side-by-side
    combo = np.concatenate([pt_img_u8, mlx_img_u8], axis=1)
    Image.fromarray(combo).save("out/stage6_side_by_side.png")
    print("\n[save] out/stage6_pt_final.png   out/stage6_mlx_final.png   out/stage6_side_by_side.png")

    # Verdict: image cos ≥ 0.995 means decode path is correct.
    if cos_image >= 0.995:
        print(f"\nimage cos: PASS ({cos_image:.6f} ≥ 0.995)")
    else:
        print(f"\nimage cos: FAIL ({cos_image:.6f} < 0.995) — bug in decode path")


if __name__ == "__main__":
    main()
