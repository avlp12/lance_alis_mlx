"""Lance text backbone — 36-layer modified Qwen2.5-VL with qk_norm + MoE-gen pairs.

STAGE 2 scope: text-only forward.  The `_moe_gen` siblings are present in the
module tree so that the full 1021-tensor Lance MLX checkpoint strict-loads,
but the forward path here only exercises the canonical Qwen2.5-VL route
(input_layernorm → self_attn{q,k,v,o_proj + q,k_norm} → residual →
post_attention_layernorm → mlp → residual).  STAGE 4 wires the MoE-gen
routing on top of the same modules.

What's borrowed from mlx-vlm (qwen2_5_vl/language.py) and what's not
----------------------------------------------------------------------
Borrowed verbatim, but re-typed in this file so we own them:
  - Qwen2RotaryEmbedding (mRoPE machinery)
  - rotate_half
  - apply_multimodal_rotary_pos_emb

NOT borrowed — Lance diverges, we implement here:
  - LanceAttention   (adds q_norm/k_norm; adds _moe_gen paired weights)
  - LanceMLP         (paired with mlp_moe_gen at the layer level)
  - LanceDecoderLayer
  - LanceQwen2Model, LanguageModel, LanceLLM (tied lm_head policy)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .moe_gen import route as _route_moe


# ----------------------------------------------------------------------------
# Config — distilled from `checkpoints/Lance/Lance_3B/llm_config.json`.
# Only the fields the backbone actually consumes are kept; vision_config is
# parked on its own dataclass (STAGE 2 doesn't touch it but the load path
# expects to see it eventually).
# ----------------------------------------------------------------------------
@dataclass
class VisionConfig:
    depth: int = 32
    hidden_size: int = 1280
    out_hidden_size: int = 2048
    num_heads: int = 16
    in_chans: int = 3
    patch_size: int = 14
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    window_size: int = 112
    intermediate_size: int = 3420
    fullatt_block_indexes: tuple[int, ...] = (7, 15, 23, 31)
    tokens_per_second: int = 2


@dataclass
class LanceTextConfig:
    """Lance text backbone hyperparameters.

    Defaults match `Lance_3B/llm_config.json`; tests / smaller variants
    override at construction.  All values are needed at module build time —
    keep this dataclass frozen-style in spirit (don't mutate after build).
    """
    hidden_size: int = 2048
    intermediate_size: int = 11008
    num_hidden_layers: int = 36
    num_attention_heads: int = 16
    num_key_value_heads: int = 2          # GQA 8:1
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 128_000
    tie_word_embeddings: bool = True
    # mRoPE: 16 dims for temporal, 24 each for H/W — totals head_dim=128 // 2.
    rope_scaling: dict = field(default_factory=lambda: {
        "type": "mrope",
        "mrope_section": [16, 24, 24],
    })
    # Lance-specific: every layer is an MoE block with the paired weights
    # listed in the docstring above.
    has_qk_norm: bool = True
    has_moe_gen: bool = True
    # Special tokens (vocab IDs) — used by tokenization + mRoPE indexing.
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    image_token_id: int = 151655
    video_token_id: int = 151656
    # Connected to VisionConfig at the wrapper level; not referenced here.
    vision_config: Optional[VisionConfig] = None

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


# ----------------------------------------------------------------------------
# RoPE — copied from mlx-vlm qwen2_5_vl/language.py.  Kept in this file so
# the Lance backbone has zero runtime dependency on mlx-vlm's internals; we
# only used mlx-vlm as a reference text, not as an import target.
# ----------------------------------------------------------------------------
class Qwen2RotaryEmbedding:
    """RoPE buffer holder.  Intentionally NOT an `nn.Module` — `inv_freq`
    is a stateless lookup buffer, not a learned parameter, and we don't
    want `tree_flatten(model.parameters())` to see it (would break the
    strict-load filter).  Downside: `model.apply(lambda a: a.astype(fp16))`
    will skip `inv_freq`; that's fine because every call site casts
    inv_freq to fp32 explicitly.
    """
    def __init__(self, dim: int, max_position_embeddings: int,
                 base: float, mrope_section: list[int]):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.mrope_section = mrope_section
        # inv_freq is dim//2 entries: 1 / base^(2i/dim)
        self.inv_freq = 1.0 / (
            base ** (mx.arange(0, dim, 2).astype(mx.float32) / dim)
        )

    def _apply_mrope(self, freqs: mx.array) -> mx.array:
        """Pack 3D mRoPE columns: [T sec | H sec | W sec], drawn from the
        three slabs of `freqs` along their respective mrope_section windows.

        freqs: (3, bs, seq_len, head_dim // 2).  Returns (bs, seq_len, head_dim // 2).

        Implemented as an explicit functional concatenation rather than
        slice-assignment on a sliced view — MLX evaluates `x[slc] = y` as
        `index_update` (functional), which works for the previous
        in-place style by accident.  The concat below makes the intent
        obvious to a reader and survives future MLX graph optimisations.
        """
        s_t, s_h, s_w = self.mrope_section
        t = freqs[0, ..., :s_t]                          # (bs, seq, s_t)
        h = freqs[1, ..., s_t:s_t + s_h]                 # (bs, seq, s_h)
        w = freqs[2, ..., s_t + s_h:s_t + s_h + s_w]     # (bs, seq, s_w)
        return mx.concatenate([t, h, w], axis=-1)

    def __call__(self, x: mx.array, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        # position_ids is either (bs, seq) or (3, bs, seq); promote to 3D.
        if position_ids.ndim == 2:
            position_ids = mx.broadcast_to(
                position_ids[None, ...],
                (3, position_ids.shape[0], position_ids.shape[1]),
            )
        inv_freq_exp = mx.broadcast_to(
            self.inv_freq[None, None, :, None].astype(mx.float32),
            (3, position_ids.shape[1], self.inv_freq.shape[0], 1),
        )
        pos_exp = position_ids[:, :, None, :].astype(mx.float32)
        freqs = inv_freq_exp @ pos_exp                  # (3, bs, dim/2, seq)
        freqs = mx.swapaxes(freqs, 2, 3)                # (3, bs, seq, dim/2)
        freqs = self._apply_mrope(freqs)                # (bs, seq, dim/2)
        emb = mx.concatenate([freqs, freqs], axis=-1)   # (bs, seq, dim)
        return mx.cos(emb).astype(x.dtype), mx.sin(emb).astype(x.dtype)


def rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_mrope(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array,
                unsqueeze_dim: int = 1) -> tuple[mx.array, mx.array]:
    """Apply mRoPE to (q, k).  cos/sin shape (bs, seq, head_dim); we
    unsqueeze one dim so it broadcasts against (bs, heads, seq, head_dim).
    """
    cos = mx.expand_dims(cos, axis=unsqueeze_dim)
    sin = mx.expand_dims(sin, axis=unsqueeze_dim)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


# ----------------------------------------------------------------------------
# Attention block — Lance flavour
#
# Difference from mlx-vlm Qwen2.5-VL Attention:
#   1. `q_norm`, `k_norm` exist (per-head RMSNorm on Q/K before RoPE).  In
#      the PT modified-Qwen2.5-VL these are real weights; in vanilla
#      Qwen2.5-VL they're absent.  Reza2kn dropped them to fit mlx-lm's
#      stock class; for a full port we keep them.
#   2. Paired `_moe_gen` weights for q/k/v/o_proj and q/k_norm exist in the
#      module tree so the checkpoint strict-loads.  STAGE 2 never *calls*
#      the _moe_gen branch — `__call__` only runs the canonical path on
#      text input.  STAGE 4 will add the routing.
# ----------------------------------------------------------------------------
class LanceAttention(nn.Module):
    def __init__(self, cfg: LanceTextConfig):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim ** -0.5
        # Branch on explicit flags rather than `hasattr(self, "q_norm")`:
        # the flag is robust against accidentally setting q_norm to None
        # somewhere downstream and explodes-loud at call rather than at
        # weight-discovery time.
        self._has_qk_norm = cfg.has_qk_norm
        self._has_moe_gen = cfg.has_moe_gen

        D = cfg.hidden_size
        Hq = self.n_heads * self.head_dim       # q proj out
        Hkv = self.n_kv_heads * self.head_dim   # k/v proj out

        # --- canonical Qwen2-style projections (text path, "und" route) ---
        self.q_proj = nn.Linear(D, Hq, bias=True)
        self.k_proj = nn.Linear(D, Hkv, bias=True)
        self.v_proj = nn.Linear(D, Hkv, bias=True)
        self.o_proj = nn.Linear(Hq, D, bias=False)

        # --- qk_norm (Lance/modified-Qwen2.5-VL specific) ---
        # PT applies RMSNorm per-head: (head_dim,) weight, eps from config.
        if cfg.has_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        # else: leave the attributes off the module entirely so strict-load
        # of a checkpoint without these keys still works.

        # --- _moe_gen siblings (held but unused at STAGE 2) ---
        if cfg.has_moe_gen:
            self.q_proj_moe_gen = nn.Linear(D, Hq, bias=True)
            self.k_proj_moe_gen = nn.Linear(D, Hkv, bias=True)
            self.v_proj_moe_gen = nn.Linear(D, Hkv, bias=True)
            self.o_proj_moe_gen = nn.Linear(Hq, D, bias=False)
            if cfg.has_qk_norm:
                self.q_norm_moe_gen = nn.RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
                self.k_norm_moe_gen = nn.RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)

        self.rotary = Qwen2RotaryEmbedding(
            self.head_dim, cfg.max_position_embeddings,
            cfg.rope_theta, cfg.rope_scaling["mrope_section"],
        )

    def __call__(
        self,
        x: mx.array,
        position_ids: mx.array,
        mask: Optional[mx.array] = None,
        cache=None,
        gen_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward with optional MoE-gen routing.

        x:            (B, L, D)
        position_ids: (3, B, L)  — 3D mRoPE per Qwen2.5-VL convention
        mask:         attention mask (additive) or None for causal.
        cache:        per-layer KV cache (mlx.nn.cache.KVCache or compatible).
        gen_mask:     (B, L) bool — True where the token routes through
                      the `_moe_gen` weights.  None or all-False uses the
                      canonical path only (text/ViT path; STAGE 2's
                      behaviour, retained as the fast common case).

        Routing strategy: compute *both* projection branches on the full
        sequence, merge with `mx.where`.  Attention itself (sdpa) is
        *not* split — the full mixed sequence attends to itself; only
        per-token projections + qk_norm + o_proj diverge.
        """
        B, L, _ = x.shape

        # ----- Q/K/V projection + per-head reshape -----
        canonical_q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        canonical_k = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        canonical_v = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)

        if self._has_qk_norm:
            canonical_q = self.q_norm(canonical_q)
            canonical_k = self.k_norm(canonical_k)

        routed = self._has_moe_gen and gen_mask is not None
        if routed:
            gen_q = self.q_proj_moe_gen(x).reshape(B, L, self.n_heads, self.head_dim)
            gen_k = self.k_proj_moe_gen(x).reshape(B, L, self.n_kv_heads, self.head_dim)
            gen_v = self.v_proj_moe_gen(x).reshape(B, L, self.n_kv_heads, self.head_dim)
            if self._has_qk_norm:
                gen_q = self.q_norm_moe_gen(gen_q)
                gen_k = self.k_norm_moe_gen(gen_k)
            # gen_mask is (B, L); q/k/v are (B, L, heads, head_dim) — need 2 extra axes.
            q = _route_moe(canonical_q, gen_q, gen_mask, broadcast_dims=2)
            k = _route_moe(canonical_k, gen_k, gen_mask, broadcast_dims=2)
            v = _route_moe(canonical_v, gen_v, gen_mask, broadcast_dims=2)
        else:
            q, k, v = canonical_q, canonical_k, canonical_v

        # ----- mRoPE + sdpa over the merged sequence -----
        q = q.transpose(0, 2, 1, 3)             # (B, n_heads, L, head_dim)
        k = k.transpose(0, 2, 1, 3)             # (B, n_kv_heads, L, head_dim)
        v = v.transpose(0, 2, 1, 3)

        cos, sin = self.rotary(v, position_ids)
        q, k = apply_mrope(q, k, cos, sin, unsqueeze_dim=1)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        attn = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask,
        )                                       # (B, n_heads, L, head_dim)
        attn = attn.transpose(0, 2, 1, 3).reshape(B, L, -1)

        # ----- o_proj branch split -----
        if routed:
            out_und = self.o_proj(attn)
            out_gen = self.o_proj_moe_gen(attn)
            return _route_moe(out_und, out_gen, gen_mask, broadcast_dims=1)
        return self.o_proj(attn)


# ----------------------------------------------------------------------------
# Qwen2 SwiGLU MLP — `gate_proj`, `up_proj`, `down_proj` (no bias in any).
#
# Lance pairs this at the layer level: every decoder layer has both `mlp`
# (text/UND route) and `mlp_moe_gen` (GEN/VAE-latent route).  STAGE 2 only
# calls the text path.
# ----------------------------------------------------------------------------
class LanceMLP(nn.Module):
    def __init__(self, cfg: LanceTextConfig):
        super().__init__()
        D, H = cfg.hidden_size, cfg.intermediate_size
        self.gate_proj = nn.Linear(D, H, bias=False)
        self.up_proj   = nn.Linear(D, H, bias=False)
        self.down_proj = nn.Linear(H, D, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


# ----------------------------------------------------------------------------
# Decoder layer — Lance MoE block.  Module tree mirrors PT
# `Qwen2MoTDecoderLayer`: both branches present, only the canonical one
# called at STAGE 2.
# ----------------------------------------------------------------------------
class LanceDecoderLayer(nn.Module):
    def __init__(self, cfg: LanceTextConfig):
        super().__init__()
        self._has_moe_gen = cfg.has_moe_gen
        self.self_attn = LanceAttention(cfg)

        self.mlp = LanceMLP(cfg)
        self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

        if cfg.has_moe_gen:
            self.mlp_moe_gen = LanceMLP(cfg)
            self.input_layernorm_moe_gen = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
            self.post_attention_layernorm_moe_gen = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        position_ids: mx.array,
        mask: Optional[mx.array] = None,
        cache=None,
        gen_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Pre-norm transformer block with optional MoE-gen routing.

        Layout split is per-token, applied independently at:
          - input_layernorm  / input_layernorm_moe_gen
          - self_attn (its own routing inside)
          - post_attention_layernorm / post_attention_layernorm_moe_gen
          - mlp / mlp_moe_gen
        """
        # Same predicate pattern as LanceAttention — single source of truth.
        routed = self._has_moe_gen and gen_mask is not None

        # ----- input_layernorm (pre-attn) -----
        if routed:
            normed_und = self.input_layernorm(x)
            normed_gen = self.input_layernorm_moe_gen(x)
            normed = _route_moe(normed_und, normed_gen, gen_mask, broadcast_dims=1)
        else:
            normed = self.input_layernorm(x)

        h = x + self.self_attn(normed, position_ids, mask, cache, gen_mask=gen_mask)

        # ----- post_attention_layernorm + MLP -----
        if routed:
            pa_und = self.post_attention_layernorm(h)
            pa_gen = self.post_attention_layernorm_moe_gen(h)
            pa = _route_moe(pa_und, pa_gen, gen_mask, broadcast_dims=1)
            mlp_und = self.mlp(pa)
            mlp_gen = self.mlp_moe_gen(pa)
            mlp_out = _route_moe(mlp_und, mlp_gen, gen_mask, broadcast_dims=1)
        else:
            mlp_out = self.mlp(self.post_attention_layernorm(h))

        return h + mlp_out


# ----------------------------------------------------------------------------
# Stack of decoder layers — module name `model` to match the checkpoint key
# prefix `language_model.model.layers.*`.
# ----------------------------------------------------------------------------
class LanceQwen2Model(nn.Module):
    def __init__(self, cfg: LanceTextConfig):
        super().__init__()
        self.cfg = cfg
        self._has_moe_gen = cfg.has_moe_gen
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = [LanceDecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        # Lance pairs the *final* RMSNorm too: UND tokens go through
        # `norm`, GEN tokens through `norm_moe_gen`.  STAGE 2 only calls
        # `norm`; the sibling is held for strict-load.
        if cfg.has_moe_gen:
            self.norm_moe_gen = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def __call__(
        self,
        input_ids: Optional[mx.array],
        position_ids: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[list] = None,
        gen_mask: Optional[mx.array] = None,
    ) -> mx.array:
        h = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        if cache is None:
            cache = [None] * len(self.layers)
        if mask is None:
            # Fallback for callers that don't supply an explicit mask
            # (text-only AR decode, unit tests).  Production T2I/T2V/TI2I
            # pipelines always pass an mx.array mask from
            # `build_lance_attention_mask` (text=causal, vision=full or
            # noise per SequenceLayout), so this branch fires only for the
            # dev paths.
            mask = "causal"
        for layer, c in zip(self.layers, cache):
            h = layer(h, position_ids, mask, c, gen_mask=gen_mask)

        # Final norm is paired too: UND→`norm`, GEN→`norm_moe_gen`.
        routed = self._has_moe_gen and gen_mask is not None
        if routed:
            n_und = self.norm(h)
            n_gen = self.norm_moe_gen(h)
            return _route_moe(n_und, n_gen, gen_mask, broadcast_dims=1)
        return self.norm(h)


# ----------------------------------------------------------------------------
# LM head + logits.  `lm_head.weight` is physically present in the Lance
# checkpoint (PT *and* MLX), but per llm_config.json `tie_word_embeddings=true`.
# We honour PT's storage (load lm_head as a real Linear) and skip the
# explicit tie at runtime — this is the path that matched RockTalk
# byte-for-byte at STAGE 1.
# ----------------------------------------------------------------------------
class LanguageModel(nn.Module):
    """Mirrors PT `Qwen2ForCausalLM`: contains `.model` (the transformer)
    and `.lm_head` (vocab projection).  Strict-load consumes keys under
    `model.*` and `lm_head.*`.
    """
    def __init__(self, cfg: LanceTextConfig):
        super().__init__()
        self.cfg = cfg
        self.model = LanceQwen2Model(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: Optional[mx.array],
        position_ids: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[list] = None,
        gen_mask: Optional[mx.array] = None,
    ) -> mx.array:
        h = self.model(input_ids, position_ids, mask=mask, cache=cache,
                       gen_mask=gen_mask)
        return self.lm_head(h)


# ----------------------------------------------------------------------------
# Top-level Lance object.  STAGE 2 ships *only* `language_model`; later
# stages will attach `vae2llm`, `llm2vae`, `time_embedder`, `latent_pos_embed`,
# the connector and the ViT here.
#
# The attribute name `language_model` is the key prefix in the safetensors:
# `language_model.model.layers.*` → `self.language_model.model.layers[i].*`.
# ----------------------------------------------------------------------------
class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep frequency → 2-layer MLP → hidden_size.

    Module tree matches RockTalk MLX keys `fc1` / `fc2` (PT stored as
    `nn.Sequential(Linear, SiLU, Linear)` with indices 0/2; renamed at
    STAGE 1 conversion).
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.fc1 = nn.Linear(frequency_embedding_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True)

    @staticmethod
    def timestep_embedding(t: mx.array, dim: int,
                           max_period: float = 10_000.0) -> mx.array:
        """Sinusoidal embedding matching DiT/PT impl exactly.

        t: (N,) float — fractional timesteps allowed.
        Returns: (N, dim).
        """
        half = dim // 2
        freqs = mx.exp(
            -mx.log(mx.array(max_period))
            * mx.arange(0, half, dtype=mx.float32) / half
        )
        args = t[:, None].astype(mx.float32) * freqs[None, :]
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        if dim % 2:
            emb = mx.concatenate([emb, mx.zeros((emb.shape[0], 1))], axis=-1)
        return emb

    def __call__(self, t: mx.array) -> mx.array:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        h = self.fc1(t_freq)
        h = nn.silu(h)
        return self.fc2(h)


class PositionEmbedding3D(nn.Module):
    """3D latent position embedding — frozen sin-cos table at training, stored
    verbatim in the checkpoint.  Forward is a flat lookup.

    Table shape: `(max_num_latent_frames * max_latent_size**2, hidden_size)`
    = (1 * 64² = 4096, 2048) for Lance-3B image variant
    (`checkpoints/Lance-3B-MLX/config.json` → max_num_latent_frames=1,
    max_latent_size=64).  The LanceConfig PT default `max_latent_size=32`
    would mean (4 * 32² = 4096, …) — same table size but *different
    interpretation*: position index becomes h*32+w instead of h*64+w.
    Mismatch silently scrambles the lookup; STAGE 6 first-image debug
    found this via image artifacts (horizontal banding) after step-cos
    misleadingly passed (compare used wrong same-value on both sides).
    """
    def __init__(self, max_num_latent_frames: int = 1,
                 max_latent_size: int = 64, hidden_size: int = 2048):
        super().__init__()
        self.max_num_latent_frames = max_num_latent_frames
        self.max_latent_size = max_latent_size
        n = max_num_latent_frames * (max_latent_size ** 2)
        self.pos_embed = mx.zeros((n, hidden_size))   # replaced by load_weights

    def __call__(self, position_ids: mx.array) -> mx.array:
        return self.pos_embed[position_ids]


class LanceLLM(nn.Module):
    def __init__(self, cfg: LanceTextConfig):
        super().__init__()
        self.language_model = LanguageModel(cfg)
        # ---- adapters (STAGE 6) -------------------------------------------------
        # VAE-latent (z_dim=48 per Lance_3B) ↔ LLM hidden (2048).  These two
        # Linears bridge the diffusion latent space to the transformer.
        self.vae2llm = nn.Linear(48, cfg.hidden_size, bias=True)
        self.llm2vae = nn.Linear(cfg.hidden_size, 48, bias=True)
        # Timestep + 3D position embeddings — added to VAE-latent embeds
        # before the GEN slab enters the transformer.
        self.time_embedder = TimestepEmbedder(cfg.hidden_size)
        self.latent_pos_embed = PositionEmbedding3D()

    def __call__(
        self,
        input_ids: mx.array,
        position_ids: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[list] = None,
        gen_mask: Optional[mx.array] = None,
    ) -> mx.array:
        return self.language_model(input_ids, position_ids, mask=mask, cache=cache,
                                   gen_mask=gen_mask)


# ----------------------------------------------------------------------------
# Strict-load helper.  Filters incoming weight dict to the set of keys this
# STAGE 2 model owns (`language_model.*`), refuses load if anything is
# missing or extra in that scope, and surfaces the count of skipped outer
# keys (adapters etc.) for later stages.
# ----------------------------------------------------------------------------
# Known outer adapters present in the Lance MLX checkpoint as of STAGE 1
# inspection.  STAGE 5/6 will reshape this set when those adapters get
# their own model classes; until then a drift here means either the
# checkpoint or our expectations are out of date.
_EXPECTED_OUTER_KEYS: frozenset[str] = frozenset({
    "latent_pos_embed.pos_embed",
    "llm2vae.bias", "llm2vae.weight",
    "vae2llm.bias", "vae2llm.weight",
    "time_embedder.fc1.bias",  "time_embedder.fc1.weight",
    "time_embedder.fc2.bias",  "time_embedder.fc2.weight",
})


def load_text_backbone(
    model: LanceLLM,
    path: str,
    *,
    allow_extra_outer: bool = True,
) -> dict:
    """Load Lance checkpoint into `model` (an instance of LanceLLM).

    Returns a small stats dict.  Raises if any `language_model.*` key is
    missing or extra; non-language_model keys are reported but not loaded
    unless they map to an attribute on `model`.
    """
    from mlx.utils import tree_flatten
    all_w = mx.load(path)

    all_ours = set(dict(tree_flatten(model.parameters())).keys())
    # STAGE 6+: model may carry adapter params (vae2llm / llm2vae /
    # time_embedder / latent_pos_embed).  This helper only loads the
    # `language_model.*` slice — scope our membership check the same way.
    ours = {k for k in all_ours if k.startswith("language_model.")}
    text_keys = {k for k in all_w if k.startswith("language_model.")}
    outer_keys = {k for k in all_w if not k.startswith("language_model.")}

    missing = ours - text_keys
    extra   = text_keys - ours
    if missing or extra:
        raise RuntimeError(
            f"strict-load mismatch: missing={sorted(missing)[:5]}... ({len(missing)}), "
            f"extra={sorted(extra)[:5]}... ({len(extra)})"
        )
    if outer_keys and not allow_extra_outer:
        raise RuntimeError(f"unexpected outer keys: {sorted(outer_keys)[:5]}...")
    # Informational drift check against the STAGE-1 baseline.  Don't raise
    # — STAGE 5/6 will legitimately extend this set — but loud-print so a
    # silent checkpoint change is noticeable.
    drift_in  = outer_keys - _EXPECTED_OUTER_KEYS
    drift_out = _EXPECTED_OUTER_KEYS - outer_keys
    if drift_in or drift_out:
        print(f"[load] outer-key drift vs baseline: "
              f"+{sorted(drift_in)} -{sorted(drift_out)}")

    # We've already explicitly verified the `language_model.*` slice
    # matches exactly (`missing`/`extra` above).  Use strict=False here
    # so adapter parameters on the model (STAGE 6+) — which we
    # intentionally do *not* load via this helper — don't trip MLX's
    # global strict check.
    model.load_weights([(k, all_w[k]) for k in text_keys], strict=False)
    mx.eval(model.parameters())   # materialize
    return {
        "loaded_keys": len(text_keys),
        "skipped_outer_keys": len(outer_keys),
        "outer_sample": sorted(outer_keys),
    }


# LanceLLM adapter modules — pinned by name because `load_full_lance` strict-
# load relies on exact attribute presence.  Renaming any of these silently
# breaks the missing/extra-key diagnostic into an opaque "checkpoint has key X
# we don't model"; this constant gives a clearer failure mode early in
# LanceLLM construction.
_ADAPTER_ATTRS: tuple[str, ...] = ("vae2llm", "llm2vae", "time_embedder", "latent_pos_embed")


def load_full_lance(model: LanceLLM, path: str) -> dict:
    """STAGE 6+: strict-load *all* Lance MLX keys including the adapters.
    Use when the model object has the adapter modules attached.

    Failure modes — any of these raise:
      - language_model.* key missing or extra
      - any param tensor not covered by the checkpoint
      - adapter attribute missing on the model (caught here, not in raw load)

    Also prints an outer-key drift line vs the STAGE-1 baseline
    (`_EXPECTED_OUTER_KEYS`) so a future checkpoint that adds/drops an
    adapter is noticeable before the strict-load swallows it.
    """
    from mlx.utils import tree_flatten

    # Early sanity: model must expose the expected adapter modules.
    for attr in _ADAPTER_ATTRS:
        if not hasattr(model, attr):
            raise AttributeError(
                f"LanceLLM missing adapter attribute {attr!r} — load_full_lance "
                f"requires all of {_ADAPTER_ATTRS}.  Did the LanceLLM definition "
                f"drift?"
            )

    all_w = mx.load(path)
    ours = set(dict(tree_flatten(model.parameters())).keys())
    ckpt = set(all_w.keys())

    missing_in_ckpt = ours - ckpt
    extra_in_ckpt   = ckpt - ours
    if missing_in_ckpt:
        raise RuntimeError(
            f"model needs keys not in checkpoint: {sorted(missing_in_ckpt)[:5]}..."
        )
    if extra_in_ckpt:
        raise RuntimeError(
            f"checkpoint has keys we don't model: {sorted(extra_in_ckpt)[:5]}..."
        )

    # Outer-key drift print (informational; strict-load above already
    # enforces exact set, but this surfaces the *delta* against the
    # STAGE-1 baseline if STAGE 7+ adds a new adapter).
    outer_keys = {k for k in all_w if not k.startswith("language_model.")}
    drift_in  = outer_keys - _EXPECTED_OUTER_KEYS
    drift_out = _EXPECTED_OUTER_KEYS - outer_keys
    if drift_in or drift_out:
        print(f"[load] outer-key drift vs baseline: "
              f"+{sorted(drift_in)} -{sorted(drift_out)}")

    model.load_weights(list(all_w.items()), strict=True)
    mx.eval(model.parameters())
    return {
        "loaded_keys": len(all_w),
        "language_model_keys": sum(1 for k in all_w if k.startswith("language_model.")),
        "adapter_keys": sum(1 for k in all_w if not k.startswith("language_model.")),
    }
