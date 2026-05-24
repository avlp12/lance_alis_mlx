"""Mixture-of-Tokens routing helpers for Lance.

Lance is NOT a classic top-k expert MoE.  Every block ships *two parallel
sets of weights* (the canonical Qwen2.5-VL set and a `_moe_gen` set).
Routing is **deterministic per token type, no gating network**:

  - UND tokens  (text / ViT image-understanding) → canonical weights
  - GEN tokens  (VAE-latent placeholders during T2I/T2V/TI2I)
                                                 → `_moe_gen` weights

This naming is RockTalk's "Mixture-of-Tokens".  Reza2kn calls the same
structure "Mixture-of-Tasks".  Same routing, two names — internal docs
elsewhere may use either.

The PT reference (`refs/Lance/modeling/lance/qwen2_navit.py`) implements
this via index-scatter on packed 1-D sequences:

    out[und_idx] = q_proj(x[und_idx])
    out[gen_idx] = q_proj_moe_gen(x[gen_idx])

We instead compute both branches on the full sequence and merge with
`mx.where` driven by a boolean mask.  Trades the per-tensor 2× FLOPs in
the projection/norm ops for clean, GPU-friendly code.  STAGE 8/9 may
revisit this if profiling shows it's the bottleneck (logged as a
candidate improvement).

What's affected per layer
-------------------------
Every decoder layer's *projections* and *norms* split by mask:
  - self_attn.q_proj / q_proj_moe_gen
  - self_attn.k_proj / k_proj_moe_gen
  - self_attn.v_proj / v_proj_moe_gen
  - self_attn.o_proj / o_proj_moe_gen
  - self_attn.q_norm / q_norm_moe_gen      (qk_norm: load-bearing — see v2.1)
  - self_attn.k_norm / k_norm_moe_gen
  - mlp / mlp_moe_gen                       (full Qwen2 SwiGLU, both branches)
  - input_layernorm / input_layernorm_moe_gen
  - post_attention_layernorm / post_attention_layernorm_moe_gen

Plus the *outer* final norm:
  - model.norm / model.norm_moe_gen

The attention operation *itself* (sdpa) is NOT split — the full mixed
sequence attends to itself.  Only the per-token projection / norm steps
diverge.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class SequenceLayout:
    """Describes which slabs of a packed sequence are UND vs GEN.

    Lance's text-to-image sequence is always one contiguous GEN slab at
    the end (the VAE-latent placeholders).  TI2I has *one* GEN slab too
    (the noise target), with the ViT placeholders and cond-VAE
    placeholders both routed UND.

    For STAGE 4 verification we'll synthesise short sequences with
    arbitrary slab placement, hence the generic interface.
    """
    seq_len: int
    # List of (start, end) half-open intervals over [0, seq_len) marking
    # the GEN slabs.  All other positions are UND.
    gen_spans: list[tuple[int, int]]


def build_gen_mask(layout: SequenceLayout, batch: int = 1) -> mx.array:
    """Return a `(B, L)` bool mask, True where the token routes through
    the `_moe_gen` weights, False where it routes through canonical.
    """
    L = layout.seq_len
    if not layout.gen_spans:
        return mx.zeros((batch, L), dtype=mx.bool_)
    cols = mx.arange(L)
    mask = mx.zeros((L,), dtype=mx.bool_)
    for start, end in layout.gen_spans:
        if not (0 <= start <= end <= L):
            raise ValueError(f"gen span ({start},{end}) out of [0,{L}]")
        if start == end:
            continue
        slab = (cols >= start) & (cols < end)
        mask = mask | slab
    return mx.broadcast_to(mask[None, :], (batch, L))


def route(canonical: mx.array, gen: mx.array, gen_mask: mx.array,
          broadcast_dims: int = 0) -> mx.array:
    """Merge two per-token branches via `mx.where`.

    `canonical` and `gen` are same-shape tensors (one branch each).
    `gen_mask` is `(B, L)` bool; we add trailing 1-axes so it broadcasts
    against the channel/head dims at the end of `canonical`'s shape.
    """
    # Expand gen_mask from (B, L) → (B, L, 1, 1, ...) to match canonical.
    expanded = gen_mask
    for _ in range(broadcast_dims):
        expanded = expanded[..., None]
    return mx.where(expanded, gen, canonical)
