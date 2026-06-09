"""3D mRoPE position-ID generation for Lance/Qwen2.5-VL.

The Qwen2.5-VL family uses *multimodal* RoPE: position ids are 3D
`(T, H, W)` rather than scalar.  For pure-text tokens, the three rows
are identical (just the sequence index); for image / video tokens, the
H- and W- rows carry the patch grid coordinates so the rotary
embedding can encode 2D / 3D spatial relations.

Lance adds two twists on top:

1. **Two kinds of image tokens** — ViT (UND) tokens for image
   understanding, and VAE-latent (GEN/noise) tokens for image
   generation.  They use the *same* mRoPE machinery but the GEN slab is
   shifted in absolute position by `pos_shift` so attention can tell
   them apart (PT Lance uses this in TI2I to keep cond-VAE-latent
   distinct from noise-VAE-latent).
2. **Latent grid is independent of pixel grid.** VAE downsamples 8× in
   pixel space and the latent patch size is `(pt=1, ph=2, pw=2)`, so a
   256×256 image lands at 16×16 latent positions.  STAGE 3 builds
   `(t, h, w)` from the latent dims and offsets by `pos_shift`.

STAGE 3 scope: pure positional layout.  Applying the rotary embedding
itself lives in `backbone.Qwen2RotaryEmbedding`.  Verification: produce
position_ids byte-identical to mlx-vlm `LanguageModel.get_rope_index`
on text-only and text+image inputs.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class VisionSpec:
    """How a single image lands in the LLM's vision slab.

    `t`, `h`, `w` are the patch-grid sizes *after* the LLM-side spatial
    merge (i.e. `vit_h // spatial_merge_size`).  `start` is the index
    of `<vision_start>` in the token sequence; `length` is the number
    of placeholder tokens between `<vision_start>` and `<vision_end>`
    (inclusive of neither tag) — should equal `t * h * w`.
    """
    start: int           # absolute index of <vision_start>
    length: int          # number of placeholder image tokens
    t: int               # post-merge temporal grid
    h: int               # post-merge spatial-H grid
    w: int               # post-merge spatial-W grid
    # Temporal mRoPE step = second_per_grid_t · tokens_per_second.  PT Lance
    # `get_rope_index` (qwen2_navit.py:1258) scales the temporal axis by this on
    # the VIDEO branch (=2 for Lance: second_per_grid_t 1.0 × tokens_per_second 2);
    # the generation path already does it (t2v.py:112).  Default 1 = image / T=1
    # (where arange(1)·k = [0] regardless) — keeps x2t-image / image_edit / the
    # STAGE-3 checks byte-identical.  x2t-video MUST pass 2 (STAGE 11 fix).
    temporal_scale: int = 1


def text_positions(seq_len: int, batch: int = 1, dtype=mx.int32) -> mx.array:
    """Pure-text mRoPE positions: (3, B, L), all three rows identical.

    For a text token at index `i`, all three mRoPE axes see position
    `i` — RoPE collapses to standard 1D rotary in this case.
    """
    pos = mx.arange(seq_len, dtype=dtype).reshape(1, -1)              # (1, L)
    pos = mx.broadcast_to(pos, (batch, seq_len))                      # (B, L)
    return mx.broadcast_to(pos[None, ...], (3, batch, seq_len))       # (3, B, L)


# ---------------------------------------------------------------------------
# Image patch grid → 3-row position fragment.
#
# Inside the `<vision_start> … <vision_end>` span of length `t*h*w`, every
# token gets its own (T, H, W) position triple — flattened in
# (t, h, w) row-major.  Outside that span text positions resume from
# `max(positions inside) + 1` to keep the global counter contiguous.
# ---------------------------------------------------------------------------
def _image_position_block(t: int, h: int, w: int,
                          base: int = 0, *, temporal_scale: int = 1,
                          dtype=mx.int32) -> mx.array:
    """Return shape (3, t*h*w) — the position triples for one image's
    placeholder tokens.  `base` is the position the *first placeholder*
    will receive on all three rows: i.e. the i-th placeholder sits at
    `base + (t_i·temporal_scale, h_i, w_i)`.  The caller is responsible for pre-
    incrementing its cursor past the `<vision_start>` token before
    calling.  Iteration order is (t, h, w) row-major.

    `temporal_scale` matches PT `get_rope_index`'s
    `time = arange(t)·second_per_grid_t·tokens_per_second` (=2 for Lance video).
    Default 1; at t=1 it is a no-op (arange(1)·k = [0]).
    """
    t_idx = (mx.arange(t, dtype=dtype) * temporal_scale).reshape(t, 1, 1)  # (t,1,1)
    h_idx = mx.arange(h, dtype=dtype).reshape(1, h, 1)               # (1,h,1)
    w_idx = mx.arange(w, dtype=dtype).reshape(1, 1, w)               # (1,1,w)

    T = mx.broadcast_to(t_idx, (t, h, w)).flatten()                  # (t*h*w,)
    H = mx.broadcast_to(h_idx, (t, h, w)).flatten()
    W = mx.broadcast_to(w_idx, (t, h, w)).flatten()

    block = mx.stack([T, H, W], axis=0) + base                       # (3, t*h*w)
    return block


def build_positions_for_layout(
    seq_len: int,
    vision_spans: list[VisionSpec],
    *,
    dtype=mx.int32,
) -> mx.array:
    """Construct (3, 1, seq_len) mRoPE positions for a sequence that
    interleaves text and image-placeholder runs.

    Algorithm (matches mlx-vlm `get_rope_index` for the text+image
    case):
      cursor = 0 in the *output* position counter.
      For each text run before / between / after image spans:
        - assign `cursor, cursor+1, ...` to each of the three rows
          (text positions are scalar — all three axes equal).
        - cursor advances by the run length.
      For each image span (vision_start ... vision_end inclusive of tags):
        - vision_start token: pos = cursor (same on all three rows)
        - cursor += 1
        - placeholder tokens of length t*h*w: pos = cursor + image_block(t,h,w)
          (so the first placeholder is at cursor + (0,0,0), last is at
          cursor + (t-1, h-1, w-1)); cursor advances by max(t,h,w)
          (NOT by t*h*w — the LLM-side "position" runs over the grid
          *extent*, not over the count of placeholders, so neighbouring
          modalities share the right corner of the previous grid).
        - vision_end token: pos = cursor (same on all three rows)
        - cursor += 1
    """
    # We'll fill an (3, seq_len) buffer in-place via concatenated slices.
    rows: list[mx.array] = []  # each piece is (3, len_i)
    text_cursor = 0
    next_token_idx = 0
    for span in sorted(vision_spans, key=lambda v: v.start):
        # Text run before the span
        text_len_before = span.start - next_token_idx
        if text_len_before > 0:
            text_block = mx.arange(text_cursor, text_cursor + text_len_before,
                                   dtype=dtype).reshape(1, -1)        # (1, len)
            text_block = mx.broadcast_to(text_block, (3, text_len_before))
            rows.append(text_block)
            text_cursor += text_len_before
            next_token_idx += text_len_before

        # <vision_start> token (scalar position = text_cursor)
        vs_pos = mx.full((3, 1), text_cursor, dtype=dtype)
        rows.append(vs_pos)
        text_cursor += 1
        next_token_idx += 1

        # Placeholder tokens (length = t*h*w)
        block = _image_position_block(span.t, span.h, span.w,
                                      base=text_cursor,
                                      temporal_scale=span.temporal_scale,
                                      dtype=dtype)                    # (3, t*h*w)
        rows.append(block)
        # Advance past the bounding grid: next token sits at max-position + 1.
        # The temporal extent is (t-1)·temporal_scale (PT scales the time axis),
        # so the advance is max((t-1)·scale, h-1, w-1) + 1.  At scale=1 this is
        # exactly max(t,h,w) (the old behaviour) — image / T=1 unchanged.
        text_cursor += max((span.t - 1) * span.temporal_scale,
                           span.h - 1, span.w - 1) + 1
        next_token_idx += span.length

        # <vision_end> token (scalar position = text_cursor)
        ve_pos = mx.full((3, 1), text_cursor, dtype=dtype)
        rows.append(ve_pos)
        text_cursor += 1
        next_token_idx += 1

    # Trailing text after the last image (or the whole sequence if no images)
    trailing = seq_len - next_token_idx
    if trailing > 0:
        text_block = mx.arange(text_cursor, text_cursor + trailing,
                               dtype=dtype).reshape(1, -1)
        text_block = mx.broadcast_to(text_block, (3, trailing))
        rows.append(text_block)

    positions = mx.concatenate(rows, axis=-1)                        # (3, seq_len)
    assert positions.shape[-1] == seq_len, (positions.shape, seq_len)
    return positions[:, None, :]                                     # (3, 1, L)


# ---------------------------------------------------------------------------
# Lance pos_shift — used to keep VAE-latent (GEN/noise) tokens position-
# distinguishable from ViT (UND) tokens that may also live in the
# sequence.  PT Lance computes a fixed safety constant
#
#     pos_shift = max_latent_size^2 * max_num_latent_frames + 1024
#
# (e.g. 32 * 32 * 7 + 1024 = 8192 for the Lance_3B/T2V defaults) and adds
# it to every position triple in the VAE-latent slab.  STAGE 3 ships the
# constant + a shifter; STAGE 6/7 will decide *which* VisionSpec gets the
# shift applied when it assembles a TI2I sequence (cond-VAE vs noise-VAE).
# ---------------------------------------------------------------------------
def lance_pos_shift(max_latent_size: int, max_num_latent_frames: int,
                    safety: int = 1024) -> int:
    """Match PT Lance: `Lance.pos_shift` in lance.py:112."""
    return max_latent_size * max_latent_size * max_num_latent_frames + safety


def shift_positions(positions: mx.array, shift: int,
                    *, col_start: int, col_end: int) -> mx.array:
    """Add `shift` to all three rows of `positions[:, :, col_start:col_end]`.

    Used to mark a contiguous span (e.g. the VAE-latent slab inside a
    TI2I sequence) so attention sees those positions as numerically
    distinct from any ViT/text positions in the same sequence.
    Functional — returns a new array, does not mutate input.
    """
    L = positions.shape[-1]
    if not (0 <= col_start <= col_end <= L):
        raise ValueError(f"col range ({col_start},{col_end}) out of [0,{L}]")
    if col_start == col_end or shift == 0:
        return positions
    add = mx.zeros_like(positions)
    # Build a (1, 1, L) mask of which columns receive the shift, broadcast
    # to (3, B, L).  Avoids in-place slice update.
    cols = mx.arange(L)
    mask = ((cols >= col_start) & (cols < col_end)).astype(positions.dtype)  # (L,)
    add = mask[None, None, :] * mx.array(shift, dtype=positions.dtype)
    return positions + add
