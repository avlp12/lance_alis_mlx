"""Lance attention mask builder.

Mirrors `refs/Lance/data/data_utils.py:create_sparse_mask` — text spans get
causal masking, latent/noise slabs get bidirectional masking within the
slab, and crossing a *noise* slab from outside it is blocked.

Algorithm (PT line-by-line, materialised on a token grid instead of via
flex_attention's predicate composition):

    for each split i with (length, mode):
        full_and_noise_seq_id[i_range] = i if mode in ("full","noise") else -1
        noise_seq_id        [i_range] = i if mode == "noise" else -1
        document_id        [i_range] = doc_idx + 1

    causal_mask        = q >= k
    full_and_noise_mask = same_slab(q, k) AND slab_id >= 0
    remove_noise_mask  = NOT (noise(k) AND noise(q) != noise(k))
    sample_mask        = same_document(q, k)

    final = (causal OR full_and_noise) AND remove_noise AND sample_mask

This module is verified bit-equivalent to PT's `create_sparse_mask`
predicate on the (L, L) grid — see `tools/stage6_pt_denoise_compare.py`.
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx


def build_lance_attention_mask(
    seq_len: int,
    split_lens: list[int],
    attn_modes: list[str],
    document_lens: list[int] | None = None,
) -> mx.array:
    """Build an (L, L) additive mask (0 where attend, -inf where blocked).

    Args mirror PT:
      - `split_lens` / `attn_modes` describe the sequence partition.
        attn_modes uses {"causal", "full", "noise", "full_noise", "full_noise_target"}.
        `full_noise` / `full_noise_target` are aliased to `full` per PT
        `process_attention_mask` (lance.py:152).
      - `document_lens`: list of per-sample lengths.  Default = single
        document spanning the whole sequence.

    Returns `(seq_len, seq_len)` MLX additive mask (float32).
    """
    assert seq_len > 0, "seq_len must be positive"
    assert split_lens, "split_lens must be non-empty"
    assert sum(split_lens) == seq_len, (
        f"split_lens sum {sum(split_lens)} != seq_len {seq_len}"
    )
    if document_lens is None:
        document_lens = [seq_len]
    assert sum(document_lens) == seq_len

    # PT line 152: full_noise variants → full
    attn_modes_ = ["full" if m in ("full_noise", "full_noise_target") else m
                   for m in attn_modes]

    # Build per-token slab/doc id arrays (PT lines 147-159).
    full_and_noise = np.full(seq_len, -1, dtype=np.int64)
    noise         = np.full(seq_len, -1, dtype=np.int64)
    cursor = 0
    for i, (length, mode) in enumerate(zip(split_lens, attn_modes_)):
        if mode in ("full", "noise"):
            full_and_noise[cursor:cursor + length] = i
        if mode == "noise":
            noise[cursor:cursor + length] = i
        cursor += length

    document_id = np.concatenate(
        [np.full(l, doc_idx + 1, dtype=np.int64)
         for doc_idx, l in enumerate(document_lens)]
    )

    # Grid evaluation (PT predicates, vectorised).
    q = np.arange(seq_len, dtype=np.int64)[:, None]    # (L, 1)
    k = np.arange(seq_len, dtype=np.int64)[None, :]    # (1, L)
    causal_mask        = q >= k
    full_and_noise_mask = (full_and_noise[q] == full_and_noise[k]) & (full_and_noise[q] >= 0)
    remove_noise_mask  = ~((noise[k] >= 0) & (noise[q] != noise[k]))
    sample_mask        = document_id[q] == document_id[k]
    attend = (causal_mask | full_and_noise_mask) & remove_noise_mask & sample_mask

    # Additive mask for sdpa: 0 where attend, -inf where blocked.
    add = np.where(attend, 0.0, -np.inf).astype(np.float32)
    return mx.array(add)
