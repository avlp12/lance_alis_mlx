"""STAGE 4 Tier 2: prove `mx.where`-merge ≡ PT-style index scatter.

The two routing implementations:

  PT scatter:
      out = zeros_like(...)
      out[und_idx] = W_canonical @ x[und_idx]
      out[gen_idx] = W_moe_gen   @ x[gen_idx]

  Ours (mx.where merge):
      out_und = W_canonical @ x          # all positions
      out_gen = W_moe_gen   @ x          # all positions
      out = where(gen_mask, out_gen, out_und)

These are mathematically equivalent: at each token position, exactly one
of the two weight matrices contributes, and `mx.where` selects per
position.  We prove it numerically on synthetic small weights and a
random sequence layout.  No PT environment needed.
"""
from __future__ import annotations

import mlx.core as mx

from lance_mlx.moe_gen import SequenceLayout, build_gen_mask, route


def _scatter_route(x: mx.array, W_c: mx.array, W_g: mx.array,
                   gen_mask: mx.array) -> mx.array:
    """PT-style: select inputs per position, apply different weights, scatter back.

    x:        (L, D)   batch dim folded for clarity
    W_c, W_g: (D, H)
    gen_mask: (L,) bool

    Implementation note: MLX supports `out[i] = v` in-place row assignment
    (verified: writing to row 2 of a zeros tensor and reading back returns
    the written value).  We rely on that, and assert at the end that *every*
    row received exactly one write — if MLX ever changes that semantics
    silently this asserts.
    """
    L, D = x.shape
    H = W_c.shape[1]
    out = mx.zeros((L, H), dtype=x.dtype)
    written = [False] * L
    # Build indices.
    und_idx = mx.array([i for i in range(L) if not bool(gen_mask[i].item())])
    gen_idx = mx.array([i for i in range(L) if bool(gen_mask[i].item())])
    # Apply per slab.
    if und_idx.size > 0:
        out_und = x[und_idx] @ W_c                  # (n_und, H)
        for i_local, i_global in enumerate(und_idx.tolist()):
            out[i_global] = out_und[i_local]
            written[i_global] = True
    if gen_idx.size > 0:
        out_gen = x[gen_idx] @ W_g
        for i_local, i_global in enumerate(gen_idx.tolist()):
            out[i_global] = out_gen[i_local]
            written[i_global] = True
    # Tripwire: every row must have been written exactly once.
    assert all(written), \
        f"scatter coverage gap — {sum(written)}/{L} rows written"
    # Tripwire 2: out must not be all-zero (defends against silent
    # in-place semantics regression in a future MLX version).
    assert float(mx.abs(out).max().item()) > 0, \
        "scatter wrote nothing detectable — MLX in-place semantics may have changed"
    return out


def _where_route(x: mx.array, W_c: mx.array, W_g: mx.array,
                 gen_mask: mx.array) -> mx.array:
    """Our impl style — both branches on full, merge with where."""
    out_c = x @ W_c
    out_g = x @ W_g
    return mx.where(gen_mask[:, None], out_g, out_c)


def main() -> None:
    mx.random.seed(0)

    # Three independent shapes to stress the equivalence.
    cases = [
        # (L, D, H, n_gen_spans)
        (32,  64, 64, 1),
        (128, 256, 128, 2),
        (97,  333, 211, 3),   # asymmetric, non-multiples
    ]

    all_ok = True
    for L, D, H, n_spans in cases:
        x  = mx.random.normal(shape=(L, D))
        Wc = mx.random.normal(shape=(D, H))
        Wg = mx.random.normal(shape=(D, H))

        # Random gen spans (non-overlapping)
        spans = []
        cursor = 0
        rng = mx.random.normal(shape=(2 * n_spans,))            # used only as a seed source
        for k in range(n_spans):
            gap = max(1, int(abs(rng[2*k].item()) * 5) + 1)
            length = max(1, int(abs(rng[2*k+1].item()) * 10) + 1)
            start = cursor + gap
            end = min(start + length, L)
            if start >= L:
                break
            spans.append((start, end))
            cursor = end
        if not spans:
            spans = [(L // 2, L)]
        layout = SequenceLayout(seq_len=L, gen_spans=spans)
        gm = build_gen_mask(layout, batch=1)[0]                  # (L,)

        ours    = _where_route(x, Wc, Wg, gm)
        scatter = _scatter_route(x, Wc, Wg, gm)

        diff = float(mx.abs(ours - scatter).max().item())
        cos = float(
            (mx.sum(ours.flatten() * scatter.flatten())
             / (mx.linalg.norm(ours) * mx.linalg.norm(scatter) + 1e-12)).item()
        )
        # f32 reduction order differs between scatter (per-slab matmul) and
        # where (full-sequence matmul, select per row).  Tolerate matmul
        # f32 precision floor; require cos≥1-1e-6 as the strong signal.
        ok = (diff < 1e-3) and (cos >= 1 - 1e-6)
        flag = "✓" if ok else "✗"
        print(f"  {flag} L={L:3d} D={D:3d} H={H:3d} spans={spans}  "
              f"max|Δ|={diff:.3e}  cos={cos:.9f}")
        all_ok = all_ok and ok

    print(f"\n[result] {'mx.where ≡ scatter (within f32 precision)' if all_ok else 'MISMATCH'}")


if __name__ == "__main__":
    main()
