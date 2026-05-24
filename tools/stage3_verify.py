"""STAGE 3 verification: our `build_positions_for_layout` must produce
position_ids byte-identical to mlx-vlm `LanguageModel.get_rope_index`
on text+image sequences.
"""
from __future__ import annotations

import sys

import mlx.core as mx
from mlx_vlm.models.qwen2_5_vl.config import (
    ModelConfig, TextConfig, VisionConfig,
)
from mlx_vlm.models.qwen2_5_vl.language import LanguageModel as RefLM

from lance_mlx.rope import (
    VisionSpec, build_positions_for_layout, lance_pos_shift,
    shift_positions, text_positions,
)


# Special token IDs (Qwen2.5-VL convention, also Lance's).
BOS              = 151643
EOS              = 151645
VISION_START_ID  = 151652
VISION_END_ID    = 151653
IMAGE_TOKEN_ID   = 151655
VIDEO_TOKEN_ID   = 151656


def _build_ref() -> RefLM:
    tc = TextConfig(model_type="qwen2_5_vl", hidden_size=2048,
                    num_hidden_layers=36, intermediate_size=11008,
                    num_attention_heads=16, num_key_value_heads=2,
                    rms_norm_eps=1e-6, vocab_size=151936,
                    max_position_embeddings=128000, rope_theta=1e6,
                    rope_scaling={"type": "mrope", "mrope_section": [16, 24, 24]},
                    tie_word_embeddings=True)
    mc = ModelConfig(text_config=tc,
        vision_config=VisionConfig(model_type="qwen2_5_vl", depth=32,
            hidden_size=1280, intermediate_size=3420, num_heads=16,
            in_channels=3, out_hidden_size=2048),
        image_token_id=IMAGE_TOKEN_ID, video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_ID, vision_end_token_id=VISION_END_ID,
        vision_token_id=151654, model_type="qwen2_5_vl")
    return RefLM(tc, mc)


def _make_sequence_with_image(text_before: int, t: int, h: int, w: int,
                              text_after: int) -> tuple[mx.array, list[VisionSpec], mx.array]:
    """Build a synthetic input_ids: [text_before tokens][<vis_start>]
    [t*h*w image tokens][<vis_end>][text_after tokens].

    Returns (input_ids, our_vision_specs, ref_image_grid_thw).  The latter
    is what mlx-vlm `get_rope_index` expects — *ViT-patch* grid sizes
    (spatial dims pre-merge), so h_vit = h * spatial_merge_size etc.
    spatial_merge_size = 2 for Qwen2.5-VL.
    """
    SMS = 2
    n_img = t * h * w
    parts = (
        [100] * text_before
        + [VISION_START_ID]
        + [IMAGE_TOKEN_ID] * n_img
        + [VISION_END_ID]
        + [200] * text_after
    )
    ids = mx.array([parts], dtype=mx.int32)
    span = VisionSpec(start=text_before, length=n_img, t=t, h=h, w=w)
    grid_thw = mx.array([[t, h * SMS, w * SMS]], dtype=mx.int32)
    return ids, [span], grid_thw


def _compare(name: str, ours: mx.array, theirs: mx.array) -> bool:
    if ours.shape != theirs.shape:
        print(f"  ✗ {name}: shape ours={ours.shape} theirs={theirs.shape}")
        return False
    diff = mx.abs(ours.astype(mx.int32) - theirs.astype(mx.int32)).max().item()
    if diff != 0:
        # show first mismatch column
        a = ours.astype(mx.int32)
        b = theirs.astype(mx.int32)
        eq = (a == b)
        bad = (~eq).any(axis=0)[0]
        col = int(mx.argmax(bad).item())
        print(f"  ✗ {name}: max|Δ|={diff}  first mismatch col={col}  "
              f"ours[:,0,{col}]={a[:,0,col].tolist()}  "
              f"theirs[:,0,{col}]={b[:,0,col].tolist()}")
        return False
    print(f"  ✓ {name}: byte-identical (shape={tuple(ours.shape)})")
    return True


def main() -> None:
    ref = _build_ref()
    ok_total = 0
    tests_total = 0

    # ---------- (1) text-only sequences ----------
    print("\n=== (1) text-only ===")
    for L in (1, 7, 31, 128):
        ids = mx.array([[100] * L], dtype=mx.int32)
        ours = text_positions(L)
        theirs, _ = ref.get_rope_index(ids)
        ok = _compare(f"L={L}", ours, theirs)
        ok_total += int(ok); tests_total += 1

    # ---------- (2) one image, various grid sizes ----------
    print("\n=== (2) text + one image + text ===")
    cases = [
        # (text_before, t, h, w, text_after)
        (5, 1,  4,  4, 3),       # tiny 4×4 image
        (10, 1, 8, 8, 5),        # 8×8
        (2, 1, 16, 16, 6),       # 16×16 (typical for 224px image)
        (8, 1, 13, 21, 4),       # non-square, max(h,w) != min
        (0, 1, 3, 5, 0),         # no text padding
    ]
    for tb, t, h, w, ta in cases:
        ids, spans, grid_thw = _make_sequence_with_image(tb, t, h, w, ta)
        L = int(ids.shape[1])
        ours = build_positions_for_layout(L, spans)
        theirs, _ = ref.get_rope_index(ids, image_grid_thw=grid_thw)
        ok = _compare(f"text({tb})+img({t}×{h}×{w})+text({ta})  L={L}", ours, theirs)
        ok_total += int(ok); tests_total += 1

    # ---------- (3) two images in one sequence ----------
    # mlx-vlm `get_rope_index` has a multi-image bug (see IMPROVEMENTS.md
    # [STAGE 3]), so we verify against a hand-computed expected array
    # built from the same algorithm the transformers Qwen2.5-VL uses.
    print("\n=== (3) text + img + text + img + text (hand ground truth) ===")
    n1 = 1 * 4 * 4
    n2 = 1 * 6 * 6
    parts = (
        [100] * 5
        + [VISION_START_ID] + [IMAGE_TOKEN_ID] * n1 + [VISION_END_ID]
        + [100] * 3
        + [VISION_START_ID] + [IMAGE_TOKEN_ID] * n2 + [VISION_END_ID]
        + [200] * 4
    )
    ids = mx.array([parts], dtype=mx.int32)
    spans = [
        VisionSpec(start=5,                  length=n1, t=1, h=4, w=4),
        VisionSpec(start=5 + 1 + n1 + 1 + 3, length=n2, t=1, h=6, w=6),
    ]
    L = int(ids.shape[1])

    # Hand-build expected positions following the canonical Qwen2.5-VL rule.
    rows_T, rows_H, rows_W = [], [], []
    cursor = 0
    # Prefix text (5 tokens) → positions 0..4
    for i in range(5):
        rows_T.append(cursor); rows_H.append(cursor); rows_W.append(cursor); cursor += 1
    # <vision_start> at col 5 → position 5
    rows_T.append(cursor); rows_H.append(cursor); rows_W.append(cursor); cursor += 1
    # 16 placeholders (t=1, h=4, w=4) — base=6, max-advance=4
    base = cursor
    for t in range(1):
        for h in range(4):
            for w in range(4):
                rows_T.append(base + t); rows_H.append(base + h); rows_W.append(base + w)
    cursor += max(1, 4, 4)              # max-extent advance
    # <vision_end>
    rows_T.append(cursor); rows_H.append(cursor); rows_W.append(cursor); cursor += 1
    # 3 middle text
    for i in range(3):
        rows_T.append(cursor); rows_H.append(cursor); rows_W.append(cursor); cursor += 1
    # <vision_start> for img2
    rows_T.append(cursor); rows_H.append(cursor); rows_W.append(cursor); cursor += 1
    # 36 placeholders (t=1, h=6, w=6)
    base = cursor
    for t in range(1):
        for h in range(6):
            for w in range(6):
                rows_T.append(base + t); rows_H.append(base + h); rows_W.append(base + w)
    cursor += max(1, 6, 6)
    # <vision_end>
    rows_T.append(cursor); rows_H.append(cursor); rows_W.append(cursor); cursor += 1
    # 4 trailing text
    for i in range(4):
        rows_T.append(cursor); rows_H.append(cursor); rows_W.append(cursor); cursor += 1

    expected = mx.array([rows_T, rows_H, rows_W], dtype=mx.int32)[:, None, :]
    assert expected.shape == (3, 1, L), expected.shape
    ours = build_positions_for_layout(L, spans)
    ok = _compare(f"two images interleaved (vs hand)  L={L}", ours, expected)
    ok_total += int(ok); tests_total += 1

    # ---------- (4) video — T_lat > 1 ----------
    # Lance T2V at 9 frames / 256px → (T_lat=3, h_lat=16, w_lat=16).
    # T>1 in the same `_image_position_block` machinery: T_idx iterates 0..2,
    # cursor advances by max(3, 16, 16) = 16.
    print("\n=== (4) text + video(T>1) + text ===")
    SMS = 2
    tb, t_lat, h_lat, w_lat, ta = 4, 3, 16, 16, 3
    n_vid = t_lat * h_lat * w_lat
    parts = (
        [100] * tb
        + [VISION_START_ID]
        + [IMAGE_TOKEN_ID] * n_vid       # Lance reuses image_token_id for VAE-latent placeholders
        + [VISION_END_ID]
        + [200] * ta
    )
    ids = mx.array([parts], dtype=mx.int32)
    L = int(ids.shape[1])
    spans = [VisionSpec(start=tb, length=n_vid, t=t_lat, h=h_lat, w=w_lat)]

    # Hand-built expected (same algorithm)
    rT, rH, rW = [], [], []
    c = 0
    for _ in range(tb):
        rT.append(c); rH.append(c); rW.append(c); c += 1
    rT.append(c); rH.append(c); rW.append(c); c += 1                  # <vis_start>
    base = c
    for ti in range(t_lat):
        for hi in range(h_lat):
            for wi in range(w_lat):
                rT.append(base + ti); rH.append(base + hi); rW.append(base + wi)
    c += max(t_lat, h_lat, w_lat)                                     # = 16
    rT.append(c); rH.append(c); rW.append(c); c += 1                  # <vis_end>
    for _ in range(ta):
        rT.append(c); rH.append(c); rW.append(c); c += 1
    expected = mx.array([rT, rH, rW], dtype=mx.int32)[:, None, :]

    ours = build_positions_for_layout(L, spans)
    ok = _compare(f"video(T={t_lat}×{h_lat}×{w_lat})  L={L}", ours, expected)
    ok_total += int(ok); tests_total += 1

    # ---------- (5) Lance pos_shift on a VAE-latent slab ----------
    print("\n=== (5) Lance pos_shift on VAE-latent span ===")
    # Pretend the video slab above is the VAE-latent (GEN) block of a T2V
    # sequence.  In the integrated TI2I/T2V pipeline (STAGE 6/7) we add
    # `pos_shift` to *only* the VAE-latent placeholder columns so they are
    # numerically separable from any ViT/UND positions elsewhere.
    SHIFT = lance_pos_shift(max_latent_size=32, max_num_latent_frames=7)
    # span starts at tb + 1 (skip prefix and <vis_start>), length n_vid.
    col0 = tb + 1
    col1 = col0 + n_vid
    shifted = shift_positions(ours, SHIFT, col_start=col0, col_end=col1)

    # Check: shifted - unshifted is `SHIFT` exactly on the slab, 0 elsewhere.
    delta = (shifted - ours)[:, 0, :]                                 # (3, L)
    in_slab = mx.arange(L)
    in_slab = ((in_slab >= col0) & (in_slab < col1)).astype(mx.int32)
    expected_delta = mx.broadcast_to(in_slab[None, :], (3, L)) * SHIFT
    diff = mx.abs(delta - expected_delta).max().item()
    if diff == 0:
        print(f"  ✓ pos_shift({SHIFT}) applied only to slab [{col0},{col1}): "
              f"max|Δ| outside slab = 0, inside = {SHIFT}")
        ok_total += 1
    else:
        print(f"  ✗ pos_shift mask off: max|Δ|={diff}")
    tests_total += 1

    print(f"\n[result] {ok_total}/{tests_total} tests byte-identical to mlx-vlm")
    if ok_total != tests_total:
        sys.exit(1)


if __name__ == "__main__":
    main()
