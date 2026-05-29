"""STAGE 9 §0 게이트 — MLX vs PT first-step velocity byte-diff.

PT fixture (`out/stage9_pt_video_*.npy`) 를 *입력*으로 그대로 사용 — MLX
측이 자체 생성하지 않음.  STAGE 7 변수 분리 패턴: 입력 동일, forward 만 비교.

Gate: cos(v_t_pt, v_t_mlx) ≥ 0.999
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, os.path.abspath("."))

from lance_mlx.backbone import (
    LanceLLM, LanceTextConfig, PositionEmbedding3D, load_full_lance,
)


PT_FIX_DIR = "out"
IMG_WEIGHTS = "checkpoints/Lance-3B-MLX/model.safetensors"
VID_SUP_PT  = "checkpoints/Lance/Lance_3B_Video/model_supplement.safetensors"

# Production video config — confirmed in §0 PT smoke
MAX_NUM_LATENT_FRAMES = 31
MAX_LATENT_SIZE = 64


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    print("=" * 70)
    print("STAGE 9 §0 GATE — MLX vs PT first-step velocity")
    print("=" * 70)

    # ---- load PT fixtures ----
    fx = {n: np.load(f"{PT_FIX_DIR}/stage9_pt_video_{n}.npy") for n in
          ("x_t_init", "text_ids", "vae_pos_ids", "current_pos_ids",
           "attn_mask", "v_t_step0")}
    with open(f"{PT_FIX_DIR}/stage9_pt_video_meta.json") as f:
        meta = json.load(f)

    L = int(meta["L_sequence"])
    noise_s, noise_e = meta["noise_span"]
    n_noise = noise_e - noise_s
    t_scalar = float(meta["timestep_first_step"])
    grid_thw = meta["video_grid_thw"]
    print(f"[fix] L={L}  noise_span=({noise_s},{noise_e})  n_noise={n_noise}  t={t_scalar}")
    print(f"[fix] grid_thw={grid_thw}  max_num_latent_frames(meta)={meta['max_num_latent_frames']}")
    print(f"[fix] PRNG seed={meta['prng_seed']}  source={meta['doctrine_source'][:80]}...")
    print(f"[fix] v_t_pt: shape={fx['v_t_step0'].shape} "
          f"||v||={np.linalg.norm(fx['v_t_step0']):.3f} "
          f"std={fx['v_t_step0'].std():.4f}")

    # ---- build MLX model (image config) + REPLACE latent_pos_embed for video ----
    print("\n[build] LanceLLM (image config) ...")
    cfg = LanceTextConfig()
    model = LanceLLM(cfg)

    print(f"[video-patch] replacing latent_pos_embed: image(1·64²=4096) → video(31·64²=126976) ...")
    img_shape = model.latent_pos_embed.pos_embed.shape
    print(f"  before: latent_pos_embed.pos_embed.shape = {img_shape}")
    model.latent_pos_embed = PositionEmbedding3D(
        max_num_latent_frames=MAX_NUM_LATENT_FRAMES,
        max_latent_size=MAX_LATENT_SIZE,
        hidden_size=cfg.hidden_size,
    )
    # ★ EXPLICIT assert (Lesson: STAGE 6 max_latent_size class)
    assert model.latent_pos_embed.max_num_latent_frames == MAX_NUM_LATENT_FRAMES, \
        f"video latent_pos_embed not rebuilt: max_num_latent_frames={model.latent_pos_embed.max_num_latent_frames}"
    assert model.latent_pos_embed.max_latent_size == MAX_LATENT_SIZE, \
        f"video latent_pos_embed wrong max_latent_size={model.latent_pos_embed.max_latent_size}"
    expected_shape = (MAX_NUM_LATENT_FRAMES * MAX_LATENT_SIZE * MAX_LATENT_SIZE, cfg.hidden_size)
    assert model.latent_pos_embed.pos_embed.shape == expected_shape, \
        f"shape {model.latent_pos_embed.pos_embed.shape} != expected {expected_shape}"
    print(f"  after:  latent_pos_embed.pos_embed.shape = {model.latent_pos_embed.pos_embed.shape}  ✓")
    print(f"  max_num_latent_frames={model.latent_pos_embed.max_num_latent_frames}  "
          f"max_latent_size={model.latent_pos_embed.max_latent_size}  ✓")

    # ---- load weights: image checkpoint backbone + video supplement override ----
    print("\n[load] image checkpoint (RockTalk MLX) + video supplement (RockTalk MLX, conversion-cross-check) ...")
    # Note: STAGE 8 doctrine — we use the *MLX* image checkpoint (already verified
    # byte-clean vs PT in STAGE 5/6/7) and the RockTalk video supplement (verified
    # byte-clean vs original PT supplement in STAGE 9 §0 doctrine-correction). The
    # PT fixture (v_t_pt) was generated from the *original PT* weights; if MLX
    # weights are faithful conversions of those, cos≥0.999 should hold.
    RT_VID_SUP = "checkpoints/Lance-3B-Video-MLX/model_supplement.safetensors"
    img_w = mx.load(IMG_WEIGHTS)
    sup_w = mx.load(RT_VID_SUP)
    merged = dict(img_w)
    override_ct = 0
    for k, v in sup_w.items():
        if k in merged and merged[k].shape != v.shape:
            override_ct += 1
        merged[k] = v
    print(f"  image keys: {len(img_w)}   supplement keys: {len(sup_w)}   merged: {len(merged)}")
    print(f"  overrides (shape diff): {override_ct} (expected 1 = latent_pos_embed.pos_embed)")
    # The strict load below will catch any extra/missing keys. Only video-only
    # `vit_model.*` keys (390) would be extras since the LanceLLM model doesn't
    # carry a ViT. Filter them out at load time.
    ours = set(dict(__import__("mlx").utils.tree_flatten(model.parameters())).keys())
    to_load = {k: v for k, v in merged.items() if k in ours}
    print(f"  filtering to model-owned keys: {len(to_load)} / {len(merged)} "
          f"(dropped {len(merged) - len(to_load)} vit_model.*)")

    # Now we MUST verify (a) every model param is covered, (b) the
    # latent_pos_embed override actually landed at (126976, 2048).
    missing = ours - set(to_load.keys())
    if missing:
        raise RuntimeError(f"strict-load: missing model params: {sorted(missing)[:5]}...")
    model.load_weights(list(to_load.items()), strict=True)
    mx.eval(model.parameters())
    # ★ EXPLICIT post-load shape assert
    loaded_shape = model.latent_pos_embed.pos_embed.shape
    assert loaded_shape == expected_shape, \
        f"after load latent_pos_embed.pos_embed.shape = {loaded_shape} (expected {expected_shape})"
    print(f"[load] post-load latent_pos_embed.pos_embed.shape = {loaded_shape}  ✓")
    # Numerical sanity: the table should look like sin/cos values, range ~[-1, 1]
    pe = np.asarray(model.latent_pos_embed.pos_embed)
    print(f"[load] latent_pos_embed stats: mean={pe.mean():+.4f} std={pe.std():.4f} "
          f"range=[{pe.min():+.4f}, {pe.max():+.4f}]")

    # ---- prepare MLX inputs (PT fixtures, NOT regenerated) ----
    text_ids = mx.array(fx["text_ids"].astype(np.int32))[None, :]   # (1, L)
    pos_ids  = mx.array(fx["current_pos_ids"].astype(np.int32))      # (3, 1, L)
    vae_pos_ids = mx.array(fx["vae_pos_ids"].astype(np.int32))       # (n_noise,)
    x_t = mx.array(fx["x_t_init"].astype(np.float32))                # (n_noise, 48)

    # attn_mask: PT fixture is bool L×L; convert to MLX additive form (0/-inf, f32).
    # STAGE 7 image_edit + MLX build_lance_attention_mask both use (L, L) shape — no broadcast.
    dense_bool = fx["attn_mask"]
    attn_add = np.where(dense_bool, 0.0, -np.inf).astype(np.float32)
    attn_mask = mx.array(attn_add)                                    # (L, L)

    print(f"\n[in] text_ids: {text_ids.shape} {text_ids.dtype}")
    print(f"[in] pos_ids: {pos_ids.shape} {pos_ids.dtype}")
    print(f"[in] vae_pos_ids: {vae_pos_ids.shape}  range=[{int(vae_pos_ids.min())}, {int(vae_pos_ids.max())}]")
    print(f"[in] x_t: {x_t.shape}  std={float(x_t.std()):.4f}")
    print(f"[in] attn_mask: {attn_mask.shape} (additive form, -inf where blocked)")

    # ---- MLX forward (with stage-by-stage diagnostic vs PT intermediates) ----
    print("\n[forward] MLX first-step velocity (with diag) ...")
    text_embed = model.language_model.model.embed_tokens(text_ids)   # (1, L, D)
    mx.eval(text_embed)

    # noise slab components — kept separate for diag
    t_arr = mx.array([t_scalar] * n_noise, dtype=mx.float32)
    vae2llm_out = model.vae2llm(x_t)                                  # (n_noise, D)
    time_embed_out = model.time_embedder(t_arr)                       # (n_noise, D)
    latent_pos_out = model.latent_pos_embed(vae_pos_ids)              # (n_noise, D)
    vae_embed_mlx = vae2llm_out + time_embed_out + latent_pos_out
    mx.eval(vae2llm_out, time_embed_out, latent_pos_out, vae_embed_mlx)

    # Splice
    embed = mx.concatenate([
        text_embed[:, :noise_s, :],
        vae_embed_mlx[None, :, :],
        text_embed[:, noise_e:, :],
    ], axis=1)
    mx.eval(embed)
    assert embed.shape == (1, L, cfg.hidden_size), f"embed shape {embed.shape}"

    # gen_mask: True at noise tokens
    cols = mx.arange(L)
    gen_mask = ((cols >= noise_s) & (cols < noise_e))[None, :]        # (1, L)

    # === stage-by-stage cos (only if PT diag dumps available) ===
    diag_dir = "out"
    if os.path.exists(f"{diag_dir}/stage9_pt_video_diag_text_embed.npy"):
        print("\n[diag] PT intermediate vs MLX (stage-by-stage cos)")
        diag_names = ["text_embed", "vae2llm_out", "time_embed_out", "latent_pos_out",
                      "vae_embed", "embed_pre_transformer"]
        mlx_arrs = {
            "text_embed":     np.asarray(text_embed[0], dtype=np.float32),
            "vae2llm_out":    np.asarray(vae2llm_out, dtype=np.float32),
            "time_embed_out": np.asarray(time_embed_out, dtype=np.float32),
            "latent_pos_out": np.asarray(latent_pos_out, dtype=np.float32),
            "vae_embed":      np.asarray(vae_embed_mlx, dtype=np.float32),
            "embed_pre_transformer": np.asarray(embed[0], dtype=np.float32),
        }
        for n in diag_names:
            pt_arr = np.load(f"{diag_dir}/stage9_pt_video_diag_{n}.npy")
            mlx_arr = mlx_arrs[n]
            c = cos(pt_arr, mlx_arr)
            norm_pt = float(np.linalg.norm(pt_arr))
            norm_mlx = float(np.linalg.norm(mlx_arr))
            ratio = norm_mlx / (norm_pt + 1e-30)
            flag = "★" if c < 0.999 else " "
            print(f"  {flag} {n:25s} cos={c:.6f}  ||pt||={norm_pt:>8.2f}  ||mlx||={norm_mlx:>8.2f}  ratio={ratio:.4f}")
    else:
        print("\n[diag] (no PT intermediate dumps — skip stage-by-stage)")

    # === single layer-0 diagnostic (only if PT dump available) ===
    if os.path.exists("out/stage9_pt_video_diag_hidden_layer0.npy"):
        print("\n[diag] layer-0 hidden cos (single-layer forward) ...")
        layer0 = model.language_model.model.layers[0]
        h0 = layer0(embed, pos_ids, attn_mask, None, gen_mask=gen_mask)
        mx.eval(h0)
        h0_np = np.asarray(h0[0], dtype=np.float32)
        h0_pt = np.load("out/stage9_pt_video_diag_hidden_layer0.npy")
        c0 = cos(h0_pt, h0_np)
        norm_pt_l0 = float(np.linalg.norm(h0_pt))
        norm_mlx_l0 = float(np.linalg.norm(h0_np))
        print(f"  layer-0 hidden cos={c0:.6f}  ||pt||={norm_pt_l0:.2f}  ||mlx||={norm_mlx_l0:.2f}  ratio={norm_mlx_l0/norm_pt_l0:.4f}")

    # === full forward ===
    hidden = model.language_model.model(
        input_ids=None, position_ids=pos_ids,
        inputs_embeds=embed,
        mask=attn_mask, gen_mask=gen_mask,
    )
    mx.eval(hidden)
    # Layer-0 diagnostic: re-run with single-layer model state? Too expensive.
    # Instead: full hidden[0] cos against PT hidden_layer0 (different stages,
    # but the layer-0 PT is from same layer index) — for now skip and only
    # compare embed + final.
    v_t_mlx = model.llm2vae(hidden[0, noise_s:noise_e, :])            # (n_noise, 48)
    mx.eval(v_t_mlx)
    v_t_mlx_np = np.asarray(v_t_mlx, dtype=np.float32)
    print(f"[forward] v_t_mlx: shape={v_t_mlx_np.shape} "
          f"||v||={np.linalg.norm(v_t_mlx_np):.3f} std={v_t_mlx_np.std():.4f} "
          f"range=[{v_t_mlx_np.min():+.3f}, {v_t_mlx_np.max():+.3f}]")

    # ---- gate ----
    v_t_pt_np = fx["v_t_step0"]
    c = cos(v_t_pt_np, v_t_mlx_np)
    diff = v_t_pt_np - v_t_mlx_np
    maxabs = float(np.abs(diff).max())
    mse = float((diff ** 2).mean())
    norm_pt = float(np.linalg.norm(v_t_pt_np))
    norm_mlx = float(np.linalg.norm(v_t_mlx_np))
    p50 = float(np.percentile(np.abs(diff), 50))
    p90 = float(np.percentile(np.abs(diff), 90))

    print(f"\n{'='*70}")
    print(f"STAGE 9 §0 GATE  (target: cos ≥ 0.999)")
    print(f"{'='*70}")
    print(f"  ||v_t_pt||  = {norm_pt:.3f}")
    print(f"  ||v_t_mlx|| = {norm_mlx:.3f}   ratio = {norm_mlx/norm_pt:.4f}")
    print(f"  cos(v_t_pt, v_t_mlx) = {c:.6f}   "
          f"[{'PASS' if c >= 0.999 else 'FAIL'}]")
    print(f"  mse={mse:.4e}  maxabs={maxabs:.4e}  p50={p50:.4e}  p90={p90:.4e}")
    print(f"{'='*70}")
    if c < 0.999:
        print("\nFAIL — STAGE 9 §0 게이트 미통과. forward 차이 진단 필요.")
        print("  주의 체크포인트: (1) latent_pos_embed shape 와 lookup, (2) attn_mask 변환,")
        print("                    (3) pos_ids form, (4) MoE routing (gen_mask)")
        sys.exit(1)
    print("\n[OK] STAGE 9 §0 게이트 PASS — MLX forward 가 PT 정답지와 일치.")


if __name__ == "__main__":
    main()
