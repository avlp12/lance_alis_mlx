"""STAGE 11 — non-blind re-verification of image_edit (TI2I) 3-component CFG velocity.

Closes the last open item of the STAGE 11 correction: the image_edit velocity
machine was only ever checked by `tools/stage7_ti2i_compare.py`, which is BLIND —
its line 685 copies *our* MLX ViT output (`visual_und`) wholesale into PT
(`visual_und_pt = from_numpy(visual_und)`).  PT never recomputed the ViT from the
raw image, so the ViT patch-order bug (raster vs 2x2 merge-grouped) was shared by
both sides and agreed at cos ~ 1.0.

This harness re-runs the SAME velocity machine (imported verbatim from
stage7_ti2i_compare: PtLanceTI2I / pt_forward_v / mlx_forward_v_shared /
build_sequences / build_positions_pt / build_mask_pt) but DE-BLINDS the ViT path:

  (a) ViT de-blind (the bug's blast site in image_edit):
        * PT builds its OWN patches via data_utils.patchify_video_with_merge from
          the normalized raw image  ->  byte-assert vs our preprocess_image patches.
        * PT runs its OWN Qwen2.5-VL ViT  ->  cos-assert vs our LanceViT output.
        * positions: PT build_positions_pt (real shift_position_ids, pro_type=10)
          derived independently  ->  byte-compare vs our build_positions_for_layout.
  (b) 3-component CFG velocity (v_full / v_t_uncond / v_tv_uncond):
        PT uses ITS OWN visual_pt; MLX uses OUR visual_und.  Nothing shared but the
        raw inputs.  Gate: each cos >= 0.999 (the exact STAGE 7 numbers, now non-blind).
  Discriminative control:
        also run the MLX velocity with the OLD-BUG raster ViT output (visual_raster).
        cos_raster vs PT tells us whether the velocity is actually sensitive to the
        ViT path (tests the hypothesis that the CFG machine barely attends to ViT-cond).

SCOPE (honest, disclosed):
  * image_edit is image-only (T=1) -> the x2t_video temporal-mRoPE bug (#2) is immune.
  * cond_flat (VAE-cond latent) is the MLX VAE encode, shared to PT.  Neither bug
    touched it; VAE encode was byte-verified independently at STAGE 8 (cos=1.0).
    Re-deriving it through a PT Wan VAE is out of scope here (like the x2t gate
    scopes out PT's bucket-resize vit_transform).
  * the attention mask is built from PT's own create_sparse_mask (PT-sourced, not an
    MLX intermediate fed to PT) and reused on both sides.
  * GIVEN identical resized+normalized cond frames and grid.
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, ".")

# (1) transformers >= 5.9 flash-probe neutraliser — BEFORE any PT modeling import
import transformers.utils.import_utils as _iu
import transformers.utils as _tu
import transformers.modeling_flash_attention_utils as _mfa
def _false(*_a, **_k):
    return False
for _m in (_iu, _tu, _mfa):
    for _fn in ("is_flash_attn_2_available", "is_flash_attn_3_available",
                "is_flash_attn_4_available", "flash_attn_supports_top_left_mask"):
        if hasattr(_m, _fn):
            setattr(_m, _fn, _false)

# (2) velocity machine — importing installs flash_attn stub + flex patch + sys.path
import tools.stage7_ti2i_compare as T7

# (3) non-blind ViT helpers + PT bits (PT env is ready after T7 import)
import numpy as np
import torch

import mlx.core as mx
from transformers import AutoTokenizer

from data.data_utils import patchify_video_with_merge
from tools.stage11_x2t_verify import _norm_image, _raster_patches

from lance_mlx.backbone import LanceLLM, LanceTextConfig, load_full_lance
from lance_mlx.vit import LanceViT, load_lance_vit
from lance_mlx.vae_wan22 import Wan2_2_VAE, Wan22VAEConfig
from lance_mlx.pipelines.x2t import (
    preprocess_image, PATCH_SIZE, TEMPORAL_PATCH_SIZE, SPATIAL_MERGE_SIZE,
)
from lance_mlx.pipelines.image_edit import (
    _vae_preprocess, _latent_position_indices, Z_DIM, SPATIAL_DOWNSAMPLE,
)
from lance_mlx.rope import VisionSpec, build_positions_for_layout


MLX_WEIGHTS = "checkpoints/Lance-3B-MLX/model.safetensors"
VIT_WEIGHTS = "checkpoints/Lance-3B-MLX/vit.safetensors"
VAE_WEIGHTS = "checkpoints/Wan2.2-VAE-MLX/model.safetensors"
IMAGE = "out/test_synthetic.png"   # same cond image STAGE 7 used -> numbers comparable
INSTRUCTION = "Make it more vibrant and saturated."
SIZE = 256
SEED = 0
T_SCALAR = 1.0                      # flow step 0 (matches STAGE 7)
OUT_JSON = "out/stage11_ti2i_velocity_verify.json"


def _cos_np(a: np.ndarray, b: np.ndarray) -> float:
    af = np.asarray(a, np.float32).flatten()
    bf = np.asarray(b, np.float32).flatten()
    return float(af @ bf / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))


def _cos_pt_mx(pt: torch.Tensor, mxa: mx.array) -> float:
    return _cos_np(pt.detach().to(torch.float32).cpu().numpy(), np.asarray(mxa))


def build_pt_image_vit():
    """PT Qwen2.5-VL image ViT loaded standalone (no redundant understanding LLM).

    Mirrors stage7_x2t_compare.build_pt_x2t_model's ViT block (config + weight map)."""
    from modeling.vit.qwen2_5_vl_vit import Qwen2_5_VisionTransformerPretrainedModel
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
    from tools.stage7_vit_compare import mlx_to_pt_vit_state
    cfg = Qwen2_5_VLVisionConfig(
        depth=32, hidden_size=1280, intermediate_size=3420, num_heads=16,
        in_channels=3, patch_size=14, spatial_patch_size=14, spatial_merge_size=2,
        temporal_patch_size=2, window_size=112, layer_norm_eps=1e-6,
        tokens_per_second=2, out_hidden_size=2048,
        fullatt_block_indexes=[7, 15, 23, 31], hidden_act="silu")
    m = Qwen2_5_VisionTransformerPretrainedModel(cfg)
    m.load_state_dict(mlx_to_pt_vit_state(mx.load(VIT_WEIGHTS)), strict=False)
    m.eval()
    return m


def main() -> None:
    print("=" * 78)
    print("STAGE 11 — image_edit 3-comp CFG velocity, NON-BLIND re-verification")
    print("=" * 78)
    print(f"[in] cond image = {IMAGE}   instruction = {INSTRUCTION!r}")
    print(f"[in] size={SIZE}  seed={SEED}  t(step0)={T_SCALAR}")

    tok = AutoTokenizer.from_pretrained("checkpoints/Lance-3B-MLX", trust_remote_code=True)

    # ---- MLX models ----
    print("\n[mlx] building LLM (gen) + ViT + Wan VAE ...")
    model = LanceLLM(LanceTextConfig()); load_full_lance(model, MLX_WEIGHTS); model.eval()
    vit = LanceViT(); load_lance_vit(vit, VIT_WEIGHTS); vit.eval()
    vae = Wan2_2_VAE(Wan22VAEConfig())
    vae.load_weights(list(mx.load(VAE_WEIGHTS).items()), strict=True)
    mx.eval(vae.parameters()); vae.eval()

    # ======================================================================
    # (a) NON-BLIND ViT de-blinding — the image_edit blast site of bug #1
    # ======================================================================
    print("\n" + "-" * 78)
    print("(a) ViT de-blind: PT recomputes patches + ViT from the raw image")
    print("-" * 78)

    # ours: preprocess_image (2x2 merge-grouped, post-fix) -> our LanceViT
    vit_patches, (T_g, H_g, W_g) = preprocess_image(IMAGE)
    grid_mlx = mx.array([[T_g, H_g, W_g]], dtype=mx.int32)
    visual_und = vit(vit_patches, grid_mlx); mx.eval(visual_und)
    n_vit = int(visual_und.shape[0])
    H_g_m, W_g_m = H_g // SPATIAL_MERGE_SIZE, W_g // SPATIAL_MERGE_SIZE
    print(f"[ours] grid=({T_g},{H_g},{W_g})  visual_und={tuple(visual_und.shape)}  n_vit={n_vit}")

    # PT independent: normalize raw image -> PT's own patchify -> byte-assert -> PT ViT
    norm = _norm_image(IMAGE)                                   # (2, H, W, 3)  (T=2 dup)
    pt_patches = patchify_video_with_merge(
        torch.from_numpy(np.transpose(norm, (3, 0, 1, 2))),     # (C, T, H, W)
        PATCH_SIZE, TEMPORAL_PATCH_SIZE).numpy()
    patches_byte_id = bool(np.array_equal(np.asarray(vit_patches), pt_patches))
    print(f"[non-blind] patches PT-recomputed byte-identical to ours: {patches_byte_id}")

    pt_vit = build_pt_image_vit()
    with torch.no_grad():
        visual_pt = pt_vit(hidden_states=torch.from_numpy(pt_patches),
                           grid_thw=torch.tensor([[T_g, H_g, W_g]], dtype=torch.long))
    vit_cos = _cos_pt_mx(visual_pt, visual_und)
    print(f"[non-blind] ViT cos (PT-own-ViT vs our-ViT): {vit_cos:.6f}")

    # raster control: feed OLD-BUG patch order into OUR ViT (wrong visual)
    raster_patches = _raster_patches(norm)
    visual_raster = vit(raster_patches, grid_mlx); mx.eval(visual_raster)
    vit_cos_raster = _cos_pt_mx(visual_pt, visual_raster)
    print(f"[control ] ViT cos (PT vs our-RASTER, old bug): {vit_cos_raster:.6f}  "
          f"(should be far below {vit_cos:.4f})")

    a_ok = patches_byte_id and (vit_cos >= 0.999)
    print(f"(a) => {'PASS' if a_ok else 'FAIL'}  "
          f"(patches byte-id={patches_byte_id}, ViT cos>=0.999={vit_cos >= 0.999})")
    if not a_ok:
        print("\n[stop] (a) failed — ViT de-blind broken; not running velocity.")
        _dump({"a_pass": a_ok, "patches_byte_identical": patches_byte_id,
               "vit_cos": vit_cos, "vit_cos_raster": vit_cos_raster, "layouts": []})
        sys.exit(1)

    # ======================================================================
    # shared raw-derived inputs (cond VAE latent, noise, latent pos ids)
    # ======================================================================
    h_lat = w_lat = SIZE // SPATIAL_DOWNSAMPLE
    t_lat = 1
    n_cond = t_lat * h_lat * w_lat
    n_noise = n_cond

    cond_latent = vae.encode(_vae_preprocess(IMAGE, size=SIZE))     # (1,1,h,w,48)
    cond_flat = cond_latent.reshape(n_cond, Z_DIM)
    cond_flat_pt = torch.from_numpy(np.asarray(cond_flat, dtype=np.float32))
    print(f"\n[vae] cond_flat={tuple(cond_flat.shape)} (MLX encode, shared+disclosed)")

    rng = np.random.default_rng(SEED)
    x_t_np = rng.standard_normal((n_noise, Z_DIM)).astype("float32")
    x_t = mx.array(x_t_np)
    x_t_pt = torch.from_numpy(x_t_np.copy())
    t_scalar = mx.array([T_SCALAR], dtype=mx.float32)

    latent_pos_ids = _latent_position_indices(t_lat, h_lat, w_lat)
    latent_pos_ids_pt = torch.from_numpy(np.asarray(latent_pos_ids, dtype=np.int64))

    visual_pt_t = visual_pt                          # PT side uses ITS OWN ViT output

    # ======================================================================
    # (b) 3-component CFG velocity — each side its own ViT output
    # ======================================================================
    print("\n" + "-" * 78)
    print("(b) 3-comp CFG velocity: PT(own ViT) vs MLX(our ViT), nothing shared but raw inputs")
    print("-" * 78)

    pt = T7.PtLanceTI2I(); pt.load_pt(); pt.to_bf16()

    def make_layout(*, include_text, include_vit):
        lay = T7.build_sequences(tok, n_vit, n_cond, n_noise,
                                 include_text=include_text, include_vit=include_vit)
        lay["_grid_thw_merged"] = (T_g, H_g_m, W_g_m)
        lay["_lat_shape"] = (t_lat, h_lat, w_lat)
        return lay

    cases = {
        "v_full":      make_layout(include_text=True,  include_vit=True),
        "v_t_uncond":  make_layout(include_text=False, include_vit=True),
        "v_tv_uncond": make_layout(include_text=False, include_vit=False),
    }

    results = []
    for name, lay in cases.items():
        L = lay["L"]
        has_vit = lay["vit_span"] is not None

        # --- positions: MLX-derived (build_positions_for_layout + pro_type=10 shifts) ---
        spans = []
        if has_vit:
            vs_, ve_ = lay["vit_span"]
            spans.append(VisionSpec(start=vs_ - 1, length=ve_ - vs_, t=T_g, h=H_g_m, w=W_g_m))
        vs_, ve_ = lay["vae_span"]
        spans.append(VisionSpec(start=vs_ - 1, length=ve_ - vs_, t=t_lat, h=h_lat, w=w_lat))
        ns_, ne_ = lay["noise_span"]
        spans.append(VisionSpec(start=ns_ - 1, length=ne_ - ns_, t=t_lat, h=h_lat, w=w_lat))
        pos_np = np.asarray(build_positions_for_layout(L, spans))
        if has_vit:
            vit_s, vit_e = lay["vit_span"]
            pos_np[0, :, vit_s:vit_e] += 1000 - int(pos_np[0, 0, vit_s])   # T axis only
        vae_s, vae_e = lay["vae_span"]
        noise_s, noise_e = lay["noise_span"]
        pos_np[:, :, noise_s:noise_e] = pos_np[:, :, vae_s:vae_e]
        pos_mlx_shifted = mx.array(pos_np)

        # --- positions: PT-independent (real shift_position_ids, pro_type=10) ---
        pos_pt = T7.build_positions_pt(lay, T_g, H_g_m, W_g_m, t_lat, h_lat, w_lat)  # (3,1,L)
        pos_byte_id = bool(np.array_equal(pos_np.astype(np.int64),
                                          pos_pt.numpy().astype(np.int64)))

        # --- mask: PT create_sparse_mask (PT-sourced), reused both sides ---
        _, dense_bool = T7.build_mask_pt(lay, num_heads=pt.cfg.num_attention_heads)
        attn_mask_pt = dense_bool
        attn_mask_mx = mx.array(np.where(dense_bool.numpy(), 0.0, -np.inf).astype(np.float32))

        # --- PT velocity (PT's own ViT + own positions) ---
        with torch.no_grad():
            v_pt = T7.pt_forward_v(pt, lay, visual_pt_t, cond_flat_pt, x_t_pt,
                                   T_SCALAR, latent_pos_ids_pt, pos_pt, attn_mask_pt)
        # --- MLX velocity FIXED (our ViT) ---
        v_mlx = T7.mlx_forward_v_shared(model, lay, visual_und, cond_flat, x_t,
                                        t_scalar, latent_pos_ids,
                                        pos_shared=pos_mlx_shifted,
                                        attn_mask_shared=attn_mask_mx)
        mx.eval(v_mlx)
        cos_fixed = _cos_pt_mx(v_pt, v_mlx)

        # --- MLX velocity RASTER control (old-bug ViT) — only when ViT present ---
        cos_raster = None
        if has_vit:
            v_mlx_r = T7.mlx_forward_v_shared(model, lay, visual_raster, cond_flat, x_t,
                                              t_scalar, latent_pos_ids,
                                              pos_shared=pos_mlx_shifted,
                                              attn_mask_shared=attn_mask_mx)
            mx.eval(v_mlx_r)
            cos_raster = _cos_pt_mx(v_pt, v_mlx_r)

        rj = {"name": name, "L": L, "has_vit": has_vit,
              "positions_byte_identical_to_PT": pos_byte_id,
              "cos_fixed": cos_fixed, "cos_raster": cos_raster,
              "v_norm_pt": float(torch.norm(v_pt)),
              "v_norm_mlx": float(np.linalg.norm(np.asarray(v_mlx))),
              "pass": bool(cos_fixed >= 0.999)}
        results.append(rj)
        rstr = f"{cos_raster:.6f}" if cos_raster is not None else "n/a (no ViT)"
        print(f"[{name:11s}] L={L:4d}  pos byte-id(PT)={pos_byte_id}  "
              f"cos_fixed={cos_fixed:.6f} {'PASS' if cos_fixed >= 0.999 else 'FAIL'}   "
              f"cos_raster={rstr}")

    # ---- discriminative read: is the velocity sensitive to the ViT path? ----
    raster_vals = [r["cos_raster"] for r in results if r["cos_raster"] is not None]
    raster_min = min(raster_vals) if raster_vals else None
    velocity_vit_sensitive = bool(raster_min is not None and raster_min < 0.999)

    all_cos_fixed_ok = all(r["pass"] for r in results)
    all_pos_byte_id = all(r["positions_byte_identical_to_PT"] for r in results)
    gate = bool(a_ok and all_cos_fixed_ok)

    summary = {
        "image": IMAGE, "instruction": INSTRUCTION, "size": SIZE, "seed": SEED,
        "grid": [T_g, H_g, W_g], "n_vit": n_vit, "n_cond": n_cond,
        "a_pass": a_ok,
        "patches_byte_identical": patches_byte_id,
        "vit_cos": vit_cos, "vit_cos_raster": vit_cos_raster,
        "positions_byte_identical_to_PT_all": all_pos_byte_id,
        "cos_fixed_min": min(r["cos_fixed"] for r in results),
        "velocity_vit_sensitive": velocity_vit_sensitive,
        "raster_cos_min": raster_min,
        "gate_pass": gate,
        "layouts": results,
        "scope": ("image-only (T=1, bug#2 immune); cond_flat=MLX VAE encode shared "
                  "(byte-verified STAGE 8); mask=PT create_sparse_mask; given identical "
                  "resized+normalized cond frames & grid"),
    }
    _dump(summary)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print(f"  (a) ViT de-blind : patches byte-id={patches_byte_id}  ViT cos={vit_cos:.6f}  "
          f"(raster control cos={vit_cos_raster:.4f})")
    print(f"  (b) velocity     : 3-comp cos_fixed min={summary['cos_fixed_min']:.6f}  "
          f"all>=0.999={all_cos_fixed_ok}")
    print(f"      positions PT byte-id (all layouts): {all_pos_byte_id}")
    if raster_min is not None:
        print(f"      raster control velocity cos min={raster_min:.6f}  -> "
              f"velocity {'IS' if velocity_vit_sensitive else 'is NOT'} sensitive to ViT path")
        if not velocity_vit_sensitive:
            print("      (note: velocity weakly attends to ViT-cond; the ViT de-blind rests")
            print("       primarily on (a)'s byte/cos check, not on velocity discrimination)")
    print("-" * 78)
    print(f"GATE stage11_ti2i_velocity_verify: {'PASS' if gate else 'FAIL'}")
    print(f"[log] {OUT_JSON}")
    sys.exit(0 if gate else 1)


def _dump(obj) -> None:
    with open(OUT_JSON, "w") as f:
        json.dump(obj, f, indent=2)


if __name__ == "__main__":
    main()
