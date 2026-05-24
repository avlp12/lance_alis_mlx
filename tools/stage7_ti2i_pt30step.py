"""STAGE 7 §3 — PT end-to-end TI2I 30-step generation.

Reuses the (now passing) single-step harness pipeline from
stage7_ti2i_compare.py, wrapping it in the full 30-step denoising loop
with 3-component CFG.  Compares per-step latent against MLX, and
saves the final PT image side-by-side with MLX.

Verdict logic (per user directive):
  - PT final image broken too  → MLX impl correct, model just reacts
                                  this way to synthetic gradient + saturated
  - PT final image clean       → 30-step accumulation path has a bug
                                  (single-step cos masked it)
"""
from __future__ import annotations

# Reuse shim install via stage7_ti2i_compare side effects
import tools.stage7_ti2i_compare as harness  # noqa: F401
from tools.stage7_ti2i_compare import (
    PtLanceTI2I, build_sequences, build_mask_pt,
    pt_forward_v,
    SPATIAL_DOWNSAMPLE, Z_DIM,
    _latent_position_indices, _vae_preprocess,
    EDIT_SYSTEM_PROMPT,
)
import time
import numpy as np
import torch
import mlx.core as mx
from PIL import Image
from transformers import AutoTokenizer
from lance_mlx.backbone import LanceLLM, LanceTextConfig, load_full_lance
from lance_mlx.vit import LanceViT, load_lance_vit
from lance_mlx.vae_wan22 import Wan2_2_VAE, Wan22VAEConfig
from lance_mlx.pipelines.x2t import preprocess_image, SPATIAL_MERGE_SIZE
from lance_mlx.rope import VisionSpec, build_positions_for_layout
from lance_mlx.scheduler import make_schedule, cfg_velocity_3comp, euler_step
from lance_mlx.pipelines.image_edit import image_edit


def cos_pt_mlx(pt: torch.Tensor, mx_: mx.array) -> float:
    a = pt.detach().to(torch.float32).cpu().numpy().flatten()
    b = np.asarray(mx_, dtype=np.float32).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))


# Match the existing CFG renorm
def cfg_3comp_pt(v_full, v_t_uncond, v_tv_uncond, cfg_text, cfg_vit, renorm_min=0.0):
    v_blend = v_tv_uncond + cfg_text*(v_full - v_t_uncond) + cfg_vit*(v_t_uncond - v_tv_uncond)
    norm_full = torch.norm(v_full)
    norm_blend = torch.norm(v_blend)
    scale = (norm_full / (norm_blend + 1e-8)).clamp(min=renorm_min, max=1.0)
    return v_blend * scale


def main():
    IMAGE = "out/test_synthetic.png"
    INSTRUCTION = "Make it more vibrant and saturated."
    SEED = 0
    SIZE = 256
    NUM_STEPS = 30
    CFG_TEXT = 3.0
    CFG_VIT = 1.5
    TIMESTEP_SHIFT = 3.5

    h_lat = w_lat = SIZE // SPATIAL_DOWNSAMPLE
    t_lat = 1
    n_cond = n_noise = t_lat * h_lat * w_lat

    print("=" * 70)
    print(f"STAGE 7 §3 PT 30-step end-to-end vs MLX")
    print(f"  cond image: {IMAGE}")
    print(f"  instruction: {INSTRUCTION}")
    print(f"  seed={SEED}  steps={NUM_STEPS}  cfg_text={CFG_TEXT}  cfg_vit={CFG_VIT}")
    print("=" * 70)

    # ---- MLX side: build ----
    print("\n[build] MLX models ...")
    model = LanceLLM(LanceTextConfig())
    load_full_lance(model, "checkpoints/Lance-3B-MLX/model.safetensors")
    model.eval()
    vit = LanceViT()
    load_lance_vit(vit, "checkpoints/Lance-3B-MLX/vit.safetensors")
    vit.eval()
    vae = Wan2_2_VAE(Wan22VAEConfig())
    vae.load_weights(list(mx.load("checkpoints/Wan2.2-VAE-MLX/model.safetensors").items()), strict=True)
    mx.eval(vae.parameters()); vae.eval()

    vit_patches, (T_g, H_g, W_g) = preprocess_image(IMAGE)
    grid_thw = mx.array([[T_g, H_g, W_g]], dtype=mx.int32)
    visual_und = vit(vit_patches, grid_thw)
    n_vit = int(visual_und.shape[0])
    H_g_m, W_g_m = H_g // SPATIAL_MERGE_SIZE, W_g // SPATIAL_MERGE_SIZE

    cond_latent = vae.encode(_vae_preprocess(IMAGE, size=SIZE))
    cond_flat = cond_latent.reshape(n_cond, Z_DIM)
    mx.eval(visual_und, cond_flat)

    visual_und_pt = torch.from_numpy(np.asarray(visual_und, dtype=np.float32))
    cond_flat_pt  = torch.from_numpy(np.asarray(cond_flat,  dtype=np.float32))

    # Initial noise (same on both sides)
    rng = np.random.default_rng(SEED)
    x_t_np = rng.standard_normal((n_noise, Z_DIM)).astype("float32")
    print(f"[init] x_t: shape={x_t_np.shape}  ||x_t||={np.linalg.norm(x_t_np):.2f}")

    # latent_pos_ids
    lat_pos_ids_mx = _latent_position_indices(t_lat, h_lat, w_lat)
    lat_pos_ids_pt = torch.from_numpy(np.asarray(lat_pos_ids_mx, dtype=np.int64))

    # Build the 3 layouts ONCE
    tok = AutoTokenizer.from_pretrained("checkpoints/Lance-3B-MLX", trust_remote_code=True)
    layouts = {}
    for name, kw in [("full",    {"include_text": True,  "include_vit": True }),
                     ("t_unc",   {"include_text": False, "include_vit": True }),
                     ("tv_unc",  {"include_text": False, "include_vit": False})]:
        lay = build_sequences(tok, n_vit, n_cond, n_noise, **kw)
        # positions (shared)
        spans = []
        if lay["vit_span"] is not None:
            vs_, ve_ = lay["vit_span"]
            spans.append(VisionSpec(start=vs_ - 1, length=ve_ - vs_,
                                     t=T_g, h=H_g_m, w=W_g_m))
        vs_, ve_ = lay["vae_span"]
        spans.append(VisionSpec(start=vs_ - 1, length=ve_ - vs_,
                                 t=t_lat, h=h_lat, w=w_lat))
        ns_, ne_ = lay["noise_span"]
        spans.append(VisionSpec(start=ns_ - 1, length=ne_ - ns_,
                                 t=t_lat, h=h_lat, w=w_lat))
        pos_mlx = build_positions_for_layout(lay["L"], spans)
        pos_np = np.asarray(pos_mlx)
        # pro_type=10 shifts
        if lay["vit_span"] is not None:
            vs_, ve_ = lay["vit_span"]
            shift = 1000 - int(pos_np[0, 0, vs_])
            pos_np[0, :, vs_:ve_] += shift
        vs_, ve_ = lay["vae_span"]
        ns_, ne_ = lay["noise_span"]
        pos_np[:, :, ns_:ne_] = pos_np[:, :, vs_:ve_]
        lay["_pos_pt"] = torch.from_numpy(pos_np.copy()).contiguous()

        # mask (bool, shared)
        _, dense_bool = build_mask_pt(lay, num_heads=16)
        lay["_dense_bool"] = dense_bool
        layouts[name] = lay
    print(f"[layout] full L={layouts['full']['L']}  t_unc L={layouts['t_unc']['L']}  "
          f"tv_unc L={layouts['tv_unc']['L']}")

    # ---- PT side: build LLM ----
    print("\n[build] PT Lance LLM (36 bf16 layers) ...")
    pt = PtLanceTI2I(); pt.load_pt(); pt.to_bf16()
    print("[pt] loaded.")

    # ---- 30-step PT denoising loop ----
    print(f"\n[loop] running PT 30 steps (3 forwards/step)...")
    sch = make_schedule(num_steps=NUM_STEPS, timestep_shift=TIMESTEP_SHIFT)
    x_t_pt = torch.from_numpy(x_t_np.copy()).to(torch.float32)

    t0 = time.time()
    latents_pt = []  # save every 5 steps for cos compare
    with torch.no_grad():
        for i in range(NUM_STEPS):
            t_val = float(sch.timesteps[i].item())
            dt_val = float(sch.dts[i].item())
            v_each = {}
            for name in ("full", "t_unc", "tv_unc"):
                lay = layouts[name]
                dense_bool = lay["_dense_bool"]
                v = pt_forward_v(pt, lay, visual_und_pt, cond_flat_pt, x_t_pt,
                                  t_val, lat_pos_ids_pt, lay["_pos_pt"],
                                  dense_bool)
                v_each[name] = v.detach()
            v_final = cfg_3comp_pt(v_each["full"], v_each["t_unc"], v_each["tv_unc"],
                                    CFG_TEXT, CFG_VIT)
            x_t_pt = (x_t_pt - v_final * dt_val).detach()
            if (i+1) % 5 == 0 or i == 0:
                print(f"  step {i+1:3d}/{NUM_STEPS}  t={t_val:.4f}  "
                      f"||v_full||={float(torch.norm(v_each['full'])):.2f}  "
                      f"||x_t||={float(torch.norm(x_t_pt)):.2f}  "
                      f"({time.time()-t0:.1f}s)")
                latents_pt.append((i+1, x_t_pt.clone()))

    print(f"[loop] PT 30 steps done in {time.time()-t0:.1f}s")

    # Save raw latent in case decode/post-process fails
    np.save("out/stage7_ti2i_pt_latent.npy", x_t_pt.detach().numpy())
    print(f"[save] out/stage7_ti2i_pt_latent.npy  ({x_t_pt.shape})")

    # ---- VAE decode PT final latent ----
    print("\n[decode] PT final latent → image (via MLX VAE)")
    pt_latent_mx = mx.array(x_t_pt.detach().numpy().reshape(1, t_lat, h_lat, w_lat, Z_DIM))
    pt_image_mx = vae.decode(pt_latent_mx)
    mx.eval(pt_image_mx)
    arr = np.asarray(pt_image_mx[0, 0])
    arr = (np.clip(arr * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr).save("out/stage7_ti2i_pt.png")
    print("[save] out/stage7_ti2i_pt.png")

    # ---- MLX side: run image_edit with same params ----
    print(f"\n[mlx] running MLX 30 steps ...")
    mlx_out = image_edit(
        model, vit, vae, tok, IMAGE, INSTRUCTION,
        size=SIZE, num_steps=NUM_STEPS, timestep_shift=TIMESTEP_SHIFT,
        cfg_text=CFG_TEXT, cfg_vit=CFG_VIT, seed=SEED,
    )
    arr = np.asarray(mlx_out.image_recon[0, 0])
    arr = (np.clip(arr * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr).save("out/stage7_ti2i_mlx.png")
    print("[save] out/stage7_ti2i_mlx.png")

    # ---- Final latent cos ----
    mlx_final_latent = mlx_out.latent.reshape(n_noise, Z_DIM)
    c_final = cos_pt_mlx(x_t_pt, mlx_final_latent)
    print(f"\n[res] PT vs MLX final latent cos = {c_final:.6f}")
    print(f"      ||PT final||  = {float(torch.norm(x_t_pt)):.2f}")
    print(f"      ||MLX final|| = {float(np.linalg.norm(np.asarray(mlx_final_latent))):.2f}")
    print(f"      {'PASS' if c_final >= 0.99 else 'FAIL'}  (gate: cos >= 0.99 for 30-step)")


if __name__ == "__main__":
    main()
