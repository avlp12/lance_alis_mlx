"""STAGE 11 — video_edit (tiv2v) NON-BLIND assembly verification.

video_edit reuses verified pieces (ViT / VAE / latent-pos / 3-comp CFG); the NEW
risk is the *assembly* — how the 3 slabs (ViT-cond T_g, VAE-cond t_lat, noise) and
their mRoPE positions COMBINE in the edit layout (slab order, base->1000 ViT shift,
noise<-cond copy, temporal x2 on BOTH ViT and VAE).  STAGE 7's lesson: verified
parts can still be mis-composed.  So we de-blind the composition.

Stage-isolated:
  (a) — light, no LLM forward:
      (a1) ViT-cond: PT patchify_video_with_merge + PT video ViT, byte-id patches + cos.
      (a2) 3-slab positions: PT production recipe (lance.py:241-249) — real
           get_rope_index(video grids, sec=1 -> temporal x2) then real
           shift_position_ids(pro_type=10) — byte-asserted vs the *actual pipeline*
           builder build_video_edit_positions.  NO pos_pt=from_numpy(pos_mlx).
  (b) — only if (a) passes:
      3-component CFG velocity (v_full / v_t_uncond / v_tv_uncond) PT vs MLX, each
      side its OWN ViT + OWN positions (raw inputs only are shared); cos >= 0.999.
      RASTER control: old patch-order ViT-cond must COLLAPSE the velocity (else the
      metric is ViT-blind and (b) is vacuous).

SCOPE (disclosed): cond_flat (VAE-cond latent) is the MLX VAE encode with the
production scale, SHARED to PT.  Wan VAE encode is byte-verified at STAGE 8; the
cond-encode *scale* match vs a PT Wan VAE is NOT re-checked here (the velocity
harness shared cond_flat too) — deferred.  PT video reference = image PT weight
(checkpoints/Lance/Lance_3B) with only latent_pos_embed swapped to the video
31x64^2 table (image & video MLX weights differ in that one key only).
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

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

import numpy as np
import torch
import mlx.core as mx
from transformers import AutoTokenizer

import tools.stage11_x2t_verify as XV          # PT env + video-ViT helpers
import tools.stage7_ti2i_compare as T7         # PtLanceTI2I / pt_forward_v / mlx_forward_v_shared / build_mask_pt
from data.common import shift_position_ids
from data.data_utils import patchify_video_with_merge

from mlx.utils import tree_flatten
from lance_mlx.vit import LanceViT
from lance_mlx.backbone import LanceLLM, LanceTextConfig, PositionEmbedding3D
from lance_mlx.vae_wan22 import Wan2_2_VAE, Wan22VAEConfig
from lance_mlx.pipelines.x2t import (
    preprocess_video, load_video_vit, SPATIAL_MERGE_SIZE, PATCH_SIZE, TEMPORAL_PATCH_SIZE,
)
from lance_mlx.pipelines.t2v import (
    vae_latent_position_indices, VAE_SCALE_MEAN, VAE_SCALE_STD,
    MAX_NUM_LATENT_FRAMES, MAX_LATENT_SIZE,
)
from lance_mlx.pipelines.image_edit import Z_DIM
from lance_mlx.scheduler import make_schedule, cfg_velocity_3comp, euler_step
from lance_mlx.pipelines.video_edit import (
    build_video_edit_layouts, build_video_edit_positions, _vae_preprocess_video,
)

VIDEO_W = "out/lance_3b_video_mlx/model.safetensors"
VAE_W   = "checkpoints/Wan2.2-VAE-MLX/model.safetensors"
FRAMES  = "out/stage11_assets/vqa01_frames.npy"
N_FRAMES = 8
VAE_H = VAE_W_PX = 128
VAE_DOWN_S, VAE_DOWN_T = 16, 4
INSTRUCTION = "Make the scene look like a snowy winter day."
MAX_PIX = XV.MAX_PIX_VIDEO
T_SCALAR = 1.0
SEED = 0
NUM_STEPS = 24          # video_edit production schedule length
K_STEPS = 5             # (c) accumulation check: first K real denoise steps


def build_pt_video_vit():
    full = mx.load(VIDEO_W)
    vt = {"vision_tower." + k[len("vit_model."):]: v
          for k, v in full.items() if k.startswith("vit_model.")}
    m = XV.S7.Qwen2_5_VisionTransformerPretrainedModel(XV.S7.Qwen2_5_VLVisionConfig(
        depth=32, hidden_size=1280, intermediate_size=3420, num_heads=16, in_channels=3,
        patch_size=14, spatial_patch_size=14, spatial_merge_size=2, temporal_patch_size=2,
        window_size=112, layer_norm_eps=1e-6, tokens_per_second=2, out_hidden_size=2048,
        fullatt_block_indexes=[7, 15, 23, 31], hidden_act="silu"))
    m.load_state_dict(XV.S7.mlx_to_pt_vit_state(vt), strict=False); m.eval()
    return m


def build_pt_video_ref():
    """PtLanceTI2I (image PT weight) with latent_pos_embed swapped to the video
    31x64^2 table (image & video weights differ only in that key)."""
    pt = T7.PtLanceTI2I()
    pt.load_pt()                                              # image PT weights
    lpe = np.asarray(mx.load(VIDEO_W)["latent_pos_embed.pos_embed"], dtype=np.float32)
    pt.latent_pos_embed = torch.nn.Embedding(lpe.shape[0], lpe.shape[1])
    pt.latent_pos_embed.weight.data = torch.from_numpy(lpe)
    pt.to_bf16()
    assert lpe.shape[0] == MAX_NUM_LATENT_FRAMES * MAX_LATENT_SIZE ** 2, "video latent_pos_embed size"
    return pt


def load_video_llm_vae():
    cfg = LanceTextConfig()
    model = LanceLLM(cfg)
    model.latent_pos_embed = PositionEmbedding3D(
        max_num_latent_frames=MAX_NUM_LATENT_FRAMES, max_latent_size=MAX_LATENT_SIZE,
        hidden_size=cfg.hidden_size)
    full = mx.load(VIDEO_W)
    ours = set(dict(tree_flatten(model.parameters())).keys())
    model.load_weights([(k, v) for k, v in full.items() if k in ours], strict=True)
    mx.eval(model.parameters()); model.eval()
    vae = Wan2_2_VAE(Wan22VAEConfig())
    vae.load_weights(list(mx.load(VAE_W).items()), strict=True)
    mx.eval(vae.parameters()); vae.eval()
    return model, vae


def _layout_meta(layout):
    L = layout["L"]
    vae_s, vae_e = layout["vae_span"]; noise_s, noise_e = layout["noise_span"]
    split_lens, attn_modes = [], []
    if layout["vit_span"] is not None:
        vit_s, vit_e = layout["vit_span"]
        split_lens += [vit_s - 1, (vit_e - vit_s) + 2]; attn_modes += ["causal", "full"]
        mid_start = vit_e + 1
    else:
        mid_start = 0
    split_lens += [(vae_s - 1) - mid_start, (vae_e - vae_s) + 2, (noise_e - noise_s) + 2]
    attn_modes += ["causal", "full_noise", "noise"]
    sl_tail = L - (noise_e + 1)
    if sl_tail > 0:
        split_lens.append(sl_tail); attn_modes.append("causal")
    assert sum(split_lens) == L
    modality = [0] * L
    if layout["vit_span"] is not None:
        for i in range(*layout["vit_span"]): modality[i] = 4
    for i in range(vae_s, vae_e): modality[i] = 2
    for i in range(noise_s, noise_e): modality[i] = 1
    return split_lens, attn_modes, modality


def pt_edit_positions(layout, *, T_g, H_g, W_g, t_lat, h_lat, w_lat):
    ids = layout["ids"]; L = layout["L"]
    grids = []
    if layout["vit_span"] is not None:
        grids.append([T_g, H_g, W_g])
    grids.append([t_lat, h_lat * 2, w_lat * 2])
    grids.append([t_lat, h_lat * 2, w_lat * 2])
    grids_t = torch.tensor(grids, dtype=torch.long)
    sec = torch.tensor([1.0] * len(grids))
    pos_base, _ = XV._qwen2_navit.Qwen2ForCausalLM.get_rope_index(
        XV._MockSelf(), input_ids=torch.tensor([ids], dtype=torch.long),
        image_grid_thw=grids_t, video_grid_thw=grids_t,
        second_per_grid_ts=sec, attention_mask=torch.ones(1, L, dtype=torch.long))
    if pos_base.ndim == 2:
        pos_base = pos_base.unsqueeze(1)
    split_lens, attn_modes, modality = _layout_meta(layout)
    shifted = shift_position_ids(
        pos_base.contiguous(), pos_shift=1000, attn_modes=attn_modes, split_lens=split_lens,
        shift_attn_mode=["full_noise", "full"], pro_type=10,
        i_sample_task=torch.tensor([2] * L), i_sample_modality=torch.tensor(modality))
    return np.asarray(shifted.numpy()).astype(np.int64)


def main():
    print("=" * 78)
    print("STAGE 11 video_edit — NON-BLIND assembly verify (a ViT+positions, b velocity)")
    print("=" * 78)
    tok = AutoTokenizer.from_pretrained("checkpoints/Lance-3B-MLX", trust_remote_code=True)
    clip = np.load(FRAMES)[:N_FRAMES]
    print(f"[in] cond frames={clip.shape} vae=({VAE_H},{VAE_W_PX}) instruction={INSTRUCTION!r}")

    vit = LanceViT(); load_video_vit(vit, VIDEO_W); vit.eval()
    patches, (T_g, H_g, W_g) = preprocess_video(clip, max_pixels=MAX_PIX)
    grid_mlx = mx.array([[T_g, H_g, W_g]], dtype=mx.int32)
    visual_und = vit(patches, grid_mlx); mx.eval(visual_und)
    n_vit = int(visual_und.shape[0])
    H_g_m, W_g_m = H_g // SPATIAL_MERGE_SIZE, W_g // SPATIAL_MERGE_SIZE
    t_lat = (N_FRAMES - 1) // VAE_DOWN_T + 1
    h_lat, w_lat = VAE_H // VAE_DOWN_S, VAE_W_PX // VAE_DOWN_S
    n_cond = n_noise = t_lat * h_lat * w_lat
    print(f"[dims] ViT=({T_g},{H_g},{W_g}) n_vit={n_vit} merged({T_g},{H_g_m},{W_g_m}) | "
          f"VAE=({t_lat},{h_lat},{w_lat}) n_cond={n_cond}")

    # ---------- (a1) ViT-cond de-blind ----------
    print("\n" + "-" * 78 + "\n(a1) ViT-cond: PT patchify + PT video ViT\n" + "-" * 78)
    norm = XV._norm_video(clip, MAX_PIX)
    pt_patches = patchify_video_with_merge(
        torch.from_numpy(np.transpose(norm, (3, 0, 1, 2))), PATCH_SIZE, TEMPORAL_PATCH_SIZE).numpy()
    patches_tie = bool(np.array_equal(np.asarray(patches), pt_patches))
    pt_vit = build_pt_video_vit()
    with torch.no_grad():
        visual_pt = pt_vit(hidden_states=torch.from_numpy(pt_patches),
                           grid_thw=torch.tensor([[T_g, H_g, W_g]], dtype=torch.long))
    vit_cos = XV._cos(visual_pt.float().numpy(), np.asarray(visual_und))
    print(f"  patches byte-identical={patches_tie}  ViT cos={vit_cos:.6f}")
    a1_ok = patches_tie and vit_cos >= 0.999

    # ---------- (a2) 3-slab positions ----------
    print("\n" + "-" * 78 + "\n(a2) positions: PT get_rope_index+shift vs pipeline\n" + "-" * 78)
    layouts = build_video_edit_layouts(tok, INSTRUCTION, n_vit, n_cond, n_noise)
    a2_ok = True
    for name, layout in layouts.items():
        pos_ours = build_video_edit_positions(
            layout, T_g=T_g, H_g_m=H_g_m, W_g_m=W_g_m, t_lat=t_lat, h_lat=h_lat, w_lat=w_lat).astype(np.int64)
        pos_pt = pt_edit_positions(layout, T_g=T_g, H_g=H_g, W_g=W_g, t_lat=t_lat, h_lat=h_lat, w_lat=w_lat)
        tie = bool(np.array_equal(pos_ours, pos_pt)); a2_ok &= tie
        print(f"  [{name:11s}] L={layout['L']:4d} positions byte-identical={tie}")
    a_ok = a1_ok and a2_ok
    print(f"\n(a) => {'PASS' if a_ok else 'FAIL'}")
    if not a_ok:
        print("-> STOP: (a) failed; fix assembly before (b)."); sys.exit(1)

    # ---------- (b) 3-comp velocity + RASTER ----------
    print("\n" + "=" * 78 + "\n(b) 3-comp CFG velocity: PT(own ViT) vs MLX(our ViT) + RASTER control\n" + "=" * 78)
    model, vae = load_video_llm_vae()
    pt = build_pt_video_ref()

    # cond_flat (MLX VAE encode, production scale) — SHARED to PT (disclosed)
    vae_scale = (mx.array(VAE_SCALE_MEAN), mx.array(1.0 / VAE_SCALE_STD))
    cond_latent = vae.encode(_vae_preprocess_video(clip, H=VAE_H, W=VAE_W_PX), scale=vae_scale)
    cond_flat = cond_latent.reshape(n_cond, Z_DIM); mx.eval(cond_flat)
    cond_flat_pt = torch.from_numpy(np.asarray(cond_flat, dtype=np.float32))

    rng = np.random.default_rng(SEED)
    x_t_np = rng.standard_normal((n_noise, Z_DIM)).astype("float32")
    x_t = mx.array(x_t_np); x_t_pt = torch.from_numpy(x_t_np.copy())
    t_scalar = mx.array([T_SCALAR], dtype=mx.float32)

    latent_pos_ids = mx.array(vae_latent_position_indices(t_lat, h_lat, w_lat, max_latent_size=MAX_LATENT_SIZE))
    latent_pos_ids_pt = torch.from_numpy(np.asarray(latent_pos_ids, dtype=np.int64))

    visual_raster = vit(XV._raster_patches(norm), grid_mlx); mx.eval(visual_raster)

    # precompute per-layout positions + masks (shared by (b) single-step + (c) loop)
    LP = {}
    for name, layout in layouts.items():
        sl, am, _ = _layout_meta(layout)
        lay = {**layout, "split_lens": sl, "attn_modes": am}
        _, dense = T7.build_mask_pt(lay, num_heads=pt.cfg.num_attention_heads)
        attn_mx = mx.array(np.where(dense.numpy(), 0.0, -np.inf).astype(np.float32))
        pos_ours = mx.array(build_video_edit_positions(
            lay, T_g=T_g, H_g_m=H_g_m, W_g_m=W_g_m, t_lat=t_lat, h_lat=h_lat, w_lat=w_lat))
        pos_pt = torch.from_numpy(pt_edit_positions(
            lay, T_g=T_g, H_g=H_g, W_g=W_g, t_lat=t_lat, h_lat=h_lat, w_lat=w_lat)).contiguous()
        LP[name] = (lay, pos_ours, pos_pt, dense, attn_mx)
    ORDER = ("v_full", "v_t_uncond", "v_tv_uncond")

    # ---- (b) single-step velocity (t=1.0) + RASTER control ----
    rows = []
    for name in ORDER:
        lay, pos_ours, pos_pt, dense, attn_mx = LP[name]
        with torch.no_grad():
            v_pt = T7.pt_forward_v(pt, lay, visual_pt, cond_flat_pt, x_t_pt,
                                   T_SCALAR, latent_pos_ids_pt, pos_pt, dense)
        v_mlx = T7.mlx_forward_v_shared(model, lay, visual_und, cond_flat, x_t, t_scalar,
                                        latent_pos_ids, pos_shared=pos_ours, attn_mask_shared=attn_mx)
        mx.eval(v_mlx)
        cos_fixed = XV._cos(v_pt.float().numpy(), np.asarray(v_mlx))
        cos_raster = None
        if lay["vit_span"] is not None:
            v_r = T7.mlx_forward_v_shared(model, lay, visual_raster, cond_flat, x_t, t_scalar,
                                          latent_pos_ids, pos_shared=pos_ours, attn_mask_shared=attn_mx)
            mx.eval(v_r)
            cos_raster = XV._cos(v_pt.float().numpy(), np.asarray(v_r))
        rows.append((name, cos_fixed, cos_raster))
        rs = f"{cos_raster:.6f}" if cos_raster is not None else "n/a"
        print(f"  [{name:11s}] cos_fixed={cos_fixed:.6f} {'PASS' if cos_fixed >= 0.999 else 'FAIL'}  cos_raster={rs}")
    cos_min = min(r[1] for r in rows)
    raster_vals = [r[2] for r in rows if r[2] is not None]
    raster_min = min(raster_vals) if raster_vals else None
    raster_discriminates = bool(raster_min is not None and raster_min < 0.999)
    b_ok = all(r[1] >= 0.999 for r in rows) and raster_discriminates

    # ---- (c) multi-step accumulation: PT-velocity traj vs MLX-velocity traj ----
    print("\n" + "-" * 78)
    print(f"(c) {K_STEPS}-step accumulation: two trajectories (PT vs MLX velocity), same noise+schedule+CFG+euler")
    print("-" * 78)
    sch = make_schedule(num_steps=NUM_STEPS, timestep_shift=3.5)

    def _v3(f):  # 3-comp CFG combine — video_edit production defaults
        return cfg_velocity_3comp(f["v_full"], f["v_t_uncond"], f["v_tv_uncond"],
                                  cfg_text=3.0, cfg_vit=1.0, renorm_type="global", renorm_min=0.0)

    x_traj_mlx = mx.array(x_t_np); x_traj_pt = mx.array(x_t_np)
    c_rows = []
    for i in range(K_STEPS):
        tv = float(sch.timesteps[i]); dt = sch.dts[i]; tsc = mx.array([tv], dtype=mx.float32)
        fm = {}
        for name in ORDER:
            lay, pos_ours, _, _, attn_mx = LP[name]
            fm[name] = T7.mlx_forward_v_shared(model, lay, visual_und, cond_flat, x_traj_mlx, tsc,
                                               latent_pos_ids, pos_shared=pos_ours, attn_mask_shared=attn_mx)
        x_traj_mlx = euler_step(x_traj_mlx, _v3(fm), dt); mx.eval(x_traj_mlx)

        x_pt_torch = torch.from_numpy(np.asarray(x_traj_pt, dtype=np.float32))
        fp = {}
        for name in ORDER:
            lay, _, pos_pt, dense, _ = LP[name]
            with torch.no_grad():
                v = T7.pt_forward_v(pt, lay, visual_pt, cond_flat_pt, x_pt_torch, tv,
                                    latent_pos_ids_pt, pos_pt, dense)
            fp[name] = mx.array(v.float().numpy())
        x_traj_pt = euler_step(x_traj_pt, _v3(fp), dt); mx.eval(x_traj_pt)

        cstep = XV._cos(np.asarray(x_traj_pt), np.asarray(x_traj_mlx))
        c_rows.append(cstep)
        print(f"  step {i+1:2d}/{K_STEPS}  t={tv:.4f}  latent cos(PT,MLX)={cstep:.6f}"
              + ("  FAIL" if cstep < 0.999 else ""))
    c_min = min(c_rows)
    c_ok = c_min >= 0.999

    raster_min_str = f"{raster_min:.6f}" if raster_min is not None else "n/a"
    print("\n" + "=" * 78)
    print(f"(a) ViT cos={vit_cos:.6f}  positions byte-id=all")
    print(f"(b) velocity cos_min={cos_min:.6f}  RASTER {'collapses' if raster_discriminates else 'NO-collapse?!'} (min={raster_min_str})")
    print(f"(c) {K_STEPS}-step accumulation latent cos_min={c_min:.6f}  "
          f"{'(no compounding)' if c_ok else '(DIVERGES - per-step error compounds)'}")
    ok = a_ok and b_ok and c_ok
    print(f"GATE stage11_video_edit_verify: {'PASS' if ok else 'FAIL'}")
    print("SCOPE: cond encode scale separately de-blind (stage11_video_edit_cond_scale, cos=1.0);")
    print("       full pixel decode implied by per-step+accumulation cos + STAGE 8 byte-clean decode; given resized frames+grid")
    print("=" * 78)

    import json
    with open("out/stage11_video_edit_verify.json", "w") as f:
        json.dump({
            "grid": {"vit": [T_g, H_g, W_g], "vae_lat": [t_lat, h_lat, w_lat],
                     "n_vit": n_vit, "n_cond": n_cond, "frames": N_FRAMES, "vae_size": [VAE_H, VAE_W_PX]},
            "a_vit_patches_byte_identical": bool(patches_tie),
            "a_vit_cos": vit_cos,
            "a_positions_byte_identical_all": bool(a2_ok),
            "b_velocity": [{"name": n, "cos_fixed": cf, "cos_raster": cr} for n, cf, cr in rows],
            "b_cos_min": cos_min, "b_raster_min": raster_min, "b_raster_discriminates": raster_discriminates,
            "c_k_steps": K_STEPS, "c_latent_cos_per_step": c_rows, "c_cos_min": c_min, "c_no_compounding": c_ok,
            "gate_pass": bool(ok),
            "scope": ("cond encode scale de-blind separate (stage11_video_edit_cond_scale, cos=1.0); "
                      "full pixel implied by accumulation cos + STAGE 8 byte-clean decode; given resized frames+grid"),
        }, f, indent=2)
    print(f"[log] out/stage11_video_edit_verify.json")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
