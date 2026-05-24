"""STAGE 8 §1.7 — Decoder3d full-assembly byte-diff vs PT.

Decoder iterates per-latent-frame (PT vae2_2.py:795-810):
  iter_ = z.shape[2]   # T_lat
  for i in range(iter_):
    self._conv_idx = [0]                           # reset feat_idx per iter
    out_i = decoder(x[:, :, i:i+1], feat_cache=..., first_chunk=(i==0))
    out = cat([out, out_i], dim=2)

New traps vs Encoder3d:
  - per-frame iter (vs encoder's 1+4 chunking)
  - `first_chunk=True` ONLY on frame 0 → DupUp3D drops factor_t-1 frames there
  - asymmetric first-frame T (1) vs subsequent (2 each)

Slot threshold: same approach as Encoder3d (distribution-driven, not pre-set).
"""
from __future__ import annotations

import os
import sys
import importlib

sys.path.insert(0, os.path.abspath("refs/Lance"))

import numpy as np
import torch
import mlx.core as mx

from lance_mlx.vae_wan22 import Decoder3d, Wan22VAEConfig, CausalConv3d
from tools.stage5_pt_compare import mlx_to_pt_state


def cos_pt_mlx(pt: torch.Tensor, mlx_: mx.array) -> float:
    a = pt.detach().to(torch.float32).cpu().numpy().flatten()
    b = np.asarray(mlx_, dtype=np.float32).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def max_abs_diff_nthwc(pt_nctw: torch.Tensor, mlx_nthwc: mx.array) -> float:
    a = np.transpose(pt_nctw.detach().to(torch.float32).cpu().numpy(), (0, 2, 3, 4, 1))
    b = np.asarray(mlx_nthwc, dtype=np.float32)
    return float(np.abs(a - b).max())


def count_mlx_conv3d(module) -> int:
    return sum(1 for m in module.modules() if isinstance(m, CausalConv3d))


def compare_slot(m_entry, p_entry, atol=1e-4):
    if m_entry is None and p_entry is None:
        return True, "both None", None
    if (m_entry is None) ^ (p_entry is None):
        return False, f"MLX={m_entry!r} PT={p_entry!r}", None
    if isinstance(m_entry, str) or isinstance(p_entry, str):
        ok = (m_entry == p_entry)
        return ok, f"MLX={m_entry!r} PT={p_entry!r}", None
    a_pt = np.transpose(p_entry.numpy(), (0, 2, 3, 4, 1))
    a_mx = np.asarray(m_entry)
    if a_pt.shape != a_mx.shape:
        return False, f"shape mismatch MLX={a_mx.shape} PT={a_pt.shape}", None
    d = float(np.abs(a_pt - a_mx).max())
    return d < atol, f"shape={tuple(a_mx.shape)} max|Δ|={d:.3e}", d


def main():
    print("STAGE 8 §1.7 — Decoder3d full-assembly byte-diff (per-frame iter)\n")

    vae_mod = importlib.import_module("modeling.vae.wan.vae2_2")
    PtDecoder3d = vae_mod.Decoder3d
    pt_count_conv3d = vae_mod.count_conv3d

    cfg = Wan22VAEConfig()
    print(f"[setup] Decoder3d(dec_dim={cfg.dec_dim}, z_dim={cfg.z_dim}, "
          f"dim_mult={cfg.dim_mult}, num_res_blocks={cfg.num_res_blocks}, "
          f"temperal_upsample={tuple(reversed(cfg.temperal_downsample))})")
    mx_dec = Decoder3d(cfg, out_channels=12)
    pt_dec = PtDecoder3d(
        dim=cfg.dec_dim, z_dim=cfg.z_dim,
        dim_mult=list(cfg.dim_mult),
        num_res_blocks=cfg.num_res_blocks,
        attn_scales=[],
        temperal_upsample=list(reversed(cfg.temperal_downsample)),
        dropout=cfg.dropout,
    )
    pt_dec.eval()

    # ---- load real Wan2.2 VAE decoder weights ----
    print("[load] loading actual Wan2.2-VAE weights ...")
    mlx_w = mx.load("checkpoints/Wan2.2-VAE-MLX/model.safetensors")
    dec_mlx_w = {k.replace("decoder.", "", 1): v for k, v in mlx_w.items() if k.startswith("decoder.")}
    print(f"[load] {len(dec_mlx_w)} decoder keys in MLX checkpoint")
    mx_dec.load_weights(list(dec_mlx_w.items()), strict=True)
    mx.eval(mx_dec.parameters())
    mx_dec.eval()
    full_pt_state = mlx_to_pt_state(mlx_w)
    dec_pt_state = {k.replace("decoder.", "", 1): v for k, v in full_pt_state.items() if k.startswith("decoder.")}
    missing, unexpected = pt_dec.load_state_dict(dec_pt_state, strict=False)
    print(f"[load] PT load: missing={len(missing)}  unexpected={len(unexpected)}")
    if missing:
        print(f"  first missing: {missing[:3]}")
    if unexpected:
        print(f"  first unexpected: {unexpected[:3]}")

    # ---- slot count ----
    pt_slot_count = pt_count_conv3d(pt_dec)
    mlx_slot_count = count_mlx_conv3d(mx_dec)
    print(f"\n[slot count]  PT={pt_slot_count}  MLX={mlx_slot_count}  "
          f"{'PASS' if pt_slot_count == mlx_slot_count else 'FAIL'}")
    if pt_slot_count != mlx_slot_count:
        print("ABORT — slot count mismatch.")
        return

    # ---- prepare input ----
    # Use the actual encoder output from §0 (out/stage8_pt_video_mu.npy):
    # shape (1, 48, 2, 8, 8).  This is post-encoder, post-conv1 latent.
    # Our standalone Decoder3d test skips Wan2_2_VAE.conv2 (1×1×1 outer
    # adapter) and feeds latent directly — that conv is tested separately
    # at §2 when we build WanVAE_.encode/decode.
    if os.path.exists("out/stage8_pt_video_mu.npy"):
        z_np = np.load("out/stage8_pt_video_mu.npy")
        print(f"\n[input] using §0 saved mu: shape={z_np.shape}  "
              f"mean={z_np.mean():+.4f}  std={z_np.std():.4f}")
    else:
        rng = np.random.default_rng(20260524)
        z_np = (rng.standard_normal((1, 48, 2, 8, 8)).astype("float32") * 1.14 + 0.04)
        print(f"\n[input] synthetic: shape={z_np.shape}")
    z_pt = torch.from_numpy(z_np.copy())                                    # NCTHW
    z_mx = mx.array(np.transpose(z_np, (0, 2, 3, 4, 1)).copy())             # NTHWC
    T_lat = z_np.shape[2]

    # ---- per-frame decode iter ----
    fc_mx = [None] * mlx_slot_count
    fi_mx = [0]
    fc_pt = [None] * pt_slot_count
    fi_pt = [0]

    cumulative_out_mx = []
    cumulative_out_pt = []
    all_slots_ok = True

    for i in range(T_lat):
        first_chunk = (i == 0)
        print("\n" + "=" * 70)
        print(f"FRAME {i}  (T_in=1, first_chunk={first_chunk})")
        print("=" * 70)

        fi_mx[0] = 0; fi_pt[0] = 0
        z_i_pt = z_pt[:, :, i:i+1, :, :]
        z_i_mx = z_mx[:, i:i+1, :, :, :]

        with torch.no_grad():
            out_pt = pt_dec(z_i_pt, feat_cache=fc_pt, feat_idx=fi_pt, first_chunk=first_chunk)
        out_mx = mx_dec(z_i_mx, feat_cache=fc_mx, feat_idx=fi_mx, first_chunk=first_chunk)
        mx.eval(out_mx)

        c = cos_pt_mlx(out_pt.permute(0, 2, 3, 4, 1).contiguous(), out_mx)
        d = max_abs_diff_nthwc(out_pt, out_mx)
        T_mx, T_pt = out_mx.shape[1], out_pt.shape[2]
        expected_T = 1 if first_chunk else 4   # PT decode pattern: first chunk T_out=1, subsequent T_out=4
        T_ok = (T_mx == T_pt == expected_T)
        print(f"  output T  MLX={T_mx}  PT={T_pt}  (expected {expected_T})  "
              f"{'PASS' if T_ok else 'FAIL'}")
        print(f"  output shape  MLX={out_mx.shape}  PT={tuple(out_pt.shape)}")
        print(f"  cos = {c:.8f}  max|Δ| = {d:.3e}  "
              f"{'PASS' if c >= 0.999 else 'FAIL'}")
        print(f"  feat_idx counter  MLX={fi_mx[0]}  PT={fi_pt[0]}  "
              f"(of {mlx_slot_count} allocated)  "
              f"{'PASS' if fi_mx[0] == fi_pt[0] else 'FAIL'}")

        cumulative_out_mx.append(out_mx)
        cumulative_out_pt.append(out_pt)

        # slot distribution (distribution-driven gating per user directive)
        n_pass = 0; n_fail = 0; first_fail = None
        deltas = []
        for s in range(mlx_slot_count):
            ok, info, delta = compare_slot(fc_mx[s], fc_pt[s], atol=1e-4)
            if delta is not None:
                deltas.append(delta)
            if ok:
                n_pass += 1
            else:
                n_fail += 1
                all_slots_ok = False
                if first_fail is None:
                    first_fail = (s, info)
        if deltas:
            d_arr = np.array(deltas)
            print(f"  [slots]  pass={n_pass}/{mlx_slot_count}  fail={n_fail}  "
                  f"max|Δ| distribution: median={np.median(d_arr):.3e}  "
                  f"p90={np.quantile(d_arr, 0.9):.3e}  max={d_arr.max():.3e}")
            if first_fail:
                print(f"           first fail slot[{first_fail[0]}]: {first_fail[1]}")
        print(f"  [slots]  {'PASS' if n_fail == 0 else 'FAIL'}")

    # ---- cumulative final pixel output ----
    print("\n" + "=" * 70)
    print("CUMULATIVE PIXEL OUTPUT (per-frame concat along T)")
    print("=" * 70)
    final_mx = mx.concatenate(cumulative_out_mx, axis=1)
    mx.eval(final_mx)
    final_pt = torch.cat(cumulative_out_pt, dim=2)
    c_final = cos_pt_mlx(final_pt.permute(0, 2, 3, 4, 1).contiguous(), final_mx)
    d_final = max_abs_diff_nthwc(final_pt, final_mx)
    print(f"  final pixel shape  MLX={final_mx.shape}  PT={tuple(final_pt.shape)}")
    print(f"  cos = {c_final:.8f}  max|Δ| = {d_final:.3e}  "
          f"{'PASS' if c_final >= 0.999 else 'FAIL'}")
    expected_total_T = 1 + 4 * (T_lat - 1)  # PT decode T_lat → pixel T pattern
    print(f"  total T = {final_mx.shape[1]}  (expected {expected_total_T} = 1 + 4*(T_lat-1))  "
          f"{'PASS' if final_mx.shape[1] == expected_total_T else 'FAIL'}")

    print("\n" + "=" * 70)
    print(f"SUMMARY  slot count + per-frame iter + first_chunk + cumulative: "
          f"{'PASS' if all_slots_ok and c_final >= 0.999 else 'FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
