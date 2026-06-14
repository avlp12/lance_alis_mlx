"""Debug: t2i step-0 single-forward cos PT(bf16) vs MLX(f32) at 512 and 768.

Q: is the 768 final-latent divergence (cos 0.9448 vs 0.999 at 512) a structural
per-forward bug, or 30-step accumulation of bf16(PT)/f32(MLX) rounding?
  - 768 step-0 cos still ~0.999  -> per-forward fine -> divergence is accumulation (numerical).
  - 768 step-0 cos noticeably low -> per-forward diverges at 768 (structural or precision-at-scale).
512 step-0 should be ~0.999 (matches STAGE 6 step 1).
"""
from __future__ import annotations
import sys; sys.path.insert(0, ".")
import transformers.utils.import_utils as _iu, transformers.utils as _tu, transformers.modeling_flash_attention_utils as _mfa
def _false(*a, **k): return False
for _m in (_iu, _tu, _mfa):
    for _fn in ("is_flash_attn_2_available","is_flash_attn_3_available","is_flash_attn_4_available","flash_attn_supports_top_left_mask"):
        if hasattr(_m, _fn): setattr(_m, _fn, _false)

import numpy as np, torch, mlx.core as mx
from transformers import AutoTokenizer
import tools.stage6_pt_denoise_compare as S6
from lance_mlx.backbone import LanceLLM, LanceTextConfig, load_full_lance
from lance_mlx.rope import VisionSpec, build_positions_for_layout
from lance_mlx.scheduler import make_schedule

PROMPT = "a young woman on the beach"; SEED, SHIFT, CFG, DS, Z, MAX = 0, 3.5, 4.0, 16, 48, 64


def _cos(a, b):
    a = np.asarray(a, np.float64).flatten(); b = np.asarray(b, np.float64).flatten()
    return float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b)+1e-12))


def main():
    tok = AutoTokenizer.from_pretrained("checkpoints/Lance-3B-MLX", trust_remote_code=True)
    pids = tok(PROMPT, add_special_tokens=False)["input_ids"]
    print("[build] MLX + PT(bf16) models (once) ...")
    mlx_model = LanceLLM(LanceTextConfig()); load_full_lance(mlx_model, S6.MLX_WEIGHTS); mlx_model.eval()
    pt = S6.PtLanceModel(); pt.to_bf16(); pt.load_from_pt_checkpoint()
    for l in pt.layers: l.eval()
    t0 = float(np.asarray(make_schedule(num_steps=30, timestep_shift=SHIFT).timesteps)[0])  # =1.0

    for H in (512, 768):
        h_lat = w_lat = H // DS; t_lat = 1; N = t_lat*h_lat*w_lat
        lay = S6.build_layout(pids, N)
        L_c = len(lay["cond_ids"]); cgs, cge = lay["cond_gen_span"]
        cbm, cdb = S6.build_pt_mask(L_c, lay["cond_split_lens"], lay["cond_attn_modes"])
        cmm = S6.pt_mask_to_mlx_additive(cdb)
        cpm = build_positions_for_layout(L_c, [VisionSpec(start=cgs-1, length=N, t=t_lat, h=h_lat, w=w_lat)])
        cpp = torch.from_numpy(np.asarray(cpm))
        ti = mx.arange(t_lat).reshape(t_lat,1,1); hi = mx.arange(h_lat).reshape(1,h_lat,1); wi = mx.arange(w_lat).reshape(1,1,w_lat)
        lpi = mx.broadcast_to(ti*MAX*MAX + hi*MAX + wi, (t_lat,h_lat,w_lat)).flatten()
        lpp = torch.from_numpy(np.asarray(lpi))
        cgm = ((mx.arange(L_c) >= cgs) & (mx.arange(L_c) < cge))[None, :]
        a = torch.arange(L_c, dtype=torch.long); cti = torch.cat([a[:cgs], a[cge:]]); cvi = a[cgs:cge]

        rng = np.random.default_rng(SEED); xn = rng.standard_normal((N, Z)).astype(np.float32)
        cim = mx.array([lay["cond_ids"]], dtype=mx.int32); cip = torch.from_numpy(np.asarray(cim))

        with torch.no_grad():
            vpt = pt.forward_to_v(cip, torch.from_numpy(xn.copy()), torch.tensor([t0]), lpp,
                                  (cgs, cge), cpp, cbm, cti, cvi).numpy()
        vmx = np.asarray(S6.mlx_forward_to_v(mlx_model, cim, mx.array(xn.copy()), mx.array([t0]),
                                             lpi, (cgs, cge), cpm, cmm, cgm))
        print(f"[res {H}] L_cond={L_c} N={N}  step-0 v_cond cos(PT-bf16, MLX-f32) = {_cos(vpt, vmx):.6f}  "
              f"max|Δ|={np.abs(vpt-vmx).max():.4f}")


if __name__ == "__main__":
    main()
