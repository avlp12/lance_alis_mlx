"""Debug: PT t2i END-TO-END at 768 vs our MLX — the comparison never done.

Generates the SAME prompt/seed on both PT (refs/Lance, non-blind — PT runs its own
denoise loop) and MLX, at 768, then decodes BOTH with the production scale and saves
images + latent/pixel cos.  Answers: is the modest 768 quality PT-faithful (model
limit) or our bug?  Reuses the PT machinery from stage6_pt_denoise_compare (which
verified the latent at 512 only).
"""
from __future__ import annotations
import sys; sys.path.insert(0, ".")

# transformers >=5.9 flash-probe neutraliser BEFORE any PT modeling import
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

import numpy as np, torch, mlx.core as mx
from transformers import AutoTokenizer
from PIL import Image

import tools.stage6_pt_denoise_compare as S6          # PT env + PtLanceModel + helpers
from lance_mlx.backbone import LanceLLM, LanceTextConfig, load_full_lance
from lance_mlx.rope import VisionSpec, build_positions_for_layout
from lance_mlx.scheduler import make_schedule, cfg_velocity, euler_step
from lance_mlx.vae_wan22 import Wan2_2_VAE, Wan22VAEConfig
from lance_mlx.pipelines.t2v import VAE_SCALE_MEAN, VAE_SCALE_STD

PROMPT = "a young woman on the beach"
SEED, STEPS, SHIFT, CFG = 0, 30, 3.5, 4.0
H = W = 768
DS, Z, MAXLAT = 16, 48, 64
h_lat = w_lat = H // DS; t_lat = 1; N = t_lat * h_lat * w_lat


def main():
    tok = AutoTokenizer.from_pretrained("checkpoints/Lance-3B-MLX", trust_remote_code=True)
    pids = tok(PROMPT, add_special_tokens=False)["input_ids"]
    lay = S6.build_layout(pids, N)
    L_c, L_u = len(lay["cond_ids"]), len(lay["uncond_ids"])
    cgs, cge = lay["cond_gen_span"]; ugs, uge = lay["uncond_gen_span"]
    print(f"[768] L_cond={L_c} L_unc={L_u} N={N}")

    cbm, cdb = S6.build_pt_mask(L_c, lay["cond_split_lens"], lay["cond_attn_modes"])
    ubm, udb = S6.build_pt_mask(L_u, lay["uncond_split_lens"], lay["uncond_attn_modes"])
    cmm = S6.pt_mask_to_mlx_additive(cdb); umm = S6.pt_mask_to_mlx_additive(udb)

    cpm = build_positions_for_layout(L_c, [VisionSpec(start=cgs-1, length=N, t=t_lat, h=h_lat, w=w_lat)])
    upm = build_positions_for_layout(L_u, [VisionSpec(start=ugs-1, length=N, t=t_lat, h=h_lat, w=w_lat)])
    cpp = torch.from_numpy(np.asarray(cpm)); upp = torch.from_numpy(np.asarray(upm))

    ti = mx.arange(t_lat).reshape(t_lat,1,1); hi = mx.arange(h_lat).reshape(1,h_lat,1); wi = mx.arange(w_lat).reshape(1,1,w_lat)
    lpi = mx.broadcast_to(ti*MAXLAT*MAXLAT + hi*MAXLAT + wi, (t_lat,h_lat,w_lat)).flatten()
    lpp = torch.from_numpy(np.asarray(lpi))

    def _gm(L, s, e): c = mx.arange(L); return ((c>=s)&(c<e))[None,:]
    cgm = _gm(L_c, cgs, cge); ugm = _gm(L_u, ugs, uge)
    def _ip(L, gs, ge): a = torch.arange(L, dtype=torch.long); return torch.cat([a[:gs], a[ge:]]), a[gs:ge]
    cti, cvi = _ip(L_c, cgs, cge); uti, uvi = _ip(L_u, ugs, uge)

    rng = np.random.default_rng(SEED); xn = rng.standard_normal((N, Z)).astype(np.float32)
    xpt = torch.from_numpy(xn.copy()); xmx = mx.array(xn.copy())
    sch = make_schedule(num_steps=STEPS, timestep_shift=SHIFT)
    ts = np.asarray(sch.timesteps); dts = np.asarray(sch.dts)

    print("[build] MLX + PT models ...")
    mlx_model = LanceLLM(LanceTextConfig()); load_full_lance(mlx_model, S6.MLX_WEIGHTS); mlx_model.eval()
    pt = S6.PtLanceModel(); pt.to_bf16(); pt.load_from_pt_checkpoint()
    for l in pt.layers: l.eval()
    cim = mx.array([lay["cond_ids"]], dtype=mx.int32); uim = mx.array([lay["uncond_ids"]], dtype=mx.int32)
    cip = torch.from_numpy(np.asarray(cim)); uip = torch.from_numpy(np.asarray(uim))

    import time; t0 = time.time()
    for i in range(STEPS):
        t = float(ts[i]); dt = float(dts[i]); tpt = torch.tensor([t]); tmx = mx.array([t])
        with torch.no_grad():
            vcp = pt.forward_to_v(cip, xpt, tpt, lpp, (cgs,cge), cpp, cbm, cti, cvi)
            vup = pt.forward_to_v(uip, xpt, tpt, lpp, (ugs,uge), upp, ubm, uti, uvi)
        vc = vcp.numpy(); vu = vup.numpy(); vb = vu + CFG*(vc-vu)
        s = float(np.clip(np.linalg.norm(vc)/(np.linalg.norm(vb)+1e-8), 0, 1))
        xpt = xpt - torch.from_numpy(vb*s)*dt
        vcm = S6.mlx_forward_to_v(mlx_model, cim, xmx, tmx, lpi, (cgs,cge), cpm, cmm, cgm)
        vum = S6.mlx_forward_to_v(mlx_model, uim, xmx, tmx, lpi, (ugs,uge), upm, umm, ugm)
        xmx = euler_step(xmx, cfg_velocity(vcm, vum, scale=CFG), dt); mx.eval(xmx)
        if (i+1) % 5 == 0:
            print(f"  step {i+1}/{STEPS}  ({time.time()-t0:.0f}s)")

    vae = Wan2_2_VAE(Wan22VAEConfig()); vae.load_weights(list(mx.load("checkpoints/Wan2.2-VAE-MLX/model.safetensors").items()), strict=True); mx.eval(vae.parameters()); vae.eval()
    sc = (mx.array(VAE_SCALE_MEAN), mx.array(1.0/VAE_SCALE_STD))
    def dec(lat_np):
        return np.asarray(vae.decode(mx.array(lat_np.reshape(1,t_lat,h_lat,w_lat,Z)), scale=sc)[0,0])
    pim = dec(xpt.numpy()); mim = dec(np.asarray(xmx))
    lc = float(np.asarray(xpt.numpy(),np.float64).flatten() @ np.asarray(xmx,np.float64).flatten() / (np.linalg.norm(xpt.numpy())*np.linalg.norm(np.asarray(xmx))+1e-12))
    pc = float(pim.flatten().astype(np.float64) @ mim.flatten().astype(np.float64) / (np.linalg.norm(pim)*np.linalg.norm(mim)+1e-12))
    print(f"\n[768 RESULT] final latent cos(PT,MLX)={lc:.6f}  pixel cos={pc:.6f}")
    for nm, im in [("pt", pim), ("mlx", mim)]:
        Image.fromarray(((np.clip(im,-1,1)*0.5+0.5)*255).astype(np.uint8)).save(f"out/debug_pt768_{nm}.png")
        print(f"  saved out/debug_pt768_{nm}.png")
    print("[interpret] PT≈MLX (high cos) + both rough -> model limit (port correct). "
          "PT clean + MLX rough -> our 768 bug.")


if __name__ == "__main__":
    main()
