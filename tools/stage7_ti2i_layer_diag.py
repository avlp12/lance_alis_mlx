"""Per-layer cos diagnostic for TI2I.

Builds the v_full sequence once, then forwards through the LLM
*step-by-step* — layer 0, layer 1, ... — comparing PT and MLX hidden
state at each layer.  Reveals which layer first diverges.
"""
from __future__ import annotations

# Reuse the same PT environment shims
import tools.stage7_ti2i_compare as harness  # auto-installs shims via side effects
from tools.stage7_ti2i_compare import (
    PtLanceTI2I, build_sequences, build_mask_pt,
    SPATIAL_DOWNSAMPLE, Z_DIM,
    EDIT_SYSTEM_PROMPT,
    _latent_position_indices, _vae_preprocess,
)
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
from lance_mlx.attn_mask import build_lance_attention_mask


def cos_npt_npmlx(pt_arr, mx_arr):
    a = pt_arr.detach().to(torch.float32).cpu().numpy().flatten()
    b = np.asarray(mx_arr, dtype=np.float32).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))


def main():
    SIZE = 256
    h_lat = w_lat = SIZE // SPATIAL_DOWNSAMPLE
    t_lat = 1
    n_cond = n_noise = t_lat * h_lat * w_lat
    IMAGE = "out/test_synthetic.png"

    # ---- MLX side build ----
    print("[build] LanceLLM (MLX) ...")
    model = LanceLLM(LanceTextConfig())
    load_full_lance(model, "checkpoints/Lance-3B-MLX/model.safetensors")
    model.eval()
    vit = LanceViT(); load_lance_vit(vit, "checkpoints/Lance-3B-MLX/vit.safetensors"); vit.eval()
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
    rng = np.random.default_rng(0)
    x_t_np = rng.standard_normal((n_noise, Z_DIM)).astype("float32")
    x_t = mx.array(x_t_np)
    visual_und_pt = torch.from_numpy(np.asarray(visual_und, dtype=np.float32))
    cond_flat_pt = torch.from_numpy(np.asarray(cond_flat, dtype=np.float32))
    x_t_pt = torch.from_numpy(x_t_np.copy()).to(torch.float32)
    lat_pos_ids = _latent_position_indices(t_lat, h_lat, w_lat)
    lat_pos_ids_pt = torch.from_numpy(np.asarray(lat_pos_ids, dtype=np.int64))
    tok = AutoTokenizer.from_pretrained("checkpoints/Lance-3B-MLX", trust_remote_code=True)

    # v_full layout
    lay = build_sequences(tok, n_vit, n_cond, n_noise, include_text=True, include_vit=True)
    L = lay["L"]
    print(f"v_full L={L}  vit={lay['vit_span']} vae={lay['vae_span']} noise={lay['noise_span']}")

    # positions
    spans = [
        VisionSpec(start=lay["vit_span"][0] - 1, length=n_vit, t=T_g, h=H_g_m, w=W_g_m),
        VisionSpec(start=lay["vae_span"][0] - 1, length=n_cond, t=t_lat, h=h_lat, w=w_lat),
        VisionSpec(start=lay["noise_span"][0] - 1, length=n_noise, t=t_lat, h=h_lat, w=w_lat),
    ]
    pos_mlx = build_positions_for_layout(L, spans)
    pos_np = np.asarray(pos_mlx)
    # pro_type=10
    vs_, ve_ = lay["vit_span"]
    shift = 1000 - int(pos_np[0, 0, vs_])
    pos_np[0, :, vs_:ve_] += shift
    vs_, ve_ = lay["vae_span"]
    ns_, ne_ = lay["noise_span"]
    pos_np[:, :, ns_:ne_] = pos_np[:, :, vs_:ve_]
    pos_pt = torch.from_numpy(pos_np.copy()).contiguous()
    pos_mx = mx.array(pos_np)

    # mask
    _, dense_bool = build_mask_pt(lay, num_heads=16)
    attn_mask_LL_pt = torch.zeros(dense_bool.shape, dtype=torch.bfloat16)
    attn_mask_LL_pt.masked_fill_(~dense_bool, float("-inf"))
    attn_mask_LL_mx = mx.array(np.where(dense_bool.numpy(), 0.0, -np.inf).astype(np.float32))

    # ---- MLX embed (full) ----
    ids = mx.array([lay["ids"]], dtype=mx.int32)
    text_embed_mx = model.language_model.model.embed_tokens(ids)
    embed_mx = text_embed_mx
    vs_vit, ve_vit = lay["vit_span"]
    embed_mx = mx.concatenate([embed_mx[:, :vs_vit, :], visual_und[None, :, :], embed_mx[:, ve_vit:, :]], axis=1)
    t_zero = mx.zeros((1,), dtype=mx.float32)
    t_scalar = mx.array([1.0], dtype=mx.float32)
    vae_cond_embed = model.vae2llm(cond_flat) + model.time_embedder(t_zero) + model.latent_pos_embed(lat_pos_ids)
    vs_vae, ve_vae = lay["vae_span"]
    embed_mx = mx.concatenate([embed_mx[:, :vs_vae, :], vae_cond_embed[None, :, :], embed_mx[:, ve_vae:, :]], axis=1)
    noise_embed = model.vae2llm(x_t) + model.time_embedder(t_scalar) + model.latent_pos_embed(lat_pos_ids)
    ns_n, ne_n = lay["noise_span"]
    embed_mx = mx.concatenate([embed_mx[:, :ns_n, :], noise_embed[None, :, :], embed_mx[:, ne_n:, :]], axis=1)
    # gen_mask
    cols = mx.arange(L)
    gen_mask = (((cols >= vs_vae) & (cols < ve_vae))
                | ((cols >= ns_n) & (cols < ne_n)))[None, :]

    # ---- PT side: build pt model and embed ----
    print("[build] PT model ...")
    pt = PtLanceTI2I(); pt.load_pt(); pt.to_bf16()

    ids_pt = torch.tensor(lay["ids"], dtype=torch.long).unsqueeze(0)
    text_embed_pt = pt.embed_tokens(ids_pt)
    embed_pt = text_embed_pt.clone()
    embed_pt[:, vs_vit:ve_vit, :] = visual_und_pt.to(torch.bfloat16).unsqueeze(0)
    x_comb = torch.cat([cond_flat_pt, x_t_pt], dim=0).to(torch.bfloat16)
    t_per = torch.zeros(n_cond + n_noise); t_per[n_cond:] = 1.0
    pos_comb = torch.cat([lat_pos_ids_pt, lat_pos_ids_pt], dim=0)
    vae_emb = pt.vae2llm(x_comb) + pt.time_embed(t_per) + pt.latent_pos_embed(pos_comb)
    embed_pt[:, vs_vae:ve_vae, :] = vae_emb[:n_cond].unsqueeze(0)
    embed_pt[:, ns_n:ne_n, :] = vae_emb[n_cond:].unsqueeze(0)

    # Routing indices
    all_idx = torch.arange(L, dtype=torch.long)
    gen_mask_pt = torch.zeros(L, dtype=torch.bool)
    gen_mask_pt[vs_vae:ve_vae] = True
    gen_mask_pt[ns_n:ne_n] = True
    packed_gen_idx = all_idx[gen_mask_pt]
    packed_und_idx = all_idx[~gen_mask_pt]
    cos_, sin_ = pt.mrope_cos_sin(pos_pt)

    # initial cos
    c0 = cos_npt_npmlx(embed_pt[0], embed_mx[0])
    print(f"\nlayer  -1 (embed)         cos = {c0:.6f}")

    # walk layer by layer
    h_pt = embed_pt[0]
    h_mx = embed_mx
    sample_lens = [L]
    with torch.no_grad():
        for li, (L_pt, L_mx) in enumerate(zip(pt.layers, model.language_model.model.layers)):
            h_pt = L_pt(
                packed_sequence=h_pt, sample_lens=sample_lens,
                attention_mask=attn_mask_LL_pt,
                packed_position_embeddings=(cos_, sin_),
                packed_und_token_indexes=packed_und_idx,
                packed_gen_token_indexes=packed_gen_idx,
                mode_forward="validation",
            )
            h_mx = L_mx(h_mx, pos_mx, attn_mask_LL_mx, None, gen_mask=gen_mask)
            mx.eval(h_mx)
            c = cos_npt_npmlx(h_pt, h_mx[0])
            # focus on noise slab too
            c_noise = cos_npt_npmlx(h_pt[ns_n:ne_n], h_mx[0, ns_n:ne_n, :])
            print(f"layer {li:3d} hidden          cos = {c:.6f}   noise_slab = {c_noise:.6f}")
            if c < 0.99 and li > 0:
                # Print but keep going
                pass


if __name__ == "__main__":
    main()
