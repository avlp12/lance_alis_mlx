"""STAGE 7 §3 — explicit verification of three suspect bugs ①②③.

User raised three specific concerns; this script verifies each against
the original PT via direct import + byte-diff:

  ①  vae2llm in *understanding direction* (cond latent → LLM space).
     Until §3 this had only ever been used in the GEN direction at
     STAGE 6 (llm2vae).  STAGE 2 outer 9 keys included vae2llm but
     forward verification had never exercised it.
  ②  Edit-mode system prompt byte-equality with refs/Lance common.py:35.
  ③  3-component CFG combine + global-norm rescale formula matches
     refs/Lance lance.py:707-724 term-by-term.

This script does NOT need the LLM forward; it just isolates each
suspect against PT byte-diff.  cos ≥ 0.999999 expected on float32
adapters with same input.
"""
from __future__ import annotations

import os
import sys
import types
import importlib
import importlib.machinery


# ----- PT environment shims (minimal — we only need adapters + tokens) -----
def _install_flash_attn_stub():
    import torch
    import torch.nn.functional as F
    def _shim(q, k, v, cu_seqlens_q, cu_seqlens_k,
              max_seqlen_q, max_seqlen_k, causal=True, **_kw):
        raise NotImplementedError("not used by this diag")
    mock = types.ModuleType("flash_attn")
    mock.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
    mock.flash_attn_varlen_func = _shim
    sys.modules["flash_attn"] = mock
    import transformers.utils.import_utils as _imp_utils
    for fn in ("is_flash_attn_2_available", "is_flash_attn_3_available",
               "is_flash_attn_4_available"):
        setattr(_imp_utils, fn, lambda: False)
    import transformers.utils as _utils
    for fn in ("is_flash_attn_2_available", "is_flash_attn_3_available",
               "is_flash_attn_4_available"):
        setattr(_utils, fn, lambda: False)


def _install_modeling_lance_stub():
    lance_dir = os.path.abspath("refs/Lance/modeling/lance")
    pkg = types.ModuleType("modeling.lance")
    pkg.__path__ = [lance_dir]
    sys.modules["modeling.lance"] = pkg


_install_flash_attn_stub()
sys.path.insert(0, os.path.abspath("refs/Lance"))
_install_modeling_lance_stub()


import numpy as np
import torch
from safetensors import safe_open

import mlx.core as mx
from transformers import AutoTokenizer

from lance_mlx.backbone import LanceLLM, LanceTextConfig, load_full_lance
from lance_mlx.vit import LanceViT, load_lance_vit
from lance_mlx.vae_wan22 import Wan2_2_VAE, Wan22VAEConfig
from lance_mlx.pipelines.x2t import preprocess_image, SPATIAL_MERGE_SIZE
from lance_mlx.pipelines.image_edit import (
    EDIT_SYSTEM_PROMPT, _vae_preprocess, _latent_position_indices, Z_DIM,
    SPATIAL_DOWNSAMPLE,
)
from lance_mlx.scheduler import cfg_velocity_3comp


PT_WEIGHTS  = "checkpoints/Lance/Lance_3B/model.safetensors"
MLX_WEIGHTS = "checkpoints/Lance-3B-MLX/model.safetensors"


def cos_pt_mlx(pt: torch.Tensor, mx_: mx.array) -> float:
    a = pt.detach().to(torch.float32).cpu().numpy().flatten()
    b = np.asarray(mx_, dtype=np.float32).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ----------------------------------------------------------------------------
# ① vae2llm in UND direction
# ----------------------------------------------------------------------------
def diag_vae2llm():
    print("=" * 70)
    print("① vae2llm in UND direction (cond latent → LLM space)")
    print("=" * 70)

    # Load PT vae2llm (Linear 48 → 2048 + bias)
    cfg = LanceTextConfig()
    pt_vae2llm = torch.nn.Linear(48, cfg.hidden_size, bias=True)
    with safe_open(PT_WEIGHTS, framework="pt", device="cpu") as f:
        pt_vae2llm.weight.data = f.get_tensor("vae2llm.weight").to(torch.float32).clone()
        pt_vae2llm.bias.data   = f.get_tensor("vae2llm.bias").to(torch.float32).clone()
    pt_vae2llm.eval()
    print(f"[pt] vae2llm: weight {tuple(pt_vae2llm.weight.shape)}, bias {tuple(pt_vae2llm.bias.shape)}")

    # Load MLX side adapter from full Lance LLM
    model = LanceLLM(LanceTextConfig())
    load_full_lance(model, MLX_WEIGHTS)
    model.eval()
    print(f"[mlx] vae2llm: weight {tuple(model.vae2llm.weight.shape)}")

    # Same input — encode the synthetic gradient via Lance VAE, take cond latent
    vae = Wan2_2_VAE(Wan22VAEConfig())
    vae.load_weights(list(mx.load("checkpoints/Wan2.2-VAE-MLX/model.safetensors").items()),
                     strict=True)
    mx.eval(vae.parameters()); vae.eval()
    vae_in = _vae_preprocess("out/test_synthetic.png", size=256)
    cond_latent = vae.encode(vae_in)
    cond_flat = cond_latent.reshape(-1, Z_DIM)
    cond_flat_np = np.asarray(cond_flat, dtype=np.float32)
    print(f"[in] cond_latent flat: {cond_flat_np.shape}  "
          f"mean={cond_flat_np.mean():+.3f}  std={cond_flat_np.std():.3f}")

    # PT direction
    cond_pt = torch.from_numpy(cond_flat_np.copy())
    with torch.no_grad():
        und_pt = pt_vae2llm(cond_pt)
    # MLX direction
    und_mx = model.vae2llm(cond_flat)
    mx.eval(und_mx)

    c = cos_pt_mlx(und_pt, und_mx)
    print(f"[res] vae2llm(cond_latent) cos(PT, MLX) = {c:.6f}  "
          f"{'PASS' if c >= 0.999999 else 'FAIL'}")
    print(f"      ||PT||  = {float(torch.norm(und_pt)):.3f}")
    print(f"      ||MLX|| = {float(np.linalg.norm(np.asarray(und_mx))):.3f}")
    print()
    return c


# ----------------------------------------------------------------------------
# ② Edit-mode system prompt byte-equality with PT common.py:35
# ----------------------------------------------------------------------------
def diag_prompt():
    print("=" * 70)
    print("② Edit-mode system prompt byte-equal with PT common.py:35")
    print("=" * 70)

    # PT source generates the string via f-string with vision_type='image'
    from data.common import generate_system_prompt
    pt_prompt = generate_system_prompt(system_prompt_type="edit",
                                       vision_type="image")
    mlx_prompt = EDIT_SYSTEM_PROMPT
    print(f"[pt]   bytes={len(pt_prompt.encode('utf-8'))}  chars={len(pt_prompt)}")
    print(f"[mlx]  bytes={len(mlx_prompt.encode('utf-8'))}  chars={len(mlx_prompt)}")
    eq = pt_prompt.encode("utf-8") == mlx_prompt.encode("utf-8")
    print(f"[res]  byte-equal: {eq}  {'PASS' if eq else 'FAIL'}")
    if not eq:
        # Show first diff
        for i, (a, b) in enumerate(zip(pt_prompt, mlx_prompt)):
            if a != b:
                print(f"       first diff at char {i}: PT={a!r} ({ord(a):#06x})  MLX={b!r} ({ord(b):#06x})")
                break

    # Also verify tokenization is byte-equal
    tok = AutoTokenizer.from_pretrained("checkpoints/Lance-3B-MLX", trust_remote_code=True)
    pt_ids  = tok(pt_prompt,  add_special_tokens=False)["input_ids"]
    mlx_ids = tok(mlx_prompt, add_special_tokens=False)["input_ids"]
    tok_eq = pt_ids == mlx_ids
    print(f"[res]  tokenized ids equal: {tok_eq}  ({len(pt_ids)} tokens)  "
          f"{'PASS' if tok_eq else 'FAIL'}")
    print()
    return eq and tok_eq


# ----------------------------------------------------------------------------
# ③ 3-comp CFG formula — diff vs PT lance.py:707-724 term-by-term
# ----------------------------------------------------------------------------
def diag_cfg_3comp():
    print("=" * 70)
    print("③ 3-component CFG formula vs PT lance.py:707-724")
    print("=" * 70)

    # PT formula reference (literal copy from lance.py):
    # line 707: v_t_ = cfg_text_vit_v_t + cfg_text_scale_ * (v_t - cfg_text_v_t) + cfg_vit_scale_  * (cfg_text_v_t - cfg_text_vit_v_t)
    # line 711-714 (global): norm_v_t = norm(v_t); norm_v_t_ = norm(v_t_); scale = clamp(norm_v_t/(norm_v_t_+1e-8), min=cfg_renorm_min, max=1.0)
    # line 724: v_t = v_t_ * scale
    #
    # Variable map:
    #   v_t              = v_full       (all conditions on)
    #   cfg_text_v_t     = v_t_uncond   (text dropped)
    #   cfg_text_vit_v_t = v_tv_uncond  (text + vit dropped)
    print("PT variable map:")
    print("  v_t              ←→ v_full       (cond_hidden_state, line 677-685)")
    print("  cfg_text_v_t     ←→ v_t_uncond   (uncond_forward,    line 690)")
    print("  cfg_text_vit_v_t ←→ v_tv_uncond  (vit-uncond_fwd,    line 705)")
    print()

    # Run with synthetic vectors — compare numerical output
    rng = np.random.default_rng(42)
    v_full        = rng.standard_normal((256, 48)).astype("float32") * 0.8
    v_t_uncond    = rng.standard_normal((256, 48)).astype("float32") * 0.8
    v_tv_uncond   = rng.standard_normal((256, 48)).astype("float32") * 0.8
    cfg_text, cfg_vit, renorm_min = 3.0, 1.5, 0.0

    # MLX
    v_mlx = cfg_velocity_3comp(
        mx.array(v_full), mx.array(v_t_uncond), mx.array(v_tv_uncond),
        cfg_text=cfg_text, cfg_vit=cfg_vit,
        renorm_type="global", renorm_min=renorm_min,
    )
    mx.eval(v_mlx)

    # PT — replicate lance.py:707 + 711-714 + 724 exactly
    v_t = torch.from_numpy(v_full).clone()
    cfg_text_v_t = torch.from_numpy(v_t_uncond).clone()
    cfg_text_vit_v_t = torch.from_numpy(v_tv_uncond).clone()
    v_t_ = cfg_text_vit_v_t + cfg_text * (v_t - cfg_text_v_t) + cfg_vit * (cfg_text_v_t - cfg_text_vit_v_t)
    norm_v_t  = torch.norm(v_t)
    norm_v_t_ = torch.norm(v_t_)
    scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=renorm_min, max=1.0)
    v_pt = v_t_ * scale

    c = cos_pt_mlx(v_pt, v_mlx)
    rel = float(torch.norm(v_pt - torch.from_numpy(np.asarray(v_mlx)))) / float(torch.norm(v_pt))
    print(f"  cfg_text={cfg_text}, cfg_vit={cfg_vit}, renorm=global")
    print(f"  cos(PT, MLX)          = {c:.8f}  {'PASS' if c >= 0.999999 else 'FAIL'}")
    print(f"  rel error ||Δ||/||v|| = {rel:.6e}")
    print(f"  scale (PT) = {float(scale):.6f}   ||v_blend||/||v_full|| = "
          f"{float(norm_v_t_)/float(norm_v_t):.4f}")

    # Also probe edge cases: cfg_text==1.0 (should equal v_full); cfg_vit==0
    print()
    print("  Edge cases:")
    for ct, cv in [(1.0, 0.0), (2.0, 0.0), (1.0, 1.5)]:
        v_mlx_e = cfg_velocity_3comp(
            mx.array(v_full), mx.array(v_t_uncond), mx.array(v_tv_uncond),
            cfg_text=ct, cfg_vit=cv,
            renorm_type="global", renorm_min=renorm_min,
        )
        v_t_ = cfg_text_vit_v_t + ct*(v_t - cfg_text_v_t) + cv*(cfg_text_v_t - cfg_text_vit_v_t)
        norm_v_t_ = torch.norm(v_t_)
        scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=renorm_min, max=1.0)
        v_pt_e = v_t_ * scale
        c_e = cos_pt_mlx(v_pt_e, v_mlx_e)
        print(f"    cfg_text={ct}  cfg_vit={cv}:  cos = {c_e:.8f}")

    print()
    return c >= 0.999999


def main():
    r1 = diag_vae2llm()
    r2 = diag_prompt()
    r3 = diag_cfg_3comp()
    print("=" * 70)
    print(f"SUMMARY")
    print(f"  ① vae2llm UND direction byte-diff   : "
          f"{'PASS' if r1 >= 0.999999 else 'FAIL'} (cos={r1:.6f})")
    print(f"  ② edit prompt byte/token equal       : "
          f"{'PASS' if r2 else 'FAIL'}")
    print(f"  ③ 3-comp CFG formula byte-diff       : "
          f"{'PASS' if r3 else 'FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
