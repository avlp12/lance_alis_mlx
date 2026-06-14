"""Gradio web UI for lance_alis_mlx — a browser front-end over the verified pipelines.

Pure UI layer (rule 3): every handler calls the verified pipeline function verbatim
(t2i / image_edit / x2t / t2v / video_edit / x2t_video) — no new model logic here.
Models are lazy-loaded once and cached: an "image bundle" (image LLM + image ViT)
and a "video bundle" (video LLM + video ViT, latent_pos_embed 31x64^2), with the
Wan 2.2 VAE shared.  First use of a tab loads its bundle (slow once, then cached).

Run:  .venv/bin/python app.py    # then open the printed http://127.0.0.1:7860
"""
from __future__ import annotations

import numpy as np
from PIL import Image
import mlx.core as mx
from transformers import AutoTokenizer

import gradio as gr

from lance_mlx.backbone import LanceLLM, LanceTextConfig, load_full_lance
from lance_mlx.vit import LanceViT, load_lance_vit
from lance_mlx.vae_wan22 import Wan2_2_VAE, Wan22VAEConfig
from lance_mlx.pipelines.t2i import t2i
from lance_mlx.pipelines.x2t import x2t, x2t_video
from lance_mlx.pipelines.image_edit import image_edit
from lance_mlx.pipelines.t2v import t2v, VAE_SCALE_MEAN, VAE_SCALE_STD
from lance_mlx.pipelines.video_edit import video_edit, load_video_edit_models
from lance_mlx.video_io import read_video_frames

# Wan VAE per-channel decode scale (mean, 1/std).  PT's WanVAE.decode ALWAYS
# un-normalizes with this; generated latents live in the normalized space, so
# decode MUST pass it (else the dynamic range is off / oversaturated).
def _vae_scale():
    return (mx.array(VAE_SCALE_MEAN), mx.array(1.0 / VAE_SCALE_STD))

IMAGE_W   = "checkpoints/Lance-3B-MLX/model.safetensors"
IMAGE_VIT = "checkpoints/Lance-3B-MLX/vit.safetensors"
VAE_W     = "checkpoints/Wan2.2-VAE-MLX/model.safetensors"
VIDEO_W   = "out/lance_3b_video_mlx/model.safetensors"
TOK_DIR   = "checkpoints/Lance-3B-MLX"

# -------- lazy model cache (load once, reuse) --------
_C: dict = {}


def _tok():
    if "tok" not in _C:
        _C["tok"] = AutoTokenizer.from_pretrained(TOK_DIR, trust_remote_code=True)
    return _C["tok"]


def _vae():
    if "vae" not in _C:
        v = Wan2_2_VAE(Wan22VAEConfig())
        v.load_weights(list(mx.load(VAE_W).items()), strict=True)
        mx.eval(v.parameters()); v.eval()
        _C["vae"] = v
    return _C["vae"]


def _image_llm():
    if "image_llm" not in _C:
        m = LanceLLM(LanceTextConfig()); load_full_lance(m, IMAGE_W); m.eval()
        _C["image_llm"] = m
    return _C["image_llm"]


def _image_vit():
    if "image_vit" not in _C:
        v = LanceViT(); load_lance_vit(v, IMAGE_VIT); v.eval()
        _C["image_vit"] = v
    return _C["image_vit"]


def _video_bundle():
    """video LLM (latent_pos_embed 31x64^2) + video ViT + VAE (lazy)."""
    if "video_llm" not in _C:
        m, vit, vae = load_video_edit_models(VIDEO_W, VAE_W)
        _C["video_llm"], _C["video_vit"] = m, vit
        _C.setdefault("vae", vae)
    return _C["video_llm"], _C["video_vit"], _C["vae"]


# -------- output converters --------
def _img_to_pil(recon) -> Image.Image:
    """(1, 1, H, W, 3) in [-1, 1] -> PIL."""
    a = np.asarray(recon[0, 0])
    a = (np.clip(a, -1, 1) * 0.5 + 0.5) * 255.0
    return Image.fromarray(a.astype(np.uint8))


def _video_to_mp4(recon, out_path: str, fps: int = 8) -> str:
    """(1, T, H, W, 3) in [-1, 1] -> mp4 file for gr.Video."""
    import imageio
    a = np.asarray(recon[0])
    frames = ((np.clip(a, -1, 1) * 0.5 + 0.5) * 255.0).astype(np.uint8)
    imageio.mimwrite(out_path, list(frames), fps=fps, codec="libx264",
                     output_params=["-pix_fmt", "yuv420p"])
    return out_path


# -------- handlers (call verified pipelines verbatim) --------
def gen_t2i(prompt, height, width, steps, cfg, seed):
    if not (prompt or "").strip():
        raise gr.Error("Enter a prompt.")
    out = t2i(_image_llm(), _tok(), prompt, height=int(height), width=int(width),
              num_steps=int(steps), cfg_scale=float(cfg), seed=int(seed))
    recon = _vae().decode(out["latent"], scale=_vae_scale()); mx.eval(recon)
    return _img_to_pil(recon)


def gen_image_edit(image_path, instruction, size, steps, cfg_text, cfg_vit, seed):
    if not image_path:
        raise gr.Error("Upload a cond image.")
    if not (instruction or "").strip():
        raise gr.Error("Enter an edit instruction.")
    res = image_edit(_image_llm(), _image_vit(), _vae(), _tok(), image_path, instruction,
                     size=int(size), num_steps=int(steps),
                     cfg_text=float(cfg_text), cfg_vit=float(cfg_vit), seed=int(seed))
    return _img_to_pil(res.image_recon)


def ask_x2t(image_path, question, max_new_tokens):
    if not image_path:
        raise gr.Error("Upload an image.")
    if not (question or "").strip():
        raise gr.Error("Enter a question.")
    return x2t(_image_llm(), _image_vit(), _tok(), image_path, question,
               max_new_tokens=int(max_new_tokens)).text


def gen_t2v(prompt, num_frames, height, width, steps, cfg, seed):
    if not (prompt or "").strip():
        raise gr.Error("Enter a prompt.")
    vmodel, _, vae = _video_bundle()
    vid = t2v(prompt, vmodel, _tok(), vae, num_frames=int(num_frames),
              H=int(height), W=int(width), num_steps=int(steps),
              cfg_text_scale=float(cfg), seed=int(seed))
    mx.eval(vid)
    return _video_to_mp4(vid, "out/gradio_t2v.mp4")


def ask_x2t_video(video_path, question, max_new_tokens):
    if not video_path:
        raise gr.Error("Upload a video.")
    if not (question or "").strip():
        raise gr.Error("Enter a question.")
    vmodel, vvit, _ = _video_bundle()
    frames, _idx = read_video_frames(video_path)
    return x2t_video(vmodel, vvit, _tok(), frames, question,
                     max_new_tokens=int(max_new_tokens)).text


def gen_video_edit(video_path, instruction, size, steps, cfg_text, cfg_vit, seed):
    if not video_path:
        raise gr.Error("Upload a cond video.")
    if not (instruction or "").strip():
        raise gr.Error("Enter an edit instruction.")
    vmodel, vvit, vae = _video_bundle()
    frames, _idx = read_video_frames(video_path)
    res = video_edit(vmodel, vvit, vae, _tok(), frames, instruction,
                     vae_size=(int(size), int(size)), num_steps=int(steps),
                     cfg_text=float(cfg_text), cfg_vit=float(cfg_vit), seed=int(seed))
    return _video_to_mp4(res.video_recon, "out/gradio_video_edit.mp4")


# -------- UI --------
def build_ui():
    with gr.Blocks(title="Lance MLX") as demo:
        gr.Markdown("# Lance MLX — local Apple-Silicon inference\n"
                    "Browser front-end over the verified MLX pipelines (t2i / image_edit / "
                    "x2t / t2v / x2t_video / video_edit). First use of an image- or video-tab "
                    "loads that bundle (slow once, then cached). "
                    "**Video tabs are heavy** — keep frames/resolution small.")
        with gr.Tabs():
            # ---- t2i ----
            with gr.Tab("t2i (text → image)"):
                with gr.Row():
                    with gr.Column():
                        p = gr.Textbox(label="Prompt", lines=4,
                                       value="A beautiful girl, half-body portrait, ultra detailed "
                                             "features, warm light on her hair, holding snowflakes, "
                                             "some falling on her head, romantic ethereal mood, "
                                             "cold atmospheric scene, cinematic.")
                        gr.Markdown("**Resolution:** Lance presets are 256/512/768 (`image_*res`) but only "
                                    "**768–1024** are clean in practice (≤512 breaks; 1024 sharper but may "
                                    "add garbled text). **Prompt:** Lance is very prompt-sensitive — short "
                                    "prompts (\"a woman on a beach\") look rough; **detailed** prompts "
                                    "(see default) match the official-example quality.")
                        with gr.Row():
                            h = gr.Slider(768, 1024, value=768, step=128, label="Height")
                            w = gr.Slider(768, 1024, value=768, step=128, label="Width")
                        with gr.Row():
                            st = gr.Slider(4, 50, value=30, step=1, label="Steps")
                            cf = gr.Slider(1.0, 8.0, value=4.0, step=0.5, label="CFG")
                            sd = gr.Number(value=0, precision=0, label="Seed")
                        b = gr.Button("Generate", variant="primary")
                    o = gr.Image(label="Result", type="pil")
                b.click(gen_t2i, [p, h, w, st, cf, sd], o)

            # ---- image_edit ----
            with gr.Tab("image_edit (image + text → image) ⚠ slow"):
                gr.Markdown("Edit at the trained **768** size (≤512 degrades, like t2i). 3-component "
                            "CFG → ~3–4 min per edit. Decode uses the production VAE scale.")
                with gr.Row():
                    with gr.Column():
                        im = gr.Image(label="Cond image", type="filepath")
                        ins = gr.Textbox(label="Edit instruction", lines=2, value="Turn it into a snowy winter scene with falling snow.")
                        with gr.Row():
                            sz = gr.Slider(512, 1024, value=768, step=128, label="Size")
                            st2 = gr.Slider(4, 40, value=24, step=1, label="Steps")
                        with gr.Row():
                            ct = gr.Slider(1.0, 8.0, value=3.0, step=0.5, label="CFG text")
                            cv = gr.Slider(0.0, 4.0, value=1.0, step=0.5, label="CFG ViT")
                            sd2 = gr.Number(value=0, precision=0, label="Seed")
                        b2 = gr.Button("Edit", variant="primary")
                    o2 = gr.Image(label="Edited", type="pil")
                b2.click(gen_image_edit, [im, ins, sz, st2, ct, cv, sd2], o2)

            # ---- x2t (image) ----
            with gr.Tab("x2t (image → text)"):
                with gr.Row():
                    with gr.Column():
                        im3 = gr.Image(label="Image", type="filepath")
                        q = gr.Textbox(label="Question", lines=2, value="Describe this image.")
                        mt = gr.Slider(8, 256, value=60, step=4, label="Max new tokens")
                        b3 = gr.Button("Ask", variant="primary")
                    o3 = gr.Textbox(label="Answer", lines=8)
                b3.click(ask_x2t, [im3, q, mt], o3)

            # ---- t2v ----
            with gr.Tab("t2v (text → video) ⚠ heavy"):
                gr.Markdown("Video runs at a **video preset** (~480 = 360p; 256=192p, 640=480p) — "
                            "lower than image. ~480 / 13 frames ≈ 90 s; higher res/frames is *much* slower. "
                            "Detailed prompts help.")
                with gr.Row():
                    with gr.Column():
                        pv = gr.Textbox(label="Prompt", lines=3,
                                        value="A red fox walking through a snowy forest, cinematic, detailed fur, soft winter light, falling snow.")
                        with gr.Row():
                            nf = gr.Slider(5, 25, value=13, step=4, label="Frames")
                            hv = gr.Slider(256, 640, value=480, step=64, label="Height")
                            wv = gr.Slider(256, 640, value=480, step=64, label="Width")
                        with gr.Row():
                            stv = gr.Slider(4, 30, value=20, step=1, label="Steps")
                            cfv = gr.Slider(1.0, 8.0, value=4.0, step=0.5, label="CFG")
                            sdv = gr.Number(value=0, precision=0, label="Seed")
                        bv = gr.Button("Generate", variant="primary")
                    ov = gr.Video(label="Result")
                bv.click(gen_t2v, [pv, nf, hv, wv, stv, cfv, sdv], ov)

            # ---- x2t_video ----
            with gr.Tab("x2t_video (video → text) ⚠ heavy"):
                with gr.Row():
                    with gr.Column():
                        vv = gr.Video(label="Video")
                        qv = gr.Textbox(label="Question", lines=2, value="What happens in this video?")
                        mtv = gr.Slider(8, 256, value=60, step=4, label="Max new tokens")
                        bvq = gr.Button("Ask", variant="primary")
                    ovq = gr.Textbox(label="Answer", lines=8)
                bvq.click(ask_x2t_video, [vv, qv, mtv], ovq)

            # ---- video_edit ----
            with gr.Tab("video_edit (video + text → video) ⚠ very heavy"):
                gr.Markdown("Edits the cond video at a **video preset** (~480). 3-component CFG + the "
                            "cond video's frame count drive the cost — **use a SHORT clip** (a long "
                            "video → many latent frames → minutes). ~480 / 13 frames ≈ 4–5 min.")
                with gr.Row():
                    with gr.Column():
                        vve = gr.Video(label="Cond video (keep it short)")
                        inse = gr.Textbox(label="Edit instruction", lines=2, value="Make the scene look like a snowy winter day.")
                        with gr.Row():
                            sze = gr.Slider(256, 640, value=480, step=64, label="Size")
                            ste = gr.Slider(4, 24, value=16, step=1, label="Steps")
                        with gr.Row():
                            cte = gr.Slider(1.0, 8.0, value=3.0, step=0.5, label="CFG text")
                            cve = gr.Slider(0.0, 4.0, value=1.0, step=0.5, label="CFG ViT")
                            sde = gr.Number(value=0, precision=0, label="Seed")
                        bve = gr.Button("Edit", variant="primary")
                    ove = gr.Video(label="Edited")
                bve.click(gen_video_edit, [vve, inse, sze, ste, cte, cve, sde], ove)
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860, show_error=True)
