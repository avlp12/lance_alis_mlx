"""STAGE 7 §2 smoke — X→T on a RockTalk sample image."""
from __future__ import annotations

import time

import mlx.core as mx
from transformers import AutoTokenizer

from lance_mlx.backbone import LanceLLM, LanceTextConfig, load_full_lance
from lance_mlx.vit import LanceViT, load_lance_vit
from lance_mlx.pipelines.x2t import x2t


def main() -> None:
    print("[build] LanceLLM ...")
    t0 = time.time()
    model = LanceLLM(LanceTextConfig())
    load_full_lance(model, "checkpoints/Lance-3B-MLX/model.safetensors")
    model.eval()
    print(f"[load] LLM in {time.time()-t0:.1f}s")

    print("[build] LanceViT ...")
    vit = LanceViT()
    load_lance_vit(vit, "checkpoints/Lance-3B-MLX/vit.safetensors")
    vit.eval()
    print("[load] ViT OK")

    tok = AutoTokenizer.from_pretrained("checkpoints/Lance-3B-MLX", trust_remote_code=True)

    IMAGE = "out/test_synthetic.png"
    QUESTION = "Describe this image briefly."
    print(f"\n[x2t] image: {IMAGE}")
    print(f"[x2t] question: {QUESTION}\n")
    t0 = time.time()
    out = x2t(model, vit, tok, IMAGE, QUESTION, max_new_tokens=60)
    dt = time.time() - t0

    print(f"\n[result] generated {len(out.tokens)} tokens in {dt:.1f}s "
          f"(~{len(out.tokens)/dt:.1f} tok/s)")
    print(f"[result] visual tokens (LLM): {out.n_visual_tokens}")
    print(f"\n>>> {out.text!r}")


if __name__ == "__main__":
    main()
