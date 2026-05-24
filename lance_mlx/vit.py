"""Qwen2.5-VL Vision Tower — wraps mlx-vlm `VisionModel` for Lance.

Why wrap rather than re-implement: Lance does *not* modify the ViT
(verified at STAGE 7 §1 — `vit.safetensors` keys are byte-identical to
the keys mlx-vlm's `VisionModel` exposes, modulo a `vision_tower.`
prefix).  The backbone (LLM) was hand-ported because Lance added
qk_norm + `_moe_gen` siblings; the ViT has no such modification, so
mlx-vlm is exactly the right reference.

What we add:
  - Strict-load helper that handles the `vision_tower.*` key prefix.
  - A `LanceViT` thin shell that owns the mlx-vlm instance — gives us a
    clean import point in our codebase (no scattered `from mlx_vlm…`).

Config defaults match `checkpoints/Lance-3B-MLX/vit_config.json` /
`refs/Lance/Lance_3B/llm_config.json:vision_config`:
  depth=32, hidden_size=1280, intermediate_size=3420, num_heads=16,
  patch_size=14, spatial_merge_size=2, temporal_patch_size=2,
  window_size=112, fullatt_block_indexes=[7, 15, 23, 31],
  out_hidden_size=2048 (→ LLM hidden after merger).
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.models.qwen2_5_vl.config import VisionConfig as _MlxVlmVisionConfig
from mlx_vlm.models.qwen2_5_vl.vision import VisionModel as _MlxVlmVisionModel


@dataclass
class LanceViTConfig:
    depth: int = 32
    hidden_size: int = 1280
    intermediate_size: int = 3420
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 14
    spatial_patch_size: int = 14
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    window_size: int = 112
    layer_norm_eps: float = 1e-6
    tokens_per_second: int = 2
    out_hidden_size: int = 2048
    # Qwen2.5-VL alternates window vs full attention.  Lance follows the
    # standard pattern (full at every 8th block).
    fullatt_block_indexes: tuple[int, ...] = (7, 15, 23, 31)

    def to_mlx_vlm(self) -> _MlxVlmVisionConfig:
        return _MlxVlmVisionConfig(
            model_type="qwen2_5_vl",
            depth=self.depth,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_heads=self.num_heads,
            in_channels=self.in_channels,
            patch_size=self.patch_size,
            spatial_patch_size=self.spatial_patch_size,
            spatial_merge_size=self.spatial_merge_size,
            temporal_patch_size=self.temporal_patch_size,
            window_size=self.window_size,
            layer_norm_eps=self.layer_norm_eps,
            tokens_per_second=self.tokens_per_second,
            out_hidden_size=self.out_hidden_size,
            fullatt_block_indexes=list(self.fullatt_block_indexes),
        )


class LanceViT(nn.Module):
    """Thin owner of an mlx-vlm `VisionModel`.

    The MLX checkpoint's keys are `vision_tower.<param>`; mlx-vlm's
    `VisionModel` exposes them as bare `<param>`.  We mirror that with
    `self.vision_tower = VisionModel(...)` so strict-load with the
    checkpoint's prefix just works.
    """
    def __init__(self, cfg: LanceViTConfig | None = None):
        super().__init__()
        cfg = cfg or LanceViTConfig()
        self.cfg = cfg
        self.vision_tower = _MlxVlmVisionModel(cfg.to_mlx_vlm())

    def __call__(self, hidden_states: mx.array,
                 grid_thw: mx.array) -> mx.array:
        """Forward Qwen2.5-VL ViT + merger.

        hidden_states: (N_patches, C·temporal_patch·patch²) flattened patches
                       (PT preprocessing layout — N_patches = ∑ T_i · H_i · W_i).
        grid_thw     : (N_images, 3) — per-image (T, H, W) grid in PATCH units.
        Returns      : (N_tokens, out_hidden_size) — LLM-ready visual tokens
                       (post 2×2 spatial merge).
        """
        return self.vision_tower(hidden_states, grid_thw)


def load_lance_vit(model: LanceViT, path: str) -> dict:
    """Strict-load the ViT checkpoint into `model`.

    Returns load stats.  Raises if key sets don't match exactly.
    """
    from mlx.utils import tree_flatten
    all_w = mx.load(path)
    ours = set(dict(tree_flatten(model.parameters())).keys())
    ckpt = set(all_w.keys())
    missing = ours - ckpt
    extra   = ckpt - ours
    if missing or extra:
        raise RuntimeError(
            f"ViT strict-load mismatch: missing={sorted(missing)[:5]} ({len(missing)})  "
            f"extra={sorted(extra)[:5]} ({len(extra)})"
        )
    model.load_weights(list(all_w.items()), strict=True)
    mx.eval(model.parameters())
    return {"loaded_keys": len(all_w)}
