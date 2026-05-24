# UPSTREAM — bugs/issues to report to upstream projects

Parked here so the porting work continues uninterrupted.  Each entry has a
*minimal repro* ready to drop into the upstream issue when the time comes.

---

## mlx-vlm: `Qwen2_5_VL.LanguageModel.get_rope_index` mis-handles multi-image inputs

**Repo:** https://github.com/Blaizzy/mlx-vlm
**File:** `mlx_vlm/models/qwen2_5_vl/language.py`
**Affected versions:** 0.5.0 (and likely earlier — the code shape goes back
to the initial Qwen2-VL port).
**Severity:** Silent incorrect output for any sequence containing more than
one image / video.  Single-image inputs are correct.

### Symptom

For a token sequence containing two `<vision_start>` … `<vision_end>` spans,
`get_rope_index` assigns image-grid positions only to the *first
placeholder* of the second span; the rest get sequential text positions.
The first span is unaffected.

Concretely, for the input layout

```
[text × 5] <vis_start> [image × 16] <vis_end> [text × 3] <vis_start> [image × 36] <vis_end> [text × 4]
```

with `image_grid_thw = [[1, 8, 8], [1, 12, 12]]` (which yields LLM-side
grids `(1, 4, 4)` and `(1, 6, 6)` after spatial-merge), the function
returns positions like:

```
col 27 (img2 placeholder 0): [15, 15, 15]      ← correct image position
col 28 (img2 placeholder 1): [16, 16, 16]      ← wrong (should be [15, 15, 16])
col 29 (img2 placeholder 2): [17, 17, 17]      ← wrong (should be [15, 15, 17])
...all remaining img2 placeholders fall into text-position scheme...
```

### Root cause

In `LanguageModel.get_rope_index` (lines ~300):

```python
vision_start_indices = mx.sum(
    mx.where(
        input_ids == vision_start_token_id,
        mx.arange(input_ids.shape[0]),
        mx.zeros_like(input_ids),
    )
)
vision_tokens = input_ids[vision_start_indices + 1]
image_nums = (vision_tokens == image_token_id).sum().item()
```

`mx.sum(...)` collapses all vision_start positions into a *scalar* (their
sum) instead of producing an array of vision_start indices.
`vision_tokens` then reads exactly one token at that summed-index
position, giving a meaningless count.  The outer loop iterates
`image_nums + video_nums` times but each iteration only processes one
placeholder from the second image onward because `st` is advanced
incorrectly.

### Minimal repro

```python
import mlx.core as mx
from mlx_vlm.models.qwen2_5_vl.config import ModelConfig, TextConfig, VisionConfig
from mlx_vlm.models.qwen2_5_vl.language import LanguageModel

tc = TextConfig(model_type="qwen2_5_vl", hidden_size=2048, num_hidden_layers=36,
    intermediate_size=11008, num_attention_heads=16, num_key_value_heads=2,
    rms_norm_eps=1e-6, vocab_size=151936, max_position_embeddings=128000,
    rope_theta=1e6, rope_scaling={"type": "mrope", "mrope_section": [16,24,24]},
    tie_word_embeddings=True)
mc = ModelConfig(text_config=tc,
    vision_config=VisionConfig(model_type="qwen2_5_vl", depth=32, hidden_size=1280,
        intermediate_size=3420, num_heads=16, in_channels=3, out_hidden_size=2048),
    image_token_id=151655, video_token_id=151656, vision_start_token_id=151652,
    vision_end_token_id=151653, vision_token_id=151654, model_type="qwen2_5_vl")
lm = LanguageModel(tc, mc)

VS, VE, IT = 151652, 151653, 151655
parts = ([100]*5 + [VS] + [IT]*16 + [VE] + [100]*3
                 + [VS] + [IT]*36 + [VE] + [200]*4)
ids = mx.array([parts], dtype=mx.int32)
grid_thw = mx.array([[1, 8, 8], [1, 12, 12]], dtype=mx.int32)

pos, _ = lm.get_rope_index(ids, image_grid_thw=grid_thw)
# Expected (transformers parity): pos[:, 0, 28] == [15, 15, 16]
# Actual mlx-vlm:                  pos[:, 0, 28] == [16, 16, 16]
print("pos[:, 0, 28] =", pos[:, 0, 28].tolist())
print("pos[:, 0, 33] =", pos[:, 0, 33].tolist())
```

### Suggested fix sketch

Replace the scalar `mx.sum` reduction with an actual collection of
positions (e.g. `mx.where(...).nonzero()` equivalent or a Python list
comprehension if MLX doesn't support nonzero on bool tensors).  The
existing inner loop already iterates per image — it just needs the right
`image_nums` and `st`-advance state per iteration.  Cross-check against
the transformers Qwen2.5-VL `Qwen2_5_VLModel.get_rope_index` reference
which handles this correctly.

### Tracking

Status: noted, not yet filed.  Apply: file an issue with the repro above
when the lance-mlx port reaches a natural pause.
