"""Convert PyTorch Lance_3B checkpoint to MLX safetensors.

Source layout (bytedance-research/Lance):
  Lance_3B/model.safetensors  — full Lance state_dict
    language_model.model.embed_tokens.weight
    language_model.model.layers.{N}.self_attn.{q,k,v,o}_proj.{weight,bias}
    language_model.model.layers.{N}.self_attn.{q,k,v,o}_proj_moe_gen.{weight,bias}
    language_model.model.layers.{N}.self_attn.{q,k}_norm.weight
    language_model.model.layers.{N}.self_attn.{q,k}_norm_moe_gen.weight
    language_model.model.layers.{N}.mlp.{gate,up,down}_proj.weight
    language_model.model.layers.{N}.mlp_moe_gen.{gate,up,down}_proj.weight
    language_model.model.layers.{N}.input_layernorm.weight
    language_model.model.layers.{N}.input_layernorm_moe_gen.weight
    language_model.model.layers.{N}.post_attention_layernorm.weight
    language_model.model.layers.{N}.post_attention_layernorm_moe_gen.weight
    language_model.model.norm.weight
    language_model.lm_head.weight                 (tied to embed_tokens in Qwen)
    connector.*                                   (MLPconnector: ViT → LLM)
    vae2llm.{weight,bias}                         (Linear: VAE patch → LLM hidden)
    llm2vae.{weight,bias}                         (Linear: LLM hidden → VAE patch)
    time_embedder.*
    latent_pos_embed.*

The LLM/adapter tensors are all 2-D (Linear) or 1-D (norms) — no conv layers
inside Lance_3B itself.  Conv layouts only matter for ViT (patch_embed) and
VAE, both of which are converted in separate scripts.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import torch
from safetensors import safe_open


# PT dtype string -> mlx dtype.  bfloat16 stays bfloat16 (mlx supports it
# directly); float16/float32 pass through; we *never* upcast here, because
# the source already chose the precision.
_DTYPE_MAP: dict[str, mx.Dtype] = {
    "bfloat16": mx.bfloat16,
    "float16":  mx.float16,
    "float32":  mx.float32,
    "float64":  mx.float32,   # collapse: mlx doesn't run f64 on Metal
    "int64":    mx.int32,
    "int32":    mx.int32,
    "int16":    mx.int16,
    "int8":     mx.int8,
    "uint8":    mx.uint8,
    "bool":     mx.bool_,
}


# ----------------------------------------------------------------------------
# Conv layout transform — UNUSED for Lance_3B (no conv inside), but parked
# here so STAGE 2 (ViT patch_embed) and STAGE 5 (Wan 2.2 VAE) can reuse the
# same convention.
#
# PyTorch conv weight layouts:
#   Conv2d :  (out_channels, in_channels, kH, kW)
#   Conv3d :  (out_channels, in_channels, kT, kH, kW)
#
# MLX conv weight layouts (channels-last):
#   conv2d :  (out_channels, kH, kW, in_channels)
#   conv3d :  (out_channels, kT, kH, kW, in_channels)
#
# i.e. move axis 1 (in_channels) to the very end.  Same rule for 2D and 3D.
# ----------------------------------------------------------------------------
def conv_pt_to_mlx(w: torch.Tensor) -> torch.Tensor:
    """Permute a PT conv kernel to MLX (channels-last) layout.

    No-op if rank < 4 (i.e. Linear weights pass through unchanged).

    Usage in a conversion pipeline: call this *before* dtype conversion,
    e.g. ``_torch_to_mlx(conv_pt_to_mlx(t), dtype)``.  Takes a torch
    tensor, returns a torch tensor.
    """
    if w.dim() == 4:
        return w.permute(0, 2, 3, 1).contiguous()
    if w.dim() == 5:
        return w.permute(0, 2, 3, 4, 1).contiguous()
    return w


_TARGET_FLOAT: dict[str, mx.Dtype] = {
    "float32":  mx.float32,
    "float16":  mx.float16,
    "bfloat16": mx.bfloat16,
}


# Exact-prefix rename map applied when converting Lance_3B.
# PT's TimestepEmbedder uses nn.Sequential, so its two Linear layers land
# as `.mlp.0.*` and `.mlp.2.*`.  RockTalk's MLX port (and our own MLX
# TimestepEmbedder we'll build at STAGE 6) name them `.fc1.*` and `.fc2.*`
# — matching the MLPconnector module in the same PT file.  Renaming here
# gives us byte-for-byte parity with RockTalk and lets the MLX class load
# either checkpoint with the same keys.
_RENAME_PREFIXES: tuple[tuple[str, str], ...] = (
    ("time_embedder.mlp.0.", "time_embedder.fc1."),
    ("time_embedder.mlp.2.", "time_embedder.fc2."),
)


def _rename(key: str) -> str:
    for src, dst in _RENAME_PREFIXES:
        if key.startswith(src):
            return dst + key[len(src):]
    return key


def _torch_to_mlx(t: torch.Tensor, target_dtype: str = "preserve") -> mx.array:
    """Materialize a torch tensor on CPU, convert to mx.array.

    bfloat16 source can't go through numpy directly; we view it as int16
    bits, hand it to mlx, then re-view as bf16.  After landing in mlx, we
    optionally cast floating tensors to target_dtype (integer/bool tensors
    are never cast).
    """
    if t.dtype == torch.bfloat16:
        bits = t.detach().contiguous().view(torch.int16).cpu().numpy()
        arr = mx.array(bits, dtype=mx.int16).view(mx.bfloat16)
    else:
        np_arr = t.detach().cpu().numpy()
        dtype = _DTYPE_MAP.get(str(t.dtype).replace("torch.", ""))
        if dtype is None:
            raise TypeError(f"unsupported dtype {t.dtype}")
        arr = mx.array(np_arr, dtype=dtype)

    if target_dtype != "preserve" and arr.dtype in (mx.bfloat16, mx.float16, mx.float32):
        arr = arr.astype(_TARGET_FLOAT[target_dtype])
    return arr


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0] or None)
    ap.add_argument("--src", type=Path, required=True,
                    help="Path to PyTorch model.safetensors (Lance_3B).")
    ap.add_argument("--dst", type=Path, required=True,
                    help="Path to output MLX safetensors.")
    ap.add_argument("--inspect-only", action="store_true",
                    help="Print PT key summary and exit (no conversion).")
    ap.add_argument("--summary-json", type=Path, default=None,
                    help="If set, dump key→shape/dtype summary as JSON.")
    ap.add_argument("--verify-against", type=Path, default=None,
                    help="Optional reference MLX safetensors (e.g. RockTalk "
                         "Lance-3B-MLX/model.safetensors).  After saving, "
                         "diff key sets and per-key shapes against it.")
    ap.add_argument("--dtype", choices=("preserve", "float32", "float16", "bfloat16"),
                    default="float32",
                    help="Target dtype.  'preserve' keeps the source dtype "
                         "(e.g. bf16 stays bf16); the explicit choices cast "
                         "every floating tensor.  Default float32 matches "
                         "RockTalk Lance-3B-MLX.")
    ap.add_argument("--drop-prefix", action="append", default=[],
                    help="Drop any source key whose dotted prefix matches this "
                         "(repeatable).  Use to exclude e.g. 'connector' when "
                         "it is converted into a separate ViT file.")
    return ap.parse_args()


def open_pt(path: Path):
    """Open a PyTorch safetensors file with safe_open (lazy, mmap)."""
    if not path.exists():
        raise FileNotFoundError(path)
    return safe_open(str(path), framework="pt", device="cpu")


def summarize_keys(f) -> dict[str, dict]:
    """Walk every key once: return {key: {shape, dtype, numel}}."""
    summary = {}
    for k in f.keys():
        t = f.get_tensor(k)
        summary[k] = {
            "shape": list(t.shape),
            "dtype": str(t.dtype).replace("torch.", ""),
            "numel": t.numel(),
        }
    return summary


def main() -> None:
    args = parse_args()

    t0 = time.time()
    with open_pt(args.src) as f:
        summary = summarize_keys(f)

    total_params = sum(v["numel"] for v in summary.values())
    print(f"[load] {args.src}")
    print(f"[load] {len(summary)} tensors, {total_params/1e9:.3f}B params, {time.time()-t0:.1f}s")

    # Bucket by top-level prefix for a high-signal overview.
    buckets: dict[str, int] = {}
    for k in summary:
        top = k.split(".", 1)[0]
        buckets[top] = buckets.get(top, 0) + 1
    print("[load] top-level buckets:")
    for top, n in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"         {n:5d}  {top}")

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2))
        print(f"[load] wrote summary to {args.summary_json}")

    if args.inspect_only:
        return

    # ---- Block 1.2: verbatim key copy + dtype preservation ----------------
    # Lance_3B is pure Linear/RMSNorm.  No conv layouts to transpose.  Every
    # PT key (including every `*_moe_gen.*`) lands in the MLX dict under the
    # same name with the same dtype.  `lm_head.weight` is the only key that
    # may legitimately be absent from PT (Qwen ties it to `embed_tokens`);
    # in that case we leave it out — the MLX runtime will tie it on load.
    drop_prefixes = tuple(args.drop_prefix)
    if drop_prefixes:
        print(f"[conv] dropping keys with prefixes: {drop_prefixes}")
    out: dict[str, mx.array] = {}
    kept_src_numel = 0
    renames: list[tuple[str, str]] = []
    dropped: list[str] = []
    t1 = time.time()
    with open_pt(args.src) as f:
        keys = list(f.keys())
        for i, k in enumerate(keys):
            if any(k == p or k.startswith(p + ".") for p in drop_prefixes):
                dropped.append(k)
                continue
            new_k = _rename(k)
            if new_k != k:
                renames.append((k, new_k))
            out[new_k] = _torch_to_mlx(f.get_tensor(k), args.dtype)
            kept_src_numel += summary[k]["numel"]
            if (i + 1) % 100 == 0:
                print(f"[conv] {i+1}/{len(keys)}  ({(time.time()-t1):.1f}s)")
    print(f"[conv] converted {len(out)} tensors in {time.time()-t1:.1f}s "
          f"(dropped {len(dropped)}, renamed {len(renames)})")
    for src, dst in renames[:10]:
        print(f"[conv]   rename  {src}  ->  {dst}")
    if "lm_head.weight" not in out and "language_model.lm_head.weight" not in out:
        print("[conv]   note: no lm_head.weight in output — MLX backbone "
              "must tie it to embed_tokens at load time")

    # ---- Block 1.3 will: write out, verify shapes/keys vs RockTalk -------
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.dst), out)
    print(f"[save] wrote {args.dst}  ({args.dst.stat().st_size/1e9:.2f} GB)")

    # ---- Block 1.3: self-check + (optional) cross-check vs reference ----
    # Self-check: round-trip load and confirm key count + total params unchanged.
    reload = mx.load(str(args.dst))
    assert isinstance(reload, dict)
    assert len(reload) == len(out), f"reload key count drift {len(reload)} vs {len(out)}"
    reload_numel = sum(int(v.size) for v in reload.values())
    assert reload_numel == kept_src_numel, \
        f"param count drift {reload_numel} vs {kept_src_numel} (kept)"
    print(f"[chk ] self-check OK: {len(reload)} tensors, {reload_numel/1e9:.3f}B params round-trip")

    if args.verify_against is not None:
        ref_path = args.verify_against
        if not ref_path.exists():
            print(f"[chk ] WARN: --verify-against {ref_path} missing, skipping diff")
            return
        ref = mx.load(str(ref_path))
        our_keys = set(reload.keys())
        ref_keys = set(ref.keys())
        only_ours = sorted(our_keys - ref_keys)
        only_ref  = sorted(ref_keys - our_keys)
        shared    = sorted(our_keys & ref_keys)
        print(f"[chk ] vs {ref_path.name}")
        print(f"[chk ]   shared keys     : {len(shared)}")
        print(f"[chk ]   only in ours    : {len(only_ours)}")
        print(f"[chk ]   only in reference: {len(only_ref)}")
        for k in only_ours[:10]:
            print(f"[chk ]     +ours    {k}  {tuple(reload[k].shape)}")
        for k in only_ref[:10]:
            print(f"[chk ]     -theirs  {k}  {tuple(ref[k].shape)}")
        # Per-key shape diff on the intersection.
        shape_mismatches = []
        for k in shared:
            ours = tuple(reload[k].shape)
            theirs = tuple(ref[k].shape)
            if ours != theirs:
                shape_mismatches.append((k, ours, theirs))
        print(f"[chk ]   shape mismatches: {len(shape_mismatches)}")
        for k, a, b in shape_mismatches[:10]:
            print(f"[chk ]     {k}  ours={a}  theirs={b}")
        if not only_ours and not only_ref and not shape_mismatches:
            print("[chk ] PARITY OK — key sets and shapes match reference.")


if __name__ == "__main__":
    main()
