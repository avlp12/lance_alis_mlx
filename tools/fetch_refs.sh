#!/usr/bin/env bash
# fetch_refs.sh — fetch the upstream PyTorch Lance snapshot used by the
# byte-diff verification harnesses in tools/stage*_compare.py.
#
# Needed ONLY for verification (PT-direct-import).  MLX inference
# (lance_mlx/) never touches refs/.
#
# Why a mirror instead of `huggingface-cli download bytedance-research/Lance`:
#   Our verified snapshot's file hashes match no commit in the *current*
#   bytedance-research/Lance HF history — likely an upstream force-push or a
#   GitHub-vs-HF divergence.  We mirror the exact files we verified STAGE 1–9
#   against so the harnesses reproduce bit-for-bit.  Upstream's latest may
#   differ; that is fine for inference, not for byte-diff.
#
# After fetching we assert the inference_lance.py anchor hash so a wrong/empty
# fetch fails loudly rather than silently passing a different snapshot to the
# harnesses (Lesson 18 — verification must not trust an unverified input).
# README.md is NOT anchored: the mirror ships its own descriptive README (the
# original ByteDance one is preserved alongside as LANCE_ORIGINAL_README.md),
# so its hash intentionally differs.  inference_lance.py is a code file we
# verified against and never modify — one stable code anchor is enough.

set -euo pipefail

MIRROR_REPO="avlp12/lance-pt-snapshot"
DEST="refs/Lance"

# Anchor hash — our verified-against snapshot (md5).  Pinned on a code file
# (inference_lance.py) we never modify, NOT README.md (the mirror's README is
# our own description of the repo, not the original ByteDance one).
ANCHOR_INFER_MD5="85fc504a0148a5e1bfe1c3da4dac914d"

_md5() {
  # macOS `md5 -q` / Linux `md5sum` — print bare hash.
  if command -v md5 >/dev/null 2>&1; then
    md5 -q "$1"
  else
    md5sum "$1" | awk '{print $1}'
  fi
}

echo "[fetch_refs] mirror: $MIRROR_REPO  →  $DEST/"

# Prefer the current `hf` CLI; fall back to the legacy `huggingface-cli`.
# (Recent huggingface_hub ships `hf`; `huggingface-cli` is deprecated and is a
#  no-op stub there — it prints a warning and downloads nothing.)
if command -v hf >/dev/null 2>&1; then
  _HF="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
  _HF="huggingface-cli"
else
  echo "[fetch_refs] ERROR: Hugging Face CLI not found. pip install -U huggingface_hub" >&2
  exit 1
fi

mkdir -p "$DEST"
# No 2>/dev/null here: a hidden stderr is exactly what masked the deprecated-CLI
# failure during STAGE-9 publish verification.  Let download errors be seen.
if ! "$_HF" download "$MIRROR_REPO" --repo-type model --local-dir "$DEST"; then
  echo "" >&2
  echo "[fetch_refs] ERROR: could not download $MIRROR_REPO." >&2
  echo "  The mirror may not be published yet.  See the repo README" >&2
  echo "  (Layout / Setup) for the current refs/Lance fetch instructions." >&2
  echo "  Until then, the tools/stage*_compare.py harnesses cannot run; MLX" >&2
  echo "  inference (lance_mlx/) is unaffected." >&2
  exit 1
fi

# --- anchor verification (Lesson 18) ---
infer="$DEST/inference_lance.py"
if [ ! -f "$infer" ]; then
  echo "[fetch_refs] ERROR: expected file missing after fetch: $infer" >&2
  exit 1
fi

got_infer=$(_md5 "$infer")
if [ "$got_infer" != "$ANCHOR_INFER_MD5" ]; then
  echo "[fetch_refs] ANCHOR MISMATCH: inference_lance.py md5=$got_infer, expected $ANCHOR_INFER_MD5" >&2
  echo "  The fetched snapshot is NOT the one STAGE 1–9 was verified against." >&2
  echo "  Refusing to proceed — harness results would be against a different PT." >&2
  exit 1
fi

echo "[fetch_refs] OK — anchor hash matches the verified snapshot."
echo "             inference_lance.py md5=$got_infer"
echo "[fetch_refs] refs/Lance ready for tools/stage*_compare.py harnesses."
