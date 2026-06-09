"""GATE (STAGE 11 step 1) — our MultiClipsFrameSampler == PT's, index-for-index.

Loads the original PT sampler from a file and runs it side-by-side with our
port (lance_mlx/video_io.py) over a battery of clip configurations + sampler
params.  The frame indices must match exactly — this is the prerequisite for
every video task verification (x2t_video / video_edit): if we sample different
frames than PT, nothing downstream can byte-diff.

PT source: data/video/sampler/frames.py from the verified snapshot
(refs/Lance/ after fetch_refs.sh, or the lance-pt-snapshot mirror).  Pass its
path as argv[1]; defaults try the usual locations.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, ".")

from lance_mlx.video_io import MultiClipsFrameSampler as OurSampler


def load_pt_sampler(path: str):
    spec = importlib.util.spec_from_file_location("pt_frames", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MultiClipsFrameSampler


def main() -> None:
    candidates = [
        sys.argv[1] if len(sys.argv) > 1 else None,
        "refs/Lance/data/video/sampler/frames.py",
        "/tmp/pt_frames.py",
    ]
    pt_path = next((p for p in candidates if p and Path(p).exists()), None)
    if pt_path is None:
        print("[gate] ERROR: PT frames.py not found. Pass its path as argv[1] "
              "(refs/Lance/data/video/sampler/frames.py after fetch_refs.sh).")
        sys.exit(1)
    PTSampler = load_pt_sampler(pt_path)
    print(f"[gate] PT sampler from {pt_path}")

    # Param sets to exercise (incl. the inference defaults: temporal=4,
    # sample_fps=12, max_duration=12, length_type=kn+1).
    param_sets = [
        dict(temporal=4, sample_fps=12, max_duration=12, length_type="kn+1"),
        dict(temporal=4, sample_fps=12, max_duration=12, length_type="kn"),
        dict(temporal=4, sample_fps=8,  max_duration=8,  length_type="kn+1"),
        dict(temporal=4, sample_fps=12, max_duration=12, length_type="kn+1", assert_seconds=False),
        dict(temporal=4, sample_fps=12, max_duration=4,  length_type="kn+1", truncate=True),
    ]
    # Single-clip (the inference case, fps=24 hard-coded) over many lengths,
    # plus a couple of multi-clip configs.
    frames_infos = [
        {"clip_indices": [(0, n)], "fps": 24}
        for n in [1, 5, 12, 17, 24, 25, 48, 96, 120, 121, 145, 240, 289, 300, 480, 721]
    ] + [
        {"clip_indices": [(0, 120), (200, 360)], "fps": 24},
        {"clip_indices": [(10, 130), (300, 500), (600, 700)], "fps": 30},
    ]

    def run(sampler, fi):
        # Compare behaviour INCLUDING exceptions: a degenerate clip can drive
        # n_frames negative in BOTH PT and our port (identical logic), so a
        # shared raise is a match, not a mismatch.
        try:
            return ("ok", sampler(fi).indices)
        except Exception as e:                       # noqa: BLE001
            return ("err", type(e).__name__)

    total = mism = shared_err = 0
    fails = []
    for ps in param_sets:
        ours = OurSampler(**ps)
        theirs = PTSampler(**ps)
        for fi in frames_infos:
            total += 1
            o = run(ours, fi)
            t = run(theirs, fi)
            if o != t:
                mism += 1
                fails.append((ps, fi, o, t))
            elif o[0] == "err":
                shared_err += 1

    print(f"[gate] cases: {total}  mismatches: {mism}  "
          f"(shared-raise edge cases that match: {shared_err})")
    for ps, fi, o, t in fails[:8]:
        def _d(r):
            return f"err:{r[1]}" if r[0] == "err" else f"n={len(r[1])}"
        print(f"  MISMATCH params={ps} clips={fi['clip_indices']} fps={fi['fps']} "
              f"ours=[{_d(o)}] pt=[{_d(t)}]")
    # Show one concrete sampled-index example (inference defaults, 121-frame clip).
    ex = OurSampler()({"clip_indices": [(0, 121)], "fps": 24}).indices
    print(f"[gate] example (defaults, 121-frame @ fps24): n={len(ex)} indices={ex}")
    print("=" * 56)
    print("GATE step 1:", "PASS — frame indices byte-identical to PT"
          if mism == 0 else f"FAIL — {mism} mismatches")
    sys.exit(0 if mism == 0 else 1)


if __name__ == "__main__":
    main()
