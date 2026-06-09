"""Video input: frame sampling + decode.

The frame *sampler* is a faithful port of the original PT
`data/video/sampler/frames.py::MultiClipsFrameSampler` (Apache-2.0, ByteDance).
It is pure NumPy and decoder-independent: it computes which frame *indices* to
take, so our indices match PT exactly regardless of the decode backend.

The frame *decode* uses imageio + imageio-ffmpeg (Apple-Silicon-friendly; no
decord).  For byte-diff verification we feed pre-extracted NumPy frame fixtures
instead, so the decoder never enters the verification path — only the model
does (Lesson 23 doctrine).

PT call site (validation_dataset.get_video_tensor_online): for a video it builds
    frames_info = {"clip_indices": [(0, total_frames)], "fps": 24}
i.e. origin fps is hard-coded to 24; the sampler's own sample_fps (target) is
its constructor arg (default 12).
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, NamedTuple

import numpy as np


class FrameSamplerOutput(NamedTuple):
    indices: List[int]
    additional_info: Dict[str, Any]


class MultiClipsFrameSampler:
    """Port of PT MultiClipsFrameSampler — deterministic, pure NumPy.

    Behaviour is byte-identical to the upstream sampler; see
    tools/stage11_frame_sampler_compare.py for the parity gate.
    """

    def __init__(
        self,
        temporal: int = 4,
        sample_fps: int = 12,
        truncate: bool = False,
        max_duration: int = 12,
        length_type: Literal["kn", "kn+1"] = "kn+1",
        assert_seconds: bool = True,
    ):
        self.temporal = temporal
        self.sample_fps = sample_fps
        self.truncate = truncate
        self.max_duration = max_duration
        self.length_type = length_type
        self.assert_seconds = assert_seconds

    def __call__(self, frames_info: Dict[str, Any]) -> FrameSamplerOutput:
        clip_indices = frames_info["clip_indices"]
        origin_fps = frames_info["fps"]

        if self.truncate:
            clip_indices = self.truncate_to_bucket(clip_indices, origin_fps)

        if self.assert_seconds:
            duration_sec = int(round(sum((end - start) / origin_fps for start, end in clip_indices)))
            if not self.truncate:
                duration_sec = min(duration_sec, self.max_duration)
            n_frames = duration_sec * self.sample_fps
            if self.length_type == "kn+1":
                n_frames += 1
        else:
            duration = sum((end - start) / origin_fps for start, end in clip_indices)
            if not self.truncate:
                duration = min(duration, self.max_duration)
            n_frames = int(round(duration * self.sample_fps))
            if self.length_type == "kn+1":
                if n_frames % self.temporal != 0:
                    n_frames = n_frames // self.temporal * self.temporal + 1
                else:
                    n_frames = n_frames // self.temporal * self.temporal + 1 - self.temporal

        clip_n_frames = self.split_n_frames_by_clip(n_frames, clip_indices)
        sample_indices = self.sample_frame_indices(clip_indices, clip_n_frames)
        clip_n_latent_frames = [(n + self.temporal - 1) // self.temporal for n in clip_n_frames]

        return FrameSamplerOutput(
            indices=sample_indices,
            additional_info={
                "clip_n_frames": clip_n_frames,
                "clip_n_latent_frames": clip_n_latent_frames,
            },
        )

    def truncate_to_bucket(self, clip_indices, fps):
        clip_indices = [tuple(index) for index in clip_indices]
        durations = [(end - start) / fps for start, end in clip_indices]
        duration = sum(durations)
        max_duration = min(int(duration), self.max_duration)
        cutoff = duration - max_duration
        if cutoff <= 0:
            return clip_indices

        if durations[-1] - cutoff > durations[0] - cutoff:
            start, end = clip_indices[-1]
            end = min(round((durations[-1] - cutoff) * fps), end) + start
            clip_indices[-1] = (start, end)
        else:
            start, end = clip_indices[0]
            start = max(end - round((durations[0] - cutoff) * fps), start)
            clip_indices[0] = (start, end)
        return clip_indices

    def split_n_frames_by_clip(self, n_frames, clip_indices):
        n_latent_frames = n_frames // self.temporal
        clip_lengths = [end - start for start, end in clip_indices]
        total_length = sum(clip_lengths)
        clip_n_latent_frames = [int(length / total_length * n_latent_frames) for length in clip_lengths]
        n_remains = n_latent_frames - sum(clip_n_latent_frames)
        for i in range(n_remains):
            clip_n_latent_frames[i] += 1
        clip_n_frames = [n * self.temporal for n in clip_n_latent_frames]
        if self.length_type == "kn+1":
            clip_n_frames[0] += 1
        return clip_n_frames

    @staticmethod
    def sample_frame_indices(clip_indices, clip_n_frames):
        shift_clip_indices = []
        accum_n_frames = 0
        for start, end in clip_indices:
            shift_start, shift_end = accum_n_frames, accum_n_frames + (end - start)
            shift_clip_indices.append((shift_start, shift_end))
            accum_n_frames += end - start

        all_sample_indices = []
        for i, ((start, end), (shift_start, shift_end), n_frames) in enumerate(
            zip(clip_indices, shift_clip_indices, clip_n_frames)
        ):
            indices = np.arange(start, end)
            next_shift_start = shift_clip_indices[i + 1][0] if i < len(clip_indices) - 1 else shift_end
            shift_sample_indices = (
                np.linspace(shift_start, next_shift_start - 1, n_frames, dtype=int) - shift_start
            )
            all_sample_indices.extend(indices[shift_sample_indices].tolist())

        return all_sample_indices


def build_video_frames_info(total_frames: int, fps: int = 24) -> Dict[str, Any]:
    """Match PT get_video_tensor_online: single clip over the whole video,
    origin fps hard-coded to 24 by the inference dataset."""
    return {"clip_indices": [(0, total_frames)], "fps": fps}


def read_video_frames(
    path: str,
    sampler: MultiClipsFrameSampler | None = None,
    *,
    origin_fps: int = 24,
) -> tuple[np.ndarray, List[int]]:
    """Decode an mp4 with imageio-ffmpeg and return (frames, indices).

    frames: (T, H, W, 3) uint8 at the sampled indices.  Returns the indices too
    so a caller can persist a NumPy fixture for decoder-independent verification.
    """
    import imageio.v3 as iio  # lazy: only needed for production decode

    # imageio-ffmpeg (FFMPEG plugin) — Apple-Silicon-friendly; no decord/pyav.
    all_frames = np.stack([f for f in iio.imiter(path, plugin="FFMPEG")])  # (N,H,W,3) uint8
    total = int(all_frames.shape[0])
    sampler = sampler or MultiClipsFrameSampler()
    idx = sampler(build_video_frames_info(total, fps=origin_fps)).indices
    return all_frames[idx], idx
