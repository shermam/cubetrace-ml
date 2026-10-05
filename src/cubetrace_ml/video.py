"""The clips' MP4s through PyAV (which bundles FFmpeg): frame counts and frames by index."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np
from PIL import Image


@dataclass
class VideoCount:
    """What the MP4 says of its video stream: the container's header (`container_frames`, `container_ms`)
    and, after a full decode, the frames decoded and their presentation times relative to the first."""

    container_frames: int | None
    container_ms: float | None
    width: int
    height: int
    decoded_frames: int | None = None
    pts_ms: np.ndarray | None = None

    @property
    def frames(self) -> int | None:
        """The count to trust: the decoded one when there is one, else the container's."""
        return self.decoded_frames if self.decoded_frames is not None else self.container_frames

    @property
    def span_ms(self) -> float | None:
        """First to last decoded frame (presentation times); None before a full decode."""
        if self.pts_ms is None or len(self.pts_ms) == 0:
            return None
        return float(self.pts_ms[-1] - self.pts_ms[0])


# Without a decode, FFmpeg need not probe the stream: an MP4's header has its sample table
# (2 ms a clip instead of 40).
HEADER_ONLY = {"probesize": "32", "analyzeduration": "0"}


def count_frames(path: str | Path, *, decode: bool = True) -> VideoCount:
    """The video stream's frame count from the container's header and, with `decode`, by decoding every
    frame (the clips are short: tens of seconds)."""
    with av.open(str(path), options={} if decode else HEADER_ONLY) as container:
        stream = container.streams.video[0]
        header = int(stream.frames) or None
        duration = (
            float(stream.duration * stream.time_base) * 1000.0
            if stream.duration is not None and stream.time_base is not None
            else None
        )
        result = VideoCount(header, duration, stream.codec_context.width, stream.codec_context.height)
        if not decode:
            return result
        stream.thread_type = "AUTO"
        pts = []
        for frame in container.decode(stream):
            pts.append(float(frame.time) * 1000.0 if frame.time is not None else np.nan)
        times = np.asarray(pts, dtype=np.float64)
        result.decoded_frames = len(times)
        result.pts_ms = times - times[0] if len(times) else times
        return result


def crop_box(crop: dict[str, Any] | None, width: int, height: int) -> tuple[int, int, int, int] | None:
    """The clip's `crop` rectangle as a PIL box, clamped to the frame; None for the whole frame."""
    if not crop:
        return None
    left = max(0, min(int(crop["x"]), width - 1))
    top = max(0, min(int(crop["y"]), height - 1))
    right = max(left + 1, min(int(crop["x"]) + int(crop["w"]), width))
    bottom = max(top + 1, min(int(crop["y"]) + int(crop["h"]), height))
    return left, top, right, bottom


def read_frames(
    path: str | Path,
    indices: Iterable[int],
    *,
    crop: dict[str, Any] | None = None,
    height: int | None = None,
) -> dict[int, Image.Image]:
    """The frames at `indices` (in presentation order, 0-based) as RGB images, cropped to `crop` and
    resized to `height` pixels high when given; decoding stops after the last one asked for."""
    wanted = sorted(set(indices))
    out: dict[int, Image.Image] = {}
    if not wanted:
        return out
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for k, frame in enumerate(container.decode(stream)):
            if k > wanted[-1]:
                break
            if k not in wanted:
                continue
            image = frame.to_image()
            box = crop_box(crop, image.width, image.height)
            if box:
                image = image.crop(box)
            if height:
                width = max(1, round(image.width * height / image.height))
                image = image.resize((width, height), Image.Resampling.BILINEAR)
            out[k] = image
    return out
