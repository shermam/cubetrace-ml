"""The clips' MP4s through PyAV (which bundles FFmpeg): frame counts, frames by index, and every frame of a
clip cut, scaled and letterboxed by an FFmpeg filter graph (in C, on the decoder's YUV frames)."""

from __future__ import annotations

from collections.abc import Callable, Iterable
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


def letterbox_size(width: int, height: int, size: int) -> tuple[int, int]:
    """A `width` × `height` rectangle scaled so that its longer side is `size` (the other one rounded)."""
    if width >= height:
        return size, max(1, round(height * size / width))
    return max(1, round(width * size / height)), size


def _graph(stream: Any, filters: list[tuple[str, str]]) -> Any:
    graph = av.filter.Graph()
    node = graph.add_buffer(template=stream)
    for name, args in filters:
        following = graph.add(name, args)
        node.link_to(following)
        node = following
    node.link_to(graph.add("buffersink"))
    graph.configure()
    return graph


def _filtered(path: str | Path, filters: Callable[[int, int], list[tuple[str, str]]]) -> list[np.ndarray]:
    """Every frame of the video through the filter chain `filters(width, height)` gives, as arrays."""
    out = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        graph = _graph(stream, filters(stream.codec_context.width, stream.codec_context.height))
        for frame in container.decode(stream):
            graph.vpush(frame)
            out.append(graph.vpull().to_ndarray())
    return out


def decode_square(path: str | Path, box: tuple[int, int, int, int] | None, size: int) -> np.ndarray:
    """Every frame of the video cut to `box` (x, y, w, h in the video's pixels; None for the whole frame),
    scaled (area averaging) so that its longer side is `size`, and letterboxed, centred on black, to `size`
    × `size`: an (n, size, size, 3) uint8 RGB array. A square box is scaled without bars."""

    def chain(width: int, height: int) -> list[tuple[str, str]]:
        x, y, w, h = box if box is not None else (0, 0, width, height)
        if min(x, y) < 0 or min(w, h) <= 0 or x + w > width or y + h > height:
            raise ValueError(f"crop {w}x{h}+{x}+{y} is not inside the {width}x{height} frame")
        sw, sh = letterbox_size(w, h, size)
        chain = []
        if (x, y, w, h) != (0, 0, width, height):
            chain.append(("crop", f"w={w}:h={h}:x={x}:y={y}:exact=1"))
        chain += [("scale", f"{sw}:{sh}:flags=area"), ("format", "rgb24")]
        if (sw, sh) != (size, size):
            chain.append(("pad", f"{size}:{size}:{(size - sw) // 2}:{(size - sh) // 2}:black"))
        return chain

    frames = _filtered(path, chain)
    return np.stack(frames) if frames else np.zeros((0, size, size, 3), dtype=np.uint8)


def gray_size(width: int, height: int, short_side: int) -> tuple[int, int]:
    """The frame's size scaled so that its shorter side is `short_side` (the longer one rounded, even)."""
    if width <= height:
        return short_side, max(2, 2 * round(height * short_side / width / 2))
    return max(2, 2 * round(width * short_side / height / 2)), short_side


def decode_gray(path: str | Path, short_side: int = 160) -> tuple[np.ndarray, tuple[int, int]]:
    """Every frame of the video in gray, scaled (area averaging) so that its shorter side is `short_side`:
    an (n, h, w) uint8 array, and the video's (width, height)."""
    size: list[tuple[int, int]] = []

    def chain(width: int, height: int) -> list[tuple[str, str]]:
        size.append((width, height))
        w, h = gray_size(width, height, min(short_side, width, height))
        return [("scale", f"{w}:{h}:flags=area"), ("format", "gray")]

    frames = _filtered(path, chain)
    gray = np.stack(frames) if frames else np.zeros((0, 1, 1), dtype=np.uint8)
    return gray, size[0]
