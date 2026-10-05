"""Where a clip's frames are cut before they are scaled to an encoder's square input.

The laptop's clips carry the framing rectangle the owner set in the app (`crop`); the phones' record the
whole frame, the cube small in it. `auto` takes the record's rectangle when there is one and otherwise
finds a square around the motion of the segment: the absolute frame differences accumulated over the
frames of the segment's window (the hands and the cube move, the room does not), blurred lightly,
thresholded at a share of their peak, the bounding box of that mass clamped to the 5th–95th percentiles
of its marginals (specks carry too little of it to move them), padded, made square and clamped to the
frame.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .align import align_clip
from .dataset import ClipRef, Dataset
from .video import crop_box, decode_gray, read_frames

CROP_MODES = ("auto", "record", "none")


@dataclass(frozen=True)
class MotionParams:
    """The motion crop's settings (recorded with every crop it finds)."""

    short_side: int = 160  # the gray frames' shorter side, in pixels
    blur_taps: int = 5  # the binomial blur of the energy map (in the gray frames' pixels)
    threshold: float = 0.15  # the share of the blurred energy's peak that counts as motion
    low: float = 5.0  # the percentiles of the mass's marginals the box is clamped to
    high: float = 95.0
    pad: float = 0.15  # each side of the box moves out by this share of the box's size
    min_side: float = 0.25  # the square's least side, as a share of the frame's shorter side

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


MOTION = MotionParams()


@dataclass(frozen=True)
class Framing:
    """A rectangle of the video's frames (x, y, w, h in its pixels), where it came from (`record`, `motion`,
    or `none` for the whole frame) and the frame's size."""

    x: int
    y: int
    w: int
    h: int
    source: str
    width: int
    height: int

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    @property
    def square(self) -> bool:
        return self.w == self.h

    def to_json(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "source": self.source,
            "letterboxed": not self.square,
            "frameWidth": self.width,
            "frameHeight": self.height,
        }

    @classmethod
    def from_json(cls, doc: dict[str, Any]) -> Framing:
        return cls(
            int(doc["x"]),
            int(doc["y"]),
            int(doc["w"]),
            int(doc["h"]),
            str(doc["source"]),
            int(doc["frameWidth"]),
            int(doc["frameHeight"]),
        )

    def describe(self) -> str:
        shape = "square" if self.square else "letterboxed"
        return f"{self.source} {self.w}x{self.h}+{self.x}+{self.y} ({shape})"


def whole_frame(width: int, height: int) -> Framing:
    return Framing(0, 0, width, height, "none", width, height)


def record_framing(crop: dict[str, Any] | None, width: int, height: int) -> Framing | None:
    """The record's `crop`, clamped to the frame; None when the clip has none."""
    box = crop_box(crop, width, height)
    if box is None:
        return None
    left, top, right, bottom = box
    return Framing(left, top, right - left, bottom - top, "record", width, height)


def blur(image: np.ndarray, taps: int = 5) -> np.ndarray:
    """A separable binomial blur (`taps` wide; 1 or less: none), the edges repeated."""
    if taps <= 1:
        return image.astype(np.float64)
    kernel = np.array([1.0])
    for _ in range(taps - 1):
        kernel = np.convolve(kernel, [1.0, 1.0])
    kernel /= kernel.sum()
    half = taps // 2
    out = image.astype(np.float64)
    for axis in (0, 1):
        padded = np.pad(out, [(half, half) if a == axis else (0, 0) for a in (0, 1)], mode="edge")
        n = out.shape[axis]
        out = sum(k * np.take(padded, np.arange(i, i + n), axis=axis) for i, k in enumerate(kernel))
    return out


def motion_energy(gray: np.ndarray, use: np.ndarray | None = None) -> np.ndarray:
    """The sum of |gray[k] − gray[k − 1]| over the frames k ≥ 1 that `use` selects (every frame when it is
    None or selects fewer than two; a frame past its end is not selected), as a float map of the frames'
    size."""
    gray = np.asarray(gray)
    if len(gray) < 2:
        return np.zeros(gray.shape[1:] if gray.ndim == 3 else (1, 1))
    pick = np.ones(len(gray), dtype=bool)
    if use is not None:
        chosen = np.asarray(use, dtype=bool)[: len(gray)]
        pick[:] = False
        pick[: len(chosen)] = chosen
    if pick.sum() < 2:
        pick[:] = True
    pick[0] = False
    energy = np.zeros(gray.shape[1:], dtype=np.float64)
    for k in np.flatnonzero(pick):
        energy += np.abs(gray[k].astype(np.int16) - gray[k - 1].astype(np.int16))
    return energy


def mass_box(
    energy: np.ndarray,
    threshold: float = MOTION.threshold,
    low: float = MOTION.low,
    high: float = MOTION.high,
) -> tuple[int, int, int, int] | None:
    """In the map's pixels, the half-open box (x0, y0, x1, y1) of the energy at or above `threshold` of its
    peak, clamped to the `low`–`high` percentiles of that mass's column and row marginals; None without
    energy."""
    peak = float(energy.max()) if energy.size else 0.0
    if peak <= 0:
        return None
    mass = np.where(energy >= threshold * peak, energy, 0.0)
    spans = []
    for axis in (0, 1):  # summing the rows gives the columns' marginal (x), then the rows' (y)
        marginal = mass.sum(axis=axis)
        present = np.flatnonzero(marginal)
        cdf = np.cumsum(marginal) / marginal.sum()
        first = int(np.searchsorted(cdf, low / 100, side="left"))
        last = int(np.searchsorted(cdf, high / 100, side="left"))
        spans.append((max(int(present[0]), first), min(int(present[-1]), last) + 1))
    (x0, x1), (y0, y1) = spans
    return x0, y0, x1, y1


def square_around(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    *,
    pad: float = MOTION.pad,
    min_side: float = MOTION.min_side,
) -> tuple[int, int, int]:
    """The square (x, y, side) around `box` (x0, y0, x1, y1 in the frame's pixels): each side moved out by
    `pad` of the box's size, the shorter dimension grown to the longer about the centre (at least
    `min_side` of the frame's shorter side, at most that side), then shifted into the frame; whole, even
    pixels."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    x0, x1, y0, y1 = x0 - pad * w, x1 + pad * w, y0 - pad * h, y1 + pad * h
    short = min(width, height)
    side = min(max(x1 - x0, y1 - y0, min_side * short), short)
    side = max(2, int(side) // 2 * 2)
    x = min(max(round((x0 + x1 - side) / 2), 0), width - side)
    y = min(max(round((y0 + y1 - side) / 2), 0), height - side)
    return x - x % 2, y - y % 2, side


def motion_framing(
    gray: np.ndarray,
    frame_size: tuple[int, int],
    use: np.ndarray | None = None,
    params: MotionParams = MOTION,
) -> tuple[Framing, np.ndarray]:
    """The square around the motion of the gray frames (`decode_gray`'s, of a video of `frame_size`) that
    `use` selects, and the blurred energy map it came from; the whole frame when nothing moves."""
    width, height = frame_size
    energy = blur(motion_energy(gray, use), params.blur_taps)
    box = mass_box(energy, params.threshold, params.low, params.high)
    if box is None:
        return whole_frame(width, height), energy
    rows, cols = energy.shape
    sx, sy = width / cols, height / rows
    x0, y0, x1, y1 = box
    x, y, side = square_around(
        (x0 * sx, y0 * sy, x1 * sx, y1 * sy), width, height, pad=params.pad, min_side=params.min_side
    )
    return Framing(x, y, side, side, "motion", width, height), energy


def find_motion(
    path: str | Path, use: np.ndarray | None = None, params: MotionParams = MOTION
) -> tuple[Framing, np.ndarray]:
    """`motion_framing` of a video file: one decode of every frame, in gray at `params.short_side`."""
    gray, size = decode_gray(path, params.short_side)
    return motion_framing(gray, size, use, params)


def framing_for(mode: str, entry: dict[str, Any]) -> Framing | None:
    """The framing `mode` gives a clip (its `video[]` entry) without decoding it: `none` the whole frame;
    `record` the record's rectangle, else the whole frame; `auto` the record's rectangle, else None (the
    caller finds the motion's)."""
    if mode not in CROP_MODES:
        raise ValueError(f"crop mode {mode!r}: one of {', '.join(CROP_MODES)}")
    width, height = int(entry["width"]), int(entry["height"])
    if mode == "none":
        return whole_frame(width, height)
    found = record_framing(entry.get("crop"), width, height)
    if found is not None or mode == "auto":
        return found
    return whole_frame(width, height)


# The preview, for the eye.

RECORD_COLOUR = (64, 196, 255)
MOTION_COLOUR = (245, 166, 35)
TEXT = (235, 235, 235)


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - Pillow without FreeType
        return ImageFont.load_default()


def _heat(energy: np.ndarray) -> Image.Image:
    """The energy map as an image: black to orange to white."""
    peak = float(energy.max()) if energy.size else 0.0
    u = np.sqrt(energy / peak) if peak > 0 else np.zeros_like(energy)
    rgb = np.stack([np.clip(2 * u, 0, 1), np.clip(2 * u - 0.5, 0, 1) * 0.85, np.clip(2 * u - 1, 0, 1)], -1)
    return Image.fromarray((255 * rgb).astype(np.uint8))


def preview_image(
    frame: Image.Image,
    energy: np.ndarray | None,
    framings: list[Framing],
    title: list[str],
    *,
    height: int = 720,
) -> Image.Image:
    """The frame scaled to `height` with each framing's rectangle drawn (the record's in blue, the motion's
    in orange), and beside it the energy map with the motion's rectangle, under the title's lines."""
    scale = height / frame.height
    left = frame.convert("RGB").resize(
        (max(1, round(frame.width * scale)), height), Image.Resampling.BILINEAR
    )
    panels = [left]
    if energy is not None and energy.size:
        panels.append(_heat(energy).resize(left.size, Image.Resampling.NEAREST))
    title_h = 8 + 18 * len(title)
    sheet = Image.new(
        "RGB", (sum(p.width for p in panels) + 8 * (len(panels) - 1), title_h + height), (17, 17, 17)
    )
    draw = ImageDraw.Draw(sheet)
    font = _font(14)
    for n, line in enumerate(title):
        draw.text((6, 4 + 18 * n), line, fill=TEXT, font=font)
    x_offset = 0
    for i, panel in enumerate(panels):
        sheet.paste(panel, (x_offset, title_h))
        for framing in framings:
            if i == 1 and framing.source != "motion":
                continue
            colour = RECORD_COLOUR if framing.source == "record" else MOTION_COLOUR
            x0 = x_offset + framing.x * scale
            y0 = title_h + framing.y * scale
            draw.rectangle(
                (x0, y0, x0 + framing.w * scale - 1, y0 + framing.h * scale - 1), outline=colour, width=3
            )
            draw.text((x0 + 6, y0 + 4), framing.source, fill=colour, font=font)
        x_offset += panel.width + 8
    return sheet


@dataclass
class CropPreview:
    image: Image.Image
    frame: int
    auto: Framing
    motion: Framing
    record: Framing | None


def crop_preview(
    dataset: Dataset,
    clip: ClipRef,
    *,
    time_base: str = "fit",
    frame: int | None = None,
    height: int = 720,
    params: MotionParams = MOTION,
) -> CropPreview:
    """One frame of the clip (by default the middle frame of the segment's window) with the record's
    rectangle and the motion's drawn on it, beside the motion's energy map: which one `auto` takes is in
    the title. The motion's is found even when the record has a rectangle, to compare them."""
    attempt = dataset.attempt(clip.session, clip.attempt)
    entry = dataset.clip_entry(clip)
    aligned = align_clip(
        attempt, dataset.frames(clip), None, camera=clip.camera, segment=clip.segment, time_base=time_base
    )
    use = aligned.track["inWindow"]
    path = dataset.video_path(clip)
    motion, energy = find_motion(path, use, params)
    record = record_framing(entry.get("crop"), motion.width, motion.height)
    auto = record or motion
    if frame is None:
        window = np.flatnonzero(use)
        frame = int(window[len(window) // 2]) if len(window) else len(use) // 2
    image = read_frames(path, [frame]).get(frame)
    if image is None:
        raise ValueError(f"{clip} has no frame {frame} ({len(use)} frames)")
    title = [
        f"{clip} | frame {frame} of {len(use)} | {int(use.sum())} frames in the {clip.segment} window",
        f"auto takes the {auto.source} crop: {auto.describe()}",
        f"motion: {motion.describe()}"
        + (f"  |  record: {record.describe()}" if record else "  |  no record crop"),
    ]
    framings = [f for f in (record, motion) if f is not None]
    return CropPreview(
        preview_image(image, energy, framings, title, height=height), frame, auto, motion, record
    )
