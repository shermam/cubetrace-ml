"""A contact sheet of one clip for the eye: a row per move onset, the frames around it, each frame with its
signed distance to the onset (frame time − (onset + lag)); the frame nearest the onset is outlined."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .align import ClipAlignment, align_clip
from .dataset import ClipRef, Dataset
from .video import read_frames

BACKGROUND = (17, 17, 17)
TEXT = (235, 235, 235)
MUTED = (150, 150, 150)
ACCENT = (245, 166, 35)
PAD = 6


@dataclass(frozen=True)
class SheetRow:
    symbol: str
    symbol_index: int  # in the attempt's symbols
    onset_ms: float  # on the frame timeline (time + lag)
    frames: list[int]
    distances_ms: list[float]


def pick_rows(aligned: ClipAlignment, onsets: int, frames: int) -> list[SheetRow]:
    """`onsets` of the segment's symbols whose onset is within the clip's frames, spread evenly over them,
    and for each the `frames` frames centred on the nearest frame."""
    t = aligned.frame_ms
    candidates = [
        i
        for i, s in enumerate(aligned.symbols)
        if s.phase == aligned.segment and t[0] <= aligned.onset_on_frames(s) <= t[-1]
    ]
    if not candidates or onsets <= 0 or frames <= 0:
        return []
    picks = sorted(
        {
            candidates[k]
            for k in np.linspace(0, len(candidates) - 1, min(onsets, len(candidates))).round().astype(int)
        }
    )
    rows = []
    for i in picks:
        symbol = aligned.symbols[i]
        onset = aligned.onset_on_frames(symbol)
        centre = int(np.argmin(np.abs(t - onset)))
        first = min(max(0, centre - frames // 2), max(0, len(t) - frames))
        indices = list(range(first, min(len(t), first + frames)))
        rows.append(SheetRow(symbol.symbol, i, onset, indices, [float(t[k] - onset) for k in indices]))
    return rows


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - Pillow without FreeType
        return ImageFont.load_default()


def render(
    title: list[str],
    rows: list[SheetRow],
    images: dict[int, Image.Image],
    window_start_ms: float,
) -> Image.Image:
    """The sheet: the title lines, then per row the symbol and its tiles with their labels."""
    big, small = _font(22), _font(13)
    tile_w = max((im.width for im in images.values()), default=64)
    tile_h = max((im.height for im in images.values()), default=64)
    label_h = 34
    left_w = 96
    columns = max((len(r.frames) for r in rows), default=1)
    title_h = PAD + 18 * len(title) + PAD
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    title_w = max((measure.textlength(line, font=small) for line in title), default=0)
    width = max(left_w + columns * (tile_w + PAD) + PAD, int(title_w) + 2 * PAD)
    height = title_h + max(1, len(rows)) * (tile_h + label_h + PAD) + PAD
    sheet = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    for n, line in enumerate(title):
        draw.text((PAD, PAD + 18 * n), line, fill=TEXT if n == 0 else MUTED, font=small)
    if not rows:
        draw.text((PAD, title_h), "no onset of the segment within the clip's frames", fill=ACCENT, font=small)
    for r, row in enumerate(rows):
        y = title_h + r * (tile_h + label_h + PAD)
        draw.text((PAD, y + 4), row.symbol, fill=TEXT, font=big)
        draw.text((PAD, y + 34), f"#{row.symbol_index}", fill=MUTED, font=small)
        draw.text((PAD, y + 52), f"{(row.onset_ms - window_start_ms) / 1000:+.2f} s", fill=MUTED, font=small)
        nearest = int(np.argmin(np.abs(row.distances_ms))) if row.distances_ms else -1
        for c, (k, d) in enumerate(zip(row.frames, row.distances_ms, strict=True)):
            x = left_w + c * (tile_w + PAD)
            image = images.get(k)
            if image is not None:
                sheet.paste(image, (x, y))
            if c == nearest:
                draw.rectangle((x - 2, y - 2, x + tile_w + 1, y + tile_h + 1), outline=ACCENT, width=3)
            draw.text(
                (x, y + tile_h + 3),
                f"{row.symbol} {d:+.0f} ms",
                fill=ACCENT if c == nearest else TEXT,
                font=small,
            )
            draw.text((x, y + tile_h + 18), f"frame {k}", fill=MUTED, font=small)
    return sheet


def contact_sheet(
    dataset: Dataset,
    clip: ClipRef,
    *,
    onsets: int = 6,
    frames: int = 5,
    tile_height: int = 160,
    time_base: str = "fit",
) -> Image.Image:
    attempt = dataset.attempt(clip.session, clip.attempt)
    entry = dataset.clip_entry(clip)
    aligned = align_clip(
        attempt,
        dataset.frames(clip),
        None,
        camera=clip.camera,
        segment=clip.segment,
        time_base=time_base,
    )
    rows = pick_rows(aligned, onsets, frames)
    wanted = sorted({k for row in rows for k in row.frames})
    images = read_frames(dataset.video_path(clip), wanted, crop=entry["crop"], height=tile_height)
    lag = "unsynced: no lag, 0 applied" if aligned.unsynced else f"lag {aligned.lag_ms:g} ms"
    covered, total = aligned.covered()
    title = [
        f"{clip.session[:8]} attempt {clip.attempt} | {clip.camera}.{clip.segment} | {lag}"
        f" | time base {time_base}",
        f"{len(rows)} of the segment's {total} onsets ({covered} within the clip's frames)",
        "under each frame: the symbol, frame time - (onset + lag), the frame's index",
    ]
    low, _ = aligned.window
    start = low + aligned.lag if np.isfinite(low) else float(aligned.frame_ms[0])
    return render(title, rows, images, start)
