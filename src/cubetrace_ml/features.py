"""Per-frame features of the clips, cached: every frame of a clip decoded, cut to its framing, scaled to the
encoder's square input and encoded, one file per clip with the frames' host times beside the features.

    <out>/<encoder>/<sessionId>/<nnnn>/<camera>.<segment>.npz
        x         float16, frames × dim: the encoder's features of every frame
        tMs       float64, the frames' host times (the frames file's t0HostMs + cumulative dtMs)
        shownMs   float64, tMs − lag (tMs for an unsynced clip): the time each frame shows
        inWindow  bool, shownMs inside the segment's window
        meta      a JSON string: the encoder, the crop, the clip, the time base and lag, the builds, the host

A clip whose file is there with the same encoder, crop mode and frame count is skipped (`force` rewrites
it); a file is written under a temporary name and renamed, so an interrupted run leaves no partial `.npz`.
The motion crops are kept under `<out>/crops/<sessionId>/<nnnn>/<camera>.<segment>.json`, so a second
encoder reuses them instead of decoding the clip twice.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import subprocess
import time
import uuid
import zipfile
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import av
import numpy as np
import polars as pl

from . import __version__
from .align import align_clip
from .dataset import SEGMENTS, ClipRef, Dataset
from .encoders import Encoder, load_encoder
from .framing import MOTION, Framing, find_motion, framing_for
from .records import RecordError
from .video import decode_square

FEATURES_ENV = "CUBETRACE_FEATURES"
FORMAT = 1
CROPS = "crops"
# What a clip's preparation may raise without stopping the run: an unreadable video, a missing file, a
# record that does not validate, frames that do not match the frames file.
CLIP_ERRORS = (av.FFmpegError, OSError, ValueError, KeyError, RecordError)


def features_root(out: str | Path | None) -> Path:
    """`out`, or `$CUBETRACE_FEATURES`."""
    root = out if out is not None else os.environ.get(FEATURES_ENV)
    if not root:
        raise ValueError(f"no features root: pass --out or set {FEATURES_ENV}")
    return Path(root).expanduser()


def _clip_rel(clip: ClipRef) -> Path:
    return Path(clip.session) / f"{clip.attempt:04d}" / f"{clip.camera}.{clip.segment}"


def feature_path(root: str | Path, encoder: str, clip: ClipRef) -> Path:
    rel = _clip_rel(clip)
    return Path(root) / encoder / rel.parent / f"{rel.name}.npz"


def crop_path(root: str | Path, clip: ClipRef) -> Path:
    rel = _clip_rel(clip)
    return Path(root) / CROPS / rel.parent / f"{rel.name}.json"


def read_features(path: str | Path) -> dict[str, Any]:
    """A features file's arrays, and its `meta` decoded."""
    with np.load(path, allow_pickle=False) as data:
        out = {name: data[name] for name in data.files}
    out["meta"] = json.loads(str(out["meta"]))
    return out


def read_meta(path: str | Path) -> dict[str, Any] | None:
    """A features file's `meta`; None when the file is missing or unreadable."""
    try:
        with np.load(path, allow_pickle=False) as data:
            return json.loads(str(data["meta"]))
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return None


def is_done(path: str | Path, *, encoder: str, crop_mode: str, frames: int) -> bool:
    """The resume rule: the file is there, readable, of this encoder and crop mode, for `frames` frames."""
    meta = read_meta(path)
    return (
        meta is not None
        and meta.get("encoder", {}).get("name") == encoder
        and meta.get("cropMode") == crop_mode
        and meta.get("clip", {}).get("frames") == frames
    )


def _replace_atomically(path: Path, write: Callable[[Any], None], mode: str = "wb") -> None:
    """`write(file)` into a temporary file beside `path`, then renamed to it: readers see the whole file or
    none (a temporary name never counts)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(partial, mode) as handle:
            write(handle)
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def write_features(path: str | Path, arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
    """The `.npz` (uncompressed: float16 features hardly compress), written atomically."""
    payload = {**arrays, "meta": np.array(json.dumps(meta, allow_nan=False))}
    _replace_atomically(Path(path), lambda handle: np.savez(handle, **payload))


def _finite(value: float | None) -> float | None:
    return float(value) if value is not None and math.isfinite(value) else None


@cache
def code_version() -> dict[str, Any]:
    """cubetrace-ml's version, and its git commit when it runs from its own checkout (`dirty` when the
    checkout's tracked files differ from it)."""
    out: dict[str, Any] = {"version": __version__, "commit": None, "dirty": None}
    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        return out

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, timeout=10, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    out["commit"] = git("rev-parse", "--short=7", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no")
    out["dirty"] = None if status is None else bool(status)
    return out


# Selecting the clips.


def select_clips(
    clips: pl.DataFrame,
    *,
    split: str | None = None,
    session: str | None = None,
    camera: str | None = None,
    segment: str | None = None,
    usable_only: bool = True,
    limit: int | None = None,
) -> list[ClipRef]:
    """The manifest's clips that the filters keep, in session, attempt, segment and camera order; `session`
    is an id or a unique prefix of one."""
    frame = clips
    if session is not None:
        ids = sorted(set(frame["sessionId"].to_list()))
        matches = [s for s in ids if s == session] or [s for s in ids if s.startswith(session)]
        if len(matches) != 1:
            raise ValueError(f"session {session!r} matches {len(matches)} sessions of the manifest")
        frame = frame.filter(pl.col("sessionId") == matches[0])
    for column, value in (("split", split), ("camera", camera), ("segment", segment)):
        if value is not None:
            frame = frame.filter(pl.col(column) == value)
    if usable_only:
        frame = frame.filter(pl.col("usable"))
    refs = sorted(
        (
            ClipRef(row["sessionId"], int(row["attemptIndex"]), row["camera"], row["segment"])
            for row in frame.select("sessionId", "attemptIndex", "camera", "segment").iter_rows(named=True)
        ),
        key=lambda c: (c.session, c.attempt, SEGMENTS.index(c.segment), c.camera),
    )
    return refs[:limit] if limit is not None else refs


def read_manifest(path: str | Path) -> pl.DataFrame:
    path = Path(path)
    return pl.read_parquet(path) if path.suffix == ".parquet" else pl.read_csv(path)


# Preparing a clip: its track from the records, then (in a worker) its framing and its frames.


@dataclass
class Clip:
    """A clip ready to be decoded: its timeline (the frames file through `align_clip`) and its framing when
    known before decoding (None: from the motion)."""

    ref: ClipRef
    entry: dict[str, Any]
    app: dict[str, Any] | None
    video: str
    t_ms: np.ndarray
    shown_ms: np.ndarray
    in_window: np.ndarray
    lag_ms: float | None
    window: tuple[float, float]
    framing: Framing | None
    crop_cached: bool = False

    @property
    def frames(self) -> int:
        return len(self.t_ms)


@dataclass
class Decoded:
    clip: Clip
    framing: Framing
    frames: np.ndarray  # n × S × S × 3 uint8
    crop_seconds: float  # the motion pass (0 when the framing was known)
    decode_seconds: float


def load_clip(dataset: Dataset, ref: ClipRef, *, time_base: str = "fit") -> Clip:
    """The clip's records: its entry, its frames file's timeline and window, from the JSON alone."""
    attempt = dataset.attempt(ref.session, ref.attempt)
    entry = dataset.clip_entry(ref)
    frames = dataset.frames(ref)
    aligned = align_clip(attempt, frames, None, camera=ref.camera, segment=ref.segment, time_base=time_base)
    track = aligned.track
    return Clip(
        ref=ref,
        entry=entry,
        app=attempt.get("app") or frames.get("app"),
        video=dataset.video_rel(ref),
        t_ms=np.array(track["tMs"], dtype=np.float64),
        shown_ms=np.array(track["shownMs"], dtype=np.float64),
        in_window=np.array(track["inWindow"], dtype=bool),
        lag_ms=aligned.lag_ms,
        window=aligned.window,
        framing=None,
    )


def cached_crop(root: Path | None, clip: Clip, time_base: str) -> Framing | None:
    """The clip's motion crop from the crop cache, when it was found with these settings and frames."""
    if root is None:
        return None
    path = crop_path(root, clip.ref)
    try:
        doc = json.loads(path.read_text())
        if (doc["frames"], doc["timeBase"], doc["motion"]) == (clip.frames, time_base, MOTION.to_json()):
            return Framing.from_json(doc["crop"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return None


def save_crop(root: Path, clip: Clip, framing: Framing, time_base: str) -> None:
    doc = {
        "crop": framing.to_json(),
        "motion": MOTION.to_json(),
        "frames": clip.frames,
        "timeBase": time_base,
    }
    _replace_atomically(crop_path(root, clip.ref), lambda h: h.write(json.dumps(doc) + "\n"), mode="w")


def decode_clip(dataset: Dataset, clip: Clip, size: int) -> Decoded:
    """The clip's framing (from the motion when it is not known yet) and every frame at `size` × `size`;
    the decoded count must be the frames file's."""
    path = dataset.video_path(clip.ref)
    crop_seconds = 0.0
    framing = clip.framing
    if framing is None:
        start = time.perf_counter()
        framing, _ = find_motion(path, clip.in_window)
        crop_seconds = time.perf_counter() - start
    start = time.perf_counter()
    frames = decode_square(path, framing.box, size)
    decode_seconds = time.perf_counter() - start
    if len(frames) != clip.frames:
        raise ValueError(f"{clip.video}: decoded {len(frames)} frames, the frames file has {clip.frames}")
    return Decoded(clip, framing, frames, crop_seconds, decode_seconds)


def prefetch(
    items: Iterable[Any], work: Callable[[Any], Any], workers: int = 1
) -> Iterator[tuple[Any, Any | Exception]]:
    """`work(item)` for each item in `workers` threads, yielded in order with its result (or the error it
    raised) while the next ones are being worked on: at most `workers` + 1 results are held at once."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    queue: deque[tuple[Any, Any]] = deque()
    source = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:

        def submit() -> None:
            for item in source:
                queue.append((item, pool.submit(work, item)))
                return

        for _ in range(workers + 1):
            submit()
        try:
            while queue:
                item, future = queue.popleft()
                try:
                    result = future.result()
                except CLIP_ERRORS as error:
                    result = error
                submit()
                yield item, result
        finally:  # an interrupted run does not decode the clips it will not encode
            for _, future in queue:
                future.cancel()


def encode_frames(encoder: Encoder, frames: np.ndarray, batch: int = 64) -> np.ndarray:
    """The encoder's features of every frame, in batches, as float16."""
    out = np.empty((len(frames), encoder.info.dim), dtype=np.float16)
    for start in range(0, len(frames), batch):
        out[start : start + batch] = encoder.encode(frames[start : start + batch])
    return out


@dataclass
class RunStats:
    """What a run did and how fast: each stage's busy time (the stages overlap: decoding runs in workers
    while the encoder works) and the wall time."""

    selected: int = 0
    written: int = 0
    skipped: int = 0
    failed: int = 0
    frames: int = 0
    crop_clips: int = 0
    crop_frames: int = 0
    crop_seconds: float = 0.0
    decode_seconds: float = 0.0
    encode_seconds: float = 0.0
    write_seconds: float = 0.0
    wall_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    @staticmethod
    def rate(frames: int, seconds: float) -> float | None:
        return frames / seconds if seconds > 0 else None

    def to_json(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "cropClips": self.crop_clips,
            "cropSeconds": round(self.crop_seconds, 3),
            "decodeSeconds": round(self.decode_seconds, 3),
            "encodeSeconds": round(self.encode_seconds, 3),
            "wallSeconds": round(self.wall_seconds, 3),
        }


def _meta(
    decoded: Decoded,
    encoder: Encoder,
    crop_mode: str,
    time_base: str,
    encode_seconds: float,
    ref_rel: str,
) -> dict[str, Any]:
    clip, framing = decoded.clip, decoded.framing
    crop = framing.to_json()
    if framing.source == "motion":
        crop["motion"] = MOTION.to_json()
    crop["cached"] = clip.crop_cached
    n = clip.frames
    low, high = clip.window
    return {
        "format": FORMAT,
        "encoder": {**encoder.info.to_json(), "loaded": encoder.weights},
        "cropMode": crop_mode,
        "crop": crop,
        "clip": {
            "sessionId": clip.ref.session,
            "attemptIndex": clip.ref.attempt,
            "camera": clip.ref.camera,
            "segment": clip.ref.segment,
            "frames": n,
            "video": clip.video,
            "width": clip.entry["width"],
            "height": clip.entry["height"],
            "fpsNominal": clip.entry["fpsNominal"],
            "windowMs": [_finite(low), _finite(high)],
        },
        "timeBase": time_base,
        "lagMs": clip.lag_ms,
        "unsynced": clip.lag_ms is None,
        "app": clip.app,
        "cubetraceMl": code_version(),
        "host": encoder.describe(),
        "timing": {
            "cropSeconds": round(decoded.crop_seconds, 3),
            "decodeSeconds": round(decoded.decode_seconds, 3),
            "encodeSeconds": round(encode_seconds, 3),
            "decodeFps": round(n / decoded.decode_seconds, 1) if decoded.decode_seconds > 0 else None,
            "encodeFps": round(n / encode_seconds, 1) if encode_seconds > 0 else None,
        },
        "file": ref_rel,
        "writtenAt": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }


def _prepare(
    dataset: Dataset, refs: list[ClipRef], time_base: str, crop_mode: str, crops: Path | None
) -> Iterator[Clip | tuple[ClipRef, Exception]]:
    for ref in refs:
        try:
            clip = load_clip(dataset, ref, time_base=time_base)
        except CLIP_ERRORS as error:
            yield ref, error
            continue
        clip.framing = framing_for(crop_mode, clip.entry)
        if clip.framing is None:
            clip.framing = cached_crop(crops, clip, time_base)
            clip.crop_cached = clip.framing is not None
        yield clip


def extract(
    dataset: Dataset,
    refs: list[ClipRef],
    encoder: Encoder,
    root: str | Path,
    *,
    crop_mode: str = "auto",
    time_base: str = "fit",
    batch: int = 64,
    workers: int = 1,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> RunStats:
    """Writes the features file of every clip of `refs` that is not done yet (all of them with `force`);
    decoding runs in `workers` threads ahead of the encoder."""
    root = Path(root)
    stats = RunStats(selected=len(refs))
    start = time.perf_counter()
    name = encoder.info.name
    todo: list[Clip] = []
    for item in _prepare(dataset, refs, time_base, crop_mode, None if force else root):
        if isinstance(item, tuple):
            ref, error = item
            stats.failed += 1
            stats.errors.append(f"{ref}: {error}")
            log(f"failed   {ref}: {error}")
            continue
        path = feature_path(root, name, item.ref)
        if not force and is_done(path, encoder=name, crop_mode=crop_mode, frames=item.frames):
            stats.skipped += 1
            continue
        todo.append(item)
    if stats.skipped:
        log(f"{stats.skipped:,} of {len(refs):,} clips already have their {name} features (--force rewrites)")

    for k, (clip, result) in enumerate(
        prefetch(todo, lambda c: decode_clip(dataset, c, encoder.info.input_size), workers), start=1
    ):
        if isinstance(result, Exception):
            stats.failed += 1
            stats.errors.append(f"{clip.ref}: {result}")
            log(f"failed   [{k}/{len(todo)}] {clip.ref}: {result}")
            continue
        decoded: Decoded = result
        tick = time.perf_counter()
        x = encode_frames(encoder, decoded.frames, batch)
        encode_seconds = time.perf_counter() - tick
        path = feature_path(root, name, clip.ref)
        tick = time.perf_counter()
        meta = _meta(decoded, encoder, crop_mode, time_base, encode_seconds, str(path.relative_to(root)))
        write_features(
            path,
            {"x": x, "tMs": clip.t_ms, "shownMs": clip.shown_ms, "inWindow": clip.in_window},
            meta,
        )
        if decoded.framing.source == "motion" and not clip.crop_cached:
            save_crop(root, clip, decoded.framing, time_base)
        stats.write_seconds += time.perf_counter() - tick
        stats.written += 1
        stats.frames += clip.frames
        stats.decode_seconds += decoded.decode_seconds
        stats.encode_seconds += encode_seconds
        if decoded.crop_seconds:
            stats.crop_clips += 1
            stats.crop_frames += clip.frames
            stats.crop_seconds += decoded.crop_seconds
        cached = " (cached)" if clip.crop_cached else ""
        log(
            f"wrote    [{k}/{len(todo)}] {clip.ref}: {clip.frames:,} frames, crop "
            f"{decoded.framing.describe()}{cached}, decode {_fps(clip.frames, decoded.decode_seconds)}, "
            f"encode {_fps(clip.frames, encode_seconds)}"
        )
    stats.wall_seconds = time.perf_counter() - start
    return stats


def _fps(frames: int, seconds: float) -> str:
    rate = RunStats.rate(frames, seconds)
    return "–" if rate is None else f"{rate:,.0f} fps"


def summary(stats: RunStats, encoder: Encoder | None, title: str = "features") -> str:
    """The run's counts and its throughput, stage by stage."""
    host = encoder.describe() if encoder is not None else None
    lines = [
        f"{title}: {stats.selected:,} clips selected: {stats.written:,} written, {stats.skipped:,} already "
        f"there, {stats.failed:,} failed; {stats.frames:,} frames in {stats.wall_seconds:,.1f} s "
        f"({_fps(stats.frames, stats.wall_seconds)} overall)"
    ]
    if stats.crop_clips:
        lines.append(
            f"  motion crop            {stats.crop_clips:,} clips, {stats.crop_frames:,} frames, "
            f"{stats.crop_seconds:,.1f} s: {_fps(stats.crop_frames, stats.crop_seconds)}"
        )
    lines.append(
        f"  decode, crop, resize   {stats.frames:,} frames, {stats.decode_seconds:,.1f} s: "
        f"{_fps(stats.frames, stats.decode_seconds)}"
    )
    if encoder is not None and host is not None:
        lines.append(
            f"  encoder {encoder.info.name:<14} {stats.frames:,} frames, {stats.encode_seconds:,.1f} s: "
            f"{_fps(stats.frames, stats.encode_seconds)} ({host['device']}, {host['deviceName']}, "
            f"{host['precision']}{', random weights' if encoder.weights == 'random' else ''})"
        )
    for error in stats.errors[:20]:
        lines.append(f"  failed: {error}")
    return "\n".join(lines)


# The throughput, without writing anything.


@dataclass
class BenchRow:
    encoder: str
    input_size: int
    stats: RunStats
    host: dict[str, Any] | None


def bench(
    dataset: Dataset,
    refs: list[ClipRef],
    encoders: list[str],
    *,
    crop_mode: str = "auto",
    time_base: str = "fit",
    batch: int = 64,
    workers: int = 1,
    device: str = "auto",
    precision: str = "auto",
    pretrained: bool = True,
    size: int = 224,
    log: Callable[[str], None] = print,
) -> list[BenchRow]:
    """For each encoder (`none`: decode, crop and resize alone, at `size`), the pipeline of `extract` over the
    clips without writing: its stages' throughput. The motion crops are found once (the first run's motion
    pass) and reused by the others."""
    clips: list[Clip] = []
    for item in _prepare(dataset, refs, time_base, crop_mode, None):
        if isinstance(item, tuple):
            raise ValueError(f"{item[0]}: {item[1]}")
        clips.append(item)
    rows = []
    for name in encoders:
        encoder = (
            None
            if name == "none"
            else load_encoder(name, device=device, precision=precision, pretrained=pretrained)
        )
        input_size = size if encoder is None else encoder.info.input_size
        stats = RunStats(selected=len(clips))
        start = time.perf_counter()
        for clip, result in prefetch(clips, lambda c, s=input_size: decode_clip(dataset, c, s), workers):
            if isinstance(result, Exception):
                stats.failed += 1
                stats.errors.append(f"{clip.ref}: {result}")
                continue
            if encoder is not None:
                tick = time.perf_counter()
                encode_frames(encoder, result.frames, batch)
                stats.encode_seconds += time.perf_counter() - tick
            stats.written += 1
            stats.frames += clip.frames
            stats.decode_seconds += result.decode_seconds
            if result.crop_seconds:
                stats.crop_clips += 1
                stats.crop_frames += clip.frames
                stats.crop_seconds += result.crop_seconds
                clip.framing = result.framing
        stats.wall_seconds = time.perf_counter() - start
        host = None if encoder is None else {**encoder.describe(), "weights": encoder.weights}
        rows.append(BenchRow(name, input_size, stats, host))
        log(summary(stats, encoder, title=f"bench {name}"))
    return rows


def bench_table(rows: list[BenchRow]) -> str:
    """The bench's rows as a Markdown table (frames per second of each stage)."""

    def rate(frames: int, seconds: float) -> str:
        value = RunStats.rate(frames, seconds)
        return "–" if value is None else f"{value:,.0f}"

    lines = [
        "| encoder | input | clips | frames | motion crop fps | decode+crop+resize fps | encoder fps "
        "| overall fps | host |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        s = row.stats
        host = (
            "–"
            if row.host is None
            else f"{row.host['device']}, {row.host['precision']}"
            + (", random weights" if row.host["weights"] == "random" else "")
        )
        lines.append(
            f"| {row.encoder} | {row.input_size} | {s.written} | {s.frames:,} | "
            f"{rate(s.crop_frames, s.crop_seconds)} | {rate(s.frames, s.decode_seconds)} | "
            f"{rate(s.frames, s.encode_seconds) if row.host else '–'} | "
            f"{rate(s.frames, s.wall_seconds)} | {host} |"
        )
    return "\n".join(lines)
