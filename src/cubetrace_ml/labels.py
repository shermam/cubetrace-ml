"""The labels of a clip for the models, from the records through `align_clip`: the frames kept (the segment's
window, and the frames nearest its first and last onsets, plus a margin on each side), the reference symbol
sequence with each onset on the frames' timeline (`onset + lag`), the per-frame target, and the gyro's
channels at the kept frames; and the clips of a split loaded with their cached features.

The classes: 0 is "no onset", 1–24 the alphabet's symbols (class = symbol index + 1). The frame nearest an
onset carries its symbol's class; when two onsets want one frame, the later takes the free neighbour
nearer its time, or no frame at all (a collision: it stays in the sequence). With `label_frames` k > 0,
the frames within k of an onset's frame get a soft target: `soft_decay ** d` on its class and the rest
on "no onset", d frames away.

The gyro's channels (`GYRO_CHANNELS`, 9 per kept frame): the cube's orientation at the frame (`qx qy qz qw`,
the track's slerp of `gyro.json` at `shownMs`, in the hemisphere w ≥ 0; zeros where the frame has none),
its change since the previous kept frame (`q_t · conj(q_{t−1})`, w ≥ 0; the identity at the first kept frame and wherever either
frame has none) and the presence flag (1 where the frame has an orientation, else 0). A model's input is
the features alone (`features`) or the features with these channels after them (`features+gyro`).
"""

from __future__ import annotations

import zipfile
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .align import align_clip, gyro_samples, relative_rotations
from .dataset import ClipRef, Dataset
from .features import feature_path, read_features, select_clips
from .moves import DOUBLE_MS, SLICE_MS, SYMBOLS
from .records import RecordError

NO_ONSET = 0
CLASSES = 1 + len(SYMBOLS)  # 25: no onset and the 24 symbols
MARGIN = 15
INPUTS = ("features", "features+gyro")
# Per kept frame: the orientation, its change since the previous kept frame, and the presence flag.
GYRO_CHANNELS = ("qx", "qy", "qz", "qw", "dqx", "dqy", "dqz", "dqw", "gyro")

# Why a selected clip has no labels: the reasons, in the order they are checked.
SKIPS = {
    "records": "its records could not be read",
    "no-window": "its segment has no move and no frame shows its window",
    "moves-outside-frames": "an onset of its segment falls outside the clip's frames",
    "no-gyro": "no gyro.json, or its samples cover none of the kept frames (data.require_gyro)",
    "no-features": "no features file for the encoder",
    "features-mismatch": "the features file's frames are not the frames file's",
}


class LabelError(ValueError):
    """A clip that cannot be labelled; `reason` is one of SKIPS."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        super().__init__(f"{reason}: {detail or SKIPS[reason]}")


@dataclass(frozen=True)
class LabelConfig:
    """How a clip is labelled."""

    margin: int = MARGIN  # frames kept on each side of the segment's window, at the clip's own rate
    fps: float = 0.0  # keep every k-th frame, k = round(the clip's rate / fps); 0: every frame
    label_frames: int = 0  # the soft target's reach, in kept frames on each side of an onset's frame
    soft_decay: float = 0.5  # the soft target's weight d frames away: soft_decay ** d
    time_base: str = "fit"
    slice_ms: float = SLICE_MS
    double_ms: float = DOUBLE_MS
    with_gyro: bool = False  # read the attempt's gyro.json for the gyro's channels (else they say "none")
    require_gyro: bool = False  # a clip whose kept frames have no orientation is skipped (no-gyro)

    @property
    def reads_gyro(self) -> bool:
        return self.with_gyro or self.require_gyro


@dataclass
class ClipLabels:
    """One clip's kept frames and labels (and their features, once loaded)."""

    ref: ClipRef
    frames: np.ndarray  # the kept frames' indices in the clip
    t_ms: np.ndarray  # their host times (the frames file's tMs)
    symbols: np.ndarray  # the reference: alphabet indices of the onsets in the kept frames, in time order
    onsets_ms: np.ndarray  # each onset on the frames' timeline: its time plus the lag
    target: np.ndarray  # per kept frame: 0, or the class of the onset it is the frame of
    near_class: np.ndarray  # per kept frame: the class of the onset frame within label_frames (0: none)
    near_weight: np.ndarray  # its weight: 1 on an onset's own frame, soft_decay ** d d frames away, else 0
    lag_ms: float | None
    tps: float | None
    facelets: str | None  # the attempt's scrambledFacelets: a solve clip's replay starts there
    stride: int = 1
    collisions: int = 0  # onsets that found no frame of their own (in `symbols`, not in `target`)
    foreign: int = 0  # onsets of the attempt's other segment inside the kept frames (in `symbols`)
    split: str = ""
    x: np.ndarray | None = None  # the kept frames' features, frames × dim, float16
    gyro: np.ndarray | None = None  # the gyro's channels at the kept frames, frames × 9, float32

    @property
    def segment(self) -> str:
        return self.ref.segment

    @property
    def gyro_frames(self) -> int:
        """How many kept frames have the cube's orientation."""
        return 0 if self.gyro is None else int(self.gyro[:, -1].sum())

    def __len__(self) -> int:
        return len(self.frames)


def frame_stride(t_ms: np.ndarray, fps: float) -> int:
    """How many of the clip's frames per kept frame for a rate of `fps` (1 when `fps` is 0 or not lower)."""
    if not fps or len(t_ms) < 2 or t_ms[-1] <= t_ms[0]:
        return 1
    native = (len(t_ms) - 1) / ((t_ms[-1] - t_ms[0]) / 1000.0)
    return max(1, round(native / fps))


def place_onsets(t_ms: np.ndarray, onsets_ms: np.ndarray, classes: np.ndarray) -> tuple[np.ndarray, int]:
    """The per-frame target: each onset (in time order) on the frame nearest it, or on the free neighbour
    nearer its time when that frame is taken; the onsets left without a frame are counted."""
    target = np.zeros(len(t_ms), dtype=np.int64)
    collisions = 0
    for onset, cls in zip(onsets_ms, classes, strict=True):
        k = int(np.argmin(np.abs(t_ms - onset)))
        neighbours = sorted(
            (j for j in (k - 1, k + 1) if 0 <= j < len(t_ms)), key=lambda j: abs(t_ms[j] - onset)
        )
        candidates = [k, *neighbours]
        free = next((j for j in candidates if target[j] == NO_ONSET), None)
        if free is None:
            collisions += 1
            continue
        target[free] = cls
    return target, collisions


def soft_targets(target: np.ndarray, label_frames: int, decay: float) -> tuple[np.ndarray, np.ndarray]:
    """For each frame, the class of the onset frame that reaches it and its weight: 1 on an onset's own
    frame, `decay ** d` d frames from the nearest one within `label_frames` (the heavier on a tie, the
    earlier of equals), 0 elsewhere."""
    near_class = target.copy()
    near_weight = (target != NO_ONSET).astype(np.float32)
    onsets = np.flatnonzero(target != NO_ONSET)
    for d in range(1, label_frames + 1):
        weight = decay**d
        for k in onsets:
            for j in (k - d, k + d):
                if 0 <= j < len(target) and near_weight[j] < weight:
                    near_class[j] = target[k]
                    near_weight[j] = weight
    return near_class, near_weight


def gyro_channels(q: np.ndarray) -> np.ndarray:
    """The gyro's 9 channels (GYRO_CHANNELS) of frames whose orientations are `q` (n × 4, x y z w, NaN where
    a frame has none), float32."""
    q = np.asarray(q, dtype=np.float64)
    present = np.isfinite(q).all(axis=1)
    out = np.zeros((len(q), len(GYRO_CHANNELS)), dtype=np.float32)
    # q and −q are one orientation: the channels take the hemisphere w ≥ 0, so that the same orientation
    # reads the same across attempts (the app's stream may land on either side).
    held = q[present]
    out[present, :4] = np.where(held[:, 3:4] < 0, -held, held)
    out[:, 4:8] = relative_rotations(q)
    out[:, 8] = present
    return out


def clip_labels(
    attempt: dict[str, Any],
    frames: dict[str, Any],
    ref: ClipRef,
    config: LabelConfig | None = None,
    *,
    gyro: dict[str, Any] | None = None,
) -> ClipLabels:
    """The clip's labels from its attempt, its frames file and the attempt's `gyro.json` (a LabelError when it
    has none)."""
    config = config or LabelConfig()
    if gyro is not None:
        try:
            gyro_samples(gyro)
        except (ValueError, KeyError) as error:
            raise LabelError("records", f"gyro.json: {error}") from error
    aligned = align_clip(
        attempt,
        frames,
        gyro,
        camera=ref.camera,
        segment=ref.segment,
        time_base=config.time_base,
        slice_ms=config.slice_ms,
        double_ms=config.double_ms,
    )
    track = aligned.track
    t_all = np.asarray(track["tMs"], dtype=np.float64)
    onsets = np.array([aligned.onset_on_frames(s) for s in aligned.symbols], dtype=np.float64)
    mine = np.array([s.phase == ref.segment for s in aligned.symbols], dtype=bool)
    # The window's frames, and the frames nearest its first and last onsets (a window shorter than a frame
    # interval, or an onset just before the window's first frame).
    edges = [int(k) for k in np.flatnonzero(track["inWindow"])[[0, -1]]] if track["inWindow"].any() else []
    if mine.any():
        edges += [
            int(np.argmin(np.abs(t_all - onsets[mine].min()))),
            int(np.argmin(np.abs(t_all - onsets[mine].max()))),
        ]
    if not edges:
        raise LabelError("no-window")
    low = max(0, min(edges) - config.margin)
    high = min(len(track) - 1, max(edges) + config.margin)
    stride = frame_stride(t_all, config.fps)
    kept = np.arange(low, high + 1, stride)
    t = t_all[kept]
    interval = float(np.median(np.diff(t_all[::stride]))) if len(t_all) > stride else 0.0
    first, last = t[0] - interval / 2, t[-1] + interval / 2
    outside = mine & ((onsets < first) | (onsets > last))
    if outside.any():
        raise LabelError("moves-outside-frames", f"{int(outside.sum())} of {int(mine.sum())} onsets")
    span = (onsets >= first) & (onsets <= last)
    order = np.argsort(onsets, kind="stable")
    order = order[span[order]]
    symbols = np.array([aligned.symbols[i].index for i in order], dtype=np.int64)
    onset_ms = onsets[order]
    target, collisions = place_onsets(t, onset_ms, symbols + 1)
    near_class, near_weight = soft_targets(target, config.label_frames, config.soft_decay)
    channels = gyro_channels(np.stack([track[name] for name in ("qx", "qy", "qz", "qw")], axis=1)[kept])
    if config.require_gyro and not channels[:, -1].any():
        raise LabelError("no-gyro", "no gyro.json" if gyro is None else "no orientation at the kept frames")
    result = attempt["result"]
    return ClipLabels(
        ref=ref,
        frames=kept,
        t_ms=t,
        symbols=symbols,
        onsets_ms=onset_ms,
        target=target,
        near_class=near_class,
        near_weight=near_weight,
        lag_ms=aligned.lag_ms,
        tps=result.get("tps"),
        facelets=attempt.get("scrambledFacelets") if ref.segment == "solve" else None,
        stride=stride,
        collisions=collisions,
        foreign=int((span & ~mine).sum()),
        gyro=channels,
    )


@dataclass
class LoadStats:
    """What loading a split found: the clips selected and loaded, the skips by reason, and the counts."""

    split: str = ""
    selected: int = 0
    unusable: int = 0  # the split's clips the manifest's filter excludes
    skipped: Counter[str] = field(default_factory=Counter)
    clips: int = 0
    frames: int = 0
    symbols: int = 0
    collisions: int = 0
    foreign: int = 0
    by_segment: Counter[str] = field(default_factory=Counter)
    gyro_clips: int | None = None  # the loaded clips with an orientation at a kept frame (None: not read)
    gyro_frames: int | None = None  # the loaded clips' kept frames with an orientation (None: not read)

    def to_json(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "selected": self.selected,
            "unusable": self.unusable,
            "skipped": dict(sorted(self.skipped.items())),
            "clips": self.clips,
            "frames": self.frames,
            "symbols": self.symbols,
            "collisions": self.collisions,
            "foreign": self.foreign,
            "bySegment": dict(sorted(self.by_segment.items())),
            "gyroClips": self.gyro_clips,
            "gyroFrames": self.gyro_frames,
        }

    def describe(self) -> str:
        skipped = ", ".join(f"{n} {reason}" for reason, n in sorted(self.skipped.items())) or "none"
        gyro = (
            f", the gyro on {self.gyro_clips:,} clips and {self.gyro_frames:,} frames"
            if self.gyro_clips is not None and self.gyro_frames is not None
            else ""
        )
        return (
            f"{self.split or 'clips'}: {self.clips:,} of {self.selected:,} clips loaded ({skipped} skipped; "
            f"{self.unusable:,} unusable in the manifest), {self.frames:,} frames, {self.symbols:,} symbols"
            f"{f', {self.collisions:,} onsets without a frame' if self.collisions else ''}"
            f"{f', {self.foreign:,} onsets of the other segment' if self.foreign else ''}{gyro}"
        )


def load_clips(
    dataset: Dataset,
    refs: Iterable[ClipRef],
    features: str | Path | None,
    encoder: str,
    config: LabelConfig | None = None,
    *,
    split: str = "",
    log: Callable[[str], None] | None = None,
) -> tuple[list[ClipLabels], LoadStats]:
    """The clips' labels with their kept frames' features (float16) read from `<features>/<encoder>/…`
    (labels alone when `features` is None) and, when the configuration reads it, the attempt's gyro.json; a
    clip that cannot be labelled or has no matching features is skipped and counted."""
    config = config or LabelConfig()
    stats = LoadStats(split=split)
    if config.reads_gyro:
        stats.gyro_clips, stats.gyro_frames = 0, 0
    out: list[ClipLabels] = []
    for ref in refs:
        stats.selected += 1
        try:
            try:
                attempt = dataset.attempt(ref.session, ref.attempt)
                frames = dataset.frames(ref)
                gyro = dataset.gyro(ref.session, ref.attempt) if config.reads_gyro else None
            except (RecordError, ValueError, OSError, KeyError) as error:
                raise LabelError("records", str(error)) from error
            labels = clip_labels(attempt, frames, ref, config, gyro=gyro)
            if features is not None:
                labels.x = _kept_features(Path(features), encoder, ref, frames, labels)
        except LabelError as error:
            stats.skipped[error.reason] += 1
            if log is not None:
                log(f"skipped  {ref}: {error}")
            continue
        labels.split = split
        out.append(labels)
        stats.clips += 1
        stats.frames += len(labels)
        stats.symbols += len(labels.symbols)
        stats.collisions += labels.collisions
        stats.foreign += labels.foreign
        stats.by_segment[ref.segment] += 1
        if stats.gyro_clips is not None and stats.gyro_frames is not None:
            stats.gyro_clips += int(labels.gyro_frames > 0)
            stats.gyro_frames += labels.gyro_frames
    return out, stats


def _kept_features(
    root: Path, encoder: str, ref: ClipRef, frames: dict[str, Any], labels: ClipLabels
) -> np.ndarray:
    path = feature_path(root, encoder, ref)
    if not path.is_file():
        raise LabelError("no-features", str(path))
    try:
        data = read_features(path)
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as error:
        raise LabelError("no-features", f"{path}: {error}") from error
    x, t = data["x"], data["tMs"]
    expected = len(frames["dtMs"])
    if len(x) != expected or len(t) != expected:
        raise LabelError("features-mismatch", f"{len(x)} feature frames, {expected} in the frames file")
    if not np.allclose(t[labels.frames], labels.t_ms, rtol=0, atol=0.01):
        raise LabelError("features-mismatch", "the frames' times differ")
    return np.ascontiguousarray(x[labels.frames], dtype=np.float16)


def model_input(clip: ClipLabels, inputs: str = "features") -> np.ndarray:
    """A clip's input to the model, frames × width, float32: its features (`features`), or its features and
    then the gyro's 9 channels (`features+gyro`)."""
    if clip.x is None:
        raise ValueError(f"{clip.ref}: no features loaded")
    if inputs == "features":
        return clip.x.astype(np.float32)
    if inputs == "features+gyro":
        if clip.gyro is None:
            raise ValueError(f"{clip.ref}: no gyro channels")
        return np.concatenate([clip.x.astype(np.float32), clip.gyro], axis=1)
    raise ValueError(f"inputs {inputs!r}: one of {', '.join(INPUTS)}")


def load_split(
    dataset: Dataset,
    manifest: pl.DataFrame,
    split: str,
    features: str | Path | None,
    encoder: str,
    config: LabelConfig | None = None,
    *,
    log: Callable[[str], None] | None = None,
) -> tuple[list[ClipLabels], LoadStats]:
    """The usable clips of the manifest's `split`, loaded."""
    refs = select_clips(manifest, split=split, usable_only=True)
    clips, stats = load_clips(dataset, refs, features, encoder, config, split=split, log=log)
    stats.unusable = len(select_clips(manifest, split=split, usable_only=False)) - len(refs)
    return clips, stats
