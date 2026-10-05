"""One clip on the host clock: its frames' times, the camera's lag, the segment's window, and the per-frame
label track (the nearest move onset, the phase, the gyro's orientation), from the JSON records alone.

The lag: a camera's frames show the cube `syncResidualMs` after the move's time (its clapperboard
measured the motion that much later), so a move's onset on the clip's frame timeline is
`move time + lag`, and a frame at `tMs` shows the cube as it was at `tMs − lag`. A clip without a sync
check has no lag (`lag_ms` None, `unsynced`): its track is computed with a lag of 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .moves import DOUBLE_MS, SLICE_MS, Symbol, attempt_symbols, move_times

PHASES = ("before", "scramble", "inspection", "solve", "after")
PHASE_INDEX = {name: i for i, name in enumerate(PHASES)}

TRACK_DTYPE = np.dtype(
    [
        ("frame", np.int32),  # the frame's index in the clip
        ("tMs", np.float64),  # its host time: t0HostMs + dtMs[0] + … + dtMs[frame]
        ("shownMs", np.float64),  # the host time of what it shows: tMs − lag (tMs when unsynced)
        ("symbol", np.int8),  # the alphabet index of the nearest onset; −1 when the attempt has no move
        ("onset", np.int32),  # that onset's index in the attempt's symbols; −1 when none
        ("distanceMs", np.float32),  # tMs − (onset + lag): negative before the onset; NaN when none
        ("phase", np.int8),  # index into PHASES of the attempt's phase at shownMs
        ("inWindow", np.bool_),  # shownMs inside the segment's window
        ("qx", np.float32),  # the gyro's orientation at shownMs (slerp of the two samples around it),
        ("qy", np.float32),  # x, y, z, w as gyro.json has them; NaN without gyro.json or outside its
        ("qz", np.float32),  # samples' span
        ("qw", np.float32),
    ]
)


def cumulative_times(t0_ms: float, dt_ms: list[float] | np.ndarray) -> np.ndarray:
    """`t0 + dt[0] + … + dt[k]` for every k: the frames files' and gyro.json's convention (dt[0] is 0)."""
    return float(t0_ms) + np.cumsum(np.asarray(dt_ms, dtype=np.float64))


def frame_times(frames: dict[str, Any]) -> np.ndarray:
    return cumulative_times(frames["t0HostMs"], frames["dtMs"])


def gyro_samples(gyro: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """The samples' host times and their quaternions (n × 4: x, y, z, w)."""
    times = cumulative_times(gyro["t0HostMs"], gyro["dtMs"])
    quats = np.asarray(gyro["q"], dtype=np.float64).reshape(-1, 4)
    if len(quats) != len(times):
        raise ValueError(f"gyro.json: {len(times)} sample times for {len(quats)} quaternions")
    return times, quats


def slerp(q0: np.ndarray, q1: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Spherical linear interpolation of unit quaternions (n × 4) at fractions u (n), along the shorter
    arc; the result is normalized, its sign that of q0's hemisphere."""
    q0 = q0 / np.linalg.norm(q0, axis=1, keepdims=True)
    q1 = q1 / np.linalg.norm(q1, axis=1, keepdims=True)
    dot = np.sum(q0 * q1, axis=1)
    q1 = np.where(dot[:, None] < 0, -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    theta = np.arccos(dot)
    sin = np.sin(theta)
    near = sin < 1e-9
    safe = np.where(near, 1.0, sin)
    w0 = np.where(near, 1.0 - u, np.sin((1.0 - u) * theta) / safe)
    w1 = np.where(near, u, np.sin(u * theta) / safe)
    out = w0[:, None] * q0 + w1[:, None] * q1
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def interpolate_orientation(sample_ms: np.ndarray, quats: np.ndarray, at_ms: np.ndarray) -> np.ndarray:
    """The orientation at each time of `at_ms` (n × 4): the sample there or the slerp of the two around it;
    NaN outside the samples' span."""
    at_ms = np.asarray(at_ms, dtype=np.float64)
    out = np.full((len(at_ms), 4), np.nan)
    if len(sample_ms) == 0:
        return out
    inside = (at_ms >= sample_ms[0]) & (at_ms <= sample_ms[-1])
    if len(sample_ms) == 1:
        out[inside] = quats[0] / np.linalg.norm(quats[0])
        return out
    x = at_ms[inside]
    k = np.clip(np.searchsorted(sample_ms, x, side="right") - 1, 0, len(sample_ms) - 2)
    span = sample_ms[k + 1] - sample_ms[k]
    u = np.where(span > 0, (x - sample_ms[k]) / np.where(span > 0, span, 1.0), 0.0)
    out[inside] = slerp(quats[k], quats[k + 1], np.clip(u, 0.0, 1.0))
    return out


def nearest_onsets(t_ms: np.ndarray, onsets_ms: np.ndarray) -> np.ndarray:
    """For each time, the index of the nearest onset (the earlier one on a tie); −1 without onsets."""
    if len(onsets_ms) == 0:
        return np.full(len(t_ms), -1, dtype=np.int64)
    order = np.argsort(onsets_ms, kind="stable")
    ordered = onsets_ms[order]
    right = np.clip(np.searchsorted(ordered, t_ms, side="left"), 0, len(ordered) - 1)
    left = np.clip(right - 1, 0, len(ordered) - 1)
    pick = np.where(np.abs(ordered[right] - t_ms) < np.abs(t_ms - ordered[left]), right, left)
    return order[pick]


def event_times(attempt: dict[str, Any], times: np.ndarray) -> dict[str, float | None]:
    """The attempt's events on the moves' time base: an event that is a move's arrival (the first and the
    last scramble move, the first and the last solve move) takes that move's time; any other (a resync's
    report, the pickup) keeps its host time."""
    events = dict(attempt["events"])
    moves = attempt["moves"]
    by_phase: dict[str, list[int]] = {}
    for i, move in enumerate(moves):
        by_phase.setdefault(move["phase"], []).append(i)
    anchors = {
        "scrambleStart": ("scramble", 0),
        "scrambleDone": ("scramble", -1),
        "solveStart": ("solve", 0),
        "solveEnd": ("solve", -1),
    }
    for name, (phase, position) in anchors.items():
        indices = by_phase.get(phase)
        if events.get(name) is not None and indices:
            i = indices[position]
            if abs(moves[i]["hostMs"] - events[name]) < 1e-6:
                events[name] = float(times[i])
    return events


def segment_window(events: dict[str, float | None], segment: str) -> tuple[float, float]:
    """The segment's window on the host clock: scramble from scrambleStart to scrambleDone, solve from
    solveStart to solveEnd; an end that did not happen (a DNF) is +inf, a start that did not happen NaN."""
    start, end = ("scrambleStart", "scrambleDone") if segment == "scramble" else ("solveStart", "solveEnd")
    low = events.get(start)
    high = events.get(end)
    return (math.nan if low is None else float(low), math.inf if high is None else float(high))


def phase_codes(shown_ms: np.ndarray, events: dict[str, float | None]) -> np.ndarray:
    """The attempt's phase at each time: before scrambleStart, scramble up to scrambleDone, inspection up to
    solveStart, solve up to solveEnd, after; an event that did not happen never begins its phase."""

    def at(name: str) -> float:
        value = events.get(name)
        return math.inf if value is None else float(value)

    codes = np.zeros(len(shown_ms), dtype=np.int8)
    codes[shown_ms >= at("scrambleStart")] = PHASE_INDEX["scramble"]
    codes[shown_ms > at("scrambleDone")] = PHASE_INDEX["inspection"]
    codes[shown_ms >= at("solveStart")] = PHASE_INDEX["solve"]
    codes[shown_ms > at("solveEnd")] = PHASE_INDEX["after"]
    return codes


@dataclass
class ClipAlignment:
    """A clip on the host clock, with its per-frame track (`track`, a TRACK_DTYPE array)."""

    camera: str
    segment: str
    lag_ms: float | None
    time_base: str
    window: tuple[float, float]
    symbols: list[Symbol]
    track: np.ndarray
    moves_off_fit: int = 0
    events: dict[str, float | None] = field(default_factory=dict)

    @property
    def unsynced(self) -> bool:
        return self.lag_ms is None

    @property
    def lag(self) -> float:
        """The lag applied: `lag_ms`, or 0 for an unsynced clip."""
        return 0.0 if self.lag_ms is None else self.lag_ms

    @property
    def frame_ms(self) -> np.ndarray:
        return self.track["tMs"]

    @property
    def segment_symbols(self) -> list[Symbol]:
        """The symbols of the clip's segment (the moves of its window)."""
        return [s for s in self.symbols if s.phase == self.segment]

    def onset_on_frames(self, symbol: Symbol) -> float:
        """The symbol's onset on the clip's frame timeline: its time plus the lag."""
        return symbol.onset_ms + self.lag

    def covered(self) -> tuple[int, int]:
        """How many of the segment's symbols have their onset within the clip's frames, and how many there
        are."""
        segment = self.segment_symbols
        if not segment or len(self.frame_ms) == 0:
            return 0, len(segment)
        first, last = self.frame_ms[0], self.frame_ms[-1]
        inside = sum(first <= self.onset_on_frames(s) <= last for s in segment)
        return inside, len(segment)


def align_clip(
    attempt: dict[str, Any],
    frames: dict[str, Any],
    gyro: dict[str, Any] | None = None,
    *,
    camera: str | None = None,
    segment: str | None = None,
    time_base: str = "fit",
    slice_ms: float = SLICE_MS,
    double_ms: float = DOUBLE_MS,
) -> ClipAlignment:
    """The clip whose frames file is `frames` (its camera and segment by default), on the host clock."""
    camera = camera or frames["camera"]
    segment = segment or frames["segment"]
    entry = next(
        (v for v in attempt["video"] if v["camera"] == camera and v["segment"] == segment),
        None,
    )
    if entry is None:
        raise KeyError(f"attempt {attempt['index']} lists no clip {camera}.{segment}")
    lag_ms = entry["syncResidualMs"]
    lag = 0.0 if lag_ms is None else float(lag_ms)

    times, on_fit = move_times(attempt, time_base)
    symbols = attempt_symbols(attempt, time_base=time_base, slice_ms=slice_ms, double_ms=double_ms)
    events = event_times(attempt, times)
    window = segment_window(events, segment)

    t = frame_times(frames)
    shown = t - lag
    track = np.zeros(len(t), dtype=TRACK_DTYPE)
    track["frame"] = np.arange(len(t))
    track["tMs"] = t
    track["shownMs"] = shown

    onsets = np.array([s.onset_ms for s in symbols], dtype=np.float64) + lag
    nearest = nearest_onsets(t, onsets)
    has = nearest >= 0
    track["onset"] = nearest
    track["symbol"] = np.where(has, np.array([s.index for s in symbols] or [0], dtype=np.int8)[nearest], -1)
    track["distanceMs"] = np.where(has, t - onsets[np.maximum(nearest, 0)] if len(onsets) else 0.0, np.nan)
    track["phase"] = phase_codes(shown, events)
    low, high = window
    track["inWindow"] = (shown >= low) & (shown <= high)

    if gyro is not None:
        sample_ms, quats = gyro_samples(gyro)
        q = interpolate_orientation(sample_ms, quats, shown)
    else:
        q = np.full((len(t), 4), np.nan)
    for i, name in enumerate(("qx", "qy", "qz", "qw")):
        track[name] = q[:, i]

    return ClipAlignment(
        camera=camera,
        segment=segment,
        lag_ms=None if lag_ms is None else float(lag_ms),
        time_base=time_base,
        window=window,
        symbols=symbols,
        track=track,
        moves_off_fit=int(len(on_fit) - on_fit.sum()) if time_base == "fit" else 0,
        events=events,
    )
