"""The gyro's frame, measured from the records (`cubetrace-ml gyro-frames`; NumPy only): which of its axes is
gravity's, how the cube is held, and how the frame's yaw moves within an attempt, within a session and
between sessions, which decides what a calibration into a camera's frame has to undo.

Per attempt with `gyro.json`, over each segment's window (the scramble's, the solve's): every sample's
rotation matrix, whose column k is the cube's axis k in the gyro's frame; per cube axis the principal
direction of those columns (sign-free: the orientation tensor's top eigenvector, so that an axis held up and
held down count alike) and its concentration (the top eigenvalue: 1 when the axis keeps one direction, 1/3
when it takes every direction alike); the segment's mean orientation (the chordal mean) and the samples'
spread around it.

Pooled over the attempts: the cube axis whose directions stay together across attempts and sessions is the
one held vertical, and its direction is gravity's in the gyro's frame (then only a yaw about it is arbitrary);
when no axis does, the frame is arbitrary in all three degrees of freedom. The yaw: the heading of each
attempt's scramble pose about the gravity axis (the app prescribes the scramble in the cube's frame, so the
solver holds the cube one way while applying it), per session (its drift per hour, its change between
consecutive attempts) and between sessions. The convention (`orientation`): the cube's angular velocity
`v`, measured in the cube's own frame, against the change between consecutive samples taken in the cube's
frame (`conj(q_t) · q_{t+1}`) and in the gyro's (`q_{t+1} · conj(q_t)`): the one it follows says which side
of q a change of reference acts on.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .align import conjugate, event_times, gyro_samples, quaternion_product, segment_window
from .dataset import Dataset
from .moves import move_times
from .orientation import AXES, angle_between, chordal_mean, heading, log_map, matrices
from .records import RecordError

SEGMENTS = ("scramble", "solve")
MIN_SAMPLES = 5  # a segment's statistics need at least this many samples in its window
GRAVITY_CONCENTRATION = 0.8  # the pooled concentration above which one cube axis is called vertical
GRAVITY_ANGLE = 15.0  # degrees: a gravity direction this near a gyro axis is that axis
CONVENTION_MARGIN = 0.05  # the body frame's mean diagonal correlation must beat the gyro frame's by this much
MIN_DRIFT_HOURS = 0.5  # a session's yaw drift per hour needs at least this long a session


def axial_mode(directions: np.ndarray) -> tuple[np.ndarray, float]:
    """The principal axis of unit vectors (n × 3), sign-free (the top eigenvector of their orientation
    tensor, its largest component positive), and its eigenvalue (1/3 for uniform directions, 1 for one)."""
    d = np.asarray(directions, dtype=np.float64)
    tensor = d.T @ d / max(len(d), 1)
    values, vectors = np.linalg.eigh(tensor)
    mode = vectors[:, -1]
    if mode[np.argmax(np.abs(mode))] < 0:
        mode = -mode
    return mode, float(values[-1])


def axial_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The angle in degrees between axes (sign-free: 0–90)."""
    dot = np.abs(np.sum(np.asarray(a) * np.asarray(b), axis=-1))
    return np.degrees(np.arccos(np.clip(dot, 0.0, 1.0)))


@dataclass
class SegmentFrames:
    """One segment's samples in the gyro's frame."""

    samples: int
    modes: np.ndarray  # 3 × 3: row k, the principal direction of cube axis k (unit, sign-free)
    concentration: np.ndarray  # 3: each cube axis's top eigenvalue
    mean: np.ndarray  # the chordal mean orientation (4)
    spread: tuple[float, float]  # the samples' median and 90th-percentile angle from the mean, degrees
    up: np.ndarray  # 3: each cube axis's mean component along its own mode's direction, signed (the sign
    # of the mode's dominant component chosen positive): +1 always along it, −1 always against it


def segment_frames(quats: np.ndarray) -> SegmentFrames:
    m = matrices(quats)
    modes, concentration, up = np.zeros((3, 3)), np.zeros(3), np.zeros(3)
    for k in range(3):
        mode, value = axial_mode(m[:, :, k])
        modes[k], concentration[k] = mode, value
        up[k] = float(np.mean(m[:, :, k] @ mode))
    mean = chordal_mean(quats)
    distance = angle_between(np.tile(mean, (len(quats), 1)), quats)
    return SegmentFrames(
        samples=len(quats),
        modes=modes,
        concentration=concentration,
        mean=mean,
        spread=(float(np.median(distance)), float(np.percentile(distance, 90))),
        up=up,
    )


@dataclass
class AttemptFrames:
    session: str
    attempt: int
    start_ms: float  # the scramble's start (its first gyro sample without one)
    samples: int
    rate_hz: float
    segments: dict[str, SegmentFrames] = field(default_factory=dict)
    halves: tuple[SegmentFrames, SegmentFrames] | None = None  # the solve's first and second half


def attempt_frames(
    attempt: dict[str, Any], gyro: dict[str, Any], time_base: str = "fit"
) -> tuple[AttemptFrames, dict[str, np.ndarray]]:
    """An attempt's frames, and its consecutive samples' changes beside its angular velocity (`body`,
    `gyro`: the rotation vectors per second of `conj(q_t) · q_{t+1}` and `q_{t+1} · conj(q_t)`; `v`: the
    velocity reported with sample t + 1), for the convention's check."""
    sample_ms, quats = gyro_samples(gyro)
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    times, _ = move_times(attempt, time_base)
    events = event_times(attempt, times)
    span = (sample_ms[-1] - sample_ms[0]) / 1000.0 if len(sample_ms) > 1 else 0.0
    start = events.get("scrambleStart")
    frames = AttemptFrames(
        session=attempt["session"],
        attempt=int(attempt["index"]),
        start_ms=float(start) if start is not None else float(sample_ms[0]),
        samples=len(sample_ms),
        rate_hz=(len(sample_ms) - 1) / span if span > 0 else math.nan,
    )
    for segment in SEGMENTS:
        low, high = segment_window(events, segment)
        inside = (sample_ms >= low) & (sample_ms <= high)
        if inside.sum() >= MIN_SAMPLES:
            frames.segments[segment] = segment_frames(quats[inside])
            if segment == "solve" and inside.sum() >= 2 * MIN_SAMPLES:
                held = quats[inside]
                half = len(held) // 2
                frames.halves = (segment_frames(held[:half]), segment_frames(held[half:]))
    changes: dict[str, np.ndarray] = {}
    v = gyro.get("v")
    if v is not None and len(quats) > 1:
        velocity = np.asarray(v, dtype=np.float64).reshape(-1, 3)
        dt = np.diff(sample_ms) / 1000.0
        ok = dt > 0
        if len(velocity) == len(quats) and ok.any():
            body = log_map(quaternion_product(conjugate(quats[:-1]), quats[1:]))
            world = log_map(quaternion_product(quats[1:], conjugate(quats[:-1])))
            changes = {
                "body": body[ok] / dt[ok, None],
                "gyro": world[ok] / dt[ok, None],
                "v": velocity[1:][ok],
            }
    return frames, changes


def spearman(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Spearman's rank correlations between the columns of a (n × p) and of b (n × q): p × q."""

    def ranks(x: np.ndarray) -> np.ndarray:
        r = np.argsort(np.argsort(x, axis=0, kind="stable"), axis=0, kind="stable").astype(np.float64)
        return r - r.mean(axis=0)

    ra, rb = ranks(np.asarray(a)), ranks(np.asarray(b))
    norm = np.sqrt(np.outer((ra * ra).sum(axis=0), (rb * rb).sum(axis=0)))
    return (ra.T @ rb) / np.where(norm > 0, norm, 1.0)


def circular_mean(radians: np.ndarray) -> float:
    return float(np.angle(np.mean(np.exp(1j * np.asarray(radians)))))


def wrap(degrees: Any) -> Any:
    """Angles in degrees wrapped to [−180, 180)."""
    return (np.asarray(degrees) + 180.0) % 360.0 - 180.0


def _quantiles(values: Iterable[float], qs: tuple[float, ...] = (10, 50, 90)) -> list[float]:
    kept = np.array([v for v in values if v is not None and math.isfinite(v)], dtype=np.float64)
    return [float(np.percentile(kept, q)) for q in qs] if len(kept) else [math.nan] * len(qs)


@dataclass
class GyroFrames:
    """The diagnostic: every attempt's frames and the pooled findings (`summary`)."""

    attempts: list[AttemptFrames]
    skipped: dict[str, int]
    sessions: dict[str, int]  # the sessions' order (by their first attempt) and how many attempts with a gyro
    summary: dict[str, Any]


def _convention(changes: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    kept = [c for c in changes if c]
    if not kept:
        return {"pairs": 0, "verdict": "no angular velocity in the files"}
    v = np.concatenate([c["v"] for c in kept])
    body = spearman(v, np.concatenate([c["body"] for c in kept]))
    world = spearman(v, np.concatenate([c["gyro"] for c in kept]))
    body_diag, world_diag = float(np.mean(np.diag(body))), float(np.mean(np.abs(np.diag(world))))
    if body_diag > world_diag + CONVENTION_MARGIN:
        verdict = "cube frame: q takes the cube's axes into the gyro's; a reference change acts on the left"
    elif world_diag > body_diag + CONVENTION_MARGIN:
        verdict = "gyro frame: q takes the gyro's axes into the cube's; a reference change acts on the right"
    else:
        verdict = "inconclusive"
    return {
        "pairs": len(v),
        "bodyFrame": body.tolist(),
        "gyroFrame": world.tolist(),
        "bodyDiagonal": body_diag,
        "gyroDiagonal": world_diag,
        "verdict": verdict,
    }


def _axes(attempts: list[AttemptFrames], order: dict[str, int]) -> dict[str, Any]:
    """Per cube axis and segment: the per-attempt concentration, and the pooled direction of the per-attempt
    modes (with its concentration, the attempts' angles to it, and the sessions' directions)."""
    out: dict[str, Any] = {}
    for segment in (*SEGMENTS, "both"):
        names = SEGMENTS if segment == "both" else (segment,)
        per_axis = {}
        for k, axis in enumerate(AXES):
            modes, values, sessions = [], [], []
            for a in attempts:
                for name in names:
                    s = a.segments.get(name)
                    if s is not None:
                        modes.append(s.modes[k])
                        values.append(s.concentration[k])
                        sessions.append(a.session)
            if not modes:
                continue
            stacked = np.stack(modes)
            direction, pooled = axial_mode(stacked)
            by_session = {}
            for session in sorted(set(sessions), key=lambda s: order[s]):
                mine = stacked[[s == session for s in sessions]]
                by_session[session] = axial_mode(mine)[0]
            directions = list(by_session.values())
            between = max(
                (float(axial_angle(a, b)) for i, a in enumerate(directions) for b in directions[i + 1 :]),
                default=0.0,
            )
            per_axis[axis] = {
                "segments": len(modes),
                "concentration": _quantiles(values, (10, 25, 50, 75, 90)),
                "direction": direction.tolist(),
                "pooledConcentration": pooled,
                "angleToDirection": _quantiles(axial_angle(stacked, direction).tolist(), (50, 90)),
                "sessions": {s: d.tolist() for s, d in by_session.items()},
                "sessionsApartMax": between,
            }
        out[segment] = per_axis
    return out


def _gravity(axes: dict[str, Any]) -> dict[str, Any]:
    both = axes.get("both", {})
    if not both:
        return {"verdict": "inconclusive", "reason": "no segment with enough samples"}
    cube_axis = max(both, key=lambda a: both[a]["pooledConcentration"])
    found = both[cube_axis]
    direction = np.asarray(found["direction"])
    nearest = int(np.argmax(np.abs(direction)))
    off = float(axial_angle(direction, np.eye(3)[nearest]))
    result = {
        "cubeAxis": cube_axis,
        "direction": found["direction"],
        "pooledConcentration": found["pooledConcentration"],
        "nearestGyroAxis": AXES[nearest],
        "angleToNearest": off,
        "others": {a: both[a]["pooledConcentration"] for a in both if a != cube_axis},
    }
    if found["pooledConcentration"] >= GRAVITY_CONCENTRATION and off <= GRAVITY_ANGLE:
        result["verdict"] = "found"
        result["gravityAxis"] = AXES[nearest]
    elif found["pooledConcentration"] >= GRAVITY_CONCENTRATION:
        result["verdict"] = "found, off the gyro's axes"
    else:
        result["verdict"] = "inconclusive"
    return result


def _held(attempts: list[AttemptFrames], gravity: dict[str, Any]) -> dict[str, Any]:
    """How the vertical cube axis sits per segment: along gravity's direction or against it, and its tilt."""
    if "cubeAxis" not in gravity:
        return {}
    k = AXES.index(gravity["cubeAxis"])
    g = np.asarray(gravity["direction"])
    out = {}
    for segment in SEGMENTS:
        signs, tilts = [], []
        for a in attempts:
            s = a.segments.get(segment)
            if s is None:
                continue
            column = matrices(s.mean)[:, k]
            signs.append(float(np.sign(column @ g)))
            tilts.append(float(axial_angle(column, g)))
        if signs:
            out[segment] = {
                "attempts": len(signs),
                "along": int(sum(x > 0 for x in signs)),
                "against": int(sum(x < 0 for x in signs)),
                "tilt": _quantiles(tilts, (10, 50, 90)),
            }
    return out


def _within(attempts: list[AttemptFrames], gravity: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for segment in SEGMENTS:
        spreads = [a.segments[segment].spread for a in attempts if segment in a.segments]
        out[f"{segment}Spread"] = {
            "median": _quantiles([s[0] for s in spreads]),
            "p90": _quantiles([s[1] for s in spreads]),
        }
    both = [a for a in attempts if all(s in a.segments for s in SEGMENTS)]
    out["scrambleToSolve"] = _quantiles(
        [float(angle_between(a.segments["scramble"].mean, a.segments["solve"].mean)) for a in both],
        (10, 50, 90),
    )
    if "cubeAxis" in gravity:
        k = AXES.index(gravity["cubeAxis"])
        out["verticalScrambleToSolve"] = _quantiles(
            [float(axial_angle(a.segments["scramble"].modes[k], a.segments["solve"].modes[k])) for a in both],
            (50, 90),
        )
        halves = [a.halves for a in attempts if a.halves is not None]
        out["verticalSolveHalves"] = _quantiles(
            [float(axial_angle(h[0].modes[k], h[1].modes[k])) for h in halves], (50, 90)
        )
    return out


def _yaw(attempts: list[AttemptFrames], order: dict[str, int], axis: str) -> dict[str, Any]:
    """The scramble pose's heading about the gravity axis, per session and between sessions."""
    sessions: dict[str, list[tuple[float, float]]] = {}
    for a in attempts:
        s = a.segments.get("scramble")
        if s is not None:
            sessions.setdefault(a.session, []).append((a.start_ms, float(heading(s.mean, axis))))
    rows = []
    steps: list[float] = []
    for session in sorted(sessions, key=lambda s: order[s]):
        points = sorted(sessions[session])
        t = np.array([p[0] for p in points])
        h = np.unwrap(np.array([p[1] for p in points]))
        hours = (t - t[0]) / 3.6e6
        consecutive = np.abs(np.degrees(np.diff(h)))
        steps += consecutive.tolist()
        row: dict[str, Any] = {
            "session": session,
            "attempts": len(points),
            "hours": float(hours[-1]),
            "mean": math.degrees(circular_mean(h)),
            "spread": float(np.degrees(np.sqrt(-2 * np.log(max(abs(np.mean(np.exp(1j * h))), 1e-12))))),
            "range": float(np.degrees(np.ptp(h))),
            "consecutiveMedian": float(np.median(consecutive)) if len(consecutive) else math.nan,
            "driftPerHour": math.nan,
            "residual": math.nan,
        }
        if len(points) > 2 and hours[-1] >= MIN_DRIFT_HOURS:
            slope, intercept = np.polyfit(hours, np.degrees(h), 1)
            row["driftPerHour"] = float(slope)
            row["residual"] = float(np.std(np.degrees(h) - (slope * hours + intercept)))
        rows.append(row)
    means = [r["mean"] for r in rows]
    apart = [abs(float(wrap(a - b))) for i, a in enumerate(means) for b in means[i + 1 :]]
    return {
        "axis": axis,
        "sessions": rows,
        "consecutive": _quantiles(steps, (50, 90)),
        "betweenSessions": {
            "pairs": len(apart),
            "max": max(apart) if apart else math.nan,
            "median": float(np.median(apart)) if apart else math.nan,
        },
    }


def diagnose(
    dataset: Dataset,
    *,
    session: str | None = None,
    time_base: str = "fit",
    log: Callable[[str], None] | None = None,
) -> GyroFrames:
    """The diagnostic over every attempt with `gyro.json` under the root (or one session's)."""
    attempts: list[AttemptFrames] = []
    changes: list[dict[str, np.ndarray]] = []
    skipped = {"no-gyro": 0, "unreadable": 0}
    for sid in [session] if session else dataset.sessions():
        for index in dataset.attempts(sid):
            try:
                attempt = dataset.attempt(sid, index)
                gyro = dataset.gyro(sid, index)
                if gyro is None:
                    skipped["no-gyro"] += 1
                    continue
                frames, change = attempt_frames(attempt, gyro, time_base)
            except (RecordError, ValueError, KeyError, OSError) as error:
                skipped["unreadable"] += 1
                if log is not None:
                    log(f"skipped  {sid}/{index:04d}: {error}")
                continue
            attempts.append(frames)
            changes.append(change)
    firsts: dict[str, float] = {}
    for a in attempts:
        firsts[a.session] = min(firsts.get(a.session, math.inf), a.start_ms)
    order = {s: i for i, s in enumerate(sorted(firsts, key=lambda s: (firsts[s], s)))}
    counts = {s: sum(a.session == s for a in attempts) for s in order}
    axes = _axes(attempts, order)
    gravity = _gravity(axes)
    yaw_axis = gravity.get("gravityAxis") or gravity.get("nearestGyroAxis") or "z"
    summary = {
        "sessions": len(order),
        "attempts": len(attempts),
        "samples": int(sum(a.samples for a in attempts)),
        "rateHz": _quantiles([a.rate_hz for a in attempts], (10, 50, 90)),
        "skipped": skipped,
        "segments": {s: sum(s in a.segments for a in attempts) for s in SEGMENTS},
        "convention": _convention(changes),
        "axes": axes,
        "gravity": gravity,
        "held": _held(attempts, gravity),
        "within": _within(attempts, gravity),
        "yaw": _yaw(attempts, order, yaw_axis),
    }
    summary["advice"] = advice(summary)
    return GyroFrames(attempts=attempts, skipped=skipped, sessions=counts, summary=summary)


def advice(summary: dict[str, Any]) -> str:
    """What the findings say about the calibration."""
    gravity = summary["gravity"]
    if gravity.get("verdict") != "found":
        return (
            "No cube axis keeps one direction across the attempts: the gyro's frame is arbitrary in three "
            "degrees of freedom, so the calibration is a whole rotation (data.calibration_dof = rotation)."
        )
    yaw = summary["yaw"]
    between = yaw["betweenSessions"]
    steps = yaw["consecutive"]
    return (
        f"Gravity is the gyro's {gravity['gravityAxis']} axis: only a yaw about it is arbitrary "
        f"(data.calibration_dof = yaw, data.gravity_axis = {gravity['gravityAxis']}). The yaw moves "
        f"{_fmt(steps[0], 1)}° between consecutive attempts (median) and its sessions sit up to "
        f"{_fmt(between['max'], 0)}° apart: one calibration per attempt, from its scramble."
    )


def _fmt(value: float, digits: int = 2) -> str:
    return "–" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def _vector(v: Iterable[float]) -> str:
    return "(" + ", ".join(f"{x:+.2f}" for x in v) + ")"


def attempts_frame(result: GyroFrames) -> pl.DataFrame:
    """One row per attempt and segment: the samples, the mean orientation, the spread, each cube axis's mode
    and concentration (the session's id as it is: the owner's)."""
    rows = []
    for a in result.attempts:
        for segment, s in a.segments.items():
            row: dict[str, Any] = {
                "sessionId": a.session,
                "attemptIndex": a.attempt,
                "segment": segment,
                "samples": s.samples,
                "rateHz": a.rate_hz,
                "qx": s.mean[0],
                "qy": s.mean[1],
                "qz": s.mean[2],
                "qw": s.mean[3],
                "spreadMedianDeg": s.spread[0],
                "spreadP90Deg": s.spread[1],
            }
            for k, axis in enumerate(AXES):
                row[f"{axis}Mode"] = s.modes[k].tolist()
                row[f"{axis}Concentration"] = float(s.concentration[k])
            rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def report_text(result: GyroFrames, root: str = "") -> str:
    """The findings as Markdown (sessions by their order and the first 8 characters of their id)."""
    s = result.summary
    order = {sid: k + 1 for k, sid in enumerate(result.sessions)}
    name = {sid: f"{order[sid]} ({sid[:8]})" for sid in order}
    lines = [
        "# The gyro's frame",
        "",
        f"`cubetrace-ml gyro-frames`{f' on {root}' if root else ''}: {s['attempts']:,} attempts with "
        f"gyro.json in {s['sessions']} sessions ({s['samples']:,} samples, "
        f"{_fmt(s['rateHz'][1], 1)} Hz median); skipped {s['skipped']['no-gyro']} without gyro.json and "
        f"{s['skipped']['unreadable']} unreadable; {s['segments']['scramble']} scrambles and "
        f"{s['segments']['solve']} solves with at least {MIN_SAMPLES} samples in their window.",
        "",
        f"**Advice.** {s['advice']}",
        "",
        "## The convention",
        "",
    ]
    c = s["convention"]
    if c["pairs"]:
        rows = []
        for k, axis in enumerate(AXES):
            rows.append(
                [
                    f"v{axis}",
                    *(_fmt(x) for x in c["bodyFrame"][k]),
                    *(_fmt(x) for x in c["gyroFrame"][k]),
                ]
            )
        lines += [
            f"Spearman's correlation of the angular velocity `v` with the change between consecutive samples "
            f"({c['pairs']:,} pairs), the change in the cube's frame `conj(q_t) · q_{{t+1}}` against the "
            "change in the gyro's `q_{t+1} · conj(q_t)` (rotation vectors per second, x y z):",
            "",
            "| | cube x | cube y | cube z | gyro x | gyro y | gyro z |",
            "|---|---|---|---|---|---|---|",
            *("| " + " | ".join(row) + " |" for row in rows),
            "",
            f"Mean diagonal: {_fmt(c['bodyDiagonal'])} in the cube's frame, {_fmt(c['gyroDiagonal'])} "
            f"(absolute) in the gyro's. **{c['verdict']}.**",
            "",
        ]
    else:
        lines += [c["verdict"], ""]
    lines += [
        "## The cube's axes in the gyro's frame",
        "",
        "Per attempt and segment, each cube axis's principal direction (sign-free) and its concentration (1: "
        "one direction throughout; 1/3: every direction alike); pooled over the attempts, the direction of "
        "those principal directions, how concentrated they are, and how far the sessions' own directions sit "
        "apart:",
        "",
        "| segment | cube axis | concentration per attempt (10/25/50/75/90%) | pooled direction | pooled "
        "concentration | angle to it (50/90%) | sessions apart (max) |",
        "|---|---|---|---|---|---|---|",
    ]
    for segment, per_axis in s["axes"].items():
        for axis, a in per_axis.items():
            lines.append(
                f"| {segment} | {axis} | {' / '.join(_fmt(x) for x in a['concentration'])} | "
                f"{_vector(a['direction'])} | {_fmt(a['pooledConcentration'])} | "
                f"{_fmt(a['angleToDirection'][0], 1)}° / {_fmt(a['angleToDirection'][1], 1)}° | "
                f"{_fmt(a['sessionsApartMax'], 1)}° |"
            )
    g = s["gravity"]
    lines += ["", "## Gravity", ""]
    if "cubeAxis" in g:
        others = ", ".join(f"{k} {_fmt(v)}" for k, v in g["others"].items())
        found = f" (gravity: the gyro's {g['gravityAxis']} axis)" if g.get("gravityAxis") else ""
        lines += [
            f"The most concentrated cube axis over both segments is **{g['cubeAxis']}** (pooled "
            f"{_fmt(g['pooledConcentration'])}; the others {others}): its direction in the gyro's frame is "
            f"{_vector(g['direction'])}, {_fmt(g['angleToNearest'], 1)}° from the gyro's "
            f"{g['nearestGyroAxis']} axis. Verdict: **{g['verdict']}**{found}.",
            "",
        ]
        held = s["held"]
        if held:
            lines += [
                f"How cube axis {g['cubeAxis']} sits (the segment's mean orientation): along the pooled "
                "direction or against it, and its tilt from it (10/50/90%):",
                "",
                "| segment | attempts | along | against | tilt |",
                "|---|---|---|---|---|",
                *(
                    f"| {seg} | {h['attempts']} | {h['along']} | {h['against']} | "
                    f"{' / '.join(_fmt(x, 1) + '°' for x in h['tilt'])} |"
                    for seg, h in held.items()
                ),
                "",
            ]
    else:
        lines += [f"Verdict: **{g['verdict']}** ({g.get('reason', '')}).", ""]
    w = s["within"]
    lines += [
        "## Within an attempt",
        "",
        f"- The samples' angle from their segment's mean orientation, median per attempt (10/50/90% of the "
        f"attempts): scramble {' / '.join(_fmt(x, 1) + '°' for x in w['scrambleSpread']['median'])}, solve "
        f"{' / '.join(_fmt(x, 1) + '°' for x in w['solveSpread']['median'])}.",
        f"- The scramble's mean orientation to the solve's (10/50/90%): "
        f"{' / '.join(_fmt(x, 1) + '°' for x in w['scrambleToSolve'])}.",
    ]
    if "verticalScrambleToSolve" in w:
        lines += [
            f"- The vertical axis's direction, scramble against solve (50/90%): "
            f"{' / '.join(_fmt(x, 1) + '°' for x in w['verticalScrambleToSolve'])}; the solve's first half "
            f"against its second (50/90%): {' / '.join(_fmt(x, 1) + '°' for x in w['verticalSolveHalves'])}.",
        ]
    y = s["yaw"]
    lines += [
        "",
        f"## The yaw about {y['axis']}",
        "",
        "Each attempt's scramble pose (the mean orientation over its scramble) and its heading about the "
        "axis; per session in the order of their first attempt (its id's first 8 characters):",
        "",
        "| session | attempts | hours | mean | spread | range | drift per hour | residual | "
        "consecutive (median) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in y["sessions"]:
        lines.append(
            f"| {name[r['session']]} | {r['attempts']} | {_fmt(r['hours'], 2)} | {_fmt(r['mean'], 0)}° | "
            f"{_fmt(r['spread'], 0)}° | {_fmt(r['range'], 0)}° | {_fmt(r['driftPerHour'], 1)}° | "
            f"{_fmt(r['residual'], 1)}° | {_fmt(r['consecutiveMedian'], 1)}° |"
        )
    b = y["betweenSessions"]
    lines += [
        "",
        f"Between consecutive attempts of a session the heading moves {_fmt(y['consecutive'][0], 1)}° "
        f"(median; 90%: {_fmt(y['consecutive'][1], 1)}°); the sessions' means sit {_fmt(b['median'], 0)}° "
        f"apart (median of {b['pairs']} pairs; at most {_fmt(b['max'], 0)}°).",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(result: GyroFrames, out: str | Path, root: str = "") -> list[Path]:
    """`gyro-frames.md` (the report), `gyro-frames.json` (the summary) and `gyro-frames.parquet` (per attempt
    and segment) in the folder."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / "gyro-frames.md", out / "gyro-frames.json", out / "gyro-frames.parquet"]
    paths[0].write_text(report_text(result, root))
    paths[1].write_text(json.dumps(_jsonable(result.summary), indent=2) + "\n")
    attempts_frame(result).write_parquet(paths[2])
    return paths


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    return value
