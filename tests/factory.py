"""Synthetic records in the app's formats (valid against `schemas/`) and tiny videos, for the tests."""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Callable, Collection, Iterable, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np

from cubetrace_ml import cube
from cubetrace_ml.align import align_clip
from cubetrace_ml.dataset import ClipRef
from cubetrace_ml.features import feature_path, write_features
from cubetrace_ml.moves import INDEX

SOLVED = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"
APP = {"version": "0.4.0", "commit": "abc1234"}
T0 = 1_790_000_000_000.0  # a host time of the right magnitude (2026)
DAY_MS = 86_400_000.0


def session_id(n: int) -> str:
    """A UUID v4 that is obviously synthetic."""
    return f"{n:08x}-0000-4000-8000-000000000000"


def frames_record(
    camera: str, segment: str, t0: float, dt: Sequence[float], *, remote: bool = False
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema": 2,
        "camera": camera,
        "segment": segment,
        "app": APP,
        "t0HostMs": t0,
        "dtMs": [float(x) for x in dt],
        "keyframes": [0],
        "arrival": {"offsetMs": 1234.5, "residualP95Ms": 4.2},
    }
    if remote:
        record["t0RemoteMs"] = t0 + 3127.4
        record["remote"] = {
            "offsetMs": 3127.4,
            "driftPpm": 0,
            "rttMs": 9.6,
            "samples": 12,
            "residualP95Ms": 2.1,
            "since": t0 - 60_000,
            "converged": True,
            "takenMs": t0 + 5000,
        }
    return record


def regular_frames(camera: str, segment: str, t0: float, n: int, interval: float = 33.3) -> dict[str, Any]:
    return frames_record(camera, segment, t0, [0.0] + [interval] * (n - 1))


def clip_entry(
    camera: str,
    segment: str,
    frames: dict[str, Any],
    *,
    lag: float | None = 50.0,
    crop: dict[str, int] | None = None,
    size: tuple[int, int] = (64, 64),
    fps: float = 30,
    bytes_: int = 1,
    truncated: bool | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "camera": camera,
        "segment": segment,
        "file": f"{camera}.{segment}.mp4",
        "bytes": bytes_,
        "codec": "avc1.640028",
        "audio": None,
        "width": size[0],
        "height": size[1],
        "crop": crop,
        "fpsNominal": fps,
        "frames": len(frames["dtMs"]),
        "firstFrameHostMs": frames["t0HostMs"],
        "framesFile": f"{camera}.{segment}.frames.json",
        "syncResidualMs": lag,
    }
    if truncated is not None:
        entry["truncatedStart"] = truncated
    return entry


def attempt_record(
    session: str,
    index: int,
    scramble: Sequence[tuple[str, float]],
    solve: Sequence[tuple[str, float]],
    *,
    video: Sequence[dict[str, Any]] = (),
    cube_ms: Sequence[float] | None = None,
    clock: dict[str, Any] | str | None = "exact",
    status: str = "solved",
    replay_ok: bool = True,
    gyro: dict[str, Any] | None = None,
    shown: float | None = None,
) -> dict[str, Any]:
    """An attempt whose moves are `scramble` then `solve`, (move, hostMs) each. By default the cube's clock
    is the host's less T0 and the clock fit is exact (a = 1, b = T0); `cube_ms` overrides the cube times."""
    moves = [(m, t, "scramble") for m, t in scramble] + [(m, t, "solve") for m, t in solve]
    cube = list(cube_ms) if cube_ms is not None else [t - T0 for _, t, _ in moves]
    records = [
        {"m": m, "hostMs": t, "cubeMs": c, "phase": p, "serial": i % 256, "packetLast": True}
        for i, ((m, t, p), c) in enumerate(zip(moves, cube, strict=True))
    ]
    if clock == "exact":
        clock = {"a": 1.0, "b": T0, "residualP95Ms": 0.0, "samples": max(2, len(moves))}
    dnf = status == "dnf"
    solve_start = solve[0][1] if solve else None
    solve_end = solve[-1][1] if solve and not dnf else None
    events = {
        "scrambleShown": shown if shown is not None else (scramble[0][1] - 1000 if scramble else T0),
        "scrambleStart": scramble[0][1] if scramble else None,
        "scrambleDone": scramble[-1][1] if scramble else None,
        "pickup": None,
        "solveStart": solve_start,
        "solveEnd": solve_end,
    }
    qtm = sum(2 if m.endswith("2") else 1 for m, _ in solve)
    time_ms = None if dnf or not solve else solve[-1][1] - solve[0][1]
    record = {
        "schema": 2,
        "session": session,
        "index": index,
        "app": APP,
        "scramble": " ".join(m for m, _ in scramble) or "R",
        "scrambledFacelets": SOLVED,
        "crossFace": None,
        "events": events,
        "moves": records,
        "clock": clock,
        "result": {
            "timeMs": time_ms,
            "inspectionMs": (solve_start - events["scrambleDone"]) if solve_start and scramble else None,
            "movesQtm": qtm,
            "tps": round(qtm / (time_ms / 1000), 2) if time_ms else None,
            "status": status,
            "replayOk": replay_ok and not dnf,
            "scrambleCorrected": False,
            "scrambleExtraMoves": 0,
        },
        "phases": [],
        "video": list(video),
        "gyro": gyro,
        "resyncs": [],
    }
    return record


def gyro_record(
    session: str, index: int, t0: float, dt: Sequence[float], quats: Sequence[Sequence[float]]
) -> dict[str, Any]:
    return {
        "schema": 1,
        "session": session,
        "index": index,
        "app": APP,
        "t0HostMs": t0,
        "dtMs": [float(x) for x in dt],
        "q": [float(v) for q in quats for v in q],
        "v": [0, 1, -1] * len(quats),
        "truncatedStart": False,
    }


def gyro_summary(gyro: dict[str, Any]) -> dict[str, Any]:
    times = gyro["t0HostMs"] + np.cumsum(gyro["dtMs"])
    span = (times[-1] - times[0]) / 1000
    return {
        "file": "gyro.json",
        "samples": len(times),
        "fromHostMs": float(times[0]),
        "toHostMs": float(times[-1]),
        "rateHz": round((len(times) - 1) / span, 1) if span > 0 else 0,
        "truncatedStart": False,
    }


def camera(label: str, *, local: bool = True, crop: dict[str, int] | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "label": label,
        "local": local,
        "facing": "user" if local else "environment",
        "deviceLabel": f"{label} camera",
        "settings": {"width": 64, "height": 64},
        "capabilities": {},
        "constraints": {},
        "crop": crop,
        "mode": "full",
    }
    if not local:
        record["remote"] = {"label": "phone", "platform": "Android"}
    return record


def session_record(
    sid: str, created_ms: float, cameras: Sequence[dict[str, Any]], lags: dict[str, float]
) -> dict:
    return {
        "schema": 2,
        "id": sid,
        "createdMs": created_ms,
        "app": APP,
        "host": {"label": "test host", "userAgent": "pytest", "platform": "Linux", "isPhone": False},
        "cube": {"model": "GAN12uiFreeplay", "hardware": "1", "firmware": "1", "gyro": True},
        "cameras": list(cameras),
        "clock": {
            "cube": {"a": 1, "b": 0, "residualP95Ms": 0, "samples": 0},
            "cameras": {
                label: {
                    "offsetMs": lag,
                    "rttMs": 0,
                    "driftPpm": 0,
                    "clapperboardResidualMs": 5.0,
                    "clapperboardSamples": 8,
                }
                for label, lag in lags.items()
            },
        },
        "audio": False,
        "settings": {"inspection15s": False, "autoAdvance": True},
        "notes": "",
        "summary": {"attempts": 1, "solved": 1, "dnf": 0},
    }


def gray(k: int) -> int:
    """The gray level of frame k in the tests' videos."""
    return 16 + 8 * k


def write_video(path: Path, n: int, size: tuple[int, int] = (64, 64), fps: int = 30) -> int:
    """An MP4 of `n` flat gray frames (frame k at gray(k)); returns its size in bytes."""
    return write_frames(
        path, (np.full((size[1], size[0], 3), gray(k), dtype=np.uint8) for k in range(n)), fps
    )


def write_frames(path: Path, frames: Iterable[np.ndarray], fps: int = 30) -> int:
    """An MP4 of the given RGB frames (h × w × 3 uint8, h and w even, at high quality); returns its size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = None
        for k, pixels in enumerate(frames):
            if stream is None:
                stream = container.add_stream("mpeg4", rate=fps, options={"b": "4M"})
                stream.height, stream.width = pixels.shape[:2]
                stream.pix_fmt = "yuv420p"
                stream.time_base = Fraction(1, fps)
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(pixels), format="rgb24")
            frame.pts = k
            for packet in stream.encode(frame):
                container.mux(packet)
        if stream is not None:
            for packet in stream.encode():
                container.mux(packet)
    return path.stat().st_size


def write_json(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))


def write_attempt(
    root: Path,
    attempt: dict[str, Any],
    frames: Sequence[dict[str, Any]],
    *,
    gyro: dict[str, Any] | None = None,
    videos: bool = True,
) -> Path:
    """Writes the attempt's folder: its frames files, an MP4 per clip (its bytes set in the record) and
    gyro.json when given."""
    folder = root / "sessions" / attempt["session"] / "attempts" / f"{attempt['index']:04d}"
    folder.mkdir(parents=True, exist_ok=True)
    for entry in attempt["video"]:
        record = next(f for f in frames if (f["camera"], f["segment"]) == (entry["camera"], entry["segment"]))
        write_json(folder / entry["framesFile"], record)
        if videos:
            entry["bytes"] = write_video(
                folder / entry["file"], len(record["dtMs"]), (entry["width"], entry["height"])
            )
    if gyro is not None:
        write_json(folder / "gyro.json", gyro)
    write_json(folder / "attempt.json", attempt)
    return folder


def short_attempt(
    sid: str,
    index: int,
    start: float,
    cameras: dict[str, float | None],
    *,
    n_frames: int = 24,
    status: str = "solved",
    with_gyro: bool = True,
    truncated: dict[str, bool] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    """A whole attempt that fits in 24-frame clips: 4 scramble turns from `start` (the last two a double:
    R U F2), the solve 1 s later (R R, L R' 5 ms apart, U, D': R2 M' U D'); per camera (label → lag) a
    scramble and a solve clip of `n_frames` frames of 33.3 ms from 100 ms before their window."""
    scramble = [("R", start), ("U", start + 120), ("F", start + 240), ("F", start + 330)]
    s0 = start + 1330
    solve = [("R", s0), ("R", s0 + 90), ("L", s0 + 250), ("R'", s0 + 255), ("U", s0 + 400), ("D'", s0 + 520)]
    frames, video = [], []
    for label, lag in cameras.items():
        for segment, w0 in (("scramble", start), ("solve", s0)):
            record = regular_frames(label, segment, w0 - 100, n_frames)
            frames.append(record)
            flag = (truncated or {}).get(f"{label}.{segment}")
            video.append(clip_entry(label, segment, record, lag=lag, truncated=flag))
    gyro = None
    if with_gyro:
        n = 40
        quats = [(0.0, 0.0, float(np.sin(k * 0.01)), float(np.cos(k * 0.01))) for k in range(n)]
        gyro = gyro_record(sid, index, start - 300, [0.0] + [50.0] * (n - 1), quats)
    if status == "dnf":
        solve = solve[:3]
    attempt = attempt_record(
        sid,
        index,
        scramble,
        solve,
        video=video,
        status=status,
        replay_ok=status == "solved",
        gyro=gyro_summary(gyro) if gyro else None,
    )
    return attempt, frames, gyro


def build_dataset(root: Path, *, videos: bool = True) -> dict[str, str]:
    """Four sessions over three days, with what the manifest must tell apart:

    - A (day 1, session.json, laptop lag 40): attempt 1 solved; attempt 2 a DNF, its solve clip truncated.
    - B (day 2, session.json, laptop lag 50 and an unsynced phone): attempt 1 solved, four clips.
    - C (day 2, no session.json, an unsynced phone, no gyro): attempt 1 solved.
    - D (day 3, the latest: test, session.json, laptop lag 30): attempt 1 solved.
    """
    ids = {name: session_id(n) for n, name in enumerate("ABCD", start=1)}
    day = {name: T0 + k * DAY_MS for k, name in enumerate("ABCD")}
    day["C"] = day["B"]
    day["D"] = T0 + 2 * DAY_MS

    write_json(
        root / "sessions" / ids["A"] / "session.json",
        session_record(ids["A"], day["A"], [camera("laptop")], {"laptop": 40.0}),
    )
    attempt, frames, gyro = short_attempt(ids["A"], 1, day["A"] + 60_000, {"laptop": 40.0})
    write_attempt(root, attempt, frames, gyro=gyro, videos=videos)
    attempt, frames, gyro = short_attempt(
        ids["A"], 2, day["A"] + 120_000, {"laptop": 40.0}, status="dnf", truncated={"laptop.solve": True}
    )
    write_attempt(root, attempt, frames, gyro=gyro, videos=videos)

    write_json(
        root / "sessions" / ids["B"] / "session.json",
        session_record(
            ids["B"], day["B"], [camera("laptop"), camera("phone-rear", local=False)], {"laptop": 50.0}
        ),
    )
    attempt, frames, gyro = short_attempt(
        ids["B"], 1, day["B"] + 60_000, {"laptop": 50.0, "phone-rear": None}
    )
    write_attempt(root, attempt, frames, gyro=gyro, videos=videos)

    attempt, frames, gyro = short_attempt(
        ids["C"], 1, day["C"] + 90_000, {"phone-front": None}, with_gyro=False
    )
    write_attempt(root, attempt, frames, gyro=gyro, videos=videos)

    write_json(
        root / "sessions" / ids["D"] / "session.json",
        session_record(ids["D"], day["D"], [camera("laptop")], {"laptop": 30.0}),
    )
    attempt, frames, gyro = short_attempt(ids["D"], 1, day["D"] + 60_000, {"laptop": 30.0})
    write_attempt(root, attempt, frames, gyro=gyro, videos=videos)
    return ids


# Synthetic features (M2): attempts without videos, whose clips' features carry a planted onset signal.

FACE_TURNS = [face + suffix for face in "URFDLB" for suffix in ("", "'")]


def move_stream(
    rng: np.random.Generator,
    start: float,
    n: int,
    *,
    gap: tuple[float, float] = (210, 330),
    doubles: float = 0.0,
) -> list[tuple[str, float]]:
    """`n` quarter turns from `start`, `gap` ms apart, no face twice in a row (so nothing merges), except
    that with probability `doubles` a turn is followed 90 ms later by its twin: a double."""
    moves: list[tuple[str, float]] = []
    t = start
    while len(moves) < n:
        previous = moves[-1][0][0] if moves else None
        move = str(rng.choice([m for m in FACE_TURNS if m[0] != previous]))
        moves.append((move, t))
        if len(moves) < n and rng.random() < doubles:
            moves.append((move, t + 90.0))
            t += 90.0
        t += float(rng.uniform(*gap))
    return moves


def synthetic_attempt(
    sid: str,
    index: int,
    start: float,
    rng: np.random.Generator,
    *,
    moves: int = 16,
    lag: float | None = 40.0,
    camera_label: str = "laptop",
    interval: float = 1000.0 / 30,
    lead_ms: float = 700.0,
    tail_ms: float = 700.0,
    doubles: float = 0.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """An attempt whose solve is `moves` random quarter turns (`doubles`: see move_stream) and whose
    scramble, from `start`, is their inverse (so the solve replays), 2 s of inspection between them; one
    camera's scramble and solve clips, each from `lead_ms` before its window to `tail_ms` after it (on the
    frames' timeline, the lag included)."""
    solve_turns = move_stream(rng, 0.0, moves, doubles=doubles)
    scramble_moves, t = [], start
    for move in cube.inverse([m for m, _ in solve_turns]):
        scramble_moves.append((move, t))
        t += float(rng.uniform(210, 330))
    solve_start = scramble_moves[-1][1] + 2000.0
    solve_moves = [(m, solve_start + u) for m, u in solve_turns]
    frames, video = [], []
    shift = lag or 0.0
    for segment, segment_moves in (("scramble", scramble_moves), ("solve", solve_moves)):
        t0 = segment_moves[0][1] + shift - lead_ms
        n = int((segment_moves[-1][1] - segment_moves[0][1] + lead_ms + tail_ms) / interval) + 1
        record = frames_record(camera_label, segment, t0, [0.0] + [interval] * (n - 1))
        frames.append(record)
        video.append(clip_entry(camera_label, segment, record, lag=lag))
    attempt = attempt_record(sid, index, scramble_moves, solve_moves, video=video)
    attempt["scrambledFacelets"] = cube.scrambled([m for m, _ in scramble_moves])
    return attempt, frames


def write_synthetic_attempt(
    root: Path,
    attempt: dict[str, Any],
    frames: Sequence[dict[str, Any]],
    gyro: dict[str, Any] | None = None,
) -> Path:
    """The attempt's folder with its frames files, gyro.json when given and, for each clip, a few bytes
    standing for its MP4 (the manifest's video check `none` reads its size only)."""
    folder = root / "sessions" / attempt["session"] / "attempts" / f"{attempt['index']:04d}"
    folder.mkdir(parents=True, exist_ok=True)
    for entry in attempt["video"]:
        (folder / entry["file"]).write_bytes(b"\0" * 16)
        entry["bytes"] = 16
    if gyro is not None:
        attempt["gyro"] = gyro_summary(gyro)
    return write_attempt(root, attempt, frames, gyro=gyro, videos=False)


# A synthetic orientation (M3): the cube turned about its U–D axis in the hands, so that the side face the
# camera sees turning is not always the face the cube reports.

SIDES = "FRBL"  # the side faces in their order clockwise around U, seen from above


def seen_symbol(symbol: str, quarters: int) -> str:
    """The symbol the camera sees when the cube is turned `quarters` quarter turns about its U axis from its
    home orientation (each a `y`: the cube's R face comes to the front): a face turn of the cube's own
    `symbol` happens on the side where that face now is. U and D stay (the synthetic streams have no
    slices)."""
    face, turn = symbol[0], symbol[1:]
    if face in "UD":
        return symbol
    assert face in SIDES, f"no slice in the synthetic streams: {symbol}"
    return SIDES[(SIDES.index(face) - quarters) % 4] + turn


def yaw_quaternion(degrees: float, tilt_degrees: float = 0.0) -> tuple[float, float, float, float]:
    """The orientation (x, y, z, w) of a cube turned `degrees` clockwise about the vertical axis (z, seen
    from above: a negative angle about +z) after a tilt of `tilt_degrees` about x."""
    yaw = -math.radians(degrees) / 2
    tilt = math.radians(tilt_degrees) / 2
    a = (0.0, 0.0, math.sin(yaw), math.cos(yaw))
    b = (math.sin(tilt), 0.0, 0.0, math.cos(tilt))
    # The Hamilton product a · b (the tilt, then the yaw).
    return (
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    )


def yaw_schedule(
    attempt: dict[str, Any],
    rng: np.random.Generator,
    quarters: Sequence[int] = (-1, 0, 1, 2),
    ramp_ms: float = 150.0,
    scramble_home: bool = False,
) -> list[tuple[float, float, int, int]]:
    """How the cube turns about U during the attempt, as ramps (from ms, to ms, from quarters, to quarters):
    one of `quarters` for the scramble (0, its home, with `scramble_home`: the solver holds it one way to
    apply the prescribed moves), one for the solve (the cube turned during the inspection), and another one
    from a pause of the solve on (between two moves at least 200 ms apart, not the first or the last)."""
    moves = attempt["moves"]
    scramble = [m["hostMs"] for m in moves if m["phase"] == "scramble"]
    solve = [m["hostMs"] for m in moves if m["phase"] == "solve"]
    first, second = (int(k) for k in rng.choice(quarters, size=2))
    if scramble_home:
        first = 0
    ramps = [(scramble[-1] + 600.0, scramble[-1] + 600.0 + ramp_ms, first, second)]
    pauses = [i for i in range(1, len(solve) - 2) if solve[i + 1] - solve[i] >= 200.0]
    others = [k for k in quarters if k != second]
    if pauses and others:
        i = pauses[int(rng.integers(len(pauses)))]
        third = int(rng.choice(others))
        ramps.append((solve[i] + 30.0, solve[i] + 30.0 + ramp_ms, second, third))
    return ramps


def yaw_at(ramps: Sequence[tuple[float, float, int, int]], t_ms: float) -> float:
    """The cube's yaw in quarter turns at a time (a ramp's two ends joined linearly)."""
    quarters = float(ramps[0][2])
    for start, end, before, after in ramps:
        if t_ms >= end:
            quarters = float(after)
        elif t_ms > start:
            quarters = before + (after - before) * (t_ms - start) / (end - start)
    return quarters


def yaw_gyro(
    attempt: dict[str, Any],
    ramps: Sequence[tuple[float, float, int, int]],
    rng: np.random.Generator,
    interval: float = 70.0,
    frame_yaw: float = 0.0,
) -> dict[str, Any]:
    """gyro.json for the attempt (about 14 samples a second, as the cube sends them, from 2 s before the
    scramble to 1 s after the solve): the schedule's yaw with a wobble of a few degrees about x, seen from a
    gyro whose frame is turned `frame_yaw` degrees about the vertical from the camera's (each sample
    `about z(−frame_yaw) · q`: the calibration that undoes it is `about z(frame_yaw)`)."""
    times = [m["hostMs"] for m in attempt["moves"]]
    t0, end = times[0] - 2000.0, times[-1] + 1000.0
    n = int((end - t0) / interval) + 1
    sample_ms = t0 + interval * np.arange(n)
    quats = [yaw_quaternion(90.0 * yaw_at(ramps, t), float(rng.normal(0.0, 3.0))) for t in sample_ms]
    if frame_yaw:
        frame = quaternion_about((0.0, 0.0, 1.0), -frame_yaw)
        quats = [tuple(quaternion_product(frame, q)) for q in quats]
    return gyro_record(attempt["session"], attempt["index"], t0, [0.0] + [interval] * (n - 1), quats)


def planted_features(
    t_ms: np.ndarray,
    onsets_ms: Sequence[float],
    symbols: Sequence[int],
    embeddings: np.ndarray,
    rng: np.random.Generator,
    *,
    noise: float = 0.3,
    shape: Sequence[float] = (1.0, 0.7, 0.4, 0.2),
) -> np.ndarray:
    """Frames × dim features: Gaussian noise, plus, from the frame nearest each onset on, its symbol's
    embedding times `shape` (the turn's motion fading over a few frames)."""
    x = rng.normal(0.0, noise, (len(t_ms), embeddings.shape[1]))
    for onset, symbol in zip(onsets_ms, symbols, strict=True):
        k = int(np.argmin(np.abs(t_ms - onset)))
        for d, a in enumerate(shape):
            if k + d < len(x):
                x[k + d] += a * embeddings[symbol]
    return x.astype(np.float16)


def build_feature_dataset(
    root: Path,
    features: Path,
    *,
    encoder: str = "synthetic",
    sessions: int = 6,
    attempts: int = 2,
    dim: int = 16,
    seed: int = 0,
    planted: bool = True,
    doubles: float = 0.0,
    moves: int = 16,
    yaw: bool = False,
    yaw_quarters: Sequence[int] = (-1, 0, 1, 2),
    without_gyro: Collection[tuple[int, int]] = (),
    scramble_home: bool = False,
    frame_yaws: Callable[[int, int], float] | None = None,
) -> dict[str, str]:
    """`sessions` sessions on as many days (each with session.json and `attempts` attempts of one laptop
    camera), and every clip's features under `<features>/<encoder>/`: the planted onset signal (each
    symbol's own embedding) or noise alone. With `yaw`, each attempt has a gyro.json (but those of
    `without_gyro`, (session position, attempt index) pairs) whose cube turns about U, among
    `yaw_quarters`, between its segments and once in its solve (`yaw_schedule`; at home through the
    scramble with `scramble_home`), and the planted signal is the symbol the camera sees (`seen_symbol`),
    the yaw drawn from a generator of its own: the moves and the noise are the same as without. M4:
    `frame_yaws(session position, attempt index)` turns that attempt's gyro frame about the vertical by
    so many degrees (`yaw_gyro`). Returns the session ids by position ("0", "1", …)."""
    rng = np.random.default_rng(seed)
    yaw_rng = np.random.default_rng([seed, 3])
    embeddings = rng.normal(0.0, 1.0, (24, dim))
    embeddings *= 2.0 / np.linalg.norm(embeddings, axis=1, keepdims=True)
    ids = {}
    for s in range(sessions):
        sid = session_id(100 + s)
        ids[str(s)] = sid
        created = T0 + s * DAY_MS
        write_json(
            root / "sessions" / sid / "session.json", session_record(sid, created, [camera("laptop")], {})
        )
        for a in range(1, attempts + 1):
            attempt, frames = synthetic_attempt(
                sid, a, created + 60_000.0 * a, rng, doubles=doubles, moves=moves
            )
            ramps = yaw_schedule(attempt, yaw_rng, yaw_quarters, scramble_home=scramble_home) if yaw else None
            frame_yaw = frame_yaws(s, a) if frame_yaws else 0.0
            gyro = (
                yaw_gyro(attempt, ramps, yaw_rng, frame_yaw=frame_yaw)
                if ramps and (s, a) not in without_gyro
                else None
            )
            write_synthetic_attempt(root, attempt, frames, gyro)
            for record in frames:
                aligned = align_clip(attempt, record, None)
                track = aligned.track
                onsets = [aligned.onset_on_frames(sym) for sym in aligned.symbols]
                indices = [
                    INDEX[seen_symbol(sym.symbol, round(yaw_at(ramps, sym.onset_ms)))] if ramps else sym.index
                    for sym in aligned.symbols
                ]
                if planted:
                    x = planted_features(track["tMs"], onsets, indices, embeddings, rng)
                else:
                    x = rng.normal(0.0, 1.0, (len(track), dim)).astype(np.float16)
                ref = ClipRef(sid, a, record["camera"], record["segment"])
                meta = {"format": 1, "encoder": {"name": encoder, "dim": dim}, "clip": {"frames": len(track)}}
                arrays = {
                    "x": x,
                    "tMs": track["tMs"],
                    "shownMs": track["shownMs"],
                    "inWindow": track["inWindow"],
                }
                write_features(feature_path(features, encoder, ref), arrays, meta)
    return ids


# The gyro's frame (M4): a cube held as the owner holds it, seen from a gyro whose frame is arbitrary.


def quaternion_about(axis: Sequence[float], degrees: float) -> np.ndarray:
    """The rotation (x, y, z, w) by `degrees` about an axis."""
    axis = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    half = math.radians(degrees) / 2
    return np.array([*(math.sin(half) * axis), math.cos(half)])


def quaternion_product(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    """The Hamilton product a · b of two quaternions (x, y, z, w)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def held_gyro(
    attempt: dict[str, Any],
    rng: np.random.Generator,
    *,
    frame: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
    scramble_heading: float = 0.0,
    interval: float = 70.0,
    velocity: str = "body",
) -> dict[str, Any]:
    """gyro.json of a cube held as the owner holds it, seen from a gyro whose frame is `frame` (each sample is
    `frame · q`, q the cube's orientation in a frame whose z is up): white (+z) up, tilted 10° (a few degrees
    either way) toward the solver at `scramble_heading` degrees about z through the scramble; turned over
    about x during the inspection; turned about the vertical a quarter at a time during the solve. About 14
    samples a second from 2 s before the scramble to 1 s after the solve, each with its angular velocity as
    4-bit integers: the change since the previous sample in the cube's frame (`velocity="body"`, as the cube
    measures it) or in the gyro's (`"gyro"`: the convention check's counter-example)."""
    moves = attempt["moves"]
    events = attempt["events"]
    t0, end = moves[0]["hostMs"] - 2000.0, moves[-1]["hostMs"] + 1000.0
    n = int((end - t0) / interval) + 1
    sample_ms = t0 + interval * np.arange(n)
    done, start = events["scrambleDone"], events["solveStart"]
    solve = [m["hostMs"] for m in moves if m["phase"] == "solve"]
    inner = solve[1:-1]
    turns = sorted(float(t) for t in rng.choice(inner, size=min(3, len(inner)), replace=False))
    quats = []
    for t in sample_ms:
        tilt = 10.0 + float(rng.normal(0.0, 3.0))
        side = float(rng.normal(0.0, 3.0))
        flip = min(1.0, max(0.0, (t - done) / max(start - done, 1.0)))
        quarters = sum(t >= u for u in turns)
        yaw = scramble_heading + 90.0 * quarters
        q = quaternion_about((0, 0, 1), yaw)
        q = quaternion_product(q, quaternion_about((1, 0, 0), tilt + 180.0 * flip))
        q = quaternion_product(q, quaternion_about((0, 1, 0), side))
        quats.append(quaternion_product(frame, q))
    quats = [q if q[3] >= 0 else -q for q in quats]
    v = [0, 0, 0]
    for previous, q in itertools.pairwise(quats):
        conj = previous * np.array([-1, -1, -1, 1.0])
        change = quaternion_product(conj, q) if velocity == "body" else quaternion_product(q, conj)
        change = change if change[3] >= 0 else -change
        rate = 2.0 * change[:3] / (interval / 1000.0)  # about the rotation vector per second
        v += [int(np.clip(round(x * 4 + rng.normal(0.0, 0.3)), -7, 7)) for x in rate]
    gyro = gyro_record(attempt["session"], attempt["index"], t0, [0.0] + [interval] * (n - 1), quats)
    gyro["q"] = [round(float(x), 5) for x in gyro["q"]]
    gyro["v"] = v
    return gyro
