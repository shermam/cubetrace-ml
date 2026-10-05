"""Synthetic records in the app's formats (valid against `schemas/`) and tiny videos, for the tests."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np

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
