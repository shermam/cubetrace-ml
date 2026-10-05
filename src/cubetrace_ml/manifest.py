"""The manifest (one row per clip, and one per attempt) and the report of what the dataset holds."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import av
import numpy as np
import polars as pl

from .align import align_clip
from .dataset import ClipRef, Dataset
from .filter import REASONS, clip_reasons
from .moves import DOUBLE_MS, SLICE_MS, attempt_symbols, move_times
from .records import RecordError
from .splits import VAL_FRACTION, assign_splits
from .video import count_frames

VIDEO_CHECKS = ("none", "fast", "full")

CLIP_SCHEMA: dict[str, Any] = {
    "sessionId": pl.String,
    "day": pl.String,
    "attemptIndex": pl.Int32,
    "camera": pl.String,
    "segment": pl.String,
    "video": pl.String,
    "fpsNominal": pl.Float64,
    "fps": pl.Float64,
    "frames": pl.Int32,
    "framesFileCount": pl.Int32,
    "videoFrames": pl.Int32,
    "seconds": pl.Float64,
    "width": pl.Int32,
    "height": pl.Int32,
    "cropX": pl.Int32,
    "cropY": pl.Int32,
    "cropW": pl.Int32,
    "cropH": pl.Int32,
    "lagMs": pl.Float64,
    "unsynced": pl.Boolean,
    "movesInWindow": pl.Int32,
    "movesCovered": pl.Int32,
    "tps": pl.Float64,
    "status": pl.String,
    "replayOk": pl.Boolean,
    "gyroRateHz": pl.Float64,
    "truncatedStart": pl.Boolean,
    "usable": pl.Boolean,
    "reasons": pl.String,
    "split": pl.String,
}

ATTEMPT_SCHEMA: dict[str, Any] = {
    "sessionId": pl.String,
    "day": pl.String,
    "attemptIndex": pl.Int32,
    "status": pl.String,
    "replayOk": pl.Boolean,
    "tps": pl.Float64,
    "timeMs": pl.Float64,
    "moves": pl.Int32,
    "solveMoves": pl.Int32,
    "symbols": pl.Int32,
    "solveSymbols": pl.Int32,
    "movesOffFit": pl.Int32,
    "clips": pl.Int32,
    "cameras": pl.String,
    "gyroRateHz": pl.Float64,
    "split": pl.String,
}

SESSION_SCHEMA: dict[str, Any] = {
    "sessionId": pl.String,
    "day": pl.String,
    "sessionRecord": pl.Boolean,
    "attempts": pl.Int32,
    "split": pl.String,
}


@dataclass
class Tables:
    """The dataset as tables: `clips` is the manifest; `problems` the records that could not be read."""

    clips: pl.DataFrame
    attempts: pl.DataFrame
    sessions: pl.DataFrame
    problems: list[str] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)


def _gyro_rate(attempt: dict[str, Any]) -> float:
    gyro = attempt.get("gyro")
    return float(gyro["rateHz"]) if gyro else math.nan


def build_tables(
    dataset: Dataset,
    *,
    time_base: str = "fit",
    slice_ms: float = SLICE_MS,
    double_ms: float = DOUBLE_MS,
    seed: int = 0,
    held_out_day: str | None = None,
    val_fraction: float = VAL_FRACTION,
    video: str = "none",
) -> Tables:
    """Reads every record under the root into the manifest's tables. `video` says whether the MP4s' frame
    counts are measured: `none` (the JSON alone), `fast` (the container's header) or `full` (a decode)."""
    if video not in VIDEO_CHECKS:
        raise ValueError(f"video check {video!r}: one of {', '.join(VIDEO_CHECKS)}")
    problems: list[str] = []
    session_rows, attempt_rows, clip_rows = [], [], []

    for session in dataset.sessions():
        try:
            day, from_record = dataset.session_day(session)
        except (RecordError, ValueError, OSError) as error:
            problems.append(str(error))
            day, from_record = None, False
        indices = dataset.attempts(session)
        session_rows.append(
            {"sessionId": session, "day": day, "sessionRecord": from_record, "attempts": len(indices)}
        )
        for index in indices:
            try:
                attempt = dataset.attempt(session, index)
            except (RecordError, ValueError, OSError) as error:
                problems.append(str(error))
                continue
            symbols = attempt_symbols(attempt, time_base=time_base, slice_ms=slice_ms, double_ms=double_ms)
            _, on_fit = move_times(attempt, time_base)
            moves = attempt["moves"]
            result = attempt["result"]
            attempt_rows.append(
                {
                    "sessionId": session,
                    "day": day,
                    "attemptIndex": index,
                    "status": result["status"],
                    "replayOk": result["replayOk"],
                    "tps": result["tps"],
                    "timeMs": result["timeMs"],
                    "moves": len(moves),
                    "solveMoves": sum(m["phase"] == "solve" for m in moves),
                    "symbols": len(symbols),
                    "solveSymbols": sum(s.phase == "solve" for s in symbols),
                    "movesOffFit": int(len(on_fit) - on_fit.sum()) if time_base == "fit" else 0,
                    "clips": len(attempt["video"]),
                    "cameras": ",".join(sorted({v["camera"] for v in attempt["video"]})),
                    "gyroRateHz": _gyro_rate(attempt),
                }
            )
            for entry in attempt["video"]:
                row, problem = _clip_row(
                    dataset, session, day, index, attempt, entry, time_base, slice_ms, double_ms, video
                )
                clip_rows.append(row)
                if problem:
                    problems.append(problem)

    days = {row["sessionId"]: row["day"] for row in session_rows}
    split = assign_splits(days, seed=seed, held_out_day=held_out_day, val_fraction=val_fraction)
    for rows in (session_rows, attempt_rows, clip_rows):
        for row in rows:
            row["split"] = split[row["sessionId"]]
    return Tables(
        clips=pl.DataFrame(clip_rows, schema=CLIP_SCHEMA),
        attempts=pl.DataFrame(attempt_rows, schema=ATTEMPT_SCHEMA),
        sessions=pl.DataFrame(session_rows, schema=SESSION_SCHEMA),
        problems=problems,
        settings={
            "timeBase": time_base,
            "sliceMs": slice_ms,
            "doubleMs": double_ms,
            "seed": seed,
            "heldOutDay": held_out_day or "latest",
            "valFraction": val_fraction,
            "video": video,
        },
    )


def _clip_row(
    dataset: Dataset,
    session: str,
    day: str | None,
    index: int,
    attempt: dict[str, Any],
    entry: dict[str, Any],
    time_base: str,
    slice_ms: float,
    double_ms: float,
    video: str,
) -> tuple[dict[str, Any], str | None]:
    """The clip's manifest row, and what could not be read of it."""
    clip = ClipRef(session, index, entry["camera"], entry["segment"])
    crop = entry["crop"] or {}
    row: dict[str, Any] = {
        "sessionId": session,
        "day": day,
        "attemptIndex": index,
        "camera": entry["camera"],
        "segment": entry["segment"],
        "video": dataset.video_rel(clip),
        "fpsNominal": float(entry["fpsNominal"]),
        "fps": None,
        "frames": entry["frames"],
        "framesFileCount": None,
        "videoFrames": None,
        "seconds": None,
        "width": entry["width"],
        "height": entry["height"],
        "cropX": crop.get("x"),
        "cropY": crop.get("y"),
        "cropW": crop.get("w"),
        "cropH": crop.get("h"),
        "lagMs": entry["syncResidualMs"],
        "unsynced": entry["syncResidualMs"] is None,
        "movesInWindow": None,
        "movesCovered": None,
        "tps": attempt["result"]["tps"],
        "status": attempt["result"]["status"],
        "replayOk": attempt["result"]["replayOk"],
        "gyroRateHz": _gyro_rate(attempt),
        "truncatedStart": bool(entry.get("truncatedStart", False)),
    }
    covered, problem = None, None
    try:
        frames = dataset.frames(clip)
    except (RecordError, ValueError, OSError) as error:
        frames, problem = None, str(error)
    if frames is not None:
        aligned = align_clip(
            attempt,
            frames,
            None,
            camera=clip.camera,
            segment=clip.segment,
            time_base=time_base,
            slice_ms=slice_ms,
            double_ms=double_ms,
        )
        t = aligned.frame_ms
        seconds = float(t[-1] - t[0]) / 1000.0
        covered = aligned.covered()
        row.update(
            framesFileCount=len(t),
            seconds=seconds,
            fps=(len(t) - 1) / seconds if seconds > 0 else None,
            movesInWindow=covered[1],
            movesCovered=covered[0],
        )
    mp4_size = dataset.video_size(clip)
    video_frames, video_error = None, False
    if video != "none" and mp4_size is not None:
        try:
            video_frames = count_frames(dataset.video_path(clip), decode=video == "full").frames
        except (av.FFmpegError, OSError, ValueError, IndexError) as error:  # corrupt, truncated, no video
            video_error, problem = True, f"{dataset.video_rel(clip)}: {error}"
    row["videoFrames"] = video_frames
    reasons = clip_reasons(
        attempt,
        entry,
        frames_count=row["framesFileCount"],
        mp4_size=mp4_size,
        video_frames=video_frames,
        video_error=video_error,
        covered=covered,
    )
    row["usable"] = not reasons
    row["reasons"] = ";".join(reasons)
    return row, problem


def write_tables(tables: Tables, out_dir: str | Path) -> list[Path]:
    """`manifest.parquet` and `manifest.csv` (the clips), `attempts.parquet` and `attempts.csv`, and
    `manifest.json` (the settings and the problems) in `out_dir`."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, frame in (("manifest", tables.clips), ("attempts", tables.attempts)):
        frame.write_parquet(out / f"{name}.parquet")
        frame.write_csv(out / f"{name}.csv")
        written += [out / f"{name}.parquet", out / f"{name}.csv"]
    meta = {"settings": tables.settings, "problems": tables.problems}
    (out / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    written.append(out / "manifest.json")
    return written


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    cells = [[str(h) for h in headers]] + [[_fmt(v) for v in row] for row in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    lines = []
    for n, row in enumerate(cells):
        lines.append(
            "  "
            + "  ".join(
                c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(row, widths, strict=True))
            )
        )
        if n == 0:
            lines.append("  " + "  ".join("-" * w for w in widths))
    return lines


def _fmt(value: Any) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return "–" if math.isnan(value) else f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _lag_range(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return "–"
    return f"{low:g}" if low == high else f"{low:g}–{high:g}"


def _share(part: int, whole: int) -> str:
    return f"{part:,}/{whole:,} ({100.0 * part / whole:.0f}%)" if whole else "0/0"


def report_text(tables: Tables, root: str = "") -> str:
    """The counts: attempts, clips, hours, moves; by camera, by day and by split; the TPS histogram."""
    clips, attempts, sessions = tables.clips, tables.attempts, tables.sessions
    lines = [f"cubetrace-ml report: {root}" if root else "cubetrace-ml report"]
    s = tables.settings
    lines.append(
        f"settings: time base {s.get('timeBase')}, slice < {s.get('sliceMs'):g} ms, double < "
        f"{s.get('doubleMs'):g} ms, split seed {s.get('seed')}, held-out day {s.get('heldOutDay')}, "
        f"video check {s.get('video')}"
    )
    no_record = int((~sessions["sessionRecord"]).sum()) if len(sessions) else 0
    with_video = int((attempts["clips"] > 0).sum()) if len(attempts) else 0
    solved = int((attempts["status"] == "solved").sum()) if len(attempts) else 0
    video_h = float(clips["seconds"].fill_null(0).sum()) / 3600 if len(clips) else 0.0
    solve_h = float(attempts["timeMs"].fill_null(0).sum()) / 3_600_000 if len(attempts) else 0.0
    lines += [
        "",
        f"sessions  {len(sessions):,} ({no_record:,} without session.json)",
        f"attempts  {len(attempts):,} ({solved:,} solved, {with_video:,} with video)",
        f"clips     {len(clips):,}: {video_h:,.2f} h of video; {solve_h:,.2f} h of solving (sum of timeMs)",
    ]
    if len(attempts):
        lines.append(
            f"moves     {int(attempts['moves'].sum()):,} quarter turns reported "
            f"({int(attempts['solveMoves'].sum()):,} in solves); {int(attempts['symbols'].sum()):,} symbols "
            f"after the normalization ({int(attempts['solveSymbols'].sum()):,} in solves)"
        )
        off = int(attempts["movesOffFit"].sum())
        if off:
            lines.append(f"          {off:,} moves placed by their arrival (off the attempt's clock fit)")
    if len(clips):
        usable = int(clips["usable"].sum())
        unsynced = int(clips["unsynced"].sum())
        lines.append(
            f"usable    {_share(usable, len(clips))} clips; unsynced (no lag) {_share(unsynced, len(clips))}"
        )
        measured = clips.filter(pl.col("videoFrames").is_not_null())
        if len(measured):
            match = int((measured["videoFrames"] == measured["framesFileCount"]).sum())
            lines.append(
                f"video     frame counts measured on {len(measured):,} clips: "
                f"{match:,} match the frames files"
            )
        reasons = Counter(r for text in clips["reasons"] if text for r in text.split(";"))
        for reason in REASONS:
            if reasons[reason]:
                lines.append(f"          {reason}: {reasons[reason]:,} clips")

        lines += ["", "by camera"]
        by_camera = (
            clips.group_by("camera")
            .agg(
                pl.len().alias("clips"),
                (pl.col("seconds").sum() / 3600).alias("hours"),
                pl.col("fpsNominal").median().alias("nominal"),
                pl.col("fps").median().alias("measured"),
                pl.col("unsynced").sum().alias("unsynced"),
                pl.col("lagMs").min().alias("lowest"),
                pl.col("lagMs").max().alias("highest"),
                pl.col("usable").sum().alias("usable"),
            )
            .sort("camera")
        )
        lines += _table(
            ["camera", "clips", "hours", "fps nominal", "fps measured", "unsynced", "lag ms", "usable"],
            [
                [
                    r["camera"],
                    r["clips"],
                    r["hours"],
                    r["nominal"],
                    r["measured"],
                    r["unsynced"],
                    _lag_range(r["lowest"], r["highest"]),
                    r["usable"],
                ]
                for r in by_camera.iter_rows(named=True)
            ],
        )

        lines += ["", "by day"]
        per_day = attempts.group_by("day").agg(
            pl.len().alias("attempts"), pl.col("sessionId").n_unique().alias("sessions")
        )
        clip_day = clips.group_by("day").agg(
            pl.len().alias("clips"), (pl.col("seconds").sum() / 3600).alias("hours")
        )
        by_day = per_day.join(clip_day, on="day", how="left", nulls_equal=True).sort("day")
        lines += _table(
            ["day", "sessions", "attempts", "clips", "hours"],
            [
                [r["day"], r["sessions"], r["attempts"], r["clips"], r["hours"]]
                for r in by_day.iter_rows(named=True)
            ],
        )

        lines += ["", "by split"]
        per_split = attempts.group_by("split").agg(
            pl.len().alias("attempts"),
            pl.col("sessionId").n_unique().alias("sessions"),
            pl.col("day").unique().sort().str.join(",").alias("days"),
        )
        clip_split = clips.group_by("split").agg(
            pl.len().alias("clips"),
            (pl.col("seconds").sum() / 3600).alias("hours"),
            pl.col("usable").sum().alias("usable"),
        )
        by_split = per_split.join(clip_split, on="split", how="left").sort(
            pl.col("split").replace_strict({"train": 0, "val": 1, "test": 2}, default=3)
        )
        lines += _table(
            ["split", "sessions", "attempts", "clips", "usable", "hours", "days"],
            [
                [r["split"], r["sessions"], r["attempts"], r["clips"], r["usable"], r["hours"], r["days"]]
                for r in by_split.iter_rows(named=True)
            ],
        )

    tps = attempts.filter(pl.col("tps").is_not_null())["tps"].to_numpy() if len(attempts) else np.array([])
    if len(tps):
        lines += [
            "",
            f"TPS of the solved attempts (n={len(tps):,}, median {np.median(tps):.2f}; bins of 0.5)",
        ]
        low = math.floor(tps.min() * 2) / 2
        edges = np.arange(low, math.floor(tps.max() * 2) / 2 + 0.25, 0.5)
        counts = [int(((tps >= a) & (tps < a + 0.5)).sum()) for a in edges]
        peak = max(counts) or 1
        for a, n in zip(edges, counts, strict=True):
            lines.append(
                f"  {a:4.1f}–{a + 0.5:<4.1f} {'#' * max(1 if n else 0, round(30 * n / peak)):<30} {n:,}"
            )
    if tables.problems:
        lines += ["", f"problems ({len(tables.problems):,} records could not be read)"]
        lines += [f"  {p}" for p in tables.problems[:20]]
        if len(tables.problems) > 20:
            lines.append(f"  … and {len(tables.problems) - 20:,} more")
    return "\n".join(lines)
