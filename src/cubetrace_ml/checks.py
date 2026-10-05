"""The checks of a dataset: every record against its schema and its folder (`validate`), and each clip's
frames file against its video and its window (`check-alignment`)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import records
from .align import align_clip, frame_times
from .dataset import ClipRef, Dataset
from .video import count_frames

T0_TOLERANCE_MS = 0.05


@dataclass(frozen=True)
class Finding:
    level: str  # "error" or "warning"
    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.level}: {self.path}: {self.message}"


def _load(dataset: Dataset, rel: str, kind: str, findings: list[Finding]) -> Any:
    try:
        doc = json.loads(dataset.store.read_bytes(rel))
    except (OSError, ValueError) as error:
        findings.append(Finding("error", rel, f"unreadable: {error}"))
        return None
    found = records.errors(kind, doc)
    findings += [Finding("error", rel, message) for message in found]
    return None if found else doc


def validate_all(dataset: Dataset) -> tuple[int, list[Finding]]:
    """Every record under the root against its schema, and each attempt's folder against its record: the
    folder's session and index, each clip's MP4 (its size) and frames file (its camera, segment, count and
    first frame), and gyro.json (its samples). Returns the records read and the findings."""
    findings: list[Finding] = []
    count = 0
    for session in dataset.sessions():
        base = f"sessions/{session}"
        if dataset.has_session_record(session):
            count += 1
            _load(dataset, f"{base}/session.json", "session", findings)
        else:
            findings.append(Finding("warning", base, "no session.json"))
        for folder in dataset.store.listdir(f"{base}/attempts")[0]:
            rel = f"{base}/attempts/{folder}"
            names = set(dataset.store.listdir(rel)[1])
            if "attempt.json" not in names:
                findings.append(Finding("warning", rel, "no attempt.json"))
                continue
            count += 1
            attempt = _load(dataset, f"{rel}/attempt.json", "attempt", findings)
            if attempt is None:
                continue
            if attempt["session"] != session:
                findings.append(
                    Finding("error", f"{rel}/attempt.json", f"session is not the folder's {session}")
                )
            if not folder.isdigit() or attempt["index"] != int(folder):
                findings.append(
                    Finding("error", f"{rel}/attempt.json", f"index {attempt['index']} in folder {folder}")
                )
            listed = {"attempt.json"}
            for entry in attempt["video"]:
                listed |= {entry["file"], entry["framesFile"]}
                count += _check_clip_files(dataset, rel, entry, names, findings)
            gyro = attempt.get("gyro")
            if gyro:
                listed.add(gyro["file"])
                count += _check_gyro(dataset, rel, gyro, names, findings)
            for name in sorted(names - listed):
                if not name.endswith(".tmp"):
                    findings.append(Finding("warning", f"{rel}/{name}", "not named by attempt.json"))
    return count, findings


def _check_clip_files(
    dataset: Dataset, rel: str, entry: dict[str, Any], names: set[str], findings: list[Finding]
) -> int:
    path = f"{rel}/{entry['file']}"
    if entry["file"] not in names:
        findings.append(Finding("error", path, "missing"))
    elif (size := dataset.store.size(path)) != entry["bytes"]:
        findings.append(Finding("error", path, f"{size} bytes, the record says {entry['bytes']}"))
    frames_rel = f"{rel}/{entry['framesFile']}"
    if entry["framesFile"] not in names:
        findings.append(Finding("error", frames_rel, "missing"))
        return 0
    frames = _load(dataset, frames_rel, "frames", findings)
    if frames is None:
        return 1
    if (frames["camera"], frames["segment"]) != (entry["camera"], entry["segment"]):
        findings.append(Finding("error", frames_rel, f"is {frames['camera']}.{frames['segment']}"))
    if len(frames["dtMs"]) != entry["frames"]:
        findings.append(
            Finding("error", frames_rel, f"{len(frames['dtMs'])} frames, the record says {entry['frames']}")
        )
    if abs(frames["t0HostMs"] - entry["firstFrameHostMs"]) > T0_TOLERANCE_MS:
        findings.append(Finding("error", frames_rel, "t0HostMs is not the record's firstFrameHostMs"))
    return 1


def _check_gyro(
    dataset: Dataset, rel: str, summary: dict[str, Any], names: set[str], findings: list[Finding]
) -> int:
    path = f"{rel}/{summary['file']}"
    if summary["file"] not in names:
        findings.append(Finding("error", path, "missing"))
        return 0
    gyro = _load(dataset, path, "gyro", findings)
    if gyro is None:
        return 1
    n = len(gyro["dtMs"])
    if (
        n != summary["samples"]
        or len(gyro["q"]) != 4 * n
        or (gyro["v"] is not None and len(gyro["v"]) != 3 * n)
    ):
        findings.append(
            Finding(
                "error",
                path,
                f"{n} sample times, {len(gyro['q'])} quaternion values; the record says "
                f"{summary['samples']} samples",
            )
        )
    return 1


@dataclass
class AlignmentCheck:
    """One clip's frames file against its record, its video and its window."""

    clip: ClipRef
    frames_file: int
    record_frames: int
    container_frames: int | None
    decoded_frames: int | None
    frames_span_ms: float
    video_span_ms: float | None
    max_pts_diff_ms: float | None
    lag_ms: float | None
    covered: int
    moves: int
    lead_ms: float | None  # the window's start (plus the lag) after the first frame; None without a start
    tail_ms: float | None  # the last frame after the window's end (plus the lag); None without an end

    @property
    def counts_match(self) -> bool:
        """The frames file, the record and the video (its decode, its header, or both) say the same count."""
        video = [n for n in (self.container_frames, self.decoded_frames) if n is not None]
        return bool(video) and len({self.frames_file, self.record_frames, *video}) == 1

    @property
    def ok(self) -> bool:
        timing = self.max_pts_diff_ms is None or self.max_pts_diff_ms <= 1.0
        return self.counts_match and timing and self.covered == self.moves


def check_alignment(
    dataset: Dataset, clip: ClipRef, *, fast: bool = False, time_base: str = "fit"
) -> AlignmentCheck:
    """The frames file's count against the record's and the video's (decoded fully, or the container's header
    with `fast`), their spans and, after a decode, the largest difference between a frame's presentation
    time and its frames-file time (both from the first frame); the segment's onsets within the clip's
    frames; the clip's margins before and after the window, on the frame timeline."""
    attempt = dataset.attempt(clip.session, clip.attempt)
    entry = dataset.clip_entry(clip)
    frames = dataset.frames(clip)
    aligned = align_clip(attempt, frames, None, camera=clip.camera, segment=clip.segment, time_base=time_base)
    t = frame_times(frames)
    video = count_frames(dataset.video_path(clip), decode=not fast)
    max_diff = None
    if video.pts_ms is not None and len(video.pts_ms):
        n = min(len(video.pts_ms), len(t))
        max_diff = float(np.max(np.abs(video.pts_ms[:n] - (t[:n] - t[0]))))
    covered, moves = aligned.covered()
    low, high = aligned.window
    return AlignmentCheck(
        clip=clip,
        frames_file=len(t),
        record_frames=entry["frames"],
        container_frames=video.container_frames,
        decoded_frames=video.decoded_frames,
        frames_span_ms=float(t[-1] - t[0]),
        video_span_ms=video.span_ms if video.span_ms is not None else video.container_ms,
        max_pts_diff_ms=max_diff,
        lag_ms=aligned.lag_ms,
        covered=covered,
        moves=moves,
        lead_ms=float(low + aligned.lag - t[0]) if math.isfinite(low) else None,
        tail_ms=float(t[-1] - (high + aligned.lag)) if math.isfinite(high) else None,
    )
