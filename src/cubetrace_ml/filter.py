"""The consistency filter: why a clip is not fit for training, as short reason codes (none: usable).

An unsynced clip (no sync check: no lag) is usable; it is flagged in the manifest instead.
"""

from __future__ import annotations

from typing import Any

REASONS = {
    "dnf": "the attempt is a DNF (result.status)",
    "replay-failed": "the solve's moves do not replay to solved (result.replayOk): moves went unseen",
    "truncated-start": "the clip begins later than asked (truncatedStart)",
    "missing-video": "the MP4 is not in the dataset",
    "bytes-mismatch": "the MP4's size is not the record's bytes",
    "missing-frames": "the frames file is missing or invalid",
    "frames-count-mismatch": "the frames file's count is not the record's frames",
    "video-frames-mismatch": "the video's frame count (measured) is not the frames file's",
    "video-unreadable": "the MP4 could not be decoded",
    "moves-outside-clip": "some of the segment's move onsets fall outside the clip's frames",
}


def clip_reasons(
    attempt: dict[str, Any],
    entry: dict[str, Any],
    *,
    frames_count: int | None,
    mp4_size: int | None,
    video_frames: int | None = None,
    video_error: bool = False,
    covered: tuple[int, int] | None = None,
) -> list[str]:
    """The reasons, in REASONS order, for one clip (`entry`, from the attempt's `video[]`).

    `frames_count` is the frames file's `len(dtMs)` (None: missing or invalid), `mp4_size` the MP4's size
    as listed (None: missing), `video_frames` the video's frame count when it was measured, `covered` the
    segment's onsets within the clip's frames and their number.
    """
    found = set()
    result = attempt["result"]
    if result["status"] != "solved":
        found.add("dnf")
    if not result["replayOk"]:
        found.add("replay-failed")
    if entry.get("truncatedStart", False):
        found.add("truncated-start")
    if mp4_size is None:
        found.add("missing-video")
    elif mp4_size != entry["bytes"]:
        found.add("bytes-mismatch")
    if frames_count is None:
        found.add("missing-frames")
    elif frames_count != entry["frames"]:
        found.add("frames-count-mismatch")
    if video_error:
        found.add("video-unreadable")
    elif video_frames is not None and frames_count is not None and video_frames != frames_count:
        found.add("video-frames-mismatch")
    if covered is not None and covered[0] < covered[1]:
        found.add("moves-outside-clip")
    return [reason for reason in REASONS if reason in found]
