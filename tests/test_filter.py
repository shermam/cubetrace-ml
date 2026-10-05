from cubetrace_ml.filter import REASONS, clip_reasons
from factory import T0, session_id, short_attempt


def attempt_and_entry(**kwargs):
    attempt, _, _ = short_attempt(session_id(1), 1, T0, {"laptop": 40.0, "phone-rear": None}, **kwargs)
    return attempt, attempt["video"][0]


def test_a_clean_clip_is_usable_and_unsynced_is_no_reason() -> None:
    attempt, _ = attempt_and_entry()
    for entry in attempt["video"]:
        assert clip_reasons(attempt, entry, frames_count=entry["frames"], mp4_size=entry["bytes"]) == []
    assert any(entry["syncResidualMs"] is None for entry in attempt["video"])


def test_each_reason() -> None:
    attempt, entry = attempt_and_entry()
    count, size = entry["frames"], entry["bytes"]
    assert clip_reasons(attempt, entry, frames_count=count, mp4_size=None) == ["missing-video"]
    assert clip_reasons(attempt, entry, frames_count=count, mp4_size=size + 1) == ["bytes-mismatch"]
    assert clip_reasons(attempt, entry, frames_count=None, mp4_size=size) == ["missing-frames"]
    assert clip_reasons(attempt, entry, frames_count=count - 1, mp4_size=size) == ["frames-count-mismatch"]
    assert clip_reasons(attempt, entry, frames_count=count, mp4_size=size, video_frames=count + 1) == [
        "video-frames-mismatch"
    ]
    assert clip_reasons(attempt, entry, frames_count=count, mp4_size=size, video_frames=count) == []
    assert clip_reasons(attempt, entry, frames_count=count, mp4_size=size, video_error=True) == [
        "video-unreadable"
    ]
    assert clip_reasons(attempt, entry, frames_count=count, mp4_size=size, covered=(5, 6)) == [
        "moves-outside-clip"
    ]
    assert clip_reasons(attempt, entry, frames_count=count, mp4_size=size, covered=(6, 6)) == []
    entry = dict(entry, truncatedStart=True)
    assert clip_reasons(attempt, entry, frames_count=count, mp4_size=size) == ["truncated-start"]


def test_the_attempts_status_and_replay() -> None:
    attempt, entry = attempt_and_entry(status="dnf")
    reasons = clip_reasons(attempt, entry, frames_count=entry["frames"], mp4_size=entry["bytes"])
    assert reasons == ["dnf", "replay-failed"]
    attempt, entry = attempt_and_entry()
    attempt["result"]["replayOk"] = False
    assert clip_reasons(attempt, entry, frames_count=entry["frames"], mp4_size=entry["bytes"]) == [
        "replay-failed"
    ]


def test_reasons_come_in_a_fixed_order() -> None:
    attempt, entry = attempt_and_entry(status="dnf")
    entry = dict(entry, truncatedStart=True)
    reasons = clip_reasons(attempt, entry, frames_count=None, mp4_size=None, covered=(0, 3))
    assert reasons == [r for r in REASONS if r in reasons]
    assert reasons == [
        "dnf",
        "replay-failed",
        "truncated-start",
        "missing-video",
        "missing-frames",
        "moves-outside-clip",
    ]
