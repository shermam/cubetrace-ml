import math
import shutil
from pathlib import Path

import polars as pl
import pytest

from cubetrace_ml.dataset import Dataset
from cubetrace_ml.manifest import ATTEMPT_SCHEMA, CLIP_SCHEMA, build_tables, report_text, write_tables
from factory import DAY_MS, T0, camera, session_id, session_record, short_attempt, write_attempt, write_json


@pytest.fixture(scope="module")
def tables(dataset_root):
    root, _ = dataset_root
    return build_tables(Dataset(root), video="fast")


def row(frame: pl.DataFrame, **match) -> dict:
    rows = frame.filter(*[pl.col(k) == v for k, v in match.items()]).to_dicts()
    assert len(rows) == 1, rows
    return rows[0]


def test_the_manifest_has_one_row_per_clip_and_its_columns(tables) -> None:
    clips = tables.clips
    assert clips.columns == list(CLIP_SCHEMA)
    assert dict(clips.schema) == CLIP_SCHEMA
    assert len(clips) == 2 + 2 + 4 + 2 + 2
    assert tables.attempts.columns == list(ATTEMPT_SCHEMA) and len(tables.attempts) == 5
    assert tables.problems == []
    for column in ("sessionId", "day", "attemptIndex", "camera", "segment", "frames", "seconds", "split"):
        assert clips[column].null_count() == 0


def test_the_rows_say_what_the_records_say(tables, dataset_root) -> None:
    _, ids = dataset_root
    clips = tables.clips
    laptop = row(clips, sessionId=ids["B"], camera="laptop", segment="solve")
    assert laptop["day"] == "2026-09-22"
    assert laptop["lagMs"] == 50.0 and laptop["unsynced"] is False
    assert laptop["frames"] == laptop["framesFileCount"] == laptop["videoFrames"] == 24
    assert laptop["seconds"] == pytest.approx(23 * 33.3 / 1000)
    assert laptop["fps"] == pytest.approx(1000 / 33.3)
    assert laptop["movesInWindow"] == 4 and laptop["movesCovered"] == 4  # R2 M' U D'
    assert laptop["tps"] is not None and laptop["status"] == "solved" and laptop["replayOk"] is True
    assert laptop["gyroRateHz"] == 20.0
    assert laptop["usable"] is True and laptop["reasons"] == ""
    assert laptop["video"] == f"sessions/{ids['B']}/attempts/0001/laptop.solve.mp4"

    phone = row(clips, sessionId=ids["B"], camera="phone-rear", segment="scramble")
    assert phone["lagMs"] is None and phone["unsynced"] is True and phone["usable"] is True
    assert phone["movesInWindow"] == 3  # R U F2

    no_gyro = row(clips, sessionId=ids["C"], segment="solve")
    assert math.isnan(no_gyro["gyroRateHz"])
    assert no_gyro["day"] == "2026-09-22"  # from its attempts: the session has no session.json

    dnf = row(clips, sessionId=ids["A"], attemptIndex=2, segment="solve")
    assert dnf["status"] == "dnf" and dnf["tps"] is None and dnf["usable"] is False
    assert dnf["reasons"] == "dnf;replay-failed;truncated-start"
    assert (
        row(clips, sessionId=ids["A"], attemptIndex=2, segment="scramble")["reasons"] == "dnf;replay-failed"
    )


def test_the_split_is_by_session_weighed_by_clips(tables, dataset_root) -> None:
    _, ids = dataset_root
    per_session = tables.clips.group_by("sessionId").agg(pl.col("split").unique())
    assert all(len(splits) == 1 for splits in per_session["split"])
    split = {r["sessionId"]: r["split"] for r in tables.sessions.iter_rows(named=True)}
    # 12 clips: day 3's 2 are the nearest to a fifth (2.4); val's 15% (1.8) is C's 2 clips.
    assert split[ids["D"]] == "test" and tables.settings["testDay"] == "2026-09-23"
    assert split[ids["C"]] == "val" and split[ids["A"]] == split[ids["B"]] == "train"


def test_a_session_without_clips_joins_no_split(dataset_root, tmp_path: Path) -> None:
    source, ids = dataset_root
    root = tmp_path / "copy"
    shutil.copytree(source, root)
    late = session_id(5)  # the latest day, but no clips: it neither is the test day nor counts for val
    write_json(
        root / "sessions" / late / "session.json",
        session_record(late, T0 + 9 * DAY_MS, [camera("laptop")], {}),
    )
    attempt, frames, gyro = short_attempt(late, 1, T0 + 9 * DAY_MS + 60_000, {})
    write_attempt(root, attempt, frames, gyro=gyro)
    tables = build_tables(Dataset(root))
    split = {r["sessionId"]: r["split"] for r in tables.sessions.iter_rows(named=True)}
    assert split[late] == "none" and split[ids["D"]] == "test"
    assert sorted(split[ids[k]] for k in "ABC") == ["train", "train", "val"]
    assert tables.attempts.filter(pl.col("sessionId") == late)["split"].to_list() == ["none"]
    assert "none" in report_text(tables).split("by split")[1]


def test_the_same_seed_gives_the_same_manifest(dataset_root, tables) -> None:
    root, _ = dataset_root
    other = build_tables(Dataset(root), video="fast")
    assert other.clips.equals(tables.clips)


def test_write_tables(tables, tmp_path: Path) -> None:
    written = write_tables(tables, tmp_path / "out")
    assert sorted(p.name for p in written) == sorted(
        ["manifest.parquet", "manifest.csv", "attempts.parquet", "attempts.csv", "manifest.json"]
    )
    back = pl.read_parquet(tmp_path / "out" / "manifest.parquet")
    assert back.equals(tables.clips)
    header = (tmp_path / "out" / "manifest.csv").read_text().splitlines()[0]
    assert header.split(",") == list(CLIP_SCHEMA)


def test_the_report_counts(tables) -> None:
    text = report_text(tables, "<root>")
    assert "sessions  4 (1 without session.json)" in text
    assert "attempts  5 (4 solved, 5 with video)" in text
    assert "clips     12:" in text
    # moves: 4 scramble + 6 solve turns per solved attempt, 4 + 3 for the DNF
    assert (
        "moves     47 quarter turns reported (27 in solves); 33 symbols after the normalization (18" in text
    )
    assert "usable    10/12 (83%) clips; unsynced (no lag) 4/12 (33%)" in text
    assert "frame counts measured on 12 clips: 12 match the frames files" in text
    for reason in ("dnf: 2 clips", "replay-failed: 2 clips", "truncated-start: 1 clips"):
        assert reason in text
    for heading in ("by camera", "by day", "by split", "TPS of the solved attempts (n=4"):
        assert heading in text
    assert "2026-09-21" in text and "2026-09-23" in text


def test_an_unreadable_video_is_a_reason_and_a_problem(dataset_root, tmp_path: Path) -> None:
    source, ids = dataset_root
    root = tmp_path / "copy"
    shutil.copytree(source, root)
    video = root / "sessions" / ids["D"] / "attempts" / "0001" / "laptop.solve.mp4"
    size = video.stat().st_size
    video.write_bytes(b"\0" * size)  # same size, no MP4 in it
    tables = build_tables(Dataset(root), video="fast")
    broken = row(tables.clips, sessionId=ids["D"], segment="solve")
    assert broken["reasons"] == "video-unreadable" and broken["videoFrames"] is None
    assert len(tables.problems) == 1 and tables.problems[0].startswith(f"sessions/{ids['D']}/attempts/0001/")
