"""The features: the clips a run selects, the file of a clip (its arrays and its meta), the resume rule, the
crop cache and the bench, on the factory's dataset with the stub encoder (no PyTorch, no download)."""

import shutil
import time
from pathlib import Path

import numpy as np
import pytest

from cubetrace_ml.align import align_clip
from cubetrace_ml.dataset import ClipRef, Dataset
from cubetrace_ml.encoders import StubEncoder, load_encoder
from cubetrace_ml.features import (
    bench,
    bench_table,
    crop_path,
    extract,
    feature_path,
    features_root,
    is_done,
    prefetch,
    read_features,
    read_manifest,
    read_meta,
    select_clips,
    write_features,
)
from cubetrace_ml.manifest import build_tables, write_tables
from factory import gray, session_id, short_attempt, write_attempt, write_frames

QUIET = {"log": lambda _: None}


@pytest.fixture(scope="module")
def tables(dataset_root):
    root, _ = dataset_root
    return build_tables(Dataset(root), video="fast")


def test_the_clips_a_run_selects(tables, dataset_root, tmp_path: Path) -> None:
    _, ids = dataset_root
    usable = select_clips(tables.clips)
    assert len(usable) == 10 and len(select_clips(tables.clips, usable_only=False)) == 12
    assert usable == sorted(usable, key=lambda c: (c.session, c.attempt, c.segment != "scramble", c.camera))
    assert {c.session for c in select_clips(tables.clips, split="test")} == {ids["D"]}
    rear = select_clips(tables.clips, camera="phone-rear")
    assert [(c.session, c.segment) for c in rear] == [(ids["B"], "scramble"), (ids["B"], "solve")]
    assert all(c.segment == "solve" for c in select_clips(tables.clips, segment="solve"))
    assert len(select_clips(tables.clips, session=ids["B"][:8])) == 4
    assert select_clips(tables.clips, limit=3) == usable[:3]
    with pytest.raises(ValueError, match="matches 0 sessions"):
        select_clips(tables.clips, session="ffff")
    write_tables(tables, tmp_path)  # a manifest read back selects the same clips
    assert select_clips(read_manifest(tmp_path / "manifest.parquet")) == usable
    assert select_clips(read_manifest(tmp_path / "manifest.csv"), split="val") == select_clips(
        tables.clips, split="val"
    )


def test_the_features_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("CUBETRACE_FEATURES", raising=False)
    with pytest.raises(ValueError, match="CUBETRACE_FEATURES"):
        features_root(None)
    monkeypatch.setenv("CUBETRACE_FEATURES", str(tmp_path))
    assert features_root(None) == tmp_path and features_root("elsewhere") == Path("elsewhere")


def test_a_clip_s_features_file(dataset_root, tmp_path: Path) -> None:
    root, ids = dataset_root
    dataset = Dataset(root)
    clip = ClipRef(ids["B"], 1, "laptop", "solve")
    stats = extract(dataset, [clip], load_encoder("stub"), tmp_path, **QUIET)
    assert (stats.written, stats.skipped, stats.failed, stats.frames) == (1, 0, 0, 24)
    path = feature_path(tmp_path, "stub", clip)
    assert path == tmp_path / "stub" / ids["B"] / "0001" / "laptop.solve.npz"
    data = read_features(path)
    assert set(data) == {"x", "tMs", "shownMs", "inWindow", "meta"}
    assert data["x"].shape == (24, 64) and data["x"].dtype == np.float16
    assert data["tMs"].dtype == np.float64 and data["inWindow"].dtype == np.bool_
    aligned = align_clip(dataset.attempt(clip.session, 1), dataset.frames(clip), None, camera="laptop")
    np.testing.assert_array_equal(data["tMs"], aligned.track["tMs"])
    np.testing.assert_allclose(data["shownMs"], data["tMs"] - 50.0)  # the laptop's lag in session B
    np.testing.assert_array_equal(data["inWindow"], aligned.track["inWindow"])
    # Every frame, in order: frame k is flat at gray(k), and the stub maps a flat frame of level g to
    # (g / 255 − 0.5) times its projection's column sums.
    sums = StubEncoder().projection.sum(axis=0)
    levels = np.array([gray(k) / 255 - 0.5 for k in range(24)])
    np.testing.assert_allclose(
        data["x"].astype(np.float64) / sums, np.repeat(levels[:, None], 64, 1), atol=0.02
    )

    meta = data["meta"]
    assert meta["format"] == 1
    assert {k: meta["encoder"][k] for k in ("name", "inputSize", "dim", "loaded")} == {
        "name": "stub",
        "inputSize": 32,
        "dim": 64,
        "loaded": "seed 0",
    }
    # No record crop: auto found the motion's square (the whole 64 × 64 frame changes evenly).
    assert meta["cropMode"] == "auto"
    assert {k: meta["crop"][k] for k in ("x", "y", "w", "h", "source", "letterboxed")} == {
        "x": 0,
        "y": 0,
        "w": 64,
        "h": 64,
        "source": "motion",
        "letterboxed": False,
    }
    assert meta["crop"]["motion"]["threshold"] == 0.15 and meta["crop"]["cached"] is False
    assert {k: meta["clip"][k] for k in ("sessionId", "attemptIndex", "camera", "segment", "frames")} == {
        "sessionId": ids["B"],
        "attemptIndex": 1,
        "camera": "laptop",
        "segment": "solve",
        "frames": 24,
    }
    assert meta["timeBase"] == "fit" and meta["lagMs"] == 50.0 and meta["unsynced"] is False
    assert meta["app"] == {"version": "0.4.0", "commit": "abc1234"}
    assert meta["cubetraceMl"]["version"] == "0.1.0"
    assert (
        meta["host"]["device"] == "cpu" and meta["host"]["precision"] == "fp32" and meta["host"]["deviceName"]
    )
    timing = meta["timing"]
    assert 0 < timing["decodeSeconds"] <= timing["wallSeconds"] and meta["writtenAt"].endswith("+00:00")
    assert read_meta(path) == meta and read_meta(tmp_path / "missing.npz") is None


def test_a_record_crop_is_letterboxed_and_the_modes_differ(tmp_path: Path) -> None:
    root = tmp_path / "data"
    sid = session_id(9)
    attempt, frames, gyro = short_attempt(sid, 1, 1_790_000_060_000.0, {"laptop": 40.0, "phone-rear": None})
    for entry in attempt["video"]:
        if entry["camera"] == "laptop":
            entry["crop"] = {"x": 8, "y": 16, "w": 48, "h": 32}
    write_attempt(root, attempt, frames, gyro=gyro)
    dataset = Dataset(root)
    out = tmp_path / "features"
    laptop, phone = ClipRef(sid, 1, "laptop", "solve"), ClipRef(sid, 1, "phone-rear", "solve")
    extract(dataset, [laptop, phone], load_encoder("stub"), out, **QUIET)
    crop = read_meta(feature_path(out, "stub", laptop))["crop"]
    assert (crop["x"], crop["y"], crop["w"], crop["h"], crop["source"], crop["letterboxed"]) == (
        8,
        16,
        48,
        32,
        "record",
        True,
    )
    assert read_meta(feature_path(out, "stub", phone))["crop"]["source"] == "motion"
    stats = extract(dataset, [laptop, phone], load_encoder("stub"), out, crop_mode="record", **QUIET)
    assert stats.written == 2  # another crop mode is another file's worth
    assert read_meta(feature_path(out, "stub", phone))["crop"]["source"] == "none"
    assert read_meta(feature_path(out, "stub", laptop))["cropMode"] == "record"


def test_resume_skips_what_is_done_and_force_rewrites_it(dataset_root, tables, tmp_path: Path) -> None:
    root, _ = dataset_root
    dataset = Dataset(root)
    refs = select_clips(tables.clips)
    encoder = load_encoder("stub")
    first = extract(dataset, refs, encoder, tmp_path, **QUIET)
    assert (first.written, first.skipped) == (10, 0)
    paths = [feature_path(tmp_path, "stub", ref) for ref in refs]
    files = {p: p.stat().st_ino for p in paths}  # a rewrite renames a new file over the old one
    lines: list[str] = []
    second = extract(dataset, refs, encoder, tmp_path, log=lines.append)
    assert (second.written, second.skipped) == (0, 10)
    assert lines == ["10 of 10 clips already have their stub features (--force rewrites)"]
    assert {p: p.stat().st_ino for p in paths} == files
    assert all(is_done(p, encoder="stub", crop_mode="auto", frames=24) for p in paths)
    assert not is_done(paths[0], encoder="stub", crop_mode="none", frames=24)
    assert not is_done(paths[0], encoder="stub", crop_mode="auto", frames=25)
    assert not is_done(paths[0], encoder="resnet18", crop_mode="auto", frames=24)

    # A temporary file never counts, a truncated file is rewritten, and nothing is left half-written.
    paths[0].unlink()
    (paths[0].parent / f"{paths[0].name}.0badf00d.tmp").write_bytes(b"PK\x03\x04 cut short")
    paths[1].write_bytes(paths[1].read_bytes()[:200])
    assert read_meta(paths[1]) is None
    third = extract(dataset, refs, encoder, tmp_path, **QUIET)
    assert (third.written, third.skipped) == (2, 8)
    assert all(is_done(p, encoder="stub", crop_mode="auto", frames=24) for p in paths)

    files = {p: p.stat().st_ino for p in paths}
    forced = extract(dataset, refs, encoder, tmp_path, force=True, **QUIET)
    assert (forced.written, forced.skipped) == (10, 0)
    assert all(p.stat().st_ino != files[p] for p in paths)


def test_an_interrupted_write_leaves_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(handle, **arrays) -> None:
        handle.write(b"PK\x03\x04 a few bytes")
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez", broken)
    with pytest.raises(OSError, match="disk full"):
        write_features(tmp_path / "x.npz", {"x": np.zeros((2, 3), np.float16)}, {"format": 1})
    assert list(tmp_path.iterdir()) == []


def test_the_motion_crops_are_cached_for_the_next_run(dataset_root, tmp_path: Path) -> None:
    root, ids = dataset_root
    dataset = Dataset(root)
    clip = ClipRef(ids["C"], 1, "phone-front", "solve")
    first = extract(dataset, [clip], load_encoder("stub"), tmp_path, **QUIET)
    assert first.crop_clips == 1 and crop_path(tmp_path, clip).is_file()
    feature_path(tmp_path, "stub", clip).unlink()
    lines: list[str] = []
    again = extract(dataset, [clip], load_encoder("stub"), tmp_path, log=lines.append)
    assert again.written == 1 and again.crop_clips == 0 and "(cached)" in lines[-1]
    meta = read_meta(feature_path(tmp_path, "stub", clip))
    assert meta["crop"]["cached"] is True and meta["crop"]["source"] == "motion"
    forced = extract(dataset, [clip], load_encoder("stub"), tmp_path, force=True, **QUIET)
    assert forced.crop_clips == 1  # --force finds it again


def test_a_clip_that_fails_does_not_stop_the_run(dataset_root, tmp_path: Path) -> None:
    source, ids = dataset_root
    root = tmp_path / "copy"
    shutil.copytree(source, root)
    video = root / "sessions" / ids["D"] / "attempts" / "0001" / "laptop.solve.mp4"
    write_frames(video, (np.full((64, 64, 3), 99, np.uint8) for _ in range(20)))  # 20 frames, not 24
    dataset = Dataset(root)
    refs = [ClipRef(ids["D"], 1, "laptop", "scramble"), ClipRef(ids["D"], 1, "laptop", "solve")]
    lines: list[str] = []
    stats = extract(dataset, refs, load_encoder("stub"), tmp_path / "out", log=lines.append)
    assert (stats.written, stats.failed) == (1, 1)
    assert "decoded 20 frames, the frames file has 24" in stats.errors[0]
    assert not feature_path(tmp_path / "out", "stub", refs[1]).exists()


def test_prefetch_keeps_the_order_and_the_errors() -> None:
    def work(n: int) -> int:
        if n == 3:
            raise ValueError("three")
        time.sleep(0.002 * (5 - n))
        return n * n

    out = list(prefetch(range(5), work, workers=3))
    assert [n for n, _ in out] == [0, 1, 2, 3, 4]
    assert [r for n, r in out if n != 3] == [0, 1, 4, 16] and isinstance(out[3][1], ValueError)
    with pytest.raises(ValueError, match="at least 1"):
        list(prefetch(range(2), work, workers=0))


def test_the_bench_measures_each_stage(dataset_root, tables) -> None:
    root, _ = dataset_root
    refs = select_clips(tables.clips, limit=2)
    rows = bench(Dataset(root), refs, ["none", "stub"], size=48, **QUIET)
    assert [(r.encoder, r.input_size, r.stats.written, r.stats.frames) for r in rows] == [
        ("none", 48, 2, 48),
        ("stub", 32, 2, 48),
    ]
    assert rows[0].stats.crop_clips == 2 and rows[1].stats.crop_clips == 0  # found once, then reused
    assert rows[0].host is None and rows[1].host["weights"] == "seed 0"
    table = bench_table(rows)
    assert table.splitlines()[0].startswith("| encoder | input |")
    assert "| none | 48 | 2 | 48 |" in table and "| stub | 32 | 2 | 48 |" in table
