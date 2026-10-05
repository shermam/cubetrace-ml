import json
import shutil
from pathlib import Path

from cubetrace_ml.checks import check_alignment, validate_all
from cubetrace_ml.dataset import ClipRef, Dataset
from factory import write_video


def test_the_factory_dataset_validates(dataset_root) -> None:
    root, ids = dataset_root
    count, findings = validate_all(Dataset(root, validate=False))
    # 4 sessions (3 with session.json), 5 attempts, 12 clips' frames files, 4 gyro files
    assert count == 3 + 5 + 12 + 4
    assert [str(f) for f in findings] == [f"warning: sessions/{ids['C']}: no session.json"]


def test_validate_finds_what_is_wrong(dataset_root, tmp_path: Path) -> None:
    source, ids = dataset_root
    root = tmp_path / "copy"
    shutil.copytree(source, root)
    folder = root / "sessions" / ids["B"] / "attempts" / "0001"
    (folder / "phone-rear.solve.mp4").unlink()
    frames = json.loads((folder / "laptop.solve.frames.json").read_text())
    frames["dtMs"] = frames["dtMs"][:-1]
    (folder / "laptop.solve.frames.json").write_text(json.dumps(frames))
    gyro = json.loads((folder / "gyro.json").read_text())
    gyro["schema"] = 3
    (folder / "gyro.json").write_text(json.dumps(gyro))
    (folder / "notes.txt").write_text("x")
    (folder / "attempt.json.1a2b.tmp").write_text("{")
    _, findings = validate_all(Dataset(root, validate=False))
    text = [str(f) for f in findings if ids["B"] in f.path]
    base = f"sessions/{ids['B']}/attempts/0001"
    assert f"error: {base}/phone-rear.solve.mp4: missing" in text
    assert f"error: {base}/laptop.solve.frames.json: 23 frames, the record says 24" in text
    assert any(t.startswith(f"error: {base}/gyro.json: /schema: ") for t in text)
    assert f"warning: {base}/notes.txt: not named by attempt.json" in text
    assert not any(".tmp" in t for t in text)


def test_check_alignment_on_a_clip(dataset_root) -> None:
    root, ids = dataset_root
    dataset = Dataset(root)
    clip = ClipRef(ids["B"], 1, "laptop", "solve")
    full = check_alignment(dataset, clip)
    assert full.ok and full.counts_match
    assert full.frames_file == full.record_frames == full.container_frames == full.decoded_frames == 24
    assert full.max_pts_diff_ms is not None and full.max_pts_diff_ms < 3.5  # the video's 1/30 s vs 33.3 ms
    assert (full.covered, full.moves) == (4, 4)
    assert full.lag_ms == 50.0
    assert full.lead_ms == 100.0 + 50.0  # the clip starts 100 ms before the window, which the lag delays
    fast = check_alignment(dataset, clip, fast=True)
    assert fast.decoded_frames is None and fast.max_pts_diff_ms is None and fast.ok


def test_check_alignment_sees_a_short_video(dataset_root, tmp_path: Path) -> None:
    source, ids = dataset_root
    root = tmp_path / "copy"
    shutil.copytree(source, root)
    write_video(root / "sessions" / ids["D"] / "attempts" / "0001" / "laptop.scramble.mp4", 20)
    check = check_alignment(Dataset(root), ClipRef(ids["D"], 1, "laptop", "scramble"))
    assert check.decoded_frames == 20 and not check.counts_match and not check.ok


def test_a_dnf_has_no_margin_after_its_open_window(dataset_root) -> None:
    root, ids = dataset_root
    check = check_alignment(Dataset(root), ClipRef(ids["A"], 2, "laptop", "solve"), fast=True)
    assert check.tail_ms is None and check.lead_ms == 100.0 + 40.0
    assert check.counts_match and check.decoded_frames is None
