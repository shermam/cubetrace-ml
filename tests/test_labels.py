"""The labels: the kept frames (the window plus the margin), the nearest-frame target, the soft target, the
collisions, the frame rate, the skips, the gyro's channels, and a split loaded with its features."""

import math
from pathlib import Path

import numpy as np
import pytest

from cubetrace_ml.dataset import ClipRef, Dataset
from cubetrace_ml.features import feature_path, write_features
from cubetrace_ml.labels import (
    GYRO_CHANNELS,
    LabelConfig,
    LabelError,
    clip_labels,
    frame_stride,
    load_clips,
    load_split,
    model_input,
    place_onsets,
    soft_targets,
)
from cubetrace_ml.manifest import build_tables
from cubetrace_ml.moves import INDEX
from factory import (
    T0,
    attempt_record,
    build_feature_dataset,
    clip_entry,
    frames_record,
    gyro_record,
    session_id,
)

SID = session_id(7)


def labelled(
    segment: str = "scramble",
    *,
    lag: float | None = 50.0,
    t0: float = T0 + 900.0,
    frames: int = 60,
    interval: float = 10.0,
    scramble=(("R", T0 + 1000.0), ("U", T0 + 1200.0)),
    solve=(("F", T0 + 3000.0),),
    config: LabelConfig | None = None,
    gyro: dict | None = None,
):
    record = frames_record("laptop", segment, t0, [0.0] + [interval] * (frames - 1))
    attempt = attempt_record(
        SID, 1, list(scramble), list(solve), video=[clip_entry("laptop", segment, record, lag=lag)]
    )
    return clip_labels(attempt, record, ClipRef(SID, 1, "laptop", segment), config, gyro=gyro)


def test_the_kept_frames_are_the_window_and_the_margin() -> None:
    # Frames every 10 ms from T0 + 900; the scramble's window is T0 + 1000 to T0 + 1200 on the cube's clock,
    # T0 + 1050 to T0 + 1250 on the frames (lag 50): frames 15 to 35.
    clip = labelled(config=LabelConfig(margin=5))
    assert clip.frames.tolist() == list(range(10, 41))
    np.testing.assert_allclose(clip.t_ms, T0 + 900.0 + 10.0 * np.arange(10, 41))
    assert labelled(config=LabelConfig(margin=15)).frames.tolist() == list(range(0, 51))
    assert labelled(config=LabelConfig(margin=100)).frames.tolist() == list(range(60))  # the clip's ends
    assert labelled().frames.tolist() == list(range(0, 51))  # 15 by default
    assert clip.segment == "scramble" and clip.stride == 1 and clip.lag_ms == 50.0
    assert clip.facelets is None and labelled("solve", t0=T0 + 2900.0).facelets is not None


def test_the_frame_nearest_an_onset_carries_its_symbol() -> None:
    clip = labelled(config=LabelConfig(margin=5))
    assert clip.symbols.tolist() == [INDEX["R"], INDEX["U"]]
    np.testing.assert_allclose(clip.onsets_ms, [T0 + 1050.0, T0 + 1250.0])  # onset + lag
    onsets = np.flatnonzero(clip.target)
    assert onsets.tolist() == [5, 25]  # frames 15 and 35 of the clip
    assert clip.target[onsets].tolist() == [INDEX["R"] + 1, INDEX["U"] + 1]
    assert clip.near_weight[onsets].tolist() == [1.0, 1.0] and clip.near_weight.sum() == 2.0
    # Unsynced: no lag, the onsets on the frames at their own times.
    unsynced = labelled(lag=None, config=LabelConfig(margin=5))
    assert unsynced.lag_ms is None
    assert (unsynced.frames[np.flatnonzero(unsynced.target)]).tolist() == [10, 30]


def test_the_soft_target_reaches_the_neighbours() -> None:
    clip = labelled(config=LabelConfig(margin=5, label_frames=2, soft_decay=0.5))
    np.testing.assert_allclose(clip.near_weight[2:9], [0, 0.25, 0.5, 1.0, 0.5, 0.25, 0])
    assert clip.near_class[3:8].tolist() == [INDEX["R"] + 1] * 5 and clip.near_class[2] == 0
    assert clip.target.tolist().count(0) == len(clip) - 2  # the hard target is unchanged
    near_class, near_weight = soft_targets(np.array([0, 3, 0, 0, 7, 0]), 1, 0.4)
    assert near_class.tolist() == [3, 3, 3, 7, 7, 7]  # a frame between two onsets: the earlier of equals
    np.testing.assert_allclose(near_weight, [0.4, 1, 0.4, 0.4, 1, 0.4])


def test_onsets_that_want_one_frame() -> None:
    t = np.arange(10) * 33.3
    # Four onsets within 9 ms of frame 5: the first takes it, the next two its neighbours (the nearer
    # first), the fourth finds none.
    target, collisions = place_onsets(t, t[5] + np.array([0.0, 3.0, 6.0, 9.0]), np.array([1, 2, 3, 4]))
    assert target[4:7].tolist() == [3, 1, 2] and collisions == 1
    clip = labelled(
        scramble=(("R", T0 + 1000.0), ("U", T0 + 1003.0), ("F", T0 + 1006.0), ("D", T0 + 1009.0)),
        interval=33.3,
        config=LabelConfig(margin=2),
    )
    assert len(clip.symbols) == 4 and clip.collisions == 1 and int((clip.target > 0).sum()) == 3


def test_the_frame_rate_keeps_every_other_frame() -> None:
    assert frame_stride(np.arange(31) * 1000 / 30, 15) == 2
    assert frame_stride(np.arange(31) * 1000 / 29.7, 15) == 2
    assert frame_stride(np.arange(31) * 1000 / 30, 10) == 3
    assert frame_stride(np.arange(31) * 1000 / 30, 0) == frame_stride(np.arange(31) * 1000 / 30, 60) == 1
    full = labelled(interval=1000 / 30, frames=40, config=LabelConfig(margin=4))
    half = labelled(interval=1000 / 30, frames=40, config=LabelConfig(margin=4, fps=15))
    assert half.stride == 2 and half.frames.tolist() == full.frames[::2].tolist()
    # The targets are re-derived from the kept frames' times: each onset on the kept frame nearest it.
    for onset, k in zip(half.onsets_ms, np.flatnonzero(half.target), strict=True):
        assert abs(half.t_ms[k] - onset) <= 1000 / 30
        assert k == np.argmin(np.abs(half.t_ms - onset))
    assert half.symbols.tolist() == full.symbols.tolist()


def test_the_clips_without_labels() -> None:
    with pytest.raises(LabelError, match="moves-outside-frames: 1 of 2"):  # the frames end before U's onset
        labelled(frames=25)
    with pytest.raises(LabelError, match="moves-outside-frames: 2 of 2"):  # every frame before the window
        labelled(t0=T0, frames=5)
    with pytest.raises(LabelError, match="no-window"):  # a segment without moves
        labelled("solve", solve=())


def test_a_window_shorter_than_a_frame_keeps_its_onsets() -> None:
    # Two turns 9 ms apart between two frames 33.3 ms apart: no frame shows the window; the frame nearest
    # its onsets is kept (and the margin around it).
    turns = (("R", T0 + 1000.0), ("U", T0 + 1009.0))
    clip = labelled(scramble=turns, interval=33.3, config=LabelConfig(margin=1))
    assert len(clip.frames) == 3 and len(clip.symbols) == 2 and clip.collisions == 0
    alone = labelled(scramble=turns, interval=33.3, config=LabelConfig(margin=0))
    assert len(alone.frames) == 1 and len(alone.symbols) == 2 and alone.collisions == 1


def test_another_segment_s_onsets_in_the_kept_frames() -> None:
    # 100 ms of inspection: the solve clip's margin reaches back to the scramble's last move, which is in
    # the reference (the video shows it) and counted.
    clip = labelled(
        "solve",
        lag=None,
        frames=90,
        scramble=(("R", T0 + 1000.0), ("U", T0 + 1200.0)),
        solve=(("F", T0 + 1300.0), ("D", T0 + 1500.0)),
    )
    assert clip.symbols.tolist() == [INDEX["U"], INDEX["F"], INDEX["D"]] and clip.foreign == 1


def about_z(degrees: float) -> list[float]:
    """The quaternion (x, y, z, w) of a rotation by `degrees` about z."""
    half = math.radians(degrees) / 2
    return [0.0, 0.0, math.sin(half), math.cos(half)]


def turning_gyro(first_ms: float = T0 + 950.0, samples: int = 9) -> dict:
    """Samples every 50 ms from `first_ms` of a cube turning about z, 10° a sample from 0° at T0 + 950."""
    start = (first_ms - (T0 + 950.0)) / 50.0
    quats = [about_z(10.0 * (start + k)) for k in range(samples)]
    return gyro_record(SID, 1, first_ms, [0.0] + [50.0] * (samples - 1), quats)


IDENTITY = [0.0, 0.0, 0.0, 1.0]


def test_the_gyro_channels_at_the_kept_frames() -> None:
    # The kept frames 10 to 40 are at T0 + 1000 to T0 + 1300 and show T0 + 950 to T0 + 1250 (lag 50), inside
    # the samples' span (T0 + 950 to T0 + 1350): the orientation turns 2° about z from one kept frame to the
    # next.
    clip = labelled(config=LabelConfig(margin=5), gyro=turning_gyro())
    assert (
        clip.gyro is not None
        and clip.gyro.shape == (31, len(GYRO_CHANNELS))
        and clip.gyro.dtype == np.float32
    )
    q, change, flag = clip.gyro[:, :4], clip.gyro[:, 4:8], clip.gyro[:, 8]
    assert q[0] == pytest.approx(IDENTITY, abs=1e-6)  # shows T0 + 950: the first sample
    assert q[5] == pytest.approx(about_z(10.0), abs=1e-6)  # shows T0 + 1000: the second sample
    assert q[2] == pytest.approx(about_z(4.0), abs=1e-6)  # shows T0 + 970: 40% of the way to it
    assert change[0] == pytest.approx(IDENTITY) and flag.tolist() == [1.0] * 31
    np.testing.assert_allclose(change[1:], np.tile(about_z(2.0), (30, 1)), atol=1e-6)
    assert clip.gyro_frames == 31
    # At half the rate, the change is between the kept frames: 4° about z.
    half = labelled(config=LabelConfig(margin=5, fps=50), gyro=turning_gyro())
    assert half.stride == 2 and half.gyro[:, :4] == pytest.approx(clip.gyro[::2, :4], abs=1e-6)
    np.testing.assert_allclose(half.gyro[1:, 4:8], np.tile(about_z(4.0), (len(half) - 1, 1)), atol=1e-6)


def test_frames_without_an_orientation() -> None:
    # Samples from T0 + 1100 only: the kept frames that show earlier have zeros and the flag 0, and the change
    # is the identity up to the first frame after one with an orientation.
    clip = labelled(config=LabelConfig(margin=5), gyro=turning_gyro(T0 + 1100.0, samples=6))
    first = 15  # the kept frame that shows T0 + 1100
    assert clip.gyro[:first, :4].tolist() == [[0.0] * 4] * first and clip.gyro[:first, 8].sum() == 0
    assert clip.gyro[first:, 8].tolist() == [1.0] * (31 - first) and clip.gyro_frames == 31 - first
    assert clip.gyro[first, :4] == pytest.approx(about_z(30.0), abs=1e-6)
    np.testing.assert_allclose(clip.gyro[: first + 1, 4:8], np.tile(IDENTITY, (first + 1, 1)))
    assert clip.gyro[first + 1, 4:8] == pytest.approx(about_z(2.0), abs=1e-6)
    # No gyro.json: zeros, the identity and the flag 0 everywhere.
    none = labelled(config=LabelConfig(margin=5))
    assert none.gyro_frames == 0 and not none.gyro[:, :4].any()
    np.testing.assert_allclose(none.gyro[:, 4:8], np.tile(IDENTITY, (31, 1)))


def test_require_gyro_skips_the_clips_without_an_orientation() -> None:
    required = LabelConfig(margin=5, require_gyro=True)
    with pytest.raises(LabelError, match=r"no-gyro: no gyro\.json"):
        labelled(config=required)
    with pytest.raises(LabelError, match="no-gyro: no orientation at the kept frames"):
        labelled(config=required, gyro=turning_gyro(T0 + 5000.0))
    assert labelled(config=required, gyro=turning_gyro(T0 + 1100.0, samples=6)).gyro_frames == 16  # partial
    broken = turning_gyro()
    broken["q"] = broken["q"][:-4]  # a sample's quaternion missing
    with pytest.raises(LabelError, match=r"records: gyro\.json"):
        labelled(config=LabelConfig(margin=5), gyro=broken)


@pytest.fixture(scope="module")
def features_dataset(tmp_path_factory: pytest.TempPathFactory):
    base = tmp_path_factory.mktemp("labels")
    root, features = base / "data", base / "features"
    ids = build_feature_dataset(root, features, sessions=3, attempts=1, dim=8)
    return root, features, ids


def test_a_split_is_loaded_with_its_features(features_dataset) -> None:
    root, features, _ = features_dataset
    dataset = Dataset(root)
    manifest = build_tables(dataset, video="none").clips
    for split in ("train", "val", "test"):
        clips, stats = load_split(dataset, manifest, split, features, "synthetic")
        assert stats.clips == stats.selected == 2 and not stats.skipped and stats.unusable == 0
        for clip in clips:
            assert clip.split == split and clip.x is not None
            assert clip.x.dtype == np.float16 and clip.x.shape == (len(clip), 8)
        assert stats.frames == sum(len(c) for c in clips) and stats.symbols == sum(
            len(c.symbols) for c in clips
        )
        assert stats.by_segment == {"scramble": 1, "solve": 1}
    clips, _ = load_split(dataset, manifest, "train", None, "synthetic")  # labels alone
    assert all(c.x is None for c in clips)
    assert "2 of 2 clips loaded" in stats.describe() and stats.to_json()["bySegment"] == {
        "scramble": 1,
        "solve": 1,
    }


def test_clips_without_features_are_skipped(features_dataset, tmp_path: Path) -> None:
    root, features, ids = features_dataset
    dataset = Dataset(root)
    refs = [ClipRef(ids["0"], 1, "laptop", "scramble"), ClipRef(ids["0"], 1, "laptop", "solve")]
    # The solve clip's features with one frame too few, the scramble clip's missing.
    good = np.load(feature_path(features, "synthetic", refs[1]))
    arrays = {k: good[k][:-1] for k in ("x", "tMs", "shownMs", "inWindow")}
    write_features(feature_path(tmp_path, "synthetic", refs[1]), arrays, {"format": 1})
    logged = []
    clips, stats = load_clips(dataset, refs, tmp_path, "synthetic", log=logged.append)
    assert clips == [] and stats.skipped == {"no-features": 1, "features-mismatch": 1}
    assert len(logged) == 2 and "features-mismatch" in logged[1]
    clips, stats = load_clips(dataset, [ClipRef(ids["0"], 9, "laptop", "solve")], features, "synthetic")
    assert stats.skipped == {"records": 1}


@pytest.fixture(scope="module")
def turned_dataset(tmp_path_factory: pytest.TempPathFactory):
    """Three sessions of two attempts with a turning cube; the second attempt of each without gyro.json."""
    base = tmp_path_factory.mktemp("turned")
    root, features = base / "data", base / "features"
    build_feature_dataset(
        root, features, sessions=3, attempts=2, dim=8, yaw=True, without_gyro={(0, 2), (1, 2), (2, 2)}
    )
    return Dataset(root), features


def test_a_split_is_loaded_with_its_gyro(turned_dataset) -> None:
    dataset, features = turned_dataset
    manifest = build_tables(dataset, video="none").clips
    for split in ("train", "val", "test"):
        clips, stats = load_split(
            dataset, manifest, split, features, "synthetic", LabelConfig(with_gyro=True)
        )
        assert stats.clips == 4 and stats.gyro_clips == 2 and not stats.skipped
        assert stats.gyro_frames == sum(len(c) for c in clips if c.ref.attempt == 1)  # every kept frame
        assert all(c.gyro_frames == 0 for c in clips if c.ref.attempt == 2)
        assert "the gyro on 2 clips" in stats.describe() and stats.to_json()["gyroClips"] == 2
        required, stats = load_split(
            dataset, manifest, split, features, "synthetic", LabelConfig(require_gyro=True)
        )
        assert [c.ref.attempt for c in required] == [1, 1] and stats.skipped == {"no-gyro": 2}
    # Without the gyro asked for, gyro.json is not read: no counts, and the channels say "none".
    clips, stats = load_split(dataset, manifest, "train", features, "synthetic")
    assert (
        stats.gyro_clips is None and stats.to_json()["gyroFrames"] is None and "gyro" not in stats.describe()
    )
    assert all(c.gyro_frames == 0 for c in clips)


def test_the_model_s_input(turned_dataset) -> None:
    dataset, features = turned_dataset
    manifest = build_tables(dataset, video="none").clips
    clips, _ = load_split(dataset, manifest, "test", features, "synthetic", LabelConfig(with_gyro=True))
    clip = clips[0]
    alone = model_input(clip, "features")
    assert alone.dtype == np.float32 and np.array_equal(alone, clip.x.astype(np.float32))
    both = model_input(clip, "features+gyro")
    assert both.shape == (len(clip), 8 + 9) and both.dtype == np.float32
    assert np.array_equal(both[:, :8], alone) and np.array_equal(both[:, 8:], clip.gyro)
    with pytest.raises(ValueError, match="inputs 'gyro'"):
        model_input(clip, "gyro")
