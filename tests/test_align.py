import math

import numpy as np
import pytest

from cubetrace_ml.align import (
    PHASE_INDEX,
    PHASES,
    TRACK_DTYPE,
    align_clip,
    conjugate,
    event_times,
    frame_times,
    interpolate_orientation,
    nearest_onsets,
    phase_codes,
    quaternion_product,
    relative_rotations,
    segment_window,
    slerp,
)
from cubetrace_ml.moves import INDEX
from factory import T0, attempt_record, clip_entry, frames_record, gyro_record, session_id

SID = session_id(1)


def one_clip(
    lag: float | None,
    *,
    t0: float = T0 + 900.0,
    dt: list[float] | None = None,
    scramble=(("R", T0 + 1000.0), ("U", T0 + 1200.0)),
    solve=(("F", T0 + 3000.0),),
    segment: str = "scramble",
    gyro: dict | None = None,
):
    """An attempt with one clip whose frames are every 10 ms from t0 unless `dt` says otherwise."""
    frames = frames_record("laptop", segment, t0, dt if dt is not None else [0.0] + [10.0] * 59)
    attempt = attempt_record(
        SID, 1, list(scramble), list(solve), video=[clip_entry("laptop", segment, frames, lag=lag)]
    )
    return align_clip(attempt, frames, gyro)


def test_frame_k_is_at_t0_plus_the_cumulative_intervals() -> None:
    frames = frames_record("laptop", "solve", 1000.0, [0.0, 33.4, 33.3, 66.7, 33.3])
    assert frame_times(frames).tolist() == pytest.approx([1000.0, 1033.4, 1066.7, 1133.4, 1166.7])
    big = frames_record("laptop", "solve", T0, [0.0] + [33.3] * 900)
    assert frame_times(big)[-1] == pytest.approx(T0 + 900 * 33.3, abs=1e-3)


def test_the_lag_puts_the_onset_later_on_the_frames() -> None:
    # The move is at T0 + 1000 on the cube's (host) clock and the camera lags 50 ms: the frame that shows
    # the move is the one at T0 + 1050, where the distance is 0; the frame at T0 + 1040 is 10 ms before.
    aligned = one_clip(50.0)
    track = aligned.track
    at = {round(t - T0): k for k, t in enumerate(track["tMs"])}
    assert track["distanceMs"][at[1050]] == pytest.approx(0.0)
    assert track["distanceMs"][at[1040]] == pytest.approx(-10.0)
    assert track["distanceMs"][at[1060]] == pytest.approx(10.0)
    assert track["symbol"][at[1050]] == INDEX["R"]
    assert track["shownMs"][at[1050]] == pytest.approx(T0 + 1000.0)
    assert aligned.onset_on_frames(aligned.symbols[0]) == pytest.approx(T0 + 1050.0)
    assert aligned.lag_ms == 50.0 and not aligned.unsynced


def test_an_unsynced_clip_has_no_lag_and_is_computed_with_zero() -> None:
    aligned = one_clip(None)
    track = aligned.track
    assert aligned.unsynced and aligned.lag_ms is None and aligned.lag == 0.0
    k = int(np.argmin(np.abs(track["tMs"] - (T0 + 1000.0))))
    assert track["distanceMs"][k] == pytest.approx(0.0)
    assert np.array_equal(track["shownMs"], track["tMs"])


def test_each_frame_takes_the_nearest_onset_and_its_signed_distance() -> None:
    # Onsets at 1000 (R) and 1200 (U) plus a lag of 20: 1020 and 1220 on the frames; the midpoint is 1120.
    track = one_clip(20.0).track
    at = {round(t - T0): k for k, t in enumerate(track["tMs"])}
    assert track["symbol"][at[1110]] == INDEX["R"] and track["distanceMs"][at[1110]] == pytest.approx(90.0)
    assert track["symbol"][at[1120]] == INDEX["R"]  # a tie goes to the earlier onset
    assert track["symbol"][at[1130]] == INDEX["U"] and track["distanceMs"][at[1130]] == pytest.approx(-90.0)
    assert track["onset"][at[1130]] == 1
    assert track.dtype == TRACK_DTYPE and track["frame"].tolist() == list(range(60))


def test_nearest_onsets_without_onsets_or_unsorted() -> None:
    assert nearest_onsets(np.array([1.0, 2.0]), np.array([])).tolist() == [-1, -1]
    assert nearest_onsets(np.array([0.0, 9.0, 21.0]), np.array([20.0, 10.0, 0.0])).tolist() == [2, 1, 0]


def test_a_clip_of_an_attempt_without_moves_has_no_symbol() -> None:
    aligned = one_clip(10.0, scramble=(), solve=())
    assert (aligned.track["symbol"] == -1).all() and (aligned.track["onset"] == -1).all()
    assert np.isnan(aligned.track["distanceMs"]).all()
    assert aligned.covered() == (0, 0)


def test_the_phase_of_each_frame_follows_the_events() -> None:
    events = {
        "scrambleShown": 0.0,
        "scrambleStart": 100.0,
        "scrambleDone": 200.0,
        "pickup": None,
        "solveStart": 300.0,
        "solveEnd": 400.0,
    }
    shown = np.array([50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0])
    names = [PHASES[k] for k in phase_codes(shown, events)]
    assert names == [
        "before",
        "scramble",
        "scramble",
        "scramble",
        "inspection",
        "solve",
        "solve",
        "solve",
        "after",
    ]
    dnf = dict(events, solveEnd=None)
    assert PHASES[phase_codes(np.array([10_000.0]), dnf)[0]] == "solve"
    unarmed = dict(events, scrambleDone=None, solveStart=None, solveEnd=None)
    assert PHASES[phase_codes(np.array([10_000.0]), unarmed)[0]] == "scramble"


def test_the_window_and_the_phase_on_a_clip_use_the_shown_time() -> None:
    # Scramble from 1000 to 1200; with a lag of 30 the frame at 1030 shows 1000, the scramble's start.
    aligned = one_clip(30.0)
    track = aligned.track
    at = {round(t - T0): k for k, t in enumerate(track["tMs"])}
    assert aligned.window == (T0 + 1000.0, T0 + 1200.0)
    assert not track["inWindow"][at[1020]] and track["phase"][at[1020]] == PHASE_INDEX["before"]
    assert track["inWindow"][at[1030]] and track["phase"][at[1030]] == PHASE_INDEX["scramble"]
    assert track["inWindow"][at[1230]] and not track["inWindow"][at[1240]]
    assert track["phase"][at[1240]] == PHASE_INDEX["inspection"]
    assert aligned.covered() == (2, 2)
    assert [s.symbol for s in aligned.segment_symbols] == ["R", "U"]


def test_segment_windows() -> None:
    events = {"scrambleStart": 1.0, "scrambleDone": 2.0, "solveStart": 3.0, "solveEnd": None}
    assert segment_window(events, "scramble") == (1.0, 2.0)
    assert segment_window(events, "solve") == (3.0, math.inf)
    low, high = segment_window({"solveStart": None, "solveEnd": None}, "solve")
    assert math.isnan(low) and high == math.inf


def test_events_that_are_moves_take_the_moves_time_base() -> None:
    attempt = attempt_record(
        SID,
        1,
        [("R", T0 + 1010.0), ("U", T0 + 1190.0)],
        [("F", T0 + 2995.0)],
        cube_ms=[1000.0, 1200.0, 3000.0],  # the fit (a = 1, b = T0) places them at 1000, 1200, 3000
    )
    times = np.array([T0 + 1000.0, T0 + 1200.0, T0 + 3000.0])
    events = event_times(attempt, times)
    assert events["scrambleStart"] == T0 + 1000.0
    assert events["scrambleDone"] == T0 + 1200.0
    assert events["solveStart"] == T0 + 3000.0 and events["solveEnd"] == T0 + 3000.0
    assert events["scrambleShown"] == attempt["events"]["scrambleShown"]


def test_slerp_and_the_orientation_at_frame_times() -> None:
    half = math.sqrt(0.5)
    quats = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, half, half]])  # identity, then 90° about z
    times = np.array([100.0, 200.0])
    out = interpolate_orientation(times, quats, np.array([100.0, 150.0, 200.0, 99.0, 201.0]))
    assert out[0] == pytest.approx([0, 0, 0, 1])  # at a sample: the sample
    assert out[2] == pytest.approx([0, 0, half, half])
    eighth = math.pi / 8  # halfway: 45° about z
    assert out[1] == pytest.approx([0, 0, math.sin(eighth), math.cos(eighth)])
    assert np.isnan(out[3]).all() and np.isnan(out[4]).all()  # outside the samples' span
    # q and −q are one rotation: the slerp takes the shorter arc.
    flipped = slerp(quats[:1], -quats[1:], np.array([0.5]))[0]
    assert abs(np.dot(flipped, [0, 0, math.sin(eighth), math.cos(eighth)])) == pytest.approx(1.0)


def test_the_track_carries_the_gyro_at_the_shown_time() -> None:
    half = math.sqrt(0.5)
    # Samples at 1000 and 1100 on the host clock: identity, then 90° about z.
    gyro = gyro_record(SID, 1, T0 + 1000.0, [0.0, 100.0], [(0, 0, 0, 1), (0, 0, half, half)])
    track = one_clip(50.0, gyro=gyro).track
    at = {round(t - T0): k for k, t in enumerate(track["tMs"])}
    q = np.stack([track[c] for c in ("qx", "qy", "qz", "qw")], axis=1)
    assert q[at[1050]] == pytest.approx([0, 0, 0, 1], abs=1e-6)  # shows 1000: the first sample
    assert q[at[1150]] == pytest.approx([0, 0, half, half], abs=1e-6)  # shows 1100: the second
    eighth = math.pi / 8
    assert q[at[1100]] == pytest.approx([0, 0, math.sin(eighth), math.cos(eighth)], abs=1e-6)
    assert np.isnan(q[at[1040]]).all() and np.isnan(q[at[1160]]).all()
    no_gyro = one_clip(50.0).track
    assert np.isnan(no_gyro["qw"]).all()


def rotation(axis: tuple[float, float, float], degrees: float) -> np.ndarray:
    """The unit quaternion (x, y, z, w) of a rotation by `degrees` about `axis` (right-handed)."""
    unit = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    half = math.radians(degrees) / 2
    return np.array([*(unit * math.sin(half)), math.cos(half)])


def rotate(q: np.ndarray, v: tuple[float, float, float]) -> np.ndarray:
    """The vector v rotated by q: q · (v, 0) · conj(q)."""
    return quaternion_product(quaternion_product(q, np.array([*v, 0.0])), conjugate(q))[:3]


def test_the_quaternion_product_composes_rotations() -> None:
    i, j, k = np.eye(4)[0], np.eye(4)[1], np.eye(4)[2]
    assert quaternion_product(i, j) == pytest.approx(k)  # i·j = k
    assert quaternion_product(j, i) == pytest.approx(-k)
    z90, x90 = rotation((0, 0, 1), 90), rotation((1, 0, 0), 90)
    assert rotate(z90, (1, 0, 0)) == pytest.approx([0, 1, 0], abs=1e-12)  # x to y about z
    # a · b turns by b first, then by a: y goes to z by x90, then stays on z under z90.
    assert rotate(quaternion_product(z90, x90), (0, 1, 0)) == pytest.approx([0, 0, 1], abs=1e-12)
    assert rotate(quaternion_product(x90, z90), (0, 1, 0)) == pytest.approx([-1, 0, 0], abs=1e-12)
    assert quaternion_product(z90, conjugate(z90)) == pytest.approx([0, 0, 0, 1])
    pairs = np.stack([z90, x90])  # row by row
    assert quaternion_product(pairs, pairs[::-1]) == pytest.approx(
        np.stack([quaternion_product(z90, x90), quaternion_product(x90, z90)])
    )


def test_the_change_of_orientation_between_rows() -> None:
    identity = [0.0, 0.0, 0.0, 1.0]
    # 30° then 50° about z: 20° about z; the first row has no predecessor.
    change = relative_rotations(np.stack([rotation((0, 0, 1), 30), rotation((0, 0, 1), 50)]))
    assert change[0] == pytest.approx(identity) and change[1] == pytest.approx(rotation((0, 0, 1), 20))
    # Turns that do not commute: the change is the rotation applied on the left, q_t = r · q_{t−1}.
    before = rotation((1, 0, 0), 90)
    turn = rotation((0.3, -0.5, 0.8), 40)
    after = quaternion_product(turn, before)
    assert relative_rotations(np.stack([before, after]))[1] == pytest.approx(turn)
    # −q is the same orientation: the change keeps w ≥ 0.
    assert relative_rotations(np.stack([before, -after]))[1] == pytest.approx(turn)
    assert relative_rotations(np.stack([before, quaternion_product(conjugate(turn), before)]))[1] == (
        pytest.approx(conjugate(turn))
    )
    # A row without an orientation: the identity there and at the row after it.
    rows = np.stack([before, np.full(4, np.nan), after, quaternion_product(turn, after)])
    change = relative_rotations(rows)
    assert change[:3] == pytest.approx(np.tile(identity, (3, 1))) and change[3] == pytest.approx(turn)
    assert relative_rotations(np.zeros((0, 4))).shape == (0, 4)
    assert relative_rotations(before[None])[0] == pytest.approx(identity)


def test_align_clip_needs_the_clip_in_the_record() -> None:
    frames = frames_record("laptop", "solve", T0, [0.0, 10.0])
    attempt = attempt_record(SID, 1, [("R", T0)], [], video=[])
    with pytest.raises(KeyError):
        align_clip(attempt, frames)
