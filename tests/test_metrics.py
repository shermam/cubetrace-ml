"""The metrics on known cases: the edit distance and WER, the onsets' one-to-one matching and F1, the replay,
and the aggregates (means over clips against pooled counts)."""

import math

import numpy as np
import pytest

from cubetrace_ml import cube
from cubetrace_ml.metrics import (
    Counts,
    aggregate,
    edit_distance,
    grouped,
    match_times,
    onset_counts,
    score_clip,
    tps_bucket,
)
from cubetrace_ml.moves import INDEX


def idx(text: str) -> np.ndarray:
    return np.array([INDEX[s] for s in text.split()], dtype=np.int64)


def test_the_edit_distance() -> None:
    assert edit_distance([], []) == 0 and edit_distance("abc", "abc") == 0
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("abc", "") == 3 and edit_distance("", "ab") == 2
    assert edit_distance("abcd", "acd") == 1 and edit_distance("abcd", "abxcd") == 1
    assert edit_distance(list(idx("R U R' U'")), list(idx("R U R U'"))) == 1


def test_the_wer_is_the_edits_over_the_reference() -> None:
    score = score_clip(
        idx("R U R' U'"),
        np.arange(4) * 100.0,
        idx("R U U'"),
        np.array([0, 100, 300.0]),
        segment="scramble",
        tps=4.2,
    )
    assert score.edits == 1 and score.wer == 0.25 and score.exact is False
    exact = score_clip(
        idx("R U"), np.array([0, 100.0]), idx("R U"), np.array([500, 900.0]), segment="scramble", tps=None
    )
    assert exact.wer == 0 and exact.exact is True  # the sequence alone: the times do not matter
    empty = score_clip(idx(""), np.zeros(0), idx("R"), np.array([0.0]), segment="scramble", tps=None)
    assert math.isnan(empty.wer) and empty.exact is None


def test_onsets_match_one_to_one_within_the_tolerance() -> None:
    ref = np.array([0.0, 100.0, 200.0])
    assert match_times(ref, np.array([25.0, 75.0, 230.0]), 25) == [(0, 0), (1, 1)]  # ±25 inclusive
    assert match_times(ref, np.array([20.0, 60.0]), 50) == [(0, 0), (1, 1)]  # one each
    assert match_times(ref, np.array([20.0, 30.0]), 50) == [(0, 0)]  # 30 is 70 ms from 100
    assert match_times(ref, np.array([5.0, 6.0, 7.0]), 50) == [(0, 0)]  # a reference is matched once
    # Greedy in time order takes the earliest reference within reach, which matches as many as can be: the
    # nearest-first choice would pair 30 with 40 and leave 75 alone.
    assert match_times(np.array([0.0, 40.0]), np.array([30.0, 75.0]), 35) == [(0, 0), (1, 1)]
    assert match_times(ref, np.array([]), 50) == [] and match_times(np.array([]), ref, 50) == []
    # Unsorted inputs are matched in time order, by their own indices.
    assert match_times(np.array([200.0, 0.0]), np.array([10.0, 190.0]), 25) == [(1, 0), (0, 1)]


def test_onset_counts_by_timing_and_by_symbol() -> None:
    ref_t, ref_s = np.array([0.0, 100.0, 200.0]), idx("R U F")
    hyp_t, hyp_s = np.array([10.0, 110.0, 400.0]), idx("R F F")
    timing = onset_counts(ref_t, ref_s, hyp_t, hyp_s, 50, "timing")
    symbol = onset_counts(ref_t, ref_s, hyp_t, hyp_s, 50, "symbol")
    assert (timing.tp, timing.fp, timing.fn) == (2, 1, 1)
    assert (symbol.tp, symbol.fp, symbol.fn) == (1, 2, 2)
    assert timing.f1 == pytest.approx(2 * 2 / (2 * 2 + 1 + 1)) and symbol.f1 == pytest.approx(2 / 6)
    assert timing.precision == pytest.approx(2 / 3) and timing.recall == pytest.approx(2 / 3)
    assert math.isnan(Counts().f1) and (Counts(1, 2, 3) + Counts(1, 1, 1)) == Counts(2, 3, 4)
    with pytest.raises(ValueError, match="mode"):
        onset_counts(ref_t, ref_s, hyp_t, hyp_s, 50, "both")


def test_the_replay_of_a_solve_clip() -> None:
    scramble = "R U F' D2 L B"
    state = cube.scrambled(scramble)
    solution = " ".join(cube.inverse(scramble))
    times = np.arange(6) * 200.0
    solved = score_clip(idx(solution), times, idx(solution), times, segment="solve", tps=3.0, facelets=state)
    assert solved.replay is True and solved.exact
    one_off = idx(solution)[:-1]
    assert (
        score_clip(idx(solution), times, one_off, times[:-1], segment="solve", tps=3.0, facelets=state).replay
        is False
    )
    # A slice replays as its pair: M is R and L'.
    sliced = score_clip(
        idx("M"), times[:1], idx("M"), times[:1], segment="solve", tps=3.0, facelets=cube.scrambled("L R'")
    )
    assert sliced.replay is True
    assert (
        score_clip(
            idx("R"), times[:1], idx("R"), times[:1], segment="scramble", tps=3.0, facelets=state
        ).replay
        is None
    )


def test_the_tps_buckets() -> None:
    assert tps_bucket(4.56) == "4.5–5.0" and tps_bucket(4.0) == "4.0–4.5" and tps_bucket(3.99) == "3.5–4.0"
    assert tps_bucket(None) == "–" and tps_bucket(float("nan")) == "–"


def test_means_over_clips_against_pooled_counts() -> None:
    long = score_clip(
        idx("R U F D L B R U F D"),
        np.arange(10) * 100.0,
        idx("R U F D L B R U F U"),
        np.arange(10) * 100.0,
        segment="solve",
        tps=4.2,
    )
    short = score_clip(idx("R U"), np.array([0, 100.0]), idx(""), np.zeros(0), segment="scramble", tps=4.7)
    assert long.wer == 0.1 and short.wer == 1.0
    agg = aggregate([long, short])
    assert agg["clips"] == 2 and agg["reference"] == 12 and agg["predicted"] == 10
    assert agg["wer"] == pytest.approx(0.55) and agg["werPooled"] == pytest.approx(3 / 12)
    symbol50 = agg["onsets"]["symbol@50"]
    assert (symbol50["tp"], symbol50["fp"], symbol50["fn"]) == (9, 1, 3)
    assert symbol50["f1Pooled"] == pytest.approx(18 / 22)
    assert symbol50["f1"] == pytest.approx((0.9 + 0.0) / 2)  # the short clip's F1 is 0, not missing
    assert agg["exact"] == 0.0 and math.isnan(agg["replay"]) and agg["replayClips"] == 0
    by_segment = grouped([long, short], "segment")
    assert list(by_segment) == ["scramble", "solve"] and by_segment["solve"]["wer"] == pytest.approx(0.1)
    assert list(grouped([long, short], "bucket")) == ["4.0–4.5", "4.5–5.0"]
    assert sorted(agg["onsets"]) == sorted(
        f"{m}@{t}" for m in ("timing", "symbol") for t in range(10, 101, 5)
    )
