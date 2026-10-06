"""The decoders: peak picking and the per-frame symbol, CTC's greedy decoding, the consistency pass and the
baseline."""

import numpy as np
import pytest

from cubetrace_ml.decode import (
    THRESHOLDS,
    Decoded,
    baseline_decode,
    baseline_shift,
    consistent,
    ctc_greedy,
    find_peaks,
    inverse_symbol,
    most_frequent,
    motion_score,
    perframe_decode,
)
from cubetrace_ml.moves import INDEX, SYMBOLS


def probs_with(onsets: dict[int, dict[str, float]], frames: int) -> np.ndarray:
    """Per-frame probabilities: "no onset" everywhere but where `onsets` puts a symbol's mass."""
    p = np.zeros((frames, 25))
    p[:, 0] = 1.0
    for k, mass in onsets.items():
        for symbol, value in mass.items():
            p[k, 1 + INDEX[symbol]] = value
        p[k, 0] = 1.0 - sum(mass.values())
    return p


def test_the_thresholds() -> None:
    assert THRESHOLDS[0] == 0.1 and THRESHOLDS[-1] == 0.9 and len(THRESHOLDS) == 17


def test_peaks_are_local_maxima_above_the_threshold() -> None:
    score = np.array([0.0, 0.6, 0.2, 0.2, 0.9, 0.9, 0.1, 0.3, 0.7])
    assert find_peaks(score, 0.5, 1).tolist() == [1, 4, 8]  # a plateau's first frame; the last frame counts
    assert find_peaks(score, 0.65, 1).tolist() == [4, 8]
    assert find_peaks(score, 0.0, 1).tolist() == [1, 4, 8]  # 2 and 3 are a plateau below its left
    # The minimum distance keeps the higher of two close peaks.
    close = np.array([0.0, 0.8, 0.1, 0.9, 0.0, 0.0, 0.7])
    assert find_peaks(close, 0.5, 1).tolist() == [1, 3, 6]
    assert find_peaks(close, 0.5, 3).tolist() == [3, 6]
    assert find_peaks(np.zeros(0), 0.5).tolist() == []


def test_a_peak_s_symbol_is_summed_over_its_neighbours() -> None:
    # Frame 3's own argmax is U, but R carries more over frames 2–4.
    p = probs_with({2: {"R": 0.3}, 3: {"U": 0.5, "R": 0.4}, 4: {"R": 0.3}, 8: {"F'": 0.9}}, 12)
    t = 1000.0 + 33.3 * np.arange(12)
    decoded = perframe_decode(p, t, 0.5)
    assert decoded.frames.tolist() == [3, 8]
    assert [SYMBOLS[s] for s in decoded.symbols] == ["R", "F'"]
    np.testing.assert_allclose(decoded.times, t[[3, 8]])
    assert [SYMBOLS[s] for s in perframe_decode(p, t, 0.5, neighbours=0).symbols] == ["U", "F'"]
    assert len(perframe_decode(p, t, 0.95)) == 0


def test_ctc_greedy_decoding() -> None:
    def frames(*classes: int) -> np.ndarray:
        p = np.full((len(classes), 25), 0.01)
        p[np.arange(len(classes)), classes] = 0.9
        return p

    r, u = INDEX["R"] + 1, INDEX["U"] + 1
    t = np.arange(8) * 10.0
    decoded = ctc_greedy(frames(r, r, 0, r, u, u, 0, 0), t)
    assert [SYMBOLS[s] for s in decoded.symbols] == ["R", "R", "U"]  # a blank separates the repeat
    assert decoded.frames.tolist() == [0, 3, 4] and decoded.times.tolist() == [0.0, 30.0, 40.0]
    assert len(ctc_greedy(frames(0, 0, 0), t[:3])) == 0 and len(ctc_greedy(np.zeros((0, 25)), t[:0])) == 0


def sequence(*items: tuple[str, float]) -> Decoded:
    return Decoded(
        np.array([INDEX[s] for s, _ in items], dtype=np.int64),
        np.array([t for _, t in items]),
        np.arange(len(items)),
    )


def symbols(d: Decoded) -> list[str]:
    return [SYMBOLS[s] for s in d.symbols]


def test_the_consistency_pass() -> None:
    # Two equal quarter turns under 200 ms are a double, at the first one's time; over it they stay.
    merged = consistent(sequence(("R", 0.0), ("R", 90.0), ("U", 400.0)))
    assert symbols(merged) == ["R2", "U"] and merged.times.tolist() == [0.0, 400.0]
    assert symbols(consistent(sequence(("R", 0.0), ("R", 300.0)))) == ["R", "R"]
    assert symbols(consistent(sequence(("R'", 0.0), ("R'", 100.0)))) == ["R2"]
    assert symbols(consistent(sequence(("R", 0.0), ("L'", 10.0)))) == ["M"]  # a slice: under 20 ms
    assert symbols(consistent(sequence(("R", 0.0), ("L'", 40.0)))) == ["R", "L'"]
    # Cancellations go, nested ones too, as a stack drops them.
    assert symbols(consistent(sequence(("R", 0.0), ("U", 300.0), ("U'", 600.0), ("R'", 900.0)))) == []
    assert symbols(consistent(sequence(("F", 0.0), ("R2", 300.0), ("R2", 600.0), ("D", 900.0)))) == ["F", "D"]
    assert symbols(consistent(sequence(("M", 0.0), ("M'", 300.0), ("U", 600.0)))) == ["U"]
    # A double merged first is not cancelled by its quarter turns: R R R' is R2 R'.
    assert symbols(consistent(sequence(("R", 0.0), ("R", 90.0), ("R'", 400.0)))) == ["R2", "R'"]
    assert len(consistent(Decoded.empty())) == 0
    assert inverse_symbol("R") == "R'" and inverse_symbol("U'") == "U" and inverse_symbol("F2") == "F2"
    assert inverse_symbol("E") == "E'"


def test_the_baseline() -> None:
    x = np.zeros((20, 4), dtype=np.float32)
    x[5:] += 1.0  # a change at frame 5
    x[12:] += 2.0  # a bigger one at frame 12
    score = motion_score(x)
    assert score[0] == 0 and score[12] == 1.0 and 0 < score[5] < 1.0
    assert motion_score(x[:1]).tolist() == [0.0] and motion_score(np.ones((5, 2))).tolist() == [0.0] * 5
    t = 100.0 * np.arange(20)
    decoded = baseline_decode(score, t, 0.2, symbol=INDEX["R"], shift_ms=30.0)
    assert decoded.frames.tolist() == [5, 12] and symbols(decoded) == ["R", "R"]
    np.testing.assert_allclose(decoded.times, [470.0, 1170.0])  # moved back by the shift
    # The shift: the peaks sit 100 ms after the onsets (within reach) or far from them (ignored).
    assert baseline_shift([score], [t], [np.array([400.0, 1100.0])]) == pytest.approx(100.0)
    assert baseline_shift([score], [t], [np.array([0.0])]) == 0.0
    assert most_frequent([[3, 3, 5], [5, 1]]) == 3 and most_frequent([]) == 0
