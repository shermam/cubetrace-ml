"""From a model's per-frame outputs (probabilities over the 25 classes) to a move sequence with onset times,
and the trivial baseline; NumPy only.

- **Per frame** (`perframe`): P(onset) = 1 − P(no onset); its peaks are the local maxima at or above a
  threshold, at least `min_distance` frames apart (the higher kept); a peak's symbol is the argmax of the
  onset classes' probabilities summed over the peak's frame and its `neighbours` on each side; its onset
  time is its frame's.
- **CTC** (`ctc`): greedy: the argmax class of every frame, repeats collapsed and blanks (class 0) dropped;
  an emitted symbol's onset time is the first frame of its spike.
- **The consistency pass** (`--consistency`): adjacent predictions merged by the normalization's rules (a
  slice: opposite faces turning the same way less than `slice_ms` apart; a double: two equal quarter turns
  of a face less than `double_ms` apart), then immediate cancellations dropped (`R R'`, `R2 R2`, `M M'`),
  repeatedly, as a stack does.
- **The baseline**: onsets at the peaks of the feature-difference norm (each frame's distance from the
  previous one in the standardized features, over the clip's 99th percentile, clipped to 1), moved by a
  constant shift learnt on the training set (the median offset from a reference onset to the nearest
  peak), each one the training set's most frequent symbol.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .moves import DOUBLE_MS, OPPOSITE, SLICE_MS, SLICES, SYMBOLS, parse_move

THRESHOLDS = tuple(round(0.1 + 0.05 * k, 2) for k in range(17))  # 0.1 to 0.9
MIN_DISTANCE = 2
NEIGHBOURS = 1
INDEX = {symbol: i for i, symbol in enumerate(SYMBOLS)}


@dataclass
class Decoded:
    """A predicted sequence: alphabet indices, their onset times (ms, on the frames' timeline) and frames."""

    symbols: np.ndarray
    times: np.ndarray
    frames: np.ndarray

    @staticmethod
    def empty() -> Decoded:
        return Decoded(np.zeros(0, np.int64), np.zeros(0), np.zeros(0, np.int64))

    def __len__(self) -> int:
        return len(self.symbols)


def find_peaks(score: np.ndarray, threshold: float, min_distance: int = MIN_DISTANCE) -> np.ndarray:
    """The frames where `score` has a local maximum (the first frame of a plateau; the ends count) at or
    above `threshold`, then thinned so that no two are closer than `min_distance` frames: the higher is
    kept (the earlier of equals)."""
    score = np.asarray(score, dtype=np.float64)
    if len(score) == 0:
        return np.zeros(0, np.int64)
    before = np.concatenate([[-np.inf], score[:-1]])
    after = np.concatenate([score[1:], [-np.inf]])
    peaks = np.flatnonzero((score > before) & (score >= after) & (score >= threshold))
    if min_distance <= 1 or len(peaks) < 2:
        return peaks
    keep = np.ones(len(peaks), dtype=bool)
    for k in np.argsort(-score[peaks], kind="stable"):
        if keep[k]:
            near = np.abs(peaks - peaks[k]) < min_distance
            near[k] = False
            keep &= ~near
    return peaks[keep]


def perframe_decode(
    probs: np.ndarray,
    t_ms: np.ndarray,
    threshold: float,
    *,
    min_distance: int = MIN_DISTANCE,
    neighbours: int = NEIGHBOURS,
) -> Decoded:
    """The peaks of P(onset) = 1 − P(no onset), each with the symbol whose probability summed over the
    peak's frame and its neighbours is the highest."""
    probs = np.asarray(probs, dtype=np.float64)
    peaks = find_peaks(1.0 - probs[:, 0], threshold, min_distance)
    symbols = np.array(
        [int(np.argmax(probs[max(0, k - neighbours) : k + neighbours + 1, 1:].sum(axis=0))) for k in peaks],
        dtype=np.int64,
    )
    return Decoded(symbols, np.asarray(t_ms, dtype=np.float64)[peaks], peaks)


def ctc_greedy(probs: np.ndarray, t_ms: np.ndarray) -> Decoded:
    """Greedy CTC decoding: each frame's argmax class, repeats collapsed, blanks (0) dropped; a symbol's
    onset is the first frame of its spike."""
    best = np.asarray(probs).argmax(axis=1)
    if len(best) == 0:
        return Decoded.empty()
    starts = np.flatnonzero((best != 0) & (np.concatenate([[-1], best[:-1]]) != best))
    return Decoded(best[starts].astype(np.int64) - 1, np.asarray(t_ms, dtype=np.float64)[starts], starts)


def inverse_symbol(symbol: str) -> str:
    """The symbol that undoes it: `R` ↔ `R'`, `R2` itself, `M` ↔ `M'`."""
    if symbol.endswith("2"):
        return symbol
    return symbol[:-1] if symbol.endswith("'") else symbol + "'"


def _quarter(symbol: str) -> tuple[str, int] | None:
    """A face's quarter turn as (face, 1 or 3); None for a double or a slice."""
    try:
        face, turns = parse_move(symbol)
    except ValueError:
        return None
    return None if turns == 2 else (face, turns)


def consistent(decoded: Decoded, *, slice_ms: float = SLICE_MS, double_ms: float = DOUBLE_MS) -> Decoded:
    """The consistency pass: adjacent quarter turns merged by the normalization's rules (a slice first, then
    a double; the merged symbol at the first one's time and frame), then adjacent cancelling pairs dropped
    until none is left."""
    merged: list[tuple[str, float, int]] = []
    items = [
        (SYMBOLS[s], float(t), int(f))
        for s, t, f in zip(decoded.symbols, decoded.times, decoded.frames, strict=True)
    ]
    i = 0
    while i < len(items):
        symbol, t, frame = items[i]
        first = _quarter(symbol)
        if first is not None and i + 1 < len(items):
            second = _quarter(items[i + 1][0])
            gap = items[i + 1][1] - t
            if second is not None:
                (face, turns), (other, other_turns) = first, second
                if other == OPPOSITE[face] and other_turns == 4 - turns and gap < slice_ms:
                    merged.append((SLICES.get((face, turns)) or SLICES[(other, other_turns)], t, frame))
                    i += 2
                    continue
                if other == face and other_turns == turns and gap < double_ms:
                    merged.append((face + "2", t, frame))
                    i += 2
                    continue
        merged.append((symbol, t, frame))
        i += 1
    stack: list[tuple[str, float, int]] = []
    for item in merged:
        if stack and inverse_symbol(stack[-1][0]) == item[0]:
            stack.pop()
        else:
            stack.append(item)
    if not stack:
        return Decoded.empty()
    return Decoded(
        np.array([INDEX[s] for s, _, _ in stack], dtype=np.int64),
        np.array([t for _, t, _ in stack], dtype=np.float64),
        np.array([f for _, _, f in stack], dtype=np.int64),
    )


# The baseline.


def motion_score(x: np.ndarray) -> np.ndarray:
    """Each frame's distance from the previous one (0 for the first), over the clip's 99th percentile,
    clipped to 1."""
    x = np.asarray(x, dtype=np.float32)
    if len(x) < 2:
        return np.zeros(len(x))
    step = np.concatenate([[0.0], np.linalg.norm(np.diff(x, axis=0), axis=1)])
    scale = float(np.percentile(step, 99))
    return np.clip(step / scale, 0.0, 1.0) if scale > 0 else np.zeros(len(x))


def most_frequent(sequences: Iterable[Sequence[int]]) -> int:
    """The most frequent symbol (the lowest index of equals); 0 (`U`) when there is none."""
    counts = Counter(int(s) for seq in sequences for s in seq)
    return min(counts, key=lambda s: (-counts[s], s)) if counts else 0


def baseline_shift(
    scores: Iterable[np.ndarray],
    times: Iterable[np.ndarray],
    onsets: Iterable[np.ndarray],
    reach_ms: float = 100,
) -> float:
    """The median offset (ms) from each reference onset to the nearest local maximum of its clip's motion
    score within `reach_ms`: what moves the baseline's peaks onto the onsets (0 without any)."""
    offsets: list[float] = []
    for score, t, refs in zip(scores, times, onsets, strict=True):
        peaks = t[find_peaks(score, 0.0, 1)]
        if len(peaks) == 0:
            continue
        for onset in refs:
            nearest = peaks[np.argmin(np.abs(peaks - onset))]
            if abs(nearest - onset) <= reach_ms:
                offsets.append(float(nearest - onset))
    return float(np.median(offsets)) if offsets else 0.0


def baseline_decode(
    score: np.ndarray,
    t_ms: np.ndarray,
    threshold: float,
    *,
    symbol: int,
    shift_ms: float = 0.0,
    min_distance: int = MIN_DISTANCE,
) -> Decoded:
    """The baseline's sequence: the motion score's peaks, moved back by the shift, all of one symbol."""
    peaks = find_peaks(score, threshold, min_distance)
    return Decoded(
        np.full(len(peaks), symbol, dtype=np.int64),
        np.asarray(t_ms, dtype=np.float64)[peaks] - shift_ms,
        peaks,
    )
