"""The metrics of a predicted move sequence against the reference, per clip and aggregated.

- **WER**: the edit distance between the symbol sequences (substitutions, insertions and deletions, each
  1) over the reference's length; a clip with an empty reference has none.
- **Onset F1** at a tolerance: the predicted onsets matched one to one to the reference onsets within
  ± the tolerance (inclusive), greedily in time order: each prediction takes the earliest unmatched
  reference onset within reach, which gives as many matches as any matching can when every onset has the
  same reach. `timing` matches any symbol; `symbol` matches only equal symbols (each symbol matched on
  its own).
- **Exact**: the edit distance is 0. **Replay** (a solve clip): the predicted sequence takes the attempt's
  `scrambledFacelets` to solved.

A split's numbers are the means over its clips (each clip's WER, F1 and so on, a clip without a value
skipped) and the pooled counts (the edits over the reference symbols, the F1 of the summed matches).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import cube
from .moves import SYMBOLS

TOLERANCES = (25, 50)
MODES = ("timing", "symbol")
CURVE_TOLERANCES = tuple(range(10, 101, 5))


def edit_distance(reference: Sequence[Any], hypothesis: Sequence[Any]) -> int:
    """Levenshtein distance: the fewest substitutions, insertions and deletions from one to the other (the
    table row by row, each row in NumPy: a deletion or a substitution from the row above, then the
    insertions as a running minimum)."""
    ref, hyp = np.asarray(list(reference)), np.asarray(list(hypothesis))
    if len(ref) == 0 or len(hyp) == 0:
        return int(len(ref) + len(hyp))
    j = np.arange(len(hyp) + 1)
    previous = j.copy()
    for r in ref:
        current = np.empty_like(previous)
        current[0] = previous[0] + 1
        current[1:] = np.minimum(previous[1:] + 1, previous[:-1] + (hyp != r))
        previous = np.minimum.accumulate(current - j) + j
    return int(previous[-1])


def match_times(reference: np.ndarray, hypothesis: np.ndarray, tolerance: float) -> list[tuple[int, int]]:
    """One-to-one pairs (reference index, hypothesis index) of onsets within ± `tolerance`, greedily in
    time order: each hypothesis, in time order, takes the earliest unmatched reference within reach."""
    ref_order = np.argsort(reference, kind="stable")
    hyp_order = np.argsort(hypothesis, kind="stable")
    pairs = []
    j = 0
    for h in hyp_order:
        t = hypothesis[h]
        while j < len(ref_order) and reference[ref_order[j]] < t - tolerance:
            j += 1
        if j < len(ref_order) and reference[ref_order[j]] <= t + tolerance:
            pairs.append((int(ref_order[j]), int(h)))
            j += 1
    return pairs


@dataclass
class Counts:
    """Matched onsets (`tp`), predictions left over (`fp`) and reference onsets missed (`fn`)."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, other: Counts) -> Counts:
        return Counts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else math.nan

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else math.nan

    @property
    def f1(self) -> float:
        """2·TP / (2·TP + FP + FN); NaN when there is nothing to match or predict."""
        total = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / total if total else math.nan


def onset_counts(
    ref_times: np.ndarray,
    ref_symbols: np.ndarray,
    hyp_times: np.ndarray,
    hyp_symbols: np.ndarray,
    tolerance: float,
    mode: str = "symbol",
) -> Counts:
    """The matches at `tolerance` (ms): `timing` ignores the symbols, `symbol` matches equal symbols only."""
    ref_times, hyp_times = np.asarray(ref_times, float), np.asarray(hyp_times, float)
    if mode == "timing":
        tp = len(match_times(ref_times, hyp_times, tolerance))
    elif mode == "symbol":
        ref_symbols, hyp_symbols = np.asarray(ref_symbols), np.asarray(hyp_symbols)
        tp = sum(
            len(match_times(ref_times[ref_symbols == s], hyp_times[hyp_symbols == s], tolerance))
            for s in np.intersect1d(ref_symbols, hyp_symbols)
        )
    else:
        raise ValueError(f"mode {mode!r}: one of {', '.join(MODES)}")
    return Counts(tp, len(hyp_times) - tp, len(ref_times) - tp)


def names(symbols: Iterable[int]) -> list[str]:
    """Alphabet indices as symbols."""
    return [SYMBOLS[int(s)] for s in symbols]


@dataclass
class ClipScore:
    """One clip's numbers for one predicted sequence."""

    segment: str
    bucket: str
    reference: int  # the reference's length
    predicted: int
    edits: int
    counts: dict[str, Counts] = field(default_factory=dict)  # by f"{mode}@{tolerance}"
    replay: bool | None = None  # solve clips with a scrambledFacelets only

    @property
    def wer(self) -> float:
        return self.edits / self.reference if self.reference else math.nan

    @property
    def exact(self) -> bool | None:
        return self.edits == 0 if self.reference else None

    def f1(self, mode: str, tolerance: int) -> float:
        return self.counts[f"{mode}@{tolerance}"].f1


def score_clip(
    ref_symbols: np.ndarray,
    ref_times: np.ndarray,
    hyp_symbols: np.ndarray,
    hyp_times: np.ndarray,
    *,
    segment: str,
    tps: float | None,
    facelets: str | None = None,
    tolerances: Iterable[int] = (*TOLERANCES, *CURVE_TOLERANCES),
) -> ClipScore:
    """A clip's WER, onset counts at each tolerance and mode, exact match and, for a solve clip with its
    scrambled state, the replay."""
    score = ClipScore(
        segment=segment,
        bucket=tps_bucket(tps),
        reference=len(ref_symbols),
        predicted=len(hyp_symbols),
        edits=edit_distance(list(ref_symbols), list(hyp_symbols)),
    )
    for tolerance in sorted(set(tolerances)):
        for mode in MODES:
            score.counts[f"{mode}@{tolerance}"] = onset_counts(
                ref_times, ref_symbols, hyp_times, hyp_symbols, tolerance, mode
            )
    if segment == "solve" and facelets:
        score.replay = cube.replay(facelets, names(hyp_symbols))
    return score


def tps_bucket(tps: float | None, width: float = 0.5) -> str:
    """The TPS bin of a clip's attempt, `4.0–4.5`; `–` without a TPS."""
    if tps is None or not math.isfinite(tps):
        return "–"
    low = math.floor(tps / width) * width
    return f"{low:.1f}–{low + width:.1f}"


def _mean(values: Iterable[float | bool | None]) -> float:
    kept = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(kept)) if kept else math.nan


def aggregate(scores: Sequence[ClipScore]) -> dict[str, Any]:
    """The means over the clips and the pooled counts: `clips`, `reference` symbols, `wer` (mean) and
    `werPooled`, per `mode@tolerance` the mean F1 (`f1`) and the pooled `f1Pooled`, `precision` and
    `recall`, `exact` (the share of clips with a reference), `replay` and `replayClips` (solve clips)."""
    out: dict[str, Any] = {
        "clips": len(scores),
        "reference": int(sum(s.reference for s in scores)),
        "predicted": int(sum(s.predicted for s in scores)),
        "wer": _mean(s.wer for s in scores),
        "werPooled": (
            sum(s.edits for s in scores if s.reference) / sum(s.reference for s in scores)
            if any(s.reference for s in scores)
            else math.nan
        ),
        "exact": _mean(s.exact for s in scores),
        "replay": _mean(s.replay for s in scores),
        "replayClips": sum(s.replay is not None for s in scores),
        "onsets": {},
    }
    keys = sorted(
        {k for s in scores for k in s.counts}, key=lambda k: (k.split("@")[0], int(k.split("@")[1]))
    )
    for key in keys:
        pooled = sum((s.counts[key] for s in scores), Counts())
        out["onsets"][key] = {
            "f1": _mean(s.counts[key].f1 for s in scores),
            "f1Pooled": pooled.f1,
            "precision": pooled.precision,
            "recall": pooled.recall,
            "tp": pooled.tp,
            "fp": pooled.fp,
            "fn": pooled.fn,
        }
    return out


def grouped(scores: Sequence[ClipScore], key: str) -> dict[str, dict[str, Any]]:
    """`aggregate` per value of the clips' `segment` or `bucket`, in sorted order."""
    groups: dict[str, list[ClipScore]] = {}
    for score in scores:
        groups.setdefault(getattr(score, key), []).append(score)
    return {name: aggregate(groups[name]) for name in sorted(groups)}
