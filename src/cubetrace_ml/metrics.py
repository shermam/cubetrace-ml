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
- **Confusions**: at the onsets matched on timing (as `timing` matches them), the predicted symbol against
  the reference's: right, the same face turned another way, the opposite face, or another face; counted by
  pair, by reference symbol and by camera.

A split's numbers are the means over its clips (each clip's WER, F1 and so on, a clip without a value
skipped) and the pooled counts (the edits over the reference symbols, the F1 of the summed matches).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import cube
from .moves import INDEX, OPPOSITE, SYMBOLS

TOLERANCES = (25, 50)
MODES = ("timing", "symbol")
CURVE_TOLERANCES = tuple(range(10, 101, 5))
CONFUSION_TOLERANCE = 50
CONFUSION_KINDS = ("right", "same face, other turn", "opposite face", "other face")
SLICE_LAYERS = "MSE"


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


# The confusions.


def confusion_kind(reference: str, predicted: str) -> str:
    """How a predicted symbol relates to the reference's (CONFUSION_KINDS): `right`; the same face (or slice)
    turned another way (`R'` or `R2` for `R`, `M'` for `M`); the opposite face (`L` for `R`), the slices
    being a family of their own (`S` for `M` counts here); any other face (a slice for a face turn too)."""
    if reference == predicted:
        return "right"
    if reference[0] == predicted[0]:
        return "same face, other turn"
    if OPPOSITE.get(reference[0]) == predicted[0] or (
        reference[0] in SLICE_LAYERS and predicted[0] in SLICE_LAYERS
    ):
        return "opposite face"
    return "other face"


@dataclass
class Confusions:
    """A system's symbols at its onsets matched one to one to the reference's within ± `tolerance` ms on
    timing (`match_times`, as F1 `timing` matches them): the matched (reference, predicted) pairs, and per
    camera the matched onsets whose symbol is right, those whose symbol is wrong, and the reference onsets
    left unmatched."""

    tolerance: float = CONFUSION_TOLERANCE
    reference: int = 0
    pairs: Counter[tuple[str, str]] = field(default_factory=Counter)
    cameras: dict[str, list[int]] = field(default_factory=dict)

    def add(
        self,
        ref_symbols: Sequence[str],
        ref_times: Sequence[float],
        hyp_symbols: Sequence[str],
        hyp_times: Sequence[float],
        camera: str = "",
    ) -> None:
        """One clip's sequences (symbols as names, onset times in ms)."""
        pairs = match_times(np.asarray(ref_times, float), np.asarray(hyp_times, float), self.tolerance)
        counts = self.cameras.setdefault(camera, [0, 0, 0])
        for i, j in pairs:
            reference, predicted = str(ref_symbols[i]), str(hyp_symbols[j])
            self.pairs[(reference, predicted)] += 1
            counts[0 if reference == predicted else 1] += 1
        counts[2] += len(ref_symbols) - len(pairs)
        self.reference += len(ref_symbols)

    @property
    def matched(self) -> int:
        return sum(self.pairs.values())

    def kinds(self) -> dict[str, int]:
        """The matched onsets by kind, in CONFUSION_KINDS' order."""
        counts: Counter[str] = Counter()
        for (r, p), n in self.pairs.items():
            counts[confusion_kind(r, p)] += n
        return {kind: counts[kind] for kind in CONFUSION_KINDS}

    def per_symbol(self) -> dict[str, tuple[int, int]]:
        """Per reference symbol with a matched onset, in the alphabet's order: (right, matched)."""
        right: Counter[str] = Counter()
        matched: Counter[str] = Counter()
        for (r, p), n in self.pairs.items():
            matched[r] += n
            right[r] += n * (r == p)
        return {s: (right[s], matched[s]) for s in sorted(matched, key=_order)}

    def top(self, count: int = 12) -> list[tuple[str, str, int]]:
        """The `count` most frequent confusions (reference ≠ predicted), the most first (then in the
        alphabet's order of the reference and of the prediction)."""
        wrong = [(r, p, n) for (r, p), n in self.pairs.items() if r != p]
        return sorted(wrong, key=lambda rpn: (-rpn[2], _order(rpn[0]), _order(rpn[1])))[:count]

    def to_json(self) -> dict[str, Any]:
        per_symbol = self.per_symbol()
        matrix: dict[str, dict[str, int]] = {}
        for (r, p), n in sorted(self.pairs.items(), key=lambda kv: (_order(kv[0][0]), _order(kv[0][1]))):
            matrix.setdefault(r, {})[p] = n
        return {
            "tolerance": self.tolerance,
            "reference": self.reference,
            "matched": self.matched,
            "kinds": self.kinds(),
            "perSymbol": {
                s: {"right": right, "matched": n, "accuracy": right / n}
                for s, (right, n) in per_symbol.items()
            },
            "top": [
                {"reference": r, "predicted": p, "onsets": n, "kind": confusion_kind(r, p)}
                for r, p, n in self.top()
            ],
            "byCamera": {
                camera: {
                    "right": right,
                    "wrong": wrong,
                    "unmatched": unmatched,
                    "accuracy": right / (right + wrong) if right + wrong else math.nan,
                    "recall": (right + wrong) / (right + wrong + unmatched)
                    if right + wrong + unmatched
                    else math.nan,
                }
                for camera, (right, wrong, unmatched) in sorted(self.cameras.items())
            },
            "matrix": matrix,
        }


def _order(symbol: str) -> int:
    """A symbol's place in the alphabet (unknown names after it)."""
    return INDEX.get(symbol, len(SYMBOLS))
