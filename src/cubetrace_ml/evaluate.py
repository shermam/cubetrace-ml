"""A run's evaluation on a split, from the model's per-frame probabilities (NumPy only): the decoding of
either head, the per-frame head's threshold chosen on `val`, the baseline fitted on `train` and `val`, each
clip scored (`metrics`) for the model, the model through the consistency pass and the baseline, and the
aggregates over the split, by segment and by TPS bucket."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import DecodeConfig
from .cube import replay
from .decode import (
    THRESHOLDS,
    Decoded,
    baseline_decode,
    baseline_shift,
    consistent,
    ctc_greedy,
    most_frequent,
    motion_score,
    perframe_decode,
)
from .labels import ClipLabels
from .metrics import ClipScore, Counts, aggregate, edit_distance, grouped, names, onset_counts, score_clip

SELECTION_TOLERANCE = 50  # F1@50 (symbol) chooses the thresholds and the per-frame head's best epoch


def onset_curve(probs: np.ndarray) -> np.ndarray:
    """P(onset) per frame: 1 − P(no onset) for the per-frame head, 1 − P(blank) for CTC."""
    return 1.0 - np.asarray(probs, dtype=np.float64)[:, 0]


def decode_clip(
    head: str, probs: np.ndarray, t_ms: np.ndarray, threshold: float, decode: DecodeConfig | None = None
) -> Decoded:
    """The head's sequence from a clip's probabilities (the per-frame head's peaks at `threshold`)."""
    decode = decode or DecodeConfig()
    if head == "perframe":
        return perframe_decode(
            probs, t_ms, threshold, min_distance=decode.min_distance, neighbours=decode.neighbours
        )
    if head == "ctc":
        return ctc_greedy(probs, t_ms)
    raise ValueError(f"no head {head!r}")


def pooled_f1(clips: Sequence[ClipLabels], decoded: Sequence[Decoded], mode: str = "symbol") -> float:
    """The pooled F1@50 (`mode`) of the decoded sequences against the clips'."""
    counts = Counts()
    for clip, d in zip(clips, decoded, strict=True):
        counts += onset_counts(clip.onsets_ms, clip.symbols, d.times, d.symbols, SELECTION_TOLERANCE, mode)
    return counts.f1


def pooled_wer(clips: Sequence[ClipLabels], decoded: Sequence[Decoded]) -> float:
    """The pooled WER: the edits over all the reference symbols."""
    edits = sum(edit_distance(c.symbols, d.symbols) for c, d in zip(clips, decoded, strict=True))
    reference = sum(len(c.symbols) for c in clips)
    return edits / reference if reference else math.nan


def pooled(
    clips: Sequence[ClipLabels], decoded: Sequence[Decoded], mode: str = "symbol"
) -> tuple[float, float]:
    """The pooled F1@50 (`mode`) and the pooled WER."""
    return pooled_f1(clips, decoded, mode), pooled_wer(clips, decoded)


def _best(candidates: dict[float, float]) -> float:
    """The threshold of the highest F1 (NaN counts as 0), the one nearest 0.5 of equals."""
    return max(candidates, key=lambda t: (np.nan_to_num(candidates[t]), -abs(t - 0.5), -t))


def choose_threshold(
    probs: Sequence[np.ndarray], clips: Sequence[ClipLabels], decode: DecodeConfig | None = None
) -> tuple[float, float, float]:
    """The per-frame head's threshold among 0.1–0.9 (step 0.05) with the best pooled F1@50 (symbol) on
    the clips; that F1 and the pooled WER there."""
    decoded = {
        t: [decode_clip("perframe", p, c.t_ms, t, decode) for p, c in zip(probs, clips, strict=True)]
        for t in THRESHOLDS
    }
    f1 = {t: pooled_f1(clips, d) for t, d in decoded.items()}
    best = _best(f1)
    return best, f1[best], pooled_wer(clips, decoded[best])


# The baseline.


@dataclass
class Baseline:
    """The baseline's settings: its one symbol (the training set's most frequent), the shift of its peaks
    (from the training set) and its threshold (chosen on val by F1@50, timing)."""

    symbol: int = 0
    shift_ms: float = 0.0
    threshold: float = 0.5
    min_distance: int = 2

    def to_json(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "shiftMs": self.shift_ms,
            "threshold": self.threshold,
            "minDistance": self.min_distance,
        }

    @staticmethod
    def from_json(doc: dict[str, Any]) -> Baseline:
        return Baseline(
            int(doc["symbol"]), float(doc["shiftMs"]), float(doc["threshold"]), int(doc["minDistance"])
        )


def standardized(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float32) - mean) / std


def baseline_clip(clip: ClipLabels, baseline: Baseline, mean: np.ndarray, std: np.ndarray) -> Decoded:
    assert clip.x is not None
    score = motion_score(standardized(clip.x, mean, std))
    return baseline_decode(
        score,
        clip.t_ms,
        baseline.threshold,
        symbol=baseline.symbol,
        shift_ms=baseline.shift_ms,
        min_distance=baseline.min_distance,
    )


def fit_baseline(
    train: Sequence[ClipLabels],
    val: Sequence[ClipLabels],
    mean: np.ndarray,
    std: np.ndarray,
    min_distance: int = 2,
) -> Baseline:
    """The baseline's symbol and shift from the training clips, its threshold from the val clips."""
    scores = [motion_score(standardized(c.x, mean, std)) for c in train if c.x is not None]
    fitted = [c for c in train if c.x is not None]
    baseline = Baseline(
        symbol=most_frequent(c.symbols for c in train),
        shift_ms=baseline_shift(scores, [c.t_ms for c in fitted], [c.onsets_ms for c in fitted]),
        min_distance=min_distance,
    )
    if val:
        f1 = {}
        for t in THRESHOLDS:
            candidate = Baseline(baseline.symbol, baseline.shift_ms, t, min_distance)
            f1[t] = pooled_f1(val, [baseline_clip(c, candidate, mean, std) for c in val], mode="timing")
        baseline.threshold = _best(f1)
    return baseline


# A split's evaluation.

SYSTEMS = ("model", "consistency", "baseline")


@dataclass
class ClipResult:
    """One clip's predicted sequences (by system) and their scores, and the model's P(onset)."""

    clip: ClipLabels
    decoded: dict[str, Decoded]
    scores: dict[str, ClipScore]
    onset: np.ndarray


@dataclass
class Evaluation:
    split: str
    head: str
    threshold: float | None
    consistency: bool
    results: list[ClipResult] = field(default_factory=list)
    calibration: Any = None  # a calibrated run's report.CalibrationResult (every mode on the same clips)

    @property
    def systems(self) -> list[str]:
        return [s for s in SYSTEMS if s != "consistency" or self.consistency]

    def scores(self, system: str) -> list[ClipScore]:
        return [r.scores[system] for r in self.results]

    def summary(self) -> dict[str, Any]:
        """Per system: the split's aggregate, and the aggregates by segment and by TPS bucket."""
        return {
            system: {
                "all": aggregate(self.scores(system)),
                "segment": grouped(self.scores(system), "segment"),
                "bucket": grouped(self.scores(system), "bucket"),
            }
            for system in self.systems
        }


def evaluate_outputs(
    clips: Sequence[ClipLabels],
    probs: Sequence[np.ndarray],
    *,
    head: str,
    threshold: float,
    baseline: Baseline,
    mean: np.ndarray,
    std: np.ndarray,
    decode: DecodeConfig | None = None,
    consistency: bool = False,
    split: str = "",
) -> Evaluation:
    """Every clip decoded and scored: the model, the model through the consistency pass (with
    `consistency`, the normalization's thresholds) and the baseline."""
    evaluation = Evaluation(split, head, threshold if head == "perframe" else None, consistency)
    for clip, p in zip(clips, probs, strict=True):
        decoded = {"model": decode_clip(head, p, clip.t_ms, threshold, decode)}
        if consistency:
            decoded["consistency"] = consistent(decoded["model"])
        decoded["baseline"] = baseline_clip(clip, baseline, mean, std)
        scores = {
            system: score_clip(
                clip.symbols,
                clip.onsets_ms,
                d.symbols,
                d.times,
                segment=clip.segment,
                tps=clip.tps,
                facelets=clip.facelets,
            )
            for system, d in decoded.items()
        }
        evaluation.results.append(ClipResult(clip, decoded, scores, onset_curve(p)))
    return evaluation


def reference_replays(clips: Sequence[ClipLabels]) -> tuple[int, int]:
    """How many solve clips' reference sequences replay to solved, of those with a scrambled state: the
    simulator's and the labels' check (every one should)."""
    solve = [c for c in clips if c.segment == "solve" and c.facelets]
    return sum(replay(c.facelets, names(c.symbols)) for c in solve), len(solve)
