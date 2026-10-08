"""A run's report on a split: `report.md` (the tables), `plots/*.png` (matplotlib, Agg), `predictions.parquet`
(one row per clip: the reference and each system's sequence with onset times, and the clip's numbers) and
`metrics.json` (the aggregates, and the model's confusions from the predictions table); NumPy, polars and
matplotlib only. The files of a split other than `test` carry its name: `report-val.md`, `plots-val/`, …"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .config import RunConfig
from .evaluate import Evaluation, reference_replays
from .metrics import (
    CONFUSION_TOLERANCE,
    CURVE_TOLERANCES,
    MODES,
    TOLERANCES,
    ClipScore,
    Confusions,
    confusion_kind,
    names,
)
from .moves import FACES, SYMBOLS

LABELS = {"model": "model", "consistency": "model + consistency", "baseline": "baseline"}
# The reference palette's first three categorical slots (light), validated together; text in ink tokens.
COLORS = {"model": "#2a78d6", "baseline": "#eb6834", "consistency": "#1baf7a"}
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
PLOT_SECONDS = 8.0


def _suffix(split: str) -> str:
    return "" if split == "test" else f"-{split}"


def _num(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}f}"


def _share(value: float) -> str:
    return "–" if value is None or math.isnan(value) else f"{100 * value:.0f}%"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    return value


def read_log(path: Path) -> list[dict[str, float]]:
    """The run's `log.csv` as numbers (NaN for an empty cell)."""
    if not path.is_file():
        return []
    with open(path, newline="") as handle:
        return [
            {k: float(v) if v not in ("", None) else math.nan for k, v in row.items()}
            for row in csv.DictReader(handle)
        ]


# The tables.


def _headline(summary: dict[str, Any], systems: Sequence[str]) -> list[str]:
    onset_keys = [f"{mode}@{t}" for t in TOLERANCES for mode in MODES]
    means = [
        [
            LABELS[s],
            summary[s]["all"]["clips"],
            _num(summary[s]["all"]["wer"]),
            *(_num(summary[s]["all"]["onsets"][k]["f1"]) for k in onset_keys),
            _share(summary[s]["all"]["exact"]),
            _share(summary[s]["all"]["replay"]),
        ]
        for s in systems
    ]
    pooled = [
        [
            LABELS[s],
            summary[s]["all"]["reference"],
            summary[s]["all"]["predicted"],
            _num(summary[s]["all"]["werPooled"]),
            *(_num(summary[s]["all"]["onsets"][k]["f1Pooled"]) for k in onset_keys),
            _num(summary[s]["all"]["onsets"]["symbol@50"]["precision"]),
            _num(summary[s]["all"]["onsets"]["symbol@50"]["recall"]),
        ]
        for s in systems
    ]
    f1_heads = [f"F1@{t} {mode}" for t in TOLERANCES for mode in MODES]
    return [
        "Means over the clips (exact: the share of clips with WER 0; replay: of the solve clips, the share "
        "whose predicted sequence solves the scrambled cube):",
        "",
        *_table(["system", "clips", "WER", *f1_heads, "exact", "replay"], means),
        "",
        "Pooled over the clips (the edits over all the reference symbols; the F1 of all the matches):",
        "",
        *_table(["system", "reference", "predicted", "WER", *f1_heads, "P@50 symbol", "R@50 symbol"], pooled),
    ]


def _by(summary: dict[str, Any], systems: Sequence[str], key: str, title: str) -> list[str]:
    groups = sorted({g for s in systems for g in summary[s][key]})
    rows = []
    for group in groups:
        for s in systems:
            agg = summary[s][key].get(group)
            if agg is None:
                continue
            rows.append(
                [
                    group,
                    LABELS[s],
                    agg["clips"],
                    agg["reference"],
                    _num(agg["wer"]),
                    _num(agg["werPooled"]),
                    _num(agg["onsets"]["symbol@25"]["f1Pooled"]),
                    _num(agg["onsets"]["symbol@50"]["f1Pooled"]),
                    _num(agg["onsets"]["timing@50"]["f1Pooled"]),
                    _share(agg["exact"]),
                    _share(agg["replay"]),
                ]
            )
    headers = [
        title,
        "system",
        "clips",
        "reference",
        "WER",
        "WER pooled",
        "F1@25 symbol",
        "F1@50 symbol",
        "F1@50 timing",
        "exact",
        "replay",
    ]
    return _table(headers, rows)


def _consistency(summary: dict[str, Any]) -> list[str]:
    raw, merged = summary["model"]["all"], summary["consistency"]["all"]
    rows = []
    for label, key, getter in (
        ("WER (mean)", "wer", lambda a: a["wer"]),
        ("WER (pooled)", "werPooled", lambda a: a["werPooled"]),
        ("F1@50 symbol (pooled)", "f1", lambda a: a["onsets"]["symbol@50"]["f1Pooled"]),
        ("F1@25 symbol (pooled)", "f1", lambda a: a["onsets"]["symbol@25"]["f1Pooled"]),
        ("predicted symbols", "n", lambda a: a["predicted"]),
        ("exact", "share", lambda a: a["exact"]),
        ("replay", "share", lambda a: a["replay"]),
    ):
        a, b = getter(raw), getter(merged)
        fmt = _share if key in ("share",) else _num
        delta = b - a if not (isinstance(a, float) and math.isnan(a)) else math.nan
        rows.append([label, fmt(a), fmt(b), fmt(delta) if key != "share" else _share_delta(delta)])
    return [
        "The consistency pass merges adjacent predictions by the normalization's rules (a double: two equal "
        "quarter turns under the double threshold; a slice: opposite faces under the slice threshold) and "
        "drops adjacent cancelling pairs (`R R'`, `R2 R2`).",
        "",
        *_table(["", "model", "model + consistency", "change"], rows),
    ]


def _share_delta(delta: float) -> str:
    return "–" if math.isnan(delta) else f"{100 * delta:+.0f} points"


def _curve_table(summary: dict[str, Any], systems: Sequence[str]) -> list[str]:
    tolerances = [t for t in CURVE_TOLERANCES if t % 10 == 0]
    rows = []
    for s in systems:
        for mode in MODES:
            onsets = summary[s]["all"]["onsets"]
            rows.append([LABELS[s], mode, *(_num(onsets[f"{mode}@{t}"]["f1Pooled"]) for t in tolerances)])
    return _table(["system", "match", *(f"±{t} ms" for t in tolerances)], rows)


def _counts_table(counts: dict[str, Any]) -> list[str]:
    splits = [split for split in ("train", "val", "test") if counts.get(split)]
    # The gyro's coverage, when the run read it: the clips with an orientation, and the share of the frames.
    gyro = any(counts[split].get("gyroClips") is not None for split in splits)
    rows = []
    for split in splits:
        c = counts[split]
        skipped = ", ".join(f"{n} {reason}" for reason, n in c["skipped"].items()) or "–"
        segments = c.get("bySegment", {})
        counted = (
            c["clips"],
            segments.get("scramble", 0),
            segments.get("solve", 0),
            c["frames"],
            c["symbols"],
        )
        row = [
            split,
            *(_num(int(v)) for v in counted),
            skipped,
            _num(int(c.get("unusable", 0))),
            _num(int(c.get("collisions", 0))),
        ]
        if gyro:
            clips, frames = c.get("gyroClips"), c.get("gyroFrames")
            row.append(
                "–"
                if clips is None or frames is None
                else f"{_num(int(clips))} clips, {_share(frames / c['frames'] if c['frames'] else math.nan)}"
                " of the frames"
            )
        rows.append(row)
    headers = [
        "split",
        "clips",
        "scramble",
        "solve",
        "frames",
        "reference symbols",
        "skipped",
        "unusable",
        "onsets without a frame",
    ]
    return _table([*headers, "with the gyro"] if gyro else headers, rows)


# The confusions.


def camera_key(camera: str, lag_ms: float | None) -> str:
    """A camera as the confusions count it: its label and whether its clips had a lag (a sync check)."""
    return f"{camera} ({'no lag' if lag_ms is None else 'lag'})"


def confusions_of(
    predictions: pl.DataFrame, system: str = "model", tolerance: float = CONFUSION_TOLERANCE
) -> Confusions:
    """The confusions of a system's sequences in a predictions table (`predictions_frame`, or a run's
    `predictions.parquet`), by camera and lag."""
    confusions = Confusions(tolerance=tolerance)
    for row in predictions.iter_rows(named=True):
        confusions.add(
            row["referenceSymbols"] or [],
            row["referenceOnsetMs"] or [],
            row[f"{system}Symbols"] or [],
            row[f"{system}OnsetMs"] or [],
            camera_key(row["camera"], row["lagMs"]),
        )
    return confusions


def _confusions(confusions: Confusions) -> list[str]:
    matched, reference = confusions.matched, confusions.reference
    kinds = confusions.kinds()
    lines = [
        f"The model's onsets matched one to one to the reference's within ±{confusions.tolerance:g} ms on "
        f"timing (the matches of F1@{confusions.tolerance:g} timing), and the symbol it gave each: "
        f"{_num(matched)} of {_num(reference)} reference onsets matched "
        f"({_share(matched / reference if reference else math.nan)}).",
        "",
        *_table(
            ["kind", "onsets", "share"],
            [[kind, _num(n), _share(n / matched if matched else math.nan)] for kind, n in kinds.items()],
        ),
        "",
        "Same face: `R'` or `R2` for `R`; opposite face: `L` for `R`, or a slice for another slice (the "
        "slices are a family of their own); other face: the rest, a slice for a face turn among them.",
        "",
    ]
    per_symbol = confusions.per_symbol()
    if per_symbol:
        rows = []
        for face in (*FACES, "M", "S", "E"):
            cells = []
            for suffix in ("", "'", "2"):
                right, n = per_symbol.get(face + suffix, (0, 0))
                cells.append(f"{_share(right / n)} ({_num(n)})" if n else "–")
            if any(cell != "–" for cell in cells):
                rows.append([f"`{face}`", *cells])
        lines += [
            "The share right at each reference symbol's matched onsets (and how many were matched):",
            "",
            *_table(["face", "X", "X'", "X2"], rows),
            "",
        ]
    top = confusions.top()
    if top:
        lines += [
            f"The {len(top)} most frequent confusions:",
            "",
            *_table(
                ["reference", "predicted", "onsets", "kind"],
                [[f"`{r}`", f"`{p}`", _num(n), confusion_kind(r, p)] for r, p, n in top],
            ),
            "",
        ]
    rows = []
    for camera, (right, wrong, unmatched) in sorted(confusions.cameras.items()):
        rows.append(
            [
                camera,
                _num(right),
                _num(wrong),
                _num(unmatched),
                _share(right / (right + wrong) if right + wrong else math.nan),
                _share(
                    (right + wrong) / (right + wrong + unmatched) if right + wrong + unmatched else math.nan
                ),
            ]
        )
    lines += [
        "By camera (lag: the clips have their sync check's lag; no lag: unsynced, taken as 0):",
        "",
        *_table(["camera", "right", "wrong", "unmatched", "right at matched", "matched"], rows),
    ]
    return lines


# The plots.


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
            "font.size": 9,
        }
    )
    return plt


def plot_log(rows: list[dict[str, float]], head: str, path: Path) -> Path | None:
    """The training and val loss by epoch, and the val metric the run selects on (two panels)."""
    if not rows:
        return None
    plt = _pyplot()
    epochs = [r["epoch"] for r in rows]
    fig, (loss_ax, metric_ax) = plt.subplots(1, 2, figsize=(9, 3.2))
    loss_ax.plot(epochs, [r["train_loss"] for r in rows], color=COLORS["model"], label="train")
    loss_ax.plot(epochs, [r["val_loss"] for r in rows], color=COLORS["baseline"], label="val")
    loss_ax.set(title="Loss by epoch", xlabel="epoch", ylabel="loss")
    loss_ax.legend(loc="upper right")
    key, name = ("val_f1_50", "Val F1@50 (symbol)") if head == "perframe" else ("val_wer", "Val WER")
    values = [r[key] for r in rows]
    metric_ax.plot(epochs, values, color=COLORS["model"])
    finite = [(e, v) for e, v in zip(epochs, values, strict=True) if not math.isnan(v)]
    if finite:
        best = (max if head == "perframe" else min)(finite, key=lambda ev: ev[1])
        metric_ax.plot(
            *best, "o", color=COLORS["model"], markersize=7, markeredgecolor=SURFACE, markeredgewidth=2
        )
        right = best[0] > (epochs[0] + epochs[-1]) / 2
        high = head == "perframe"  # the best F1 is the curve's top, the best WER its bottom
        metric_ax.annotate(
            f"best {best[1]:.3f} (epoch {best[0]:.0f})",
            best,
            textcoords="offset points",
            xytext=(-8 if right else 8, -10 if high else 10),
            ha="right" if right else "left",
            va="top" if high else "bottom",
            color=INK,
            fontsize=8,
        )
    metric_ax.set(title=name + " by epoch", xlabel="epoch", ylabel=name.split(" (")[0].replace("Val ", ""))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_tolerance(summary: dict[str, Any], systems: Sequence[str], path: Path) -> Path:
    """Pooled F1 against the tolerance (10–100 ms), symbol and timing, per system."""
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, mode in zip(axes, MODES, strict=True):
        for s in systems:
            onsets = summary[s]["all"]["onsets"]
            ax.plot(
                CURVE_TOLERANCES,
                [onsets[f"{mode}@{t}"]["f1Pooled"] for t in CURVE_TOLERANCES],
                color=COLORS[s],
                label=LABELS[s],
                zorder=3
                if s == "model"
                else 2,  # the model on top where the consistency pass changes nothing
            )
        for t in TOLERANCES:
            ax.axvline(t, color=GRID, linewidth=1.2, zorder=0)
        ax.set(title=f"F1 ({mode}) against the tolerance", xlabel="tolerance (± ms)", ylim=(0, 1.02))
    axes[0].set_ylabel("F1 (pooled)")
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_wer_by_tps(summary: dict[str, Any], systems: Sequence[str], path: Path) -> Path:
    """The mean WER per TPS bucket, the systems side by side (each bar's value at its tip), the clips counted
    under each bucket."""
    plt = _pyplot()
    buckets = sorted({b for s in systems for b in summary[s]["bucket"]})
    fig, ax = plt.subplots(figsize=(2.0 + 1.5 * len(buckets), 3.4))
    width = 0.2
    x = np.arange(len(buckets))
    top = 1.0
    for k, s in enumerate(systems):
        values = [summary[s]["bucket"].get(b, {}).get("wer", math.nan) for b in buckets]
        where = x + (k - (len(systems) - 1) / 2) * (width + 0.03)
        ax.bar(where, values, width, color=COLORS[s], label=LABELS[s], zorder=2)
        for xi, v in zip(where, values, strict=True):
            if not math.isnan(v):
                ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=INK_2)
                top = max(top, v)
    clips = [summary[systems[0]]["bucket"].get(b, {}).get("clips", 0) for b in buckets]
    ax.set_xticks(x, [f"{b}\n{n} clips" for b, n in zip(buckets, clips, strict=True)])
    ax.set(title="WER by TPS bucket (mean over clips)", xlabel="the attempt's TPS", ylabel="WER")
    ax.set_xlim(-0.7, len(buckets) - 0.3)
    ax.set_ylim(0, top * 1.25)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncols=len(systems))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def pick_clip(evaluation: Evaluation) -> int:
    """The clip to draw: the solve clip with the median model WER (the first of equals), else the first."""
    results = evaluation.results
    solve = [i for i, r in enumerate(results) if r.clip.segment == "solve" and r.scores["model"].reference]
    if not solve:
        return 0
    ordered = sorted(solve, key=lambda i: (results[i].scores["model"].wer, i))
    return ordered[(len(ordered) - 1) // 2]


def plot_clip(evaluation: Evaluation, index: int, threshold: float | None, path: Path) -> Path:
    """One clip's P(onset) over its first seconds, its reference onsets (vertical lines, their symbols on two
    staggered rows above the plot) and the model's predicted onsets (dots, their symbols above them)."""
    plt = _pyplot()
    result = evaluation.results[index]
    clip = result.clip
    t0 = clip.t_ms[0]
    seconds = (clip.t_ms - t0) / 1000.0
    end = min(seconds[-1], PLOT_SECONDS)
    shown = seconds <= end
    curve = "P(onset)" if evaluation.head == "perframe" else "1 − P(blank)"
    fig, ax = plt.subplots(figsize=(11, 3.6))
    above = ax.get_xaxis_transform()  # x in seconds, y in the axes' height
    for k, (onset, symbol) in enumerate(zip(clip.onsets_ms, clip.symbols, strict=True)):
        s = (onset - t0) / 1000.0
        if s <= end:
            ax.axvline(s, color=MUTED, linewidth=1, zorder=1)
            y = 1.02 + 0.06 * (k % 2)
            ax.text(
                s, y, SYMBOLS[int(symbol)], transform=above, ha="center", va="bottom", fontsize=7, color=INK_2
            )
    ax.plot(seconds[shown], result.onset[shown], color=COLORS["model"], linewidth=1.6, zorder=2, label=curve)
    decoded = result.decoded["model"]
    keep = (decoded.times - t0) / 1000.0 <= end
    xs = (decoded.times[keep] - t0) / 1000.0
    ys = result.onset[decoded.frames[keep]]
    ax.plot(
        xs,
        ys,
        "o",
        color=COLORS["model"],
        markersize=7,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
        zorder=3,
        label="predicted onset",
    )
    for k, (x, y, symbol) in enumerate(zip(xs, ys, decoded.symbols[keep], strict=True)):
        lift = 0.035 + 0.05 * (k % 2)
        ax.text(x, y + lift, SYMBOLS[int(symbol)], ha="center", va="bottom", fontsize=7, color=INK)
    if threshold is not None:
        ax.axhline(
            threshold,
            color=INK_2,
            linewidth=0.8,
            linestyle=(0, (4, 3)),
            zorder=0,
            label=f"threshold {threshold:g}",
        )
    ax.plot([], [], color=MUTED, linewidth=1, label="reference onset (symbol above)")
    ax.set(xlim=(0, end), ylim=(0, 1.15), xlabel="seconds from the first kept frame", ylabel=curve)
    tps = "–" if clip.tps is None else f"{clip.tps:.2f}"
    label = f"attempt {clip.ref.attempt}, {clip.ref.camera}.{clip.ref.segment}, TPS {tps}"
    ax.set_title(f"One {evaluation.split} clip ({label}): WER {result.scores['model'].wer:.2f}", pad=28)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, -0.17), ncols=4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# The files.


def predictions_frame(evaluation: Evaluation) -> pl.DataFrame:
    """One row per clip: the clip, the reference and each system's symbols and onset times (ms on the frames'
    timeline), and each system's numbers."""
    rows = []
    for r in evaluation.results:
        c = r.clip
        row: dict[str, Any] = {
            "sessionId": c.ref.session,
            "attemptIndex": c.ref.attempt,
            "camera": c.ref.camera,
            "segment": c.ref.segment,
            "split": evaluation.split,
            "tps": c.tps,
            "bucket": r.scores["model"].bucket,
            "frames": len(c),
            "stride": c.stride,
            "lagMs": c.lag_ms,
            "referenceSymbols": names(c.symbols),
            "referenceOnsetMs": [float(t) for t in c.onsets_ms],
        }
        for system, d in r.decoded.items():
            score: ClipScore = r.scores[system]
            row[f"{system}Symbols"] = names(d.symbols)
            row[f"{system}OnsetMs"] = [float(t) for t in d.times]
            row[f"{system}Wer"] = score.wer
            for t in TOLERANCES:
                for mode in MODES:
                    row[f"{system}F1At{t}{mode.capitalize()}"] = score.f1(mode, t)
            row[f"{system}Exact"] = score.exact
            row[f"{system}Replay"] = score.replay
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def write_report(
    run: str | Path,
    evaluation: Evaluation,
    *,
    config: RunConfig,
    record: dict[str, Any],
    counts: dict[str, Any],
    checkpoint: str = "best",
) -> list[Path]:
    """Writes the split's report, plots, predictions and metrics into the run folder; returns their paths."""
    run = Path(run)
    suffix = _suffix(evaluation.split)
    plots = run / f"plots{suffix}"
    plots.mkdir(parents=True, exist_ok=True)
    summary = evaluation.summary()
    systems = evaluation.systems
    replayed, solve_clips = reference_replays([r.clip for r in evaluation.results])
    log_rows = read_log(run / "log.csv")

    written = []
    figures = {}
    loss = plot_log(log_rows, evaluation.head, plots / "loss.png")
    if loss:
        figures["The loss and the val metric by epoch"] = loss
    figures["F1 against the tolerance"] = plot_tolerance(summary, systems, plots / "f1-tolerance.png")
    figures["WER by TPS bucket"] = plot_wer_by_tps(summary, systems, plots / "wer-tps.png")
    if evaluation.results:
        index = pick_clip(evaluation)
        figures["One clip's P(onset) over its reference onsets"] = plot_clip(
            evaluation, index, evaluation.threshold, plots / "onsets.png"
        )
    written += list(figures.values())

    predictions = run / f"predictions{suffix}.parquet"
    frame = predictions_frame(evaluation)
    frame.write_parquet(predictions)
    written.append(predictions)
    confusions = confusions_of(frame)

    metrics_path = run / f"metrics{suffix}.json"
    metrics = {
        "run": config.name,
        "split": evaluation.split,
        "head": evaluation.head,
        "body": config.model.body,
        "encoder": config.paths.encoder,
        "fps": config.data.fps,
        "inputs": config.data.inputs,
        "requireGyro": config.data.require_gyro,
        "checkpoint": {"name": checkpoint, "epoch": record.get("epoch"), "metrics": record.get("metrics")},
        "threshold": evaluation.threshold,
        "baseline": record.get("baseline"),
        "consistency": evaluation.consistency,
        "referenceReplays": {"replayed": replayed, "solveClips": solve_clips},
        "counts": counts,
        "systems": summary,
        "confusions": {"system": "model", **confusions.to_json()},
    }
    metrics_path.write_text(json.dumps(_jsonable(metrics), indent=2) + "\n")
    written.append(metrics_path)

    report_path = run / f"report{suffix}.md"
    report_path.write_text(
        "\n".join(
            _report_lines(
                evaluation,
                summary,
                systems,
                config,
                record,
                counts,
                checkpoint,
                replayed,
                solve_clips,
                figures,
                run,
                confusions,
            )
        )
        + "\n"
    )
    written.append(report_path)
    return written


def _report_lines(
    evaluation: Evaluation,
    summary: dict[str, Any],
    systems: Sequence[str],
    config: RunConfig,
    record: dict[str, Any],
    counts: dict[str, Any],
    checkpoint: str,
    replayed: int,
    solve_clips: int,
    figures: dict[str, Path],
    run: Path,
    confusions: Confusions,
) -> list[str]:
    baseline = record.get("baseline") or {}
    metrics = record.get("metrics") or {}
    rate = (
        "every frame (the clips' own rate, about 30 fps)"
        if not config.data.fps
        else f"{config.data.fps:g} fps (every {_strides(evaluation)} frame of the clips)"
    )
    selection = "F1@50 (symbol)" if evaluation.head == "perframe" else "WER"
    head_line = (
        f"per frame, peaks of P(onset) at or above {evaluation.threshold:g} (chosen on val by F1@50, "
        f"symbol), at least {config.decode.min_distance} frames apart, the symbol summed over the peak and "
        f"{config.decode.neighbours} frame on each side"
        if evaluation.head == "perframe"
        else "CTC, greedy (repeats collapsed, blanks dropped; an onset at the first frame of its spike)"
    )
    baseline_symbol = SYMBOLS[int(baseline.get("symbol", 0))]
    inputs = (
        "the features and the gyro's 9 channels (the orientation, its change since the previous kept frame, "
        "the presence flag)"
        if config.data.inputs == "features+gyro"
        else "the features"
    )
    if config.data.require_gyro:
        inputs += "; the clips without a gyro skipped"
    lines = [
        f"# {config.name}: {evaluation.split}",
        "",
        f"`cubetrace-ml evaluate --run {run} --split {evaluation.split}"
        f"{' --consistency' if evaluation.consistency else ''}` with `{checkpoint}.pt`.",
        "",
        *_table(
            ["", ""],
            [
                ["model", f"{config.model.body} body, {evaluation.head} head"],
                ["features", f"{config.paths.encoder}, {rate}"],
                ["inputs", inputs],
                [
                    "checkpoint",
                    f"{checkpoint}.pt, epoch {record.get('epoch')} (best on val {selection}; val F1@50 "
                    f"{_num(metrics.get('valF1At50'))}, val WER {_num(metrics.get('valWer'))})",
                ],
                ["decoding", head_line],
                [
                    "baseline",
                    f"the peaks of the feature-difference norm at or above {baseline.get('threshold', '–')} "
                    f"(chosen on val by F1@50, timing), moved by {_num(-baseline.get('shiftMs', 0.0), 1)} "
                    f"ms (train's median offset), every onset `{baseline_symbol}` (train's most frequent "
                    "symbol)",
                ],
                [
                    "the references",
                    f"{replayed} of {solve_clips} solve clips' reference sequences replay to solved "
                    "(the simulator's check)",
                ],
            ],
        ),
        "",
        "## The data",
        "",
        *_counts_table(counts),
        "",
        f"## {evaluation.split}: the model against the baseline",
        "",
        *_headline(summary, systems),
        "",
        "## By segment",
        "",
        *_by(summary, systems, "segment", "segment"),
        "",
        "## By TPS bucket",
        "",
        "The attempt's `result.tps` in bins of 0.5 (a scramble clip takes its attempt's).",
        "",
        *_by(summary, systems, "bucket", "TPS"),
        "",
    ]
    if evaluation.consistency:
        lines += ["## The consistency pass", "", *_consistency(summary), ""]
    lines += ["## Confusions", "", *_confusions(confusions), ""]
    lines += [
        "## F1 against the tolerance",
        "",
        "Pooled F1 by the matching's tolerance:",
        "",
        *_curve_table(summary, systems),
        "",
        "## Plots",
        "",
    ]
    for title, path in figures.items():
        lines += [f"![{title}]({path.parent.name}/{path.name})", ""]
    return lines


def _strides(evaluation: Evaluation) -> str:
    strides = sorted({r.clip.stride for r in evaluation.results})
    ordinal = {1: "", 2: "2nd", 3: "3rd"}
    return " or ".join(ordinal.get(s, f"{s}th") or "" for s in strides) or "–"
