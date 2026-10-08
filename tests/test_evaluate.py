"""The evaluation without a model: the threshold chosen on val, the baseline fitted on train and val, the
scores of known outputs, the report's files (tables, plots, predictions, metrics) and its confusions."""

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from PIL import Image

from cubetrace_ml.config import RunConfig
from cubetrace_ml.dataset import Dataset
from cubetrace_ml.decode import Decoded, most_frequent
from cubetrace_ml.evaluate import (
    Baseline,
    choose_threshold,
    evaluate_outputs,
    fit_baseline,
    pooled,
    reference_replays,
)
from cubetrace_ml.labels import ClipLabels, load_split
from cubetrace_ml.manifest import build_tables
from cubetrace_ml.moves import INDEX, SYMBOLS
from cubetrace_ml.report import CalibrationResult, confusions_of, pick_clip, write_report
from factory import build_feature_dataset


@pytest.fixture(scope="module")
def splits(tmp_path_factory: pytest.TempPathFactory) -> dict[str, list[ClipLabels]]:
    base = tmp_path_factory.mktemp("evaluate")
    root, features = base / "data", base / "features"
    build_feature_dataset(root, features, sessions=5, attempts=2, dim=8, doubles=0.2)
    dataset = Dataset(root)
    manifest = build_tables(dataset, video="none").clips
    return {s: load_split(dataset, manifest, s, features, "synthetic")[0] for s in ("train", "val", "test")}


def oracle(clip: ClipLabels, confidence: float = 0.8) -> np.ndarray:
    """Probabilities that put `confidence` on each target frame's class."""
    p = np.zeros((len(clip), 25))
    p[:, 0] = 1.0
    for k in np.flatnonzero(clip.target):
        p[k, 0] = 1.0 - confidence
        p[k, clip.target[k]] = confidence
    return p


def test_the_threshold_is_chosen_on_val(splits) -> None:
    val = splits["val"]
    threshold, f1, wer = choose_threshold([oracle(c, 0.6) for c in val], val)
    assert (
        f1 == 1.0 and wer == 0.0 and threshold == 0.5
    )  # every threshold up to 0.6 is perfect: the nearest 0.5
    threshold, f1, _ = choose_threshold([oracle(c, 0.3) for c in val], val)
    assert threshold == 0.3 and f1 == 1.0
    f1, wer = pooled(val, [Decoded.empty() for _ in val])
    assert f1 == 0.0 and wer == 1.0


def test_the_baseline_is_fitted_on_train_and_val(splits) -> None:
    train, val = splits["train"], splits["val"]
    dim = train[0].x.shape[1]
    baseline = fit_baseline(train, val, np.zeros(dim, np.float32), np.ones(dim, np.float32))
    assert baseline.symbol == most_frequent(c.symbols for c in train)
    # The planted signal starts on the frame nearest each onset: the motion's peaks are there.
    assert abs(baseline.shift_ms) <= 17
    assert 0.1 <= baseline.threshold <= 0.9
    assert Baseline.from_json(baseline.to_json()) == baseline


def test_known_outputs_are_scored(splits, tmp_path: Path) -> None:
    test = splits["test"]
    dim = test[0].x.shape[1]
    mean, std = np.zeros(dim, np.float32), np.ones(dim, np.float32)
    evaluation = evaluate_outputs(
        test,
        [oracle(c) for c in test],
        head="perframe",
        threshold=0.5,
        baseline=Baseline(symbol=3, threshold=0.4),
        mean=mean,
        std=std,
        consistency=True,
        split="test",
    )
    summary = evaluation.summary()
    assert evaluation.systems == ["model", "consistency", "baseline"]
    model = summary["model"]["all"]
    assert model["clips"] == len(test) and model["wer"] == 0.0 and model["exact"] == 1.0
    assert model["onsets"]["symbol@50"]["f1Pooled"] == 1.0 and model["onsets"]["timing@25"]["f1Pooled"] == 1.0
    assert model["replay"] == 1.0 and model["replayClips"] == sum(c.segment == "solve" for c in test)
    assert summary["baseline"]["all"]["wer"] > 0.5 and summary["baseline"]["all"]["replay"] == 0.0
    assert set(summary["model"]["segment"]) == {"scramble", "solve"}
    assert reference_replays(test) == (model["replayClips"], model["replayClips"])
    # The consistency pass merges no correct double (they are one symbol already) and drops nothing here.
    assert summary["consistency"]["all"]["wer"] == 0.0

    config = RunConfig(name="oracle")
    config.paths.encoder = "synthetic"
    (tmp_path / "log.csv").write_text(
        "epoch,train_loss,val_loss,val_f1_50,val_wer,lr,seconds,threshold\n"
        "1,2.0,1.9,0.5,0.6,0.001,1.0,0.5\n2,1.0,1.1,0.7,0.4,0.0005,1.0,0.45\n"
    )
    record = {"epoch": 2, "metrics": {"valF1At50": 0.7, "valWer": 0.4}, "baseline": Baseline(3).to_json()}
    counts = {
        "test": {"clips": len(test), "frames": 1, "symbols": 2, "skipped": {}, "bySegment": {"solve": 1}}
    }
    written = write_report(tmp_path, evaluation, config=config, record=record, counts=counts)
    names = sorted(p.relative_to(tmp_path).as_posix() for p in written)
    assert names == [
        "metrics.json",
        "plots/f1-tolerance.png",
        "plots/loss.png",
        "plots/onsets.png",
        "plots/wer-tps.png",
        "predictions.parquet",
        "report.md",
    ]
    for png in tmp_path.glob("plots/*.png"):
        assert Image.open(png).size[0] > 400
    report = (tmp_path / "report.md").read_text()
    for heading in (
        "# oracle: test",
        "## The data",
        "## By segment",
        "## By TPS bucket",
        "## The consistency pass",
    ):
        assert heading in report
    assert "![F1 against the tolerance](plots/f1-tolerance.png)" in report
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["systems"]["model"]["all"]["wer"] == 0.0 and metrics["referenceReplays"]["replayed"] >= 1
    assert "NaN" not in (tmp_path / "metrics.json").read_text()
    frame = pl.read_parquet(tmp_path / "predictions.parquet")
    assert len(frame) == len(test)
    assert frame["referenceSymbols"].to_list() == frame["modelSymbols"].to_list()
    assert {"baselineOnsetMs", "consistencyWer", "modelF1At50Symbol", "modelReplay"} <= set(frame.columns)
    assert 0 <= pick_clip(evaluation) < len(test) and test[pick_clip(evaluation)].segment == "solve"


def test_a_split_other_than_test_has_its_own_files(splits, tmp_path: Path) -> None:
    val = splits["val"]
    dim = val[0].x.shape[1]
    evaluation = evaluate_outputs(
        val,
        [oracle(c) for c in val],
        head="ctc",
        threshold=0.5,
        baseline=Baseline(),
        mean=np.zeros(dim, np.float32),
        std=np.ones(dim, np.float32),
        split="val",
    )
    assert evaluation.threshold is None and evaluation.systems == ["model", "baseline"]
    written = write_report(tmp_path, evaluation, config=RunConfig(), record={}, counts={})
    assert {p.name for p in written} >= {"report-val.md", "metrics-val.json", "predictions-val.parquet"}
    assert (tmp_path / "plots-val" / "onsets.png").is_file() and not (
        tmp_path / "plots-val" / "loss.png"
    ).exists()
    # CTC's greedy decoding of the oracle: one emission per onset frame, at its time.
    model = evaluation.summary()["model"]["all"]
    assert model["wer"] == 0.0 and not math.isnan(model["onsets"]["timing@25"]["f1Pooled"])


def test_the_confusions_of_known_predictions(splits, tmp_path: Path) -> None:
    test = splits["test"]
    dim = test[0].x.shape[1]
    # The oracle's outputs with every F given as B and every U as U' (the onsets where they are).
    swap = {INDEX["F"]: INDEX["B"], INDEX["U"]: INDEX["U'"]}
    probs = []
    for clip in test:
        p = oracle(clip)
        for k in np.flatnonzero(clip.target):
            symbol = int(clip.target[k]) - 1
            if symbol in swap:
                p[k, symbol + 1], p[k, swap[symbol] + 1] = 0.0, 0.8
        probs.append(p)
    evaluation = evaluate_outputs(
        test,
        probs,
        head="perframe",
        threshold=0.5,
        baseline=Baseline(),
        mean=np.zeros(dim, np.float32),
        std=np.ones(dim, np.float32),
        consistency=True,
        split="test",
    )
    reference = Counter(SYMBOLS[s] for c in test for s in c.symbols)
    total, f, u = sum(reference.values()), reference["F"], reference["U"]
    assert f and u and f != u
    counts = {
        "test": {
            "clips": len(test),
            "frames": 200,
            "symbols": total,
            "skipped": {"no-gyro": 1},
            "bySegment": {"scramble": 2, "solve": 2},
            "gyroClips": 3,
            "gyroFrames": 150,
        }
    }
    config = RunConfig(name="swapped")
    config.data.inputs, config.data.require_gyro = "features+gyro", True
    write_report(tmp_path, evaluation, config=config, record={}, counts=counts)
    doc = json.loads((tmp_path / "metrics.json").read_text())
    confusions = doc["confusions"]
    assert doc["inputs"] == "features+gyro" and doc["requireGyro"] is True
    assert confusions["system"] == "model" and confusions["reference"] == confusions["matched"] == total
    assert confusions["kinds"] == {
        "right": total - f - u,
        "same face, other turn": u,
        "opposite face": f,
        "other face": 0,
    }
    assert confusions["perSymbol"]["F"] == {"right": 0, "matched": f, "accuracy": 0.0}
    assert confusions["perSymbol"]["R"]["accuracy"] == 1.0
    assert [(t["reference"], t["predicted"]) for t in confusions["top"]] == sorted(
        [("F", "B"), ("U", "U'")], key=lambda rp: -reference[rp[0]]
    )
    assert confusions["byCamera"] == {
        "laptop (lag)": {
            "right": total - f - u,
            "wrong": f + u,
            "unmatched": 0,
            "accuracy": pytest.approx((total - f - u) / total),
            "recall": 1.0,
        }
    }
    # The same from the predictions table the run wrote.
    table = confusions_of(pl.read_parquet(tmp_path / "predictions.parquet")).to_json()
    assert {"system": "model", **table} == json.loads(json.dumps(confusions))
    report = (tmp_path / "report.md").read_text()
    assert (
        report.index("## The consistency pass")
        < report.index("## Confusions")
        < report.index("## F1 against the tolerance")
    )
    assert f"{total:,} of {total:,} reference onsets matched (100%)" in report
    assert f"| opposite face | {f:,} | {100 * f / total:.0f}% |" in report
    assert f"| `F` | `B` | {f:,} | opposite face |" in report
    assert f"| `F` | 0% ({f:,}) |" in report  # the per-symbol table: F's quarter turn, none right
    assert f"| laptop (lag) | {total - f - u:,} | {f + u:,} | 0 | " in report
    assert "| inputs | the features and the gyro's 9 channels" in report
    assert "the clips without a gyro skipped" in report
    assert "with the gyro |" in report and "| 3 clips, 75% of the frames |" in report


def test_the_confusions_of_a_model_that_predicts_nothing(splits, tmp_path: Path) -> None:
    # CTC on its blank plateau: every frame blank, nothing matched; the section and metrics.json still hold.
    val = splits["val"]
    dim = val[0].x.shape[1]
    blank = []
    for clip in val:
        p = np.zeros((len(clip), 25))
        p[:, 0] = 1.0
        blank.append(p)
    evaluation = evaluate_outputs(
        val,
        blank,
        head="ctc",
        threshold=0.5,
        baseline=Baseline(),
        mean=np.zeros(dim, np.float32),
        std=np.ones(dim, np.float32),
        split="val",
    )
    write_report(tmp_path, evaluation, config=RunConfig(), record={}, counts={})
    text = (tmp_path / "metrics-val.json").read_text()
    confusions = json.loads(text)["confusions"]
    reference = sum(len(c.symbols) for c in val)
    assert confusions["matched"] == 0 and confusions["reference"] == reference and "NaN" not in text
    assert confusions["perSymbol"] == {} and confusions["top"] == [] and confusions["matrix"] == {}
    assert confusions["byCamera"]["laptop (lag)"]["accuracy"] is None
    report = (tmp_path / "report-val.md").read_text()
    assert f"0 of {reference:,} reference onsets matched (0%)" in report
    assert (
        "most frequent confusions" not in report
        and f"| laptop (lag) | 0 | 0 | {reference:,} | – | 0% |" in report
    )


def test_the_calibration_section(splits, tmp_path: Path) -> None:
    # Four ways of giving the test clips their rotation: at the identity every F reads as B (the frame turned
    # half way), the three others right.
    test = splits["test"]
    dim = test[0].x.shape[1]
    swapped = []
    for clip in test:
        p = oracle(clip)
        for k in np.flatnonzero(clip.target):
            if int(clip.target[k]) - 1 == INDEX["F"]:
                p[k, INDEX["F"] + 1], p[k, INDEX["B"] + 1] = 0.0, 0.8
        swapped.append(p)
    common = {
        "head": "perframe",
        "threshold": 0.5,
        "baseline": Baseline(),
        "mean": np.zeros(dim, np.float32),
        "std": np.ones(dim, np.float32),
        "split": "test",
    }
    evaluations = {
        "none": evaluate_outputs(test, swapped, **common),
        **{
            mode: evaluate_outputs(test, [oracle(c) for c in test], **common)
            for mode in ("pose", "scramble", "all")
        },
    }
    keys = sorted({f"{c.ref.session}/{c.ref.attempt}" for c in test})
    rows = []
    for mode, yaw in (("pose", 170.0), ("scramble", 178.0), ("all", 180.0)):
        for key in keys:
            half = np.radians(yaw) / 2
            rows.append(
                {
                    "key": key,
                    "sessionId": key.split("/")[0],
                    "attemptIndex": int(key.split("/")[1]),
                    "camera": None,
                    "mode": mode,
                    "qx": 0.0,
                    "qy": 0.0,
                    "qz": float(np.sin(half)),
                    "qw": float(np.cos(half)),
                    "yawDeg": yaw,
                    "angleToGuessDeg": yaw - 170.0,
                    "loss": math.nan if mode == "pose" else 0.1,
                    "identityLoss": math.nan if mode == "pose" else 1.2,
                    "guessLoss": math.nan if mode == "pose" else 0.3,
                    "clips": 0 if mode == "pose" else 1 if mode == "scramble" else 2,
                    "frames": 0,
                    "onsets": 0,
                    "source": "pose" if mode == "pose" else "grid",
                }
            )
    result = CalibrationResult(
        headline="scramble",
        settings={
            "kind": "attempt",
            "dof": "yaw",
            "axis": "z",
            "orientation": "matrix",
            "init": "pose",
            "by": "attempt",
        },
        evaluations=evaluations,
        rows=rows,
    )
    evaluation = evaluations["scramble"]
    evaluation.calibration = result
    written = write_report(tmp_path, evaluation, config=RunConfig(name="calibrated"), record={}, counts={})
    assert "calibration.parquet" in {p.name for p in written}
    frame = pl.read_parquet(tmp_path / "calibration.parquet")
    assert len(frame) == 3 * len(keys) and set(frame["mode"]) == {"pose", "scramble", "all"}
    report = (tmp_path / "report.md").read_text()
    assert report.index("## Confusions") < report.index("## Calibration") < report.index("## F1 against")
    assert (
        "one rotation c per attempt (a yaw about z; in training `attempt`, learnt with the network" in report
    )
    assert "The sections above are `scramble`'s." in report
    # The side faces' table: a row per mode, the cells R, R', R2, F, … after the mode's.
    f = sum(SYMBOLS[s] == "F" for c in test if c.segment == "solve" for s in c.symbols)
    lines = report.splitlines()
    first = next(i for i, line in enumerate(lines) if line.startswith("The side faces on the solve clips"))
    rows = {line.split(" | ")[0].strip("| `"): line.split(" | ") for line in lines[first + 4 : first + 8]}
    assert list(rows) == ["none", "pose", "scramble", "all"] and f > 0
    assert (
        rows["none"][4] == f"0% ({f:,})" and rows["scramble"][4] == f"100% ({f:,})"
    )  # F read as B, or right
    assert f"The honest rotation sits 2.0° from the oracle's (median over {len(keys)} keys" in report
    doc = json.loads((tmp_path / "metrics.json").read_text())["calibration"]
    assert (
        doc["headline"] == "scramble"
        and doc["modes"]["none"]["confusionsSolve"]["perSymbol"]["F"]["accuracy"] == 0.0
    )
    assert doc["fits"]["scramble"] == {
        "keys": len(keys),
        "sources": {"grid": len(keys)},
        "angleToGuessDeg": 8.0,
        "lossGainOverIdentity": pytest.approx(1.1),
    }
    assert doc["honestToOracleDeg"]["median"] == pytest.approx(2.0)
