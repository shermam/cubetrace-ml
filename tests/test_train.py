"""The models and their training on synthetic features with a planted onset signal (PyTorch: skipped without
the features extra): the shapes and the padding, the losses, the planted signal learnt well beyond the
baseline in a few CPU epochs, CTC, determinism, the `train` and `evaluate` commands, and the gyro's
channels telling the side faces apart when the cube turns in the hands."""

import csv
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")

from cubetrace_ml.cli import main  # noqa: E402
from cubetrace_ml.config import BODIES, HEADS, RunConfig, load_config  # noqa: E402
from cubetrace_ml.dataset import Dataset  # noqa: E402
from cubetrace_ml.labels import GYRO_CHANNELS  # noqa: E402
from cubetrace_ml.manifest import build_tables, write_tables  # noqa: E402
from cubetrace_ml.models import MoveModel, class_weights, ctc_loss, perframe_loss  # noqa: E402
from cubetrace_ml.train import evaluate_run, load_run, train_run  # noqa: E402
from factory import build_feature_dataset  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SMALL = ["model.width=64", "model.gru_hidden=32", "train.batch=4", "train.lr=3e-3"]
QUIET = {"log": lambda _: None}


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Eight sessions of two attempts (32 clips: 24 train, 4 val, 4 test) with the planted signal."""
    base = tmp_path_factory.mktemp("train")
    root, features = base / "data", base / "features"
    build_feature_dataset(root, features, sessions=8, attempts=2, dim=16, doubles=0.2)
    write_tables(build_tables(Dataset(root), video="none"), base / "manifest")
    return {"root": root, "features": features, "manifest": base / "manifest" / "manifest.parquet"}


def configured(paths: dict[str, Path], *overrides: str, path: Path | None = None) -> RunConfig:
    config = load_config(path, list(overrides))
    config.paths.root, config.paths.features = str(paths["root"]), str(paths["features"])
    config.paths.manifest, config.paths.encoder = str(paths["manifest"]), "synthetic"
    return config


def systems(evaluation) -> dict[str, dict]:
    return {name: values["all"] for name, values in evaluation.summary().items()}


@pytest.mark.parametrize("head", HEADS)
@pytest.mark.parametrize("body", BODIES)
def test_the_shapes_and_the_padding(head: str, body: str) -> None:
    torch.manual_seed(0)
    config = load_config(
        None, [f"model.head={head}", f"model.body={body}", "model.width=32", "model.gru_hidden=16"]
    )
    model = MoveModel(12, config.model).eval()
    short, long = torch.randn(1, 9, 12), torch.randn(1, 14, 12)
    alone = model(short, torch.ones(1, 9, dtype=torch.bool))
    assert alone.shape == (1, 9, 25)
    # In a batch beside a longer clip, padded at its end: the same outputs on its own frames.
    x = torch.zeros(2, 14, 12)
    x[0, :9], x[1] = short[0], long[0]
    mask = torch.zeros(2, 14, dtype=torch.bool)
    mask[0, :9], mask[1] = True, True
    batched = model(x, mask)
    torch.testing.assert_close(batched[0, :9], alone[0], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(
        batched[1], model(long, torch.ones(1, 14, dtype=torch.bool))[0], rtol=1e-4, atol=1e-5
    )
    # The time mask zeroes the standardized input: masking every frame is the input at the mean.
    model.set_normalization(torch.full((12,), 0.5), torch.full((12,), 2.0))
    everything = torch.ones(1, 9, dtype=torch.bool)
    torch.testing.assert_close(
        model(short, everything, everything),
        model(torch.full((1, 9, 12), 0.5), everything),
        rtol=1e-4,
        atol=1e-5,
    )


def test_the_losses() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 6, 25)
    target = torch.tensor([[0, 4, 0, 0, 9, 0], [0, 0, 0, 2, 0, 0]])
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[1, 5] = False
    weights = class_weights([target[0].numpy(), target[1, :5].numpy()])
    assert weights[0] == 1.0 and weights[1] == pytest.approx(8 / 3)  # 8 frames without an onset, 3 with
    # Hard targets: PyTorch's weighted cross-entropy over the real frames.
    expected = torch.nn.functional.cross_entropy(logits[mask], target[mask], weight=weights)
    hard = perframe_loss(logits, target, (target > 0).float(), mask, weights)
    torch.testing.assert_close(hard, expected)
    # A soft frame: half its weight on the onset class, half on "no onset".
    soft_class, soft_weight = target.clone(), (target > 0).float()
    soft_class[0, 3], soft_weight[0, 3] = 9, 0.5
    logp = torch.log_softmax(logits, -1)
    terms = -(weights[target] * logp.gather(-1, target[..., None]).squeeze(-1))
    norms = weights[target].clone()
    terms[0, 3] = -(0.5 * weights[0] * logp[0, 3, 0] + 0.5 * weights[9] * logp[0, 3, 9])
    norms[0, 3] = 0.5 * weights[0] + 0.5 * weights[9]
    torch.testing.assert_close(
        perframe_loss(logits, soft_class, soft_weight, mask, weights), terms[mask].sum() / norms[mask].sum()
    )
    # CTC: PyTorch's, blank 0, each clip's loss over its reference's length.
    targets = [torch.tensor([4, 9]), torch.tensor([2])]
    expected = torch.nn.functional.ctc_loss(
        torch.log_softmax(logits, -1).transpose(0, 1),
        torch.tensor([4, 9, 2]),
        torch.tensor([6, 5]),
        torch.tensor([2, 1]),
        blank=0,
    )
    torch.testing.assert_close(ctc_loss(logits, mask, targets), expected)


def test_the_planted_signal_is_learnt_beyond_the_baseline(synthetic, tmp_path: Path) -> None:
    start = time.perf_counter()
    config = configured(synthetic, *SMALL, "train.epochs=8")
    summary = train_run(config, tmp_path / "run", **QUIET)
    evaluation, written = evaluate_run(tmp_path / "run", **QUIET)
    assert time.perf_counter() - start < 60
    result = systems(evaluation)
    model, baseline = result["model"], result["baseline"]
    assert model["onsets"]["symbol@50"]["f1Pooled"] > 0.7 > 0.2 > baseline["onsets"]["symbol@50"]["f1Pooled"]
    assert model["werPooled"] < 0.4 < baseline["werPooled"]
    assert model["onsets"]["timing@25"]["f1Pooled"] > baseline["onsets"]["timing@25"]["f1Pooled"]
    assert 0.1 <= summary["threshold"] <= 0.9 and summary["bestEpoch"] >= 1
    with open(tmp_path / "run" / "log.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == [
        "epoch",
        "train_loss",
        "val_loss",
        "val_f1_50",
        "val_wer",
        "lr",
        "seconds",
        "threshold",
    ]
    assert len(rows) == 8 and float(rows[-1]["train_loss"]) < float(rows[0]["train_loss"])
    assert {p.name for p in (tmp_path / "run").iterdir()} >= {"config.json", "log.csv", "best.pt", "last.pt"}
    assert {p.name for p in written} >= {"report.md", "metrics.json", "predictions.parquet"}
    config_, _, record = load_run(tmp_path / "run")
    assert config_.paths.encoder == "synthetic" and record["epoch"] == summary["bestEpoch"]
    assert record["data"]["train"]["clips"] == 24 and record["baseline"]["minDistance"] == 2


def test_ctc_trains_and_decodes(synthetic, tmp_path: Path) -> None:
    config = configured(synthetic, *SMALL, "model.head=ctc", "train.epochs=3")
    train_run(config, tmp_path / "run", **QUIET)
    with open(tmp_path / "run" / "log.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert float(rows[-1]["train_loss"]) < float(rows[0]["train_loss"]) and rows[0]["threshold"] == ""
    evaluation, _ = evaluate_run(tmp_path / "run", split="val", **QUIET)
    assert evaluation.head == "ctc" and evaluation.threshold is None
    assert (tmp_path / "run" / "report-val.md").is_file()


def test_the_transformer_body_trains(synthetic, tmp_path: Path) -> None:
    config = configured(
        synthetic, *SMALL, "model.body=transformer", "model.transformer_ff=64", "train.epochs=1"
    )
    train_run(config, tmp_path / "run", **QUIET)
    _, model, _ = load_run(tmp_path / "run")
    assert model.body == "transformer" and len(model.encoder.layers) == 4


def test_training_is_deterministic(synthetic, tmp_path: Path) -> None:
    states = []
    for name, seed in (("a", 0), ("b", 0), ("c", 1)):
        train_run(
            configured(synthetic, *SMALL, "train.epochs=1", f"train.seed={seed}"), tmp_path / name, **QUIET
        )
        states.append(torch.load(tmp_path / name / "last.pt", weights_only=True)["state"])
    assert all(torch.equal(states[0][k], states[1][k]) for k in states[0])
    assert not all(torch.equal(states[0][k], states[2][k]) for k in states[0])


def test_the_train_and_evaluate_commands(
    synthetic, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = tmp_path / "run15"
    argv = [
        *("train", "--config", str(CONFIGS / "perframe-bigru.toml"), "--root", str(synthetic["root"])),
        *("--features", str(synthetic["features"]), "--encoder", "synthetic"),
        *("--manifest", str(synthetic["manifest"]), "--out", str(run), "--fps", "15"),
        *("--set", "train.epochs=2", "--set", "model.width=32", "--set", "model.gru_hidden=16"),
        *("--label-frames", "1"),
    ]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert f"run {run}: best epoch" in out
    config, _, _ = load_run(run)
    assert config.data.fps == 15.0 and config.train.epochs == 2 and config.name == "perframe-bigru"
    assert config.data.label_frames == 1
    assert main(argv) == 2 and "already holds a run" in capsys.readouterr().err
    assert main(["evaluate", "--run", str(run), "--split", "val"]) == 0 and (run / "report-val.md").is_file()
    assert main([*argv, "--force"]) == 0  # a forced run starts clean: the old reports go
    assert not (run / "report-val.md").exists() and not (run / "plots-val").exists()
    capsys.readouterr()
    assert main(["evaluate", "--run", str(run), "--consistency"]) == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0].startswith("test: 4 of 4 clips loaded")
    assert [line.split()[0] for line in lines[1:4]] == ["model", "consistency", "baseline"]
    report = (run / "report.md").read_text()
    assert "15 fps (every 2nd frame of the clips)" in report and "## The consistency pass" in report
    predictions = pl.read_parquet(run / "predictions.parquet")
    assert len(predictions) == 4 and set(predictions["stride"]) == {2}


def test_a_run_needs_its_paths(synthetic, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "train",
            "--root",
            str(synthetic["root"]),
            "--features",
            str(synthetic["features"]),
            "--out",
            str(tmp_path / "x"),
        ]
    )
    assert code == 2 and "no encoder" in capsys.readouterr().err
    code = main(
        [
            "train",
            "--root",
            str(synthetic["root"]),
            "--features",
            str(tmp_path / "none"),
            "--encoder",
            "synthetic",
            "--manifest",
            str(synthetic["manifest"]),
            "--out",
            str(tmp_path / "y"),
        ]
    )
    assert code == 2 and "no training clip" in capsys.readouterr().err


@pytest.fixture(scope="module")
def turned(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Ten sessions of three attempts (60 clips: 48 train, 6 val, 6 test) whose cube is turned half way about
    U or not, between the segments and once in the solve: the planted signal is the face the camera sees, so
    that F and B, R and L, look alike unless the orientation tells them apart."""
    base = tmp_path_factory.mktemp("turned")
    root, features = base / "data", base / "features"
    build_feature_dataset(
        root, features, sessions=10, attempts=3, dim=16, doubles=0.2, yaw=True, yaw_quarters=(0, 2)
    )
    write_tables(build_tables(Dataset(root), video="none"), base / "manifest")
    return {"root": root, "features": features, "manifest": base / "manifest" / "manifest.parquet"}


# The comparison's model: the small transformer, which learns the face-and-orientation pairs in fewer CPU
# seconds than the BiGRU (the inputs reach both bodies through the same projection), without the
# regularizers (the test measures what the inputs carry).
TURNED = [
    "model.body=transformer",
    "model.width=64",
    "model.transformer_ff=128",
    "model.dropout=0.0",
    "train.time_masks=0",
    "train.batch=4",
    "train.lr=3e-3",
    "train.epochs=15",
    "data.require_gyro=true",
]


def test_the_orientation_tells_the_side_faces_apart(turned, tmp_path: Path) -> None:
    start = time.perf_counter()
    result = {}
    for inputs in ("features", "features+gyro"):
        config = configured(turned, *TURNED, f'data.inputs="{inputs}"')
        train_run(config, tmp_path / inputs, **QUIET)
        evaluation, _ = evaluate_run(tmp_path / inputs, **QUIET)
        result[inputs] = systems(evaluation)["model"]
    assert time.perf_counter() - start < 60
    alone, both = result["features"], result["features+gyro"]
    # The features alone find the onsets but not which of two opposite faces turned; the gyro's channels do.
    assert alone["onsets"]["timing@50"]["f1Pooled"] > 0.8
    assert both["onsets"]["symbol@50"]["f1Pooled"] >= alone["onsets"]["symbol@50"]["f1Pooled"] + 0.2
    assert both["werPooled"] < alone["werPooled"]
    # The same clips; the gyro's 9 channels after the 16 features, the flag not standardized.
    _, model, record = load_run(tmp_path / "features+gyro")
    assert record["dim"] == 16 + len(GYRO_CHANNELS) == model.proj.in_features
    assert (float(model.mean[-1]), float(model.std[-1])) == (0.0, 1.0)
    assert load_run(tmp_path / "features")[2]["dim"] == 16
    assert record["data"] == load_run(tmp_path / "features")[2]["data"]
    assert record["data"]["train"]["gyroClips"] == record["data"]["train"]["clips"] == 48


def test_a_gyro_run_through_the_commands(turned, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = tmp_path / "gyro"
    argv = [
        *("train", "--config", str(CONFIGS / "perframe-bigru.toml"), "--root", str(turned["root"])),
        *("--features", str(turned["features"]), "--encoder", "synthetic"),
        *("--manifest", str(turned["manifest"]), "--out", str(run)),
        *("--set", "data.inputs=features+gyro", "--set", "data.require_gyro=true"),
        *("--set", "train.epochs=1", "--set", "model.width=32", "--set", "model.gru_hidden=16"),
    ]
    assert main(argv) == 0
    capsys.readouterr()
    assert main(["evaluate", "--run", str(run)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert (
        lines[0].startswith("test: 6 of 6 clips loaded (none skipped") and "the gyro on 6 clips" in lines[0]
    )
    assert lines[3].startswith("confusions (model, ±50 ms): ") and "onsets matched" in lines[3]
    config, model, _ = load_run(run)
    assert config.data.inputs == "features+gyro" and model.proj.in_features == 16 + 9
    report = (run / "report.md").read_text()
    assert "## Confusions" in report and "| inputs | the features and the gyro's 9 channels" in report
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["inputs"] == "features+gyro" and metrics["confusions"]["reference"] > 0


# M4: the orientation in the camera's frame.


@pytest.fixture(scope="module")
def reframed(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """The turned fixture whose test session's gyro frames are turned 180° about the vertical, a frame the
    training never saw (its 48 training and 6 val clips keep the camera's)."""
    base = tmp_path_factory.mktemp("reframed")
    root, features = base / "data", base / "features"
    build_feature_dataset(
        root,
        features,
        sessions=10,
        attempts=3,
        dim=16,
        doubles=0.2,
        yaw=True,
        yaw_quarters=(0, 2),
        frame_yaws=lambda s, a: 180.0 if s == 9 else 0.0,
    )
    write_tables(build_tables(Dataset(root), video="none"), base / "manifest")
    return {"root": root, "features": features, "manifest": base / "manifest" / "manifest.parquet"}


def f1_symbol(evaluation) -> float:
    return evaluation.summary()["model"]["all"]["onsets"]["symbol@50"]["f1Pooled"]


def test_the_calibration_recovers_a_frame_the_training_never_saw(
    reframed, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    start = time.perf_counter()
    gyro = 'data.inputs="features+gyro"'
    train_run(configured(reframed, *TURNED, gyro), tmp_path / "m3", **QUIET)
    m3, _ = evaluate_run(tmp_path / "m3", **QUIET)
    calibrated = configured(
        reframed, *TURNED, gyro, "data.calibration=attempt", "data.calibration_init=identity"
    )
    train_run(calibrated, tmp_path / "cal", **QUIET)
    evaluation, written = evaluate_run(tmp_path / "cal", **QUIET)
    assert time.perf_counter() - start < 60
    modes = {mode: f1_symbol(e) for mode, e in evaluation.calibration.evaluations.items()}
    # M3's channels learn the training's frame and turn every side face around in the test's; so does the
    # calibrated model at the identity. The fit on the scramble's labels finds the frame: most of the oracle.
    assert f1_symbol(m3) < 0.5 and modes["none"] < 0.5
    assert modes["scramble"] >= modes["none"] + 0.3 and modes["scramble"] >= 0.9 * modes["all"]
    assert evaluation.calibration.headline == "scramble" and f1_symbol(evaluation) == modes["scramble"]
    fits = evaluation.calibration.frame()
    honest = fits.filter(pl.col("mode") == "scramble")
    assert len(fits) == 9 and len(honest) == 3 and set(honest["source"]) <= {"grid", "guess"}
    assert all(abs(abs(y) - 180.0) < 10.0 for y in honest["yawDeg"])  # the frame turned 180°, found
    # The run's record and the report.
    _, model, record = load_run(tmp_path / "cal")
    info = record["calibration"]
    assert (info["kind"], info["dof"], info["axis"], info["by"]) == ("attempt", "yaw", "z", "attempt")
    assert len(info["keys"]) == len(info["learnt"]) == 24 and model.proj.in_features == 16 + 14
    assert {p.name for p in written} >= {"report.md", "metrics.json", "calibration.parquet"}
    report = (tmp_path / "cal" / "report.md").read_text()
    assert "## Calibration" in report and "The side faces on the solve clips" in report
    metrics = json.loads((tmp_path / "cal" / "metrics.json").read_text())
    assert (
        metrics["calibration"]["headline"] == "scramble"
        and metrics["calibration"]["honestToOracleDeg"]["keys"] == 3
    )
    # M3's run takes no calibration.
    assert main(["evaluate", "--run", str(tmp_path / "m3"), "--calibrate", "scramble"]) == 2
    assert "take no calibration" in capsys.readouterr().err


@pytest.fixture(scope="module")
def posed(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """The turned fixture with the cube at home through every scramble (the solver applies the prescribed
    moves holding it one way) and every session's gyro frame at its own yaw, drifting 3° an attempt."""
    base = tmp_path_factory.mktemp("posed")
    root, features = base / "data", base / "features"
    yaws = np.random.default_rng(7).uniform(0.0, 360.0, 10)
    build_feature_dataset(
        root,
        features,
        sessions=10,
        attempts=3,
        dim=16,
        doubles=0.2,
        yaw=True,
        yaw_quarters=(0, 2),
        scramble_home=True,
        frame_yaws=lambda s, a: float(yaws[s]) + 3.0 * a,
    )
    write_tables(build_tables(Dataset(root), video="none"), base / "manifest")
    return {"root": root, "features": features, "manifest": base / "manifest" / "manifest.parquet"}


def test_the_scramble_pose_calibrates_without_labels(
    posed, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    start = time.perf_counter()
    run = tmp_path / "posed"
    overrides = [*TURNED, "data.inputs=features+gyro", "data.calibration=attempt"]
    argv = [
        *("train", "--root", str(posed["root"]), "--features", str(posed["features"])),
        *("--encoder", "synthetic", "--manifest", str(posed["manifest"]), "--out", str(run)),
        *(item for override in overrides for item in ("--set", override)),
    ]
    assert main(argv) == 0
    capsys.readouterr()
    assert main(["evaluate", "--run", str(run), "--calibrate", "pose"]) == 0
    assert time.perf_counter() - start < 60
    lines = capsys.readouterr().out.splitlines()
    modes = {line.split()[1]: line for line in lines if line.startswith("calibration ")}
    assert list(modes) == ["none", "pose", "scramble", "all"] and "(the sections above)" in modes["pose"]
    _, _, record = load_run(run)
    guess = record["calibration"]["guess"]
    assert record["calibration"]["init"] == "pose" and guess["usesPose"] and guess["agreement"] >= 0.8
    metrics = json.loads((run / "metrics.json").read_text())["calibration"]
    f1 = {mode: m["all"]["onsets"]["symbol@50"]["f1Pooled"] for mode, m in metrics["modes"].items()}
    # Every session's frame at its own yaw: the identity is lost, the scramble's pose alone finds the frame.
    assert f1["none"] < 0.6 and f1["pose"] >= 0.85 and f1["scramble"] >= 0.85
    assert metrics["headline"] == "pose" and metrics["fits"]["pose"]["sources"] == {"pose": 3}
