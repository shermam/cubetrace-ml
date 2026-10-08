"""The gyro-frames diagnostic on synthetic gyro files: the convention from the angular velocity, gravity's
when the cube is held upright in a gyro frame whose yaw alone is arbitrary, none when the whole frame is,
the hold, the yaw per session, and the command's files."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from cubetrace_ml.cli import main
from cubetrace_ml.dataset import Dataset
from cubetrace_ml.gyroframes import axial_mode, diagnose, report_text, spearman, wrap
from factory import (
    DAY_MS,
    T0,
    camera,
    held_gyro,
    quaternion_about,
    session_id,
    session_record,
    synthetic_attempt,
    write_json,
    write_synthetic_attempt,
)

FRAME_YAWS = {0: 30.0, 1: -100.0, 2: 160.0}  # each session's gyro frame, a yaw about z (degrees)
DRIFT = 2.0  # degrees a frame turns from one attempt to the next


def held_dataset(root: Path, *, frames=None, velocity: str = "body", seed: int = 0) -> dict[int, str]:
    """Three sessions of four attempts, the cube held as the owner holds it; each attempt's gyro frame is
    `frames(session, attempt)` (a quaternion), by default the session's yaw plus the drift."""
    rng = np.random.default_rng(seed)
    ids = {}
    for s, yaw in FRAME_YAWS.items():
        sid = session_id(200 + s)
        ids[s] = sid
        created = T0 + s * DAY_MS
        write_json(
            root / "sessions" / sid / "session.json", session_record(sid, created, [camera("laptop")], {})
        )
        for a in range(1, 5):
            attempt, clips = synthetic_attempt(sid, a, created + 60_000.0 * a, rng, moves=10)
            frame = frames(s, a) if frames else quaternion_about((0, 0, 1), yaw + DRIFT * a)
            gyro = held_gyro(attempt, rng, frame=frame, velocity=velocity)
            write_synthetic_attempt(root, attempt, clips, gyro)
    return ids


@pytest.fixture(scope="module")
def held(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[int, str]]:
    root = tmp_path_factory.mktemp("held")
    return root, held_dataset(root)


def test_the_helpers() -> None:
    mode, value = axial_mode(np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.1, 0.995]]))
    assert mode[2] > 0.99 and value > 0.99  # an axis held up and held down count alike
    assert axial_mode(np.eye(3))[1] == pytest.approx(1 / 3)
    x = np.arange(10.0)[:, None]
    np.testing.assert_allclose(spearman(x, np.concatenate([x**3, -x], axis=1)), [[1.0, -1.0]])
    assert wrap(190.0) == pytest.approx(-170.0) and wrap(-180.0) == pytest.approx(-180.0)


def test_gravity_and_the_yaw_of_a_cube_held_upright(held) -> None:
    root, ids = held
    result = diagnose(Dataset(root))
    s = result.summary
    assert (s["sessions"], s["attempts"], s["segments"]) == (3, 12, {"scramble": 12, "solve": 12})
    assert list(result.sessions) == [ids[0], ids[1], ids[2]]  # by their first attempt
    # The angular velocity follows the change in the cube's frame: a reference change acts on the left.
    convention = s["convention"]
    assert convention["verdict"].startswith("cube frame")
    assert convention["bodyDiagonal"] > 0.5 > convention["gyroDiagonal"]
    # The cube's z axis keeps one direction, the gyro's z; x and y take every heading.
    gravity = s["gravity"]
    assert (gravity["verdict"], gravity["cubeAxis"], gravity["gravityAxis"]) == ("found", "z", "z")
    assert gravity["pooledConcentration"] > 0.95 and all(v < 0.8 for v in gravity["others"].values())
    assert gravity["angleToNearest"] < 3.0
    # White up through every scramble, down through every solve, about 10° off the vertical.
    held_ = s["held"]
    assert (held_["scramble"]["along"], held_["scramble"]["against"]) == (12, 0)
    assert (held_["solve"]["along"], held_["solve"]["against"]) == (0, 12)
    assert 5.0 < held_["scramble"]["tilt"][1] < 15.0
    assert 170.0 - 30.0 < s["within"]["scrambleToSolve"][1] <= 180.0
    # The scramble pose's heading is each session's frame yaw (plus the drift): the sessions apart, the
    # consecutive attempts DRIFT apart.
    yaw = s["yaw"]
    assert yaw["axis"] == "z"
    for row, frame_yaw in zip(yaw["sessions"], FRAME_YAWS.values(), strict=True):
        assert row["attempts"] == 4
        assert abs(wrap(row["mean"] - (frame_yaw + 2.5 * DRIFT))) < 3.0
    assert yaw["consecutive"][0] == pytest.approx(DRIFT, abs=1.5)
    assert yaw["betweenSessions"]["max"] == pytest.approx(130.0, abs=3.0)  # 30 against 160
    assert "data.gravity_axis = z" in s["advice"]
    text = report_text(result, "synthetic")
    assert "Verdict: **found** (gravity: the gyro's z axis)" in text
    assert f"| 1 ({ids[0][:8]}) | 4 |" in text and "**cube frame" in text


def test_no_gravity_when_the_whole_frame_is_arbitrary(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    frames = {}

    def arbitrary(s: int, a: int) -> np.ndarray:
        frames[(s, a)] = q = rng.normal(size=4)
        return q / np.linalg.norm(q)

    held_dataset(tmp_path, frames=arbitrary)
    s = diagnose(Dataset(tmp_path)).summary
    assert s["gravity"]["verdict"] == "inconclusive" and s["gravity"]["pooledConcentration"] < 0.8
    assert "data.calibration_dof = rotation" in s["advice"]


def test_the_velocity_of_the_other_convention(tmp_path: Path) -> None:
    held_dataset(tmp_path, velocity="gyro")
    convention = diagnose(Dataset(tmp_path)).summary["convention"]
    assert convention["verdict"].startswith("gyro frame")


def test_the_command(held, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root, ids = held
    code = main(["gyro-frames", "--root", str(root), "--out", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert code == 0 and out.startswith("# The gyro's frame") and "gyro-frames.parquet" in out
    doc = json.loads((tmp_path / "out" / "gyro-frames.json").read_text())
    assert (
        doc["gravity"]["gravityAxis"] == "z"
        and "NaN" not in (tmp_path / "out" / "gyro-frames.json").read_text()
    )
    assert (tmp_path / "out" / "gyro-frames.md").read_text().startswith("# The gyro's frame")
    import polars as pl

    rows = pl.read_parquet(tmp_path / "out" / "gyro-frames.parquet")
    assert len(rows) == 24 and set(rows["segment"]) == {"scramble", "solve"}
    assert math.isclose(float(np.median(rows["zConcentration"].to_numpy())), 1.0, abs_tol=0.05)
    code = main(["gyro-frames", "--root", str(root), "--session", ids[1]])
    assert code == 0 and "1 sessions" in capsys.readouterr().out
    empty = tmp_path / "empty"
    (empty / "sessions").mkdir(parents=True)
    assert main(["gyro-frames", "--root", str(empty)]) == 2
    assert "no attempt with a readable gyro.json" in capsys.readouterr().err
