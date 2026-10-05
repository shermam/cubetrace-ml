import os
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest
from PIL import Image

from cubetrace_ml.cli import main
from cubetrace_ml.dataset import ROOT_ENV


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def test_validate(dataset_root, capsys) -> None:
    root, _ = dataset_root
    code, out, _ = run(capsys, "validate", "--root", str(root))
    assert code == 0
    assert out.strip().endswith("0 errors, 1 warnings")


def test_report_reads_the_root_from_the_environment(dataset_root, capsys, monkeypatch) -> None:
    root, _ = dataset_root
    monkeypatch.setenv(ROOT_ENV, str(root))
    code, out, _ = run(capsys, "report")
    assert code == 0
    assert "attempts  5 (4 solved, 5 with video)" in out and "video check fast" in out
    code, out, _ = run(capsys, "report", "--video", "none", "--time-base", "arrival")
    assert "time base arrival" in out and "frame counts measured" not in out


def test_no_root_is_an_error(capsys) -> None:
    code, _, err = run(capsys, "report")
    assert code == 2 and ROOT_ENV in err


def test_manifest_writes_parquet_and_csv(dataset_root, capsys, tmp_path: Path) -> None:
    root, _ = dataset_root
    code, out, _ = run(capsys, "manifest", "--root", str(root), "--out", str(tmp_path / "m"), "--seed", "3")
    assert code == 0 and "12 clips, 5 attempts; 0 problems" in out
    clips = pl.read_parquet(tmp_path / "m" / "manifest.parquet")
    assert len(clips) == 12 and set(clips["split"]) <= {"train", "val", "test"}
    assert len(pl.read_csv(tmp_path / "m" / "manifest.csv")) == 12


def test_splits(dataset_root, capsys) -> None:
    root, ids = dataset_root
    code, out, _ = run(capsys, "splits", "--root", str(root))
    assert code == 0
    lines = out.splitlines()
    assert any(line.startswith("test ") and ids["D"] in line for line in lines)
    assert any(ids["C"] in line and "no session.json" in line for line in lines)
    assert lines[-1] == "test: 1 sessions  train: 2 sessions  val: 1 sessions"
    code, out, _ = run(capsys, "splits", "--root", str(root), "--held-out-day", "2026-09-21")
    assert any(line.startswith("test ") and ids["A"] in line for line in out.splitlines())
    code, _, err = run(capsys, "splits", "--root", str(root), "--held-out-day", "1999-01-01")
    assert code == 2 and "no session on 1999-01-01" in err


def test_inspect_by_clip_and_by_row(dataset_root, capsys, tmp_path: Path) -> None:
    root, ids = dataset_root
    out_png = tmp_path / "sheet.png"
    code, out, _ = run(
        capsys,
        *("inspect", "--root", str(root), "--session", ids["B"][:8], "--attempt", "1"),
        *(
            "--camera",
            "laptop",
            "--segment",
            "solve",
            "--onsets",
            "2",
            "--frames",
            "3",
            "--out",
            str(out_png),
        ),
    )
    assert code == 0 and out.strip() == str(out_png)
    assert Image.open(out_png).size[0] > 0
    run(capsys, "manifest", "--root", str(root), "--out", str(tmp_path / "m"))
    code, out, _ = run(
        capsys,
        *("inspect", "--root", str(root), "--row", "0", "--manifest", str(tmp_path / "m" / "manifest.csv")),
        *("--out", str(tmp_path / "row0.png")),
    )
    assert code == 0 and (tmp_path / "row0.png").is_file()
    code, _, err = run(capsys, "inspect", "--root", str(root), "--camera", "laptop")
    assert code == 2 and "clips match" in err


def test_inspect_writes_under_out_by_default(dataset_root, capsys, tmp_path: Path, monkeypatch) -> None:
    root, ids = dataset_root
    monkeypatch.chdir(tmp_path)
    code, out, _ = run(
        capsys,
        *("inspect", "--root", str(root), "--session", ids["D"], "--attempt", "1", "--camera", "laptop"),
        *("--segment", "scramble", "--onsets", "1", "--frames", "1"),
    )
    assert code == 0
    assert out.strip() == f"out/inspect-{ids['D'][:8]}-0001-laptop-scramble.png"
    assert (tmp_path / out.strip()).is_file()


def test_check_alignment(dataset_root, capsys) -> None:
    root, ids = dataset_root
    code, out, _ = run(capsys, "check-alignment", "--root", str(root), "--session", ids["B"])
    assert code == 0
    assert out.splitlines()[-1] == "4/4 clips ok"
    code, out, _ = run(capsys, "check-alignment", "--root", str(root), "--fast", "--camera", "phone-front")
    assert code == 0 and "unsynced" in out and out.splitlines()[-1] == "2/2 clips ok"


def test_the_console_script(dataset_root) -> None:
    root, _ = dataset_root
    env = {k: v for k, v in os.environ.items() if k != ROOT_ENV}
    done = subprocess.run(
        [sys.executable, "-m", "cubetrace_ml", "validate", "--root", str(root)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "0 errors" in done.stdout
    script = Path(sys.executable).with_name("cubetrace-ml")
    if not script.exists():
        pytest.skip("the console script is not installed beside this interpreter")
    helped = subprocess.run([str(script), "--help"], capture_output=True, text=True, env=env, check=False)
    assert helped.returncode == 0
    for command in ("report", "manifest", "splits", "inspect", "check-alignment", "validate"):
        assert command in helped.stdout
