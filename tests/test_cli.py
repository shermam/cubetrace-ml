import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from PIL import Image

from cubetrace_ml.cli import main
from cubetrace_ml.dataset import ROOT_ENV
from cubetrace_ml.features import FEATURES_ENV, read_meta
from factory import write_frames


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
    commands = ("report", "manifest", "splits", "inspect", "check-alignment", "validate", "features", "bench")
    for command in (*commands, "crop-preview"):
        assert command in helped.stdout


def test_features_writes_resumes_and_reports(dataset_root, capsys, tmp_path: Path, monkeypatch) -> None:
    root, ids = dataset_root
    monkeypatch.setenv(FEATURES_ENV, str(tmp_path / "features"))
    argv = ("features", "--root", str(root), "--encoder", "stub", "--split", "test")
    code, out, _ = run(capsys, *argv)
    assert code == 0
    assert "features: 2 clips selected: 2 written, 0 already there, 0 failed; 48 frames" in out
    assert "decode, crop, resize" in out and "encoder stub" in out
    folder = tmp_path / "features" / "stub" / ids["D"] / "0001"
    assert sorted(p.name for p in folder.iterdir()) == ["laptop.scramble.npz", "laptop.solve.npz"]
    code, out, _ = run(capsys, *argv)
    assert code == 0 and "2 of 2 clips already have their stub features" in out
    assert "0 written, 2 already there" in out
    code, out, _ = run(capsys, *argv, "--force", "--crop", "none", "--workers", "2", "--batch", "5")
    assert code == 0 and "2 written" in out
    assert read_meta(folder / "laptop.solve.npz")["cropMode"] == "none"


def test_features_from_a_manifest_with_its_filters(dataset_root, capsys, tmp_path: Path) -> None:
    root, ids = dataset_root
    run(capsys, "manifest", "--root", str(root), "--out", str(tmp_path / "m"))
    code, out, _ = run(
        capsys,
        *("features", "--root", str(root), "--out", str(tmp_path / "f"), "--encoder", "stub"),
        *("--manifest", str(tmp_path / "m" / "manifest.parquet"), "--camera", "phone-rear", "--limit", "1"),
    )
    assert code == 0 and "1 clips selected: 1 written" in out
    assert (tmp_path / "f" / "stub" / ids["B"] / "0001" / "phone-rear.scramble.npz").is_file()
    code, out, _ = run(
        capsys,
        *("features", "--root", str(root), "--out", str(tmp_path / "f"), "--encoder", "stub"),
        *("--session", ids["A"][:8], "--segment", "solve", "--no-usable-only"),
    )
    assert code == 0 and "2 clips selected" in out  # the DNF's solve clip too
    code, _, err = run(
        capsys, "features", "--root", str(root), "--out", str(tmp_path), "--encoder", "stub", "--camera", "x"
    )
    assert code == 2 and "no clip matches" in err


def test_features_needs_its_root_and_its_extra(dataset_root, capsys, tmp_path: Path, monkeypatch) -> None:
    root, _ = dataset_root
    monkeypatch.delenv(FEATURES_ENV, raising=False)
    code, _, err = run(capsys, "features", "--root", str(root), "--encoder", "stub")
    assert code == 2 and FEATURES_ENV in err
    monkeypatch.setitem(sys.modules, "torch", None)
    code, _, err = run(
        capsys, "features", "--root", str(root), "--out", str(tmp_path), "--encoder", "resnet18"
    )
    assert code == 2 and "install the features extra" in err


def test_features_exits_1_when_a_clip_fails(dataset_root, capsys, tmp_path: Path) -> None:
    source, ids = dataset_root
    root = tmp_path / "copy"
    shutil.copytree(source, root)
    video = root / "sessions" / ids["D"] / "attempts" / "0001" / "laptop.solve.mp4"
    write_frames(video, (np.full((64, 64, 3), 99, np.uint8) for _ in range(20)))
    code, out, _ = run(
        capsys,
        *("features", "--root", str(root), "--out", str(tmp_path / "f"), "--encoder", "stub"),
        *("--session", ids["D"], "--no-usable-only"),
    )
    assert code == 1 and "1 written, 0 already there, 1 failed" in out
    assert "decoded 20 frames, the frames file has 24" in out


def test_bench(dataset_root, capsys) -> None:
    root, _ = dataset_root
    code, out, _ = run(
        capsys, "bench", "--root", str(root), "--encoders", "none,stub", "--limit", "2", "--size", "32"
    )
    assert code == 0
    assert "| none | 32 | 2 | 48 |" in out and "| stub | 32 | 2 | 48 |" in out
    assert "bench none: 2 clips selected" in out and "motion crop" in out
    code, _, err = run(capsys, "bench", "--root", str(root), "--encoders", "vgg")
    assert code == 2 and "--encoders 'vgg'" in err


def test_crop_preview(dataset_root, capsys, tmp_path: Path) -> None:
    root, ids = dataset_root
    png = tmp_path / "crop.png"
    code, out, _ = run(
        capsys,
        *("crop-preview", "--root", str(root), "--session", ids["B"][:8], "--attempt", "1"),
        *("--camera", "phone-rear", "--segment", "solve", "--height", "128", "--out", str(png)),
    )
    assert code == 0 and out.splitlines()[0] == str(png)
    assert "auto: motion 64x64+0+0 (square)" in out and "record: none" in out
    assert Image.open(png).size[1] > 128
    run(capsys, "manifest", "--root", str(root), "--out", str(tmp_path / "m"))
    code, out, _ = run(
        capsys,
        *(
            "crop-preview",
            "--root",
            str(root),
            "--row",
            "0",
            "--manifest",
            str(tmp_path / "m" / "manifest.csv"),
        ),
        *("--frame", "3", "--out", str(tmp_path / "row0.png")),
    )
    assert code == 0 and "frame 3;" in out and (tmp_path / "row0.png").is_file()
