"""The PyTorch encoders with their pretrained weights, end to end: `cubetrace-ml features` on the tests'
synthetic dataset (its test split: two 24-frame clips), each file's arrays and meta checked.

The weights download on first use (download.pytorch.org, huggingface.co), so this runs where the network
allows it: CI's manual `weights` job (Actions, "Run workflow") and the GPU machine (`--device cuda`). The
tests never download anything.

    uv run python scripts/check_encoders.py [--device auto|cpu|cuda] [--encoders resnet18,dinov2-vits14]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from cubetrace_ml.cli import main as cubetrace_ml
from cubetrace_ml.dataset import ClipRef
from cubetrace_ml.encoders import ENCODERS
from cubetrace_ml.features import feature_path, read_features
from factory import build_dataset


def check(out: Path, name: str, clips: list[ClipRef]) -> list[str]:
    """What is wrong with the encoder's files of `clips` (nothing: an empty list)."""
    problems = []
    info = ENCODERS[name]
    for clip in clips:
        data = read_features(feature_path(out, name, clip))
        x, meta = data["x"].astype(np.float32), data["meta"]
        if x.shape != (len(data["tMs"]), info.dim) or not np.isfinite(x).all():
            problems.append(f"{name} {clip}: features {x.shape}, finite {np.isfinite(x).all()}")
        expected = "pretrained" if info.extra else "seed 0"  # the stub has no weights to download
        if meta["encoder"]["loaded"] != expected:
            problems.append(f"{name} {clip}: weights {meta['encoder']['loaded']}")
        if np.abs(np.diff(x, axis=0)).max() == 0:  # the frames differ (frame k is gray(k)): so must they
            problems.append(f"{name} {clip}: every frame has the same features")
        norm = np.linalg.norm(x, axis=1).mean()
        change = np.linalg.norm(np.diff(x, axis=0), axis=1).mean()
        host = meta["host"]
        print(
            f"{name} {clip.camera}.{clip.segment}: x {x.shape} {data['x'].dtype}, norm {norm:.2f}, "
            f"frame-to-frame change {change:.3f}; {host['device']} ({host['deviceName']}), "
            f"{host['precision']}, torch {host['torch']}, encode {meta['timing']['encodeFps']} fps"
        )
    return problems


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--encoders", default="resnet18,dinov2-vits14")
    args = parser.parse_args(argv)
    problems: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        root, out = Path(tmp) / "data", Path(tmp) / "features"
        ids = build_dataset(root)
        clips = [ClipRef(ids["D"], 1, "laptop", segment) for segment in ("scramble", "solve")]
        for name in args.encoders.split(","):
            argv = ["features", "--root", str(root), "--out", str(out), "--encoder", name, "--split", "test"]
            code = cubetrace_ml([*argv, "--device", args.device])
            if code != 0:
                problems.append(f"{name}: cubetrace-ml features exited {code}")
                continue
            problems += check(out, name, clips)
    for problem in problems:
        print(f"PROBLEM {problem}")
    print("ok" if not problems else f"{len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(run())
