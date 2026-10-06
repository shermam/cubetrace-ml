"""Training a model on the cached features, and evaluating a trained run (PyTorch: the `features` extra).

`train_run` reads the train and val clips once (features as float16, cast per batch), fits the features'
normalization, the class weights and the baseline on them, then trains with AdamW and a per-step cosine
schedule, batches of whole clips (padded, masked) in an order drawn from the seed, time masking, gradient
clipping and early stopping on val (F1@50, symbol, for the per-frame head, with its threshold chosen on val
every epoch; WER for CTC). The run folder: `config.json` (the resolved configuration), `log.csv` (one row
per epoch), `best.pt` and `last.pt` (the model's state with the configuration, the threshold, the
baseline and the data's counts). `evaluate_run` decodes a split with a run's checkpoint and writes the
report (`report`).
"""

from __future__ import annotations

import csv
import math
import random
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from . import report
from .config import RunConfig, read_config, write_config
from .dataset import Dataset
from .decode import ctc_greedy
from .encoders import DEVICES
from .evaluate import (
    Baseline,
    Evaluation,
    choose_threshold,
    evaluate_outputs,
    fit_baseline,
    pooled,
)
from .features import read_manifest
from .labels import ClipLabels, LoadStats, load_split
from .manifest import build_tables, write_tables
from .models import MoveModel, class_weights, ctc_loss, perframe_loss

LOG_COLUMNS = ("epoch", "train_loss", "val_loss", "val_f1_50", "val_wer", "lr", "seconds", "threshold")
Log = Callable[[str], None]


def pick_device(name: str = "auto") -> torch.device:
    if name not in DEVICES:
        raise ValueError(f"device {name!r}: one of {', '.join(DEVICES)}")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda, but PyTorch sees no CUDA device here")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass
class Batch:
    x: torch.Tensor  # B × T × D, float32
    mask: torch.Tensor  # B × T, True on the clips' frames
    near_class: torch.Tensor  # B × T, long
    near_weight: torch.Tensor  # B × T, float32
    targets: list[torch.Tensor]  # each clip's reference sequence as classes (symbol + 1)
    lengths: list[int]


def collate(clips: Sequence[ClipLabels], device: torch.device) -> Batch:
    """Whole clips padded at the end to the longest, with their mask and targets."""
    lengths = [len(c) for c in clips]
    longest, dim = max(lengths), clips[0].x.shape[1]  # type: ignore[union-attr]
    x = torch.zeros(len(clips), longest, dim)
    mask = torch.zeros(len(clips), longest, dtype=torch.bool)
    near_class = torch.zeros(len(clips), longest, dtype=torch.long)
    near_weight = torch.zeros(len(clips), longest)
    for i, clip in enumerate(clips):
        n = lengths[i]
        x[i, :n] = torch.from_numpy(clip.x.astype(np.float32))  # type: ignore[union-attr]
        mask[i, :n] = True
        near_class[i, :n] = torch.from_numpy(clip.near_class)
        near_weight[i, :n] = torch.from_numpy(clip.near_weight.astype(np.float32))
    return Batch(
        x.to(device),
        mask.to(device),
        near_class.to(device),
        near_weight.to(device),
        [torch.from_numpy(c.symbols + 1) for c in clips],
        lengths,
    )


def batch_order(count: int, size: int, rng: np.random.Generator) -> list[np.ndarray]:
    """The clips' indices shuffled by the run's generator, in batches of `size`."""
    order = rng.permutation(count)
    return [order[k : k + size] for k in range(0, count, size)]


def time_masks(
    lengths: Sequence[int], longest: int, count: int, frames: int, rng: np.random.Generator
) -> np.ndarray:
    """B × T: `count` spans of 1 to `frames` frames per clip, at random places inside it."""
    out = np.zeros((len(lengths), longest), dtype=bool)
    if count <= 0 or frames <= 0:
        return out
    for i, n in enumerate(lengths):
        for _ in range(count):
            width = int(rng.integers(1, frames + 1))
            start = int(rng.integers(0, max(1, n - width + 1)))
            out[i, start : min(n, start + width)] = True
    return out


def batch_loss(
    model: MoveModel, batch: Batch, head: str, weights: torch.Tensor, time_mask: Any = None
) -> torch.Tensor:
    logits = model(batch.x, batch.mask, time_mask)
    if head == "perframe":
        return perframe_loss(logits, batch.near_class, batch.near_weight, batch.mask, weights)
    return ctc_loss(logits, batch.mask, batch.targets)


@torch.no_grad()
def predict(
    model: MoveModel,
    clips: Sequence[ClipLabels],
    device: torch.device,
    *,
    batch: int = 8,
    head: str | None = None,
    weights: torch.Tensor | None = None,
) -> tuple[list[np.ndarray], float]:
    """Each clip's per-frame probabilities (T × 25), and the mean loss over the batches when `head` is
    given."""
    model.eval()
    probs: list[np.ndarray] = []
    losses: list[tuple[float, int]] = []
    for k in range(0, len(clips), batch):
        chunk = clips[k : k + batch]
        b = collate(chunk, device)
        logits = model(b.x, b.mask)
        if head is not None:
            loss = (
                perframe_loss(logits, b.near_class, b.near_weight, b.mask, weights)
                if head == "perframe"
                else ctc_loss(logits, b.mask, b.targets)
            )
            losses.append((float(loss), len(chunk)))
        p = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        probs.extend(p[i, :n] for i, n in enumerate(b.lengths))
    loss = sum(v * n for v, n in losses) / sum(n for _, n in losses) if losses else math.nan
    return probs, loss


def _manifest(config: RunConfig, dataset: Dataset, out: Path | None, log: Log) -> pl.DataFrame:
    """The run's manifest: `paths.manifest`, or built from the root now (and written into the run folder,
    so that `evaluate` reads the same splits)."""
    if config.paths.manifest:
        return read_manifest(config.paths.manifest)
    video = "none" if dataset.root.startswith("gs://") else "fast"
    tables = build_tables(dataset, time_base=config.data.time_base, video=video)
    if out is not None:
        write_tables(tables, out / "manifest")
        config.paths.manifest = str(out / "manifest" / "manifest.parquet")
        log(f"manifest built from {dataset.root}: {out / 'manifest'}")
    return tables.clips


def _dataset(config: RunConfig, cache: str | None, validate: bool) -> Dataset:
    if not config.paths.root:
        raise ValueError("no dataset root: pass --root or set CUBETRACE_DATA")
    return Dataset(config.paths.root, cache_dir=cache, validate=validate)


def _normalization(clips: Sequence[ClipLabels]) -> tuple[np.ndarray, np.ndarray]:
    """The features' mean and standard deviation over the frames of the clips (float64 sums)."""
    total = sum(len(c) for c in clips)
    dim = clips[0].x.shape[1]  # type: ignore[union-attr]
    s, ss = np.zeros(dim), np.zeros(dim)
    for c in clips:
        x = c.x.astype(np.float64)  # type: ignore[union-attr]
        s += x.sum(axis=0)
        ss += (x * x).sum(axis=0)
    mean = s / total
    std = np.sqrt(np.maximum(ss / total - mean * mean, 0.0))
    return mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32)


def _better(head: str, value: float, best: float | None) -> bool:
    if math.isnan(value):
        return False
    if best is None:
        return True
    return value > best if head == "perframe" else value < best


def _clear(out: Path) -> None:
    """Removes a run's files from its folder (a forced run starts clean): the configuration, the log, the
    checkpoints, the manifest it built and every split's report, plots, predictions and metrics."""
    if not out.is_dir():
        return
    for path in out.iterdir():
        name = path.name
        if name in ("config.json", "log.csv", "best.pt", "last.pt") or name.startswith(
            ("report", "metrics", "predictions")
        ):
            path.unlink()
        elif path.is_dir() and (name == "manifest" or name.startswith("plots")):
            shutil.rmtree(path)


def _checkpoint(model: MoveModel, config: RunConfig, **extra: Any) -> dict[str, Any]:
    return {"state": model.state_dict(), "config": config.to_json(), "dim": model.dim, **extra}


def train_run(
    config: RunConfig,
    out: str | Path,
    *,
    cache: str | None = None,
    validate: bool = True,
    force: bool = False,
    log: Log = print,
) -> dict[str, Any]:
    """Trains the configured model into the run folder `out`; returns the summary (the best epoch, its val
    metrics, the threshold and the baseline)."""
    out = Path(out)
    if (out / "config.json").exists() and not force:
        raise ValueError(f"{out} already holds a run (--force overwrites it)")
    _clear(out)
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(config.train.seed)
    device = pick_device(config.train.device)
    dataset = _dataset(config, cache, validate)
    manifest = _manifest(config, dataset, out, log)
    labels = config.data.labels()
    loaded: dict[str, tuple[list[ClipLabels], LoadStats]] = {}
    for split in ("train", "val"):
        loaded[split] = load_split(
            dataset, manifest, split, config.paths.features, config.paths.encoder, labels
        )
        log(loaded[split][1].describe())
    train, val = loaded["train"][0], loaded["val"][0]
    if not train:
        raise ValueError("no training clip: check the manifest's splits and the features root")
    write_config(config, out / "config.json")

    mean, std = _normalization(train)
    model = MoveModel(train[0].x.shape[1], config.model)  # type: ignore[union-attr]
    model.set_normalization(mean, std)
    model.to(device)
    head = config.model.head
    weights = class_weights([c.target for c in train]).to(device)
    baseline = fit_baseline(train, val, mean, std, config.decode.min_distance)
    log(
        f"baseline: symbol {baseline.symbol} (the most frequent), shift {baseline.shift_ms:+.1f} ms, "
        f"threshold {baseline.threshold:g} (val, F1@50 timing)"
    )
    data = {split: stats.to_json() for split, (_, stats) in loaded.items()}

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay
    )
    steps_per_epoch = math.ceil(len(train) / config.train.batch)
    total = max(1, config.train.epochs * steps_per_epoch)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 0.5 * (1.0 + math.cos(math.pi * min(step, total) / total))
    )
    rng = np.random.default_rng(config.train.seed)
    best: float | None = None
    best_epoch, stale = 0, 0
    threshold = config.decode.threshold
    summary: dict[str, Any] = {}
    with open(out / "log.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(LOG_COLUMNS)
        for epoch in range(1, config.train.epochs + 1):
            start = time.perf_counter()
            model.train()
            losses = []
            for indices in batch_order(len(train), config.train.batch, rng):
                chunk = [train[i] for i in indices]
                batch = collate(chunk, device)
                mask = time_masks(
                    batch.lengths,
                    batch.x.shape[1],
                    config.train.time_masks,
                    config.train.time_mask_frames,
                    rng,
                )
                loss = batch_loss(model, batch, head, weights, torch.from_numpy(mask).to(device))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad)
                optimizer.step()
                schedule.step()
                losses.append(loss.item())
            lr = optimizer.param_groups[0]["lr"]
            train_loss = float(np.mean(losses))
            if val:
                probs, val_loss = predict(
                    model, val, device, batch=config.train.batch, head=head, weights=weights
                )
                if head == "perframe":
                    epoch_threshold, f1, wer = choose_threshold(probs, val, config.decode)
                else:
                    epoch_threshold = None
                    f1, wer = pooled(val, [ctc_greedy(p, c.t_ms) for p, c in zip(probs, val, strict=True)])
            else:
                val_loss, f1, wer, epoch_threshold = math.nan, math.nan, math.nan, None
            seconds = time.perf_counter() - start
            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{val_loss:.6f}",
                    f"{f1:.6f}",
                    f"{wer:.6f}",
                    f"{lr:.8f}",
                    f"{seconds:.2f}",
                    "" if epoch_threshold is None else f"{epoch_threshold:g}",
                ]
            )
            handle.flush()
            metric = f1 if head == "perframe" else wer
            improved = _better(head, metric, best) or (not val and epoch == config.train.epochs)
            if improved:
                best, best_epoch, stale = metric, epoch, 0
                if epoch_threshold is not None:
                    threshold = epoch_threshold
            else:
                stale += 1
            metrics = {"trainLoss": train_loss, "valLoss": val_loss, "valF1At50": f1, "valWer": wer}
            extra = {"baseline": baseline.to_json(), "data": data}
            # Each checkpoint carries the threshold chosen on val for its own epoch (the last one standing
            # for an epoch without val clips).
            own = threshold if epoch_threshold is None else epoch_threshold
            torch.save(
                _checkpoint(model, config, epoch=epoch, metrics=metrics, threshold=own, **extra),
                out / "last.pt",
            )
            if improved:
                torch.save(
                    _checkpoint(model, config, epoch=epoch, metrics=metrics, threshold=threshold, **extra),
                    out / "best.pt",
                )
                summary = {"bestEpoch": epoch, **metrics, "threshold": threshold}
            chosen = "" if epoch_threshold is None else f", threshold {epoch_threshold:g}"
            log(
                f"epoch {epoch:>3}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, "
                f"val F1@50 {f1:.3f}, val WER {wer:.3f}{chosen}, lr {lr:.2e}, {seconds:.1f} s"
                f"{' *' if improved else ''}"
            )
            if val and epoch >= config.train.min_epochs and stale >= config.train.patience:
                log(f"early stop: no better val {'F1@50' if head == 'perframe' else 'WER'} in {stale} epochs")
                break
    if not (out / "best.pt").exists():  # no epoch had a val metric: the last one stands
        shutil.copyfile(out / "last.pt", out / "best.pt")
    summary.setdefault("bestEpoch", best_epoch)
    return {**summary, "baseline": baseline.to_json(), "data": data, "run": str(out)}


def load_run(
    run: str | Path, checkpoint: str = "best", device: torch.device | str = "cpu"
) -> tuple[RunConfig, MoveModel, dict[str, Any]]:
    """A run's configuration, its model at the checkpoint (`best` or `last`) and the checkpoint's record."""
    run = Path(run)
    config = read_config(run / "config.json")
    record = torch.load(run / f"{checkpoint}.pt", map_location="cpu", weights_only=True)
    model = MoveModel(int(record["dim"]), config.model)
    model.load_state_dict(record["state"])
    model.to(device)
    model.eval()
    return config, model, record


def evaluate_run(
    run: str | Path,
    *,
    split: str = "test",
    checkpoint: str = "best",
    consistency: bool = False,
    features: str | None = None,
    root: str | None = None,
    manifest: str | None = None,
    cache: str | None = None,
    validate: bool = True,
    device: str = "auto",
    log: Log = print,
) -> tuple[Evaluation, list[Path]]:
    """The run's model on a split: decoded, scored against the baseline, and the report written into the
    run folder (`report.md`, `plots/`, `predictions.parquet`, `metrics.json`; suffixed by the split when it
    is not `test`)."""
    run = Path(run)
    where = pick_device(device)
    config, model, record = load_run(run, checkpoint, where)
    for name, value in (("features", features), ("root", root), ("manifest", manifest)):
        if value:
            setattr(config.paths, name, value)
    dataset = _dataset(config, cache, validate)
    frame = _manifest(config, dataset, None, log)
    clips, stats = load_split(
        dataset, frame, split, config.paths.features, config.paths.encoder, config.data.labels()
    )
    log(stats.describe())
    if not clips:
        raise ValueError(f"no {split} clip to evaluate")
    probs, _ = predict(model, clips, where, batch=config.train.batch)
    mean = model.mean.detach().cpu().numpy()
    std = model.std.detach().cpu().numpy()
    evaluation = evaluate_outputs(
        clips,
        probs,
        head=config.model.head,
        threshold=float(record["threshold"]),
        baseline=Baseline.from_json(record["baseline"]),
        mean=mean,
        std=std,
        decode=config.decode,
        consistency=consistency,
        split=split,
    )
    counts = {**record.get("data", {}), split: stats.to_json()}
    written = report.write_report(
        run, evaluation, config=config, record=record, counts=counts, checkpoint=checkpoint
    )
    return evaluation, written
