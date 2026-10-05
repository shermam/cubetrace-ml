"""The `cubetrace-ml` command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from .checks import check_alignment, validate_all
from .contact_sheet import contact_sheet
from .dataset import ROOT_ENV, SEGMENTS, ClipRef, Dataset
from .encoders import DEVICES, ENCODERS, PRECISIONS, EncoderError, load_encoder
from .features import FEATURES_ENV, bench, bench_table, extract, features_root, read_manifest, select_clips
from .features import summary as features_summary
from .framing import CROP_MODES, crop_preview
from .manifest import build_tables, report_text, write_tables
from .moves import DOUBLE_MS, SLICE_MS, TIME_BASES
from .splits import VAL_FRACTION


def _common() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--root", help=f"the dataset root: a folder or gs://bucket/prefix (default: ${ROOT_ENV})"
    )
    common.add_argument("--cache", help="the cache of a gs:// root (default: ~/.cache/cubetrace-ml)")
    common.add_argument("--no-validate", action="store_true", help="read the records without their schemas")
    return common


def _labels(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--time-base", choices=TIME_BASES, default="fit", help="the moves' times (default: fit)"
    )
    parser.add_argument(
        "--slice-ms", type=float, default=SLICE_MS, help=f"slice threshold (default {SLICE_MS:g})"
    )
    parser.add_argument(
        "--double-ms", type=float, default=DOUBLE_MS, help=f"double threshold (default {DOUBLE_MS:g})"
    )


def _split_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=0, help="the split's seed (default 0)")
    parser.add_argument(
        "--held-out-day",
        help="the test day: YYYY-MM-DD, or latest (default: the day whose clips are nearest a fifth of all)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=VAL_FRACTION,
        help=f"the share of the clips val takes, in whole sessions (default {VAL_FRACTION:g})",
    )


def _clip_options(parser: argparse.ArgumentParser, required: bool = False) -> None:
    parser.add_argument("--session", help="a session id (or a unique prefix of one)")
    parser.add_argument("--attempt", type=int, help="an attempt's index")
    parser.add_argument("--camera", help="a camera's label")
    parser.add_argument("--segment", choices=SEGMENTS)


def _feature_options(parser: argparse.ArgumentParser) -> None:
    """The clips a features or bench run takes, and how it runs."""
    parser.add_argument(
        "--manifest", help="the manifest (parquet or CSV) to select from (default: built from the root now)"
    )
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--session", help="a session id (or a unique prefix of one)")
    parser.add_argument("--camera", help="a camera's label")
    parser.add_argument("--segment", choices=SEGMENTS)
    parser.add_argument(
        "--usable-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="only the clips the consistency filter keeps (default: on)",
    )
    parser.add_argument("--limit", type=int, help="the first N clips only")
    parser.add_argument(
        "--crop",
        choices=CROP_MODES,
        default="auto",
        help="the framing: the record's rectangle, else one from the motion (auto, the default); the "
        "record's only (record); the whole frame (none)",
    )
    parser.add_argument("--batch", type=int, default=64, help="frames per encoder call (default 64)")
    parser.add_argument(
        "--workers", type=int, default=1, help="clips decoded at once, ahead of the encoder (default 1)"
    )
    parser.add_argument("--device", choices=DEVICES, default="auto", help="auto: cuda when there is one")
    parser.add_argument(
        "--precision", choices=PRECISIONS, default="auto", help="auto: fp16 on cuda, fp32 on the CPU"
    )
    parser.add_argument("--time-base", choices=TIME_BASES, default="fit")
    _split_options(parser)


def parser() -> argparse.ArgumentParser:
    common = _common()
    top = argparse.ArgumentParser(prog="cubetrace-ml", description="The cubetrace recordings as a dataset.")
    commands = top.add_subparsers(dest="command", required=True)

    report = commands.add_parser("report", parents=[common], help="print what the dataset holds")
    _labels(report)
    _split_options(report)
    report.add_argument(
        "--video",
        choices=("auto", "none", "fast", "full"),
        default="auto",
        help="measure the videos' frame counts: none, fast (the container), full (a decode); "
        "auto: fast for a folder, none for gs:// (which would download every MP4)",
    )

    manifest = commands.add_parser("manifest", parents=[common], help="write the manifest (parquet and CSV)")
    _labels(manifest)
    _split_options(manifest)
    manifest.add_argument("--video", choices=("auto", "none", "fast", "full"), default="auto")
    manifest.add_argument("--out", default="out", help="the folder to write into (default: out)")

    splits = commands.add_parser("splits", parents=[common], help="print each session's split")
    _split_options(splits)

    inspect = commands.add_parser("inspect", parents=[common], help="a contact sheet (PNG) of one clip")
    _clip_options(inspect)
    inspect.add_argument("--row", type=int, help="the clip of this manifest row (0-based) instead")
    inspect.add_argument("--manifest", help="the manifest (parquet or CSV) --row reads (default: built now)")
    inspect.add_argument("--onsets", type=int, default=6, help="rows: onsets of the segment (default 6)")
    inspect.add_argument("--frames", type=int, default=5, help="frames around each onset (default 5)")
    inspect.add_argument("--tile-height", type=int, default=160)
    inspect.add_argument("--time-base", choices=TIME_BASES, default="fit")
    inspect.add_argument(
        "--out", help="the PNG (default: out/inspect-<session>-<attempt>-<camera>-<segment>.png)"
    )

    check = commands.add_parser(
        "check-alignment", parents=[common], help="each clip's frames file against its video and its window"
    )
    _clip_options(check)
    check.add_argument("--fast", action="store_true", help="trust the container's frame count (no decode)")
    check.add_argument("--time-base", choices=TIME_BASES, default="fit")

    commands.add_parser("validate", parents=[common], help="every record against its schema and its folder")

    features = commands.add_parser(
        "features", parents=[common], help="a frozen encoder's per-frame features of each clip (cached)"
    )
    features.add_argument("--encoder", required=True, choices=list(ENCODERS))
    features.add_argument("--out", help=f"the features root (default: ${FEATURES_ENV})")
    features.add_argument("--force", action="store_true", help="rewrite the clips already done")
    _feature_options(features)

    benchmark = commands.add_parser(
        "bench", parents=[common], help="the throughput of decoding, cropping and each encoder (no files)"
    )
    benchmark.add_argument(
        "--encoders",
        default="none,stub,resnet18,dinov2-vits14",
        help="comma-separated; none: decode, crop and resize alone (default: none and every encoder)",
    )
    benchmark.add_argument(
        "--random-weights",
        action="store_true",
        help="the PyTorch encoders' architectures with random weights (no download; the same compute)",
    )
    benchmark.add_argument("--size", type=int, default=224, help="the frames' size for none (default 224)")
    _feature_options(benchmark)

    preview = commands.add_parser(
        "crop-preview", parents=[common], help="a frame with the clip's crop rectangles drawn (PNG)"
    )
    _clip_options(preview)
    preview.add_argument("--row", type=int, help="the clip of this manifest row (0-based) instead")
    preview.add_argument("--manifest", help="the manifest (parquet or CSV) --row reads (default: built now)")
    preview.add_argument("--frame", type=int, help="the frame (default: the middle of the segment's window)")
    preview.add_argument("--height", type=int, default=720, help="the preview's height in pixels")
    preview.add_argument("--time-base", choices=TIME_BASES, default="fit")
    preview.add_argument(
        "--out", help="the PNG (default: out/crop-<session>-<attempt>-<camera>-<segment>.png)"
    )
    return top


def _dataset(args: argparse.Namespace) -> Dataset:
    return Dataset(args.root, cache_dir=args.cache, validate=not args.no_validate)


def _video_check(args: argparse.Namespace, dataset: Dataset) -> str:
    if args.video != "auto":
        return args.video
    return "none" if dataset.root.startswith("gs://") else "fast"


def _session(dataset: Dataset, text: str) -> str:
    sessions = dataset.sessions()
    if text in sessions:
        return text
    matches = [s for s in sessions if s.startswith(text)]
    if len(matches) != 1:
        raise ValueError(f"session {text!r} matches {len(matches)} sessions")
    return matches[0]


def _select(dataset: Dataset, args: argparse.Namespace) -> list[ClipRef]:
    session = _session(dataset, args.session) if args.session else None
    return [
        clip
        for clip in dataset.clips(session)
        if (args.attempt is None or clip.attempt == args.attempt)
        and (args.camera is None or clip.camera == args.camera)
        and (args.segment is None or clip.segment == args.segment)
    ]


def _tables(args: argparse.Namespace, dataset: Dataset, video: str):
    return build_tables(
        dataset,
        time_base=args.time_base,
        slice_ms=args.slice_ms,
        double_ms=args.double_ms,
        seed=args.seed,
        held_out_day=args.held_out_day,
        val_fraction=args.val_fraction,
        video=video,
    )


def cmd_report(args: argparse.Namespace) -> int:
    dataset = _dataset(args)
    tables = _tables(args, dataset, _video_check(args, dataset))
    print(report_text(tables, dataset.root))
    return 0


def cmd_manifest(args: argparse.Namespace) -> int:
    dataset = _dataset(args)
    tables = _tables(args, dataset, _video_check(args, dataset))
    for path in write_tables(tables, args.out):
        print(path)
    print(
        f"{len(tables.clips):,} clips, {len(tables.attempts):,} attempts; {len(tables.problems):,} problems"
    )
    return 0


def cmd_splits(args: argparse.Namespace) -> int:
    dataset = _dataset(args)
    tables = build_tables(
        dataset, seed=args.seed, held_out_day=args.held_out_day, val_fraction=args.val_fraction
    )
    sessions = tables.sessions.sort(["split", "day", "sessionId"])
    for row in sessions.iter_rows(named=True):
        record = "" if row["sessionRecord"] else "  (day from its attempts: no session.json)"
        day = row["day"] or "?"
        print(f"{row['split']:<5}  {day:<10}  {row['sessionId']}  {row['attempts']:>4} attempts{record}")
    counts = sessions.group_by("split").agg(pl.len().alias("n")).sort("split")
    print("  ".join(f"{r['split']}: {r['n']} sessions" for r in counts.iter_rows(named=True)))
    if not (sessions["split"] == "train").any():
        print("warning: no session left for train (one recording day?)", file=sys.stderr)
    return 0


def _one_clip(dataset: Dataset, args: argparse.Namespace) -> ClipRef:
    """The clip of `--row` (of `--manifest`, or of the manifest built now), or the one clip `--session`,
    `--attempt`, `--camera` and `--segment` select."""
    if args.row is not None:
        frame = read_manifest(args.manifest) if args.manifest else build_tables(dataset).clips
        if not 0 <= args.row < len(frame):
            raise ValueError(f"row {args.row}: the manifest has {len(frame)} rows")
        row = frame.row(args.row, named=True)
        return ClipRef(row["sessionId"], int(row["attemptIndex"]), row["camera"], row["segment"])
    clips = _select(dataset, args)
    if len(clips) != 1:
        raise ValueError(f"{len(clips)} clips match: name one with --session --attempt --camera --segment")
    return clips[0]


def cmd_inspect(args: argparse.Namespace) -> int:
    dataset = _dataset(args)
    clip = _one_clip(dataset, args)
    sheet = contact_sheet(
        dataset,
        clip,
        onsets=args.onsets,
        frames=args.frames,
        tile_height=args.tile_height,
        time_base=args.time_base,
    )
    out = Path(
        args.out or f"out/inspect-{clip.session[:8]}-{clip.attempt:04d}-{clip.camera}-{clip.segment}.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(out)
    return 0


def _seconds(ms: float | None) -> str:
    return "–" if ms is None else f"{ms / 1000:.2f} s"


def cmd_check_alignment(args: argparse.Namespace) -> int:
    dataset = _dataset(args)
    clips = _select(dataset, args)
    if not clips:
        raise ValueError("no clip matches")
    failed = 0
    for clip in clips:
        c = check_alignment(dataset, clip, fast=args.fast, time_base=args.time_base)
        decoded = "-" if c.decoded_frames is None else str(c.decoded_frames)
        timing = (
            f"pts vs frames file max {c.max_pts_diff_ms:.2f} ms"
            if c.max_pts_diff_ms is not None
            else f"span {c.frames_span_ms / 1000:.2f} s vs container {c.video_span_ms / 1000:.2f} s"
            if c.video_span_ms is not None
            else "no video span"
        )
        lag = "unsynced" if c.lag_ms is None else f"lag {c.lag_ms:g} ms"
        verdict = "ok" if c.ok else "MISMATCH"
        print(
            f"{verdict:<8} {clip}: frames file {c.frames_file}, record {c.record_frames}, container "
            f"{c.container_frames}, decoded {decoded}; {timing}; {lag}; "
            f"onsets in clip {c.covered}/{c.moves}; "
            f"margins {_seconds(c.lead_ms)} before, {_seconds(c.tail_ms)} after the window"
        )
        failed += not c.ok
    print(f"{len(clips) - failed}/{len(clips)} clips ok")
    return 1 if failed else 0


def cmd_validate(args: argparse.Namespace) -> int:
    dataset = Dataset(args.root, cache_dir=args.cache, validate=False)
    count, findings = validate_all(dataset)
    for finding in findings:
        print(finding)
    errors = sum(f.level == "error" for f in findings)
    print(f"{count:,} records under {dataset.root}: {errors:,} errors, {len(findings) - errors:,} warnings")
    return 1 if errors else 0


def _feature_clips(args: argparse.Namespace, dataset: Dataset) -> list[ClipRef]:
    if args.manifest:
        clips = read_manifest(args.manifest)
    else:
        video = "none" if dataset.root.startswith("gs://") else "fast"
        clips = build_tables(
            dataset,
            time_base=args.time_base,
            seed=args.seed,
            held_out_day=args.held_out_day,
            val_fraction=args.val_fraction,
            video=video,
        ).clips
    refs = select_clips(
        clips,
        split=args.split,
        session=args.session,
        camera=args.camera,
        segment=args.segment,
        usable_only=args.usable_only,
        limit=args.limit,
    )
    if not refs:
        raise ValueError("no clip matches")
    return refs


def cmd_features(args: argparse.Namespace) -> int:
    dataset = _dataset(args)
    root = features_root(args.out)
    refs = _feature_clips(args, dataset)
    encoder = load_encoder(args.encoder, device=args.device, precision=args.precision)
    stats = extract(
        dataset,
        refs,
        encoder,
        root,
        crop_mode=args.crop,
        time_base=args.time_base,
        batch=args.batch,
        workers=args.workers,
        force=args.force,
    )
    print(features_summary(stats, encoder))
    print(f"features root: {root}")
    return 1 if stats.failed else 0


def cmd_bench(args: argparse.Namespace) -> int:
    names = [name.strip() for name in args.encoders.split(",") if name.strip()]
    unknown = [name for name in names if name != "none" and name not in ENCODERS]
    if unknown or not names:
        raise ValueError(f"--encoders {args.encoders!r}: none or {', '.join(ENCODERS)}")
    dataset = _dataset(args)
    refs = _feature_clips(args, dataset)
    rows = bench(
        dataset,
        refs,
        names,
        crop_mode=args.crop,
        time_base=args.time_base,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        precision=args.precision,
        pretrained=not args.random_weights,
        size=args.size,
    )
    print()
    print(bench_table(rows))
    return 0


def cmd_crop_preview(args: argparse.Namespace) -> int:
    dataset = _dataset(args)
    clip = _one_clip(dataset, args)
    preview = crop_preview(dataset, clip, time_base=args.time_base, frame=args.frame, height=args.height)
    out = Path(args.out or f"out/crop-{clip.session[:8]}-{clip.attempt:04d}-{clip.camera}-{clip.segment}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    preview.image.save(out)
    print(out)
    record = preview.record.describe() if preview.record else "none"
    print(
        f"frame {preview.frame}; auto: {preview.auto.describe()}; motion: {preview.motion.describe()}; "
        f"record: {record}"
    )
    return 0


COMMANDS = {
    "report": cmd_report,
    "manifest": cmd_manifest,
    "splits": cmd_splits,
    "inspect": cmd_inspect,
    "check-alignment": cmd_check_alignment,
    "validate": cmd_validate,
    "features": cmd_features,
    "bench": cmd_bench,
    "crop-preview": cmd_crop_preview,
}


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except (ValueError, KeyError, FileNotFoundError, EncoderError) as error:
        print(f"cubetrace-ml {args.command}: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
