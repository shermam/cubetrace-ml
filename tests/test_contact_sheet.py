import numpy as np

from cubetrace_ml.align import align_clip
from cubetrace_ml.contact_sheet import contact_sheet, pick_rows
from cubetrace_ml.dataset import ClipRef, Dataset
from factory import gray


def test_rows_are_onsets_of_the_segment_with_the_frames_around_them(dataset_root) -> None:
    root, ids = dataset_root
    dataset = Dataset(root)
    clip = ClipRef(ids["B"], 1, "laptop", "solve")
    attempt = dataset.attempt(clip.session, clip.attempt)
    aligned = align_clip(attempt, dataset.frames(clip), None, camera="laptop", segment="solve")
    rows = pick_rows(aligned, onsets=6, frames=5)
    assert [r.symbol for r in rows] == ["R2", "M'", "U", "D'"]  # fewer onsets than asked: all of them
    for r in rows:
        assert len(r.frames) == 5 and r.frames == list(range(r.frames[0], r.frames[0] + 5))
        nearest = int(np.argmin(np.abs(r.distances_ms)))
        assert nearest == 2  # centred on the frame nearest the onset
        assert all(d2 > d1 for d1, d2 in zip(r.distances_ms, r.distances_ms[1:], strict=False))
    assert [r.symbol for r in pick_rows(aligned, onsets=2, frames=3)] == ["R2", "D'"]  # spread evenly
    assert pick_rows(aligned, onsets=0, frames=3) == []


def test_the_sheet_shows_the_frames_asked_for(dataset_root) -> None:
    root, ids = dataset_root
    dataset = Dataset(root)
    clip = ClipRef(ids["B"], 1, "laptop", "solve")
    sheet = contact_sheet(dataset, clip, onsets=2, frames=3, tile_height=32)
    assert sheet.mode == "RGB" and sheet.width > 3 * 32 and sheet.height > 2 * 32
    attempt = dataset.attempt(clip.session, clip.attempt)
    aligned = align_clip(attempt, dataset.frames(clip), None, camera="laptop", segment="solve")
    rows = pick_rows(aligned, onsets=2, frames=3)
    # The first tile of the first row is the frame rows[0].frames[0]: its gray level is its index's.
    pixels = np.asarray(sheet)
    title_h = 6 + 18 * 3 + 6
    tile = pixels[title_h + 8 : title_h + 24, 96 + 8 : 96 + 24]
    assert abs(tile.mean() - gray(rows[0].frames[0])) < 4
