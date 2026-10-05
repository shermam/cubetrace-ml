from pathlib import Path

import numpy as np
import pytest

from cubetrace_ml.video import count_frames, crop_box, read_frames
from factory import gray, write_video


@pytest.fixture(scope="module")
def video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("video") / "laptop.solve.mp4"
    write_video(path, 20, (64, 48), fps=30)
    return path


def test_the_frame_count_from_the_header_and_from_a_decode(video: Path) -> None:
    fast = count_frames(video, decode=False)
    assert fast.container_frames == 20 and fast.decoded_frames is None and fast.frames == 20
    assert (fast.width, fast.height) == (64, 48)
    full = count_frames(video)
    assert full.decoded_frames == 20 and full.frames == 20
    assert full.pts_ms is not None and full.pts_ms[0] == 0.0
    assert np.allclose(np.diff(full.pts_ms), 1000 / 30)
    assert full.span_ms == pytest.approx(19 * 1000 / 30)


def test_frames_by_index_cropped_and_resized(video: Path) -> None:
    images = read_frames(video, [3, 0, 17, 3])
    assert sorted(images) == [0, 3, 17]
    for k, image in images.items():
        assert image.size == (64, 48)
        assert abs(np.asarray(image).mean() - gray(k)) < 4  # each frame is its own gray level
    cropped = read_frames(video, [5], crop={"x": 8, "y": 4, "w": 32, "h": 16}, height=8)
    assert cropped[5].size == (16, 8)
    assert read_frames(video, []) == {}


def test_crop_box_is_clamped_to_the_frame() -> None:
    assert crop_box(None, 64, 48) is None
    assert crop_box({"x": 8, "y": 4, "w": 32, "h": 16}, 64, 48) == (8, 4, 40, 20)
    assert crop_box({"x": 60, "y": 40, "w": 32, "h": 16}, 64, 48) == (60, 40, 64, 48)
