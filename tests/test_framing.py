"""The framing: the resize and the letterbox of the decode, and the crop found from the motion."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cubetrace_ml.framing import (
    MOTION,
    Framing,
    blur,
    find_motion,
    framing_for,
    mass_box,
    motion_energy,
    motion_framing,
    preview_image,
    record_framing,
    square_around,
)
from cubetrace_ml.video import decode_gray, decode_square, letterbox_size
from factory import write_frames

RED, BLUE = (220, 30, 30), (30, 30, 220)


@pytest.fixture(scope="module")
def halves(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """96 × 64 frames: red left of x = 48, blue from it."""
    pixels = np.zeros((64, 96, 3), np.uint8)
    pixels[:, :48], pixels[:, 48:] = RED, BLUE
    path = tmp_path_factory.mktemp("halves") / "halves.mp4"
    write_frames(path, [pixels] * 3)
    return path


def colour(region: np.ndarray) -> np.ndarray:
    return region.reshape(-1, 3).mean(axis=0)


def test_letterbox_size() -> None:
    assert letterbox_size(816, 703, 224) == (224, 193)
    assert letterbox_size(703, 816, 224) == (193, 224)
    assert letterbox_size(64, 64, 32) == (32, 32)


def test_a_square_box_is_scaled_without_bars(halves: Path) -> None:
    frames = decode_square(halves, (16, 0, 64, 64), 32)  # x 16..79: half red, half blue
    assert frames.shape == (3, 32, 32, 3) and frames.dtype == np.uint8
    assert np.abs(colour(frames[:, :, 1:15]) - RED).max() < 12
    assert np.abs(colour(frames[:, :, 17:31]) - BLUE).max() < 12


def test_a_rectangle_is_letterboxed_on_black(halves: Path) -> None:
    frames = decode_square(halves, (24, 16, 48, 24), 32)  # 48 × 24 → 32 × 16, 8 black rows above and below
    assert frames.shape == (3, 32, 32, 3)
    assert frames[:, :8].max() < 8 and frames[:, 24:].max() < 8
    assert np.abs(colour(frames[:, 9:23, 1:15]) - RED).max() < 12
    assert np.abs(colour(frames[:, 9:23, 17:31]) - BLUE).max() < 12
    whole = decode_square(halves, None, 32)  # 96 × 64 → 32 × 21, 5 rows above and 6 below
    assert whole[:, :5].max() < 8 and whole[:, 26:].max() < 8 and whole[:, 6:25].min() > 8
    with pytest.raises(ValueError, match="not inside the 96x64 frame"):
        decode_square(halves, (64, 0, 48, 64), 32)


def test_the_gray_frames_for_the_motion(halves: Path) -> None:
    gray, size = decode_gray(halves, 32)
    assert size == (96, 64) and gray.shape == (3, 32, 48) and gray.dtype == np.uint8
    assert decode_gray(halves, 160)[0].shape == (3, 64, 96)  # never larger than the video


def blob_frames(n: int = 48, size: tuple[int, int] = (192, 128)) -> tuple[list[np.ndarray], np.ndarray]:
    """A 12 × 12 bright blob going round a circle of radius 20 about (130, 70) on a dark, textured
    background, and a 2 × 2 speck flickering faintly at (10, 10); the blob's centres."""
    width, height = size
    rng = np.random.default_rng(1)
    background = rng.integers(10, 40, (height, width, 1), dtype=np.uint8).repeat(3, axis=2)
    centres = np.array(
        [(130 + 20 * np.cos(2 * np.pi * k / n), 70 + 20 * np.sin(2 * np.pi * k / n)) for k in range(n)]
    )
    out = []
    for k, (cx, cy) in enumerate(centres):
        pixels = background.copy()
        x, y = round(cx), round(cy)
        pixels[y - 6 : y + 6, x - 6 : x + 6] = 235
        pixels[10:12, 10:12] = 60 if (k // 4) % 2 else 20
        out.append(pixels)
    return out, centres


def test_the_motion_crop_frames_the_moving_blob(tmp_path: Path) -> None:
    frames, centres = blob_frames()
    path = tmp_path / "blob.mp4"
    write_frames(path, frames)
    framing, energy = find_motion(path)
    assert framing.source == "motion" and framing.square
    assert (framing.width, framing.height) == (192, 128)
    assert energy.shape == (128, 192)  # 128 is the shorter side: no scaling under 160
    x0, y0 = centres.min(axis=0) - 6
    x1, y1 = centres.max(axis=0) + 6
    assert framing.x <= x0 and framing.y <= y0  # the blob's whole path is inside the square
    assert framing.x + framing.w >= x1 and framing.y + framing.h >= y1
    assert framing.x > 12 and framing.y > 12  # the speck is not
    assert min(framing.x, framing.y) >= 0 and framing.x + framing.w <= 192 and framing.y + framing.h <= 128
    assert framing.w < 128  # padded, not the whole height


def test_the_motion_of_the_window_only() -> None:
    gray = np.zeros((6, 8, 8), np.uint8)
    gray[1, 0, 0] = 100  # changes between frames 0, 1 and 2
    gray[4, 7, 7] = 100  # changes between frames 3, 4 and 5
    every = motion_energy(gray)
    assert every[0, 0] == 200 and every[7, 7] == 200
    late = motion_energy(gray, np.array([0, 0, 0, 1, 1, 1], bool))  # differences 4−3 and 5−4
    assert late[0, 0] == 0 and late[7, 7] == 200
    np.testing.assert_array_equal(motion_energy(gray, np.zeros(6, bool)), every)  # none selected: every one
    assert blur(every, 1)[0, 0] == 200 and blur(every, 5).sum() == pytest.approx(every.sum(), rel=0.3)


def test_the_mass_box_ignores_a_speck_above_the_threshold() -> None:
    energy = np.zeros((100, 100))
    energy[40:60, 50:80] = 1.0
    energy[5, 5] = 0.9  # above 15% of the peak, but a sliver of the mass
    # The 5th–95th percentiles drop the speck (and trim the block's 30 columns by about 1.5 on each side).
    assert mass_box(energy) == (51, 40, 79, 59)  # half-open
    assert mass_box(energy, low=0, high=100) == (5, 5, 80, 60)  # without the percentiles: the speck's box
    assert mass_box(np.zeros((4, 4))) is None


def test_the_square_is_padded_and_clamped_to_the_frame() -> None:
    assert square_around((40, 40, 60, 50), 200, 100, pad=0.0, min_side=0.0) == (40, 34, 20)
    assert square_around((40, 40, 60, 50), 200, 100, pad=0.5, min_side=0.0) == (30, 24, 40)
    assert square_around((0, 0, 10, 10), 200, 100, pad=0.5, min_side=0.0) == (0, 0, 20)  # shifted in
    assert square_around((190, 90, 200, 100), 200, 100, pad=0.0, min_side=0.0) == (190, 90, 10)
    assert square_around((10, 10, 190, 20), 200, 100, pad=0.0, min_side=0.0) == (50, 0, 100)  # at most 100
    assert square_around((48, 48, 52, 52), 200, 100, pad=0.0, min_side=0.25) == (38, 38, 24)  # at least 25
    energy = np.zeros((50, 100))
    energy[:5, 95:] = 1.0  # motion in a corner of a 200 × 100 frame
    framing, _ = motion_framing(np.zeros((1, 50, 100), np.uint8), (200, 100))
    assert framing == Framing(0, 0, 200, 100, "none", 200, 100)  # nothing moves: the whole frame
    corner = np.stack([np.zeros((50, 100), np.uint8), (energy * 200).astype(np.uint8)])
    framing, _ = motion_framing(corner, (200, 100))
    assert framing.square and framing.x + framing.w == 200 and framing.y == 0


def test_the_framing_of_each_mode() -> None:
    entry = {"width": 1920, "height": 1080, "crop": {"x": 477, "y": 154, "w": 816, "h": 703}}
    record = framing_for("auto", entry)
    assert record == Framing(477, 154, 816, 703, "record", 1920, 1080) and not record.square
    assert framing_for("record", entry) == record
    assert framing_for("none", entry) == Framing(0, 0, 1920, 1080, "none", 1920, 1080)
    phone = {"width": 1080, "height": 1920, "crop": None}
    assert framing_for("auto", phone) is None  # from the motion
    assert framing_for("record", phone) == Framing(0, 0, 1080, 1920, "none", 1080, 1920)
    assert record_framing({"x": 1900, "y": 0, "w": 100, "h": 50}, 1920, 1080) == Framing(
        1900, 0, 20, 50, "record", 1920, 1080
    )
    assert Framing.from_json(record.to_json()) == record
    assert record.to_json()["letterboxed"] is True
    with pytest.raises(ValueError, match="crop mode 'tight'"):
        framing_for("tight", entry)
    assert MOTION.to_json()["threshold"] == 0.15


def test_the_preview_draws_on_the_frame_and_the_energy() -> None:
    frame = Image.new("RGB", (192, 128), (40, 40, 40))
    energy = np.zeros((64, 96))
    energy[20:40, 50:70] = 1
    framings = [Framing(100, 40, 40, 40, "motion", 192, 128), Framing(0, 0, 50, 40, "record", 192, 128)]
    image = preview_image(frame, energy, framings, ["a title", "a second line"], height=256)
    assert image.size == (2 * 384 + 8, 8 + 2 * 18 + 256)
