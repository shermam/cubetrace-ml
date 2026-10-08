"""The rotations (NumPy): matrices and quaternions, the cube's 24 symmetries and the estimation's grids, the
exponential map, headings, the scramble's pose, and the calibrated channels (a yaw of 90° maps F's
direction to R's)."""

import math

import numpy as np
import pytest

from cubetrace_ml.align import conjugate, quaternion_product, relative_rotations
from cubetrace_ml.orientation import (
    IDENTITY,
    about,
    angle,
    angle_between,
    body_rotations,
    calibrated_channels,
    calibration_grid,
    channel_names,
    chordal_mean,
    cube_rotations,
    distinct,
    exp_map,
    from_matrices,
    heading,
    hemisphere,
    log_map,
    matrices,
    pose_calibration,
    project,
    rotation_grid,
    scramble_pose,
    unit,
    yaw_grid,
)
from factory import T0, attempt_record, gyro_record, session_id

RNG = np.random.default_rng(11)


def random_rotations(n: int) -> np.ndarray:
    return hemisphere(unit(RNG.normal(size=(n, 4))))


def test_matrices_and_quaternions() -> None:
    q = random_rotations(200)
    m = matrices(q)
    np.testing.assert_allclose(m @ np.swapaxes(m, -1, -2), np.tile(np.eye(3), (200, 1, 1)), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(m), 1.0, atol=1e-12)
    np.testing.assert_allclose(from_matrices(m), q, atol=1e-12)  # both in w ≥ 0
    # The product is the matrices' product, and column k is the image of axis k: q · e_k · q*.
    a, b = q[:100], q[100:]
    np.testing.assert_allclose(matrices(quaternion_product(a, b)), matrices(a) @ matrices(b), atol=1e-12)
    pure = np.concatenate([np.tile([0.0, 1.0, 0.0], (100, 1)), np.zeros((100, 1))], axis=1)
    image = quaternion_product(quaternion_product(a, pure), conjugate(a))[:, :3]
    np.testing.assert_allclose(image, m[:100, :, 1], atol=1e-12)


def test_the_cube_s_24_rotations_and_the_grids() -> None:
    cube = cube_rotations()
    assert cube.shape == (24, 4) and len(distinct(cube)) == 24
    np.testing.assert_allclose(cube[0], IDENTITY)
    m = matrices(cube)
    assert np.allclose(np.round(m), m, atol=1e-12)  # signed permutation matrices
    products = quaternion_product(cube[:, None, :], cube[None, :, :]).reshape(-1, 4)
    assert len(distinct(np.concatenate([cube, products]))) == 24  # closed under composition
    yaws = yaw_grid("z")
    assert yaws.shape == (24, 4)
    np.testing.assert_allclose(np.degrees(heading(yaws, "z")) % 360, np.arange(0, 360, 15), atol=1e-9)
    grid = rotation_grid("z")
    assert grid.shape == (144, 4) and len(distinct(grid)) == 144
    # Each grid rotation takes one of the six axis directions to the vertical (its matrix's last row), each
    # with the 24 yaws about it.
    up = np.round(matrices(grid)[:, 2, :]).astype(int)
    places = {tuple(row) for row in up}
    assert len(places) == 6 and all((up == p).all(axis=1).sum() == 24 for p in places)
    assert calibration_grid("yaw", "y").shape == (24, 4) and calibration_grid("rotation").shape == (144, 4)
    with pytest.raises(ValueError, match="calibration dof 'tilt'"):
        calibration_grid("tilt")


def test_the_exponential_map() -> None:
    r = RNG.normal(size=(100, 3))
    r *= (RNG.uniform(0.0, math.pi * 0.999, 100) / np.linalg.norm(r, axis=1))[:, None]
    np.testing.assert_allclose(log_map(exp_map(r)), r, atol=1e-10)
    np.testing.assert_allclose(exp_map(np.zeros(3)), IDENTITY)
    np.testing.assert_allclose(exp_map(np.array([1e-9, 0, 0])), [5e-10, 0, 0, 1], atol=1e-15)
    np.testing.assert_allclose(angle(exp_map(np.array([0.0, 0.0, math.pi / 2]))), 90.0)
    np.testing.assert_allclose(exp_map(np.array([0.0, 0.0, 0.3])), about("z", 0.3))


def test_headings_and_the_pose_s_calibration() -> None:
    for degrees in (0.0, 30.0, 100.0, -170.0):
        tilted = quaternion_product(about("z", math.radians(degrees)), about("x", math.radians(15)))
        assert math.degrees(float(heading(tilted, "z"))) == pytest.approx(degrees)
        assert math.degrees(float(heading(about("y", math.radians(degrees)), "y"))) == pytest.approx(degrees)
    pose = quaternion_product(about("z", math.radians(77)), about("x", math.radians(10)))
    yaw = pose_calibration(pose, "yaw", "z")
    np.testing.assert_allclose(yaw, about("z", math.radians(-77)), atol=1e-12)
    assert float(heading(quaternion_product(yaw, pose), "z")) == pytest.approx(0.0, abs=1e-12)
    whole = pose_calibration(pose, "rotation")
    assert float(angle(quaternion_product(whole, pose))) == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(project(pose, "yaw", "z"), about("z", math.radians(77)), atol=1e-12)
    np.testing.assert_allclose(project(-pose, "rotation"), pose)
    assert float(angle_between(about("z", 0.3), -about("z", -0.2))) == pytest.approx(math.degrees(0.5))
    # The chordal mean is sign-free.
    samples = np.stack([about("z", math.radians(d)) for d in (-5.0, 0.0, 5.0)])
    samples[1] *= -1
    np.testing.assert_allclose(chordal_mean(samples), IDENTITY, atol=1e-12)


def test_a_yaw_of_90_degrees_maps_f_s_direction_to_r_s() -> None:
    # The cube's axes (the app's schema): +x through R, +y through B, +z through U; F is −y.
    raw = np.zeros((3, 9))
    raw[:, :4] = IDENTITY  # the cube at the gyro frame's reference
    raw[:, 4:8] = IDENTITY
    raw[:, 8] = 1.0
    raw[2] = [0, 0, 0, 0, 0, 0, 0, 1, 0]  # a frame without an orientation
    c = about("z", math.radians(90))
    channels = calibrated_channels(raw, c, "matrix")
    assert channels.shape == (3, 14) and len(channel_names("matrix")) == 14
    before = matrices(IDENTITY)
    after = channels[0, :9].reshape(3, 3)
    f, r = np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(after @ f, before @ r, atol=1e-7)  # F now points where R pointed
    np.testing.assert_allclose(channels[0, 9:13], IDENTITY)  # the cube's own change, untouched
    assert channels[2, :9].tolist() == [0.0] * 9 and channels[2, 13] == 0.0  # no orientation: zeros
    # Sign-free: −q gives the same matrix; the quaternion form takes w ≥ 0 after the product.
    flipped = raw.copy()
    flipped[:, :4] *= -1
    np.testing.assert_array_equal(calibrated_channels(flipped, c, "matrix"), channels)
    quat = calibrated_channels(flipped, c, "quat")
    assert quat.shape == (3, 9) and channel_names("quat")[:4] == ("cqx", "cqy", "cqz", "cqw")
    np.testing.assert_allclose(quat[0, :4], c, atol=1e-7)
    with pytest.raises(ValueError, match="orientation 'euler'"):
        calibrated_channels(raw, c, "euler")


def test_the_cube_s_own_change_is_free_of_any_calibration() -> None:
    q = random_rotations(20)
    q[5] = np.nan
    c = random_rotations(1)[0]
    turned = quaternion_product(c[None, :], q)
    np.testing.assert_allclose(body_rotations(turned), body_rotations(q), atol=1e-12)
    np.testing.assert_allclose(body_rotations(q)[[0, 5, 6]], np.tile(IDENTITY, (3, 1)))
    # M3's change, in the gyro's frame, turns with the calibration: c · Δ · c*.
    world = relative_rotations(q)
    expected = hemisphere(quaternion_product(quaternion_product(c[None, :], world), conjugate(c)[None, :]))
    np.testing.assert_allclose(relative_rotations(turned), expected, atol=1e-12)


def test_the_scramble_pose() -> None:
    sid = session_id(9)
    scramble = [("R", T0 + 1000.0), ("U", T0 + 1300.0), ("F", T0 + 1600.0)]
    attempt = attempt_record(sid, 1, scramble, [("F'", T0 + 4000.0), ("U'", T0 + 4300.0)])
    pose = quaternion_product(about("z", math.radians(40)), about("x", math.radians(12)))
    # Samples every 100 ms from T0 + 500: the cube at the pose through the scramble, turned elsewhere.
    times = T0 + 500.0 + 100.0 * np.arange(45)
    quats = [pose if T0 + 1000 <= t <= T0 + 1600 else about("x", math.radians(150)) for t in times]
    quats = [q if k % 2 else -q for k, q in enumerate(quats)]  # either sign
    gyro = gyro_record(sid, 1, float(times[0]), [0.0] + [100.0] * 44, quats)
    np.testing.assert_allclose(scramble_pose(attempt, gyro), hemisphere(pose), atol=1e-12)
    assert scramble_pose(attempt, None) is None
    sparse = gyro_record(sid, 1, T0 + 1100.0, [0.0, 1000.0], [pose, pose])  # one sample in the window
    assert scramble_pose(attempt, sparse) is None
