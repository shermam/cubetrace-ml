"""The cube's orientation as rotations (NumPy only): unit quaternions (x, y, z, w, the Hamilton product, as
`gyro.json` has them), their rotation matrices, the 24 rotations of the cube's symmetry group, rotations
about an axis, the exponential map, chordal means, and the calibration of an orientation into a camera's
frame.

The convention: a sample's quaternion q takes the cube's own axes into the gyro's frame (a vector v of the
cube is `q · v · q*` in the gyro's frame), so column k of its rotation matrix is the cube's axis k seen
from the gyro's frame, and the cube's own change between two samples is `conj(q_t) · q_{t+1}`
(`cubetrace-ml gyro-frames` checks it on the records: the cube's angular velocity `v`, a body-frame
quantity, follows that change and not `q_{t+1} · conj(q_t)`). A change of the reference frame therefore
acts on the left: the orientation in a camera's frame is `q_cam = c · q` for one rotation `c` between the
gyro's frame and the camera's.

The cube's axes (the app's schema): +x through the red face, +y through the blue, +z through the white;
`gyro-frames` finds which of the gyro's axes is gravity's (z on the records).
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np

from .align import conjugate, event_times, gyro_samples, quaternion_product, segment_window
from .moves import move_times

AXES = ("x", "y", "z")
IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])
# The calibration's degrees of freedom: a whole rotation, or a yaw about the gravity axis alone.
DOFS = ("yaw", "rotation")
ORIENTATIONS = ("matrix", "quat")
YAW_STEPS = 24  # the estimation's yaw grid: 15° steps
MIN_POSE_SAMPLES = 3  # a scramble pose needs at least this many gyro samples in the scramble's window


def axis_index(axis: str) -> int:
    if axis not in AXES:
        raise ValueError(f"axis {axis!r}: one of {', '.join(AXES)}")
    return AXES.index(axis)


def unit(q: np.ndarray) -> np.ndarray:
    """Quaternions (…, 4) scaled to unit length."""
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def hemisphere(q: np.ndarray) -> np.ndarray:
    """q or −q, whichever has w ≥ 0: one representative of the rotation (the identity's sign at w = 0 is
    whatever the input had)."""
    q = np.asarray(q, dtype=np.float64)
    return np.where(q[..., 3:4] < 0, -q, q)


def matrices(q: np.ndarray) -> np.ndarray:
    """The rotation matrices (…, 3, 3) of unit quaternions (…, 4): column k is the image of axis k."""
    x, y, z, w = np.moveaxis(np.asarray(q, dtype=np.float64), -1, 0)
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=-2,
    )


def from_matrices(m: np.ndarray) -> np.ndarray:
    """The unit quaternions (…, 4, w ≥ 0) of rotation matrices (…, 3, 3): the eigenvector of the largest
    eigenvalue of each matrix's symmetric 4 × 4 form (Bar-Itzhack), which is exact for a rotation and the
    nearest rotation's quaternion otherwise."""
    m = np.asarray(m, dtype=np.float64)
    shape = m.shape[:-2]
    m = m.reshape(-1, 3, 3)
    xx, xy, xz = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    yx, yy, yz = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    zx, zy, zz = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]
    k = (
        np.stack(
            [
                np.stack([xx - yy - zz, yx + xy, zx + xz, zy - yz], axis=-1),
                np.stack([yx + xy, yy - xx - zz, zy + yz, xz - zx], axis=-1),
                np.stack([zx + xz, zy + yz, zz - xx - yy, yx - xy], axis=-1),
                np.stack([zy - yz, xz - zx, yx - xy, xx + yy + zz], axis=-1),
            ],
            axis=-2,
        )
        / 3.0
    )
    _, vectors = np.linalg.eigh(k)
    q = hemisphere(vectors[..., -1])
    return q.reshape(*shape, 4)


def about(axis: str | np.ndarray, radians: Any) -> np.ndarray:
    """The rotations by `radians` (a number or an array) about an axis (`x`, `y`, `z` or a vector)."""
    vector = np.eye(3)[axis_index(axis)] if isinstance(axis, str) else np.asarray(axis, dtype=np.float64)
    vector = vector / np.linalg.norm(vector)
    half = np.asarray(radians, dtype=np.float64)[..., None] / 2.0
    return np.concatenate([np.sin(half) * vector, np.cos(half)], axis=-1)


def exp_map(r: np.ndarray) -> np.ndarray:
    """The unit quaternions of rotation vectors (…, 3): the rotation by |r| about r / |r|."""
    r = np.asarray(r, dtype=np.float64)
    theta = np.linalg.norm(r, axis=-1, keepdims=True)
    small = theta < 1e-8
    scale = np.where(small, 0.5 - theta**2 / 48.0, np.sin(theta / 2.0) / np.where(small, 1.0, theta))
    return np.concatenate([scale * r, np.cos(theta / 2.0)], axis=-1)


def log_map(q: np.ndarray) -> np.ndarray:
    """The rotation vectors (…, 3, at most π long) of unit quaternions (…, 4): exp_map's inverse."""
    q = hemisphere(unit(q))
    v = q[..., :3]
    s = np.linalg.norm(v, axis=-1, keepdims=True)
    theta = 2.0 * np.arctan2(s, q[..., 3:4])
    return np.where(s > 1e-12, v / np.where(s > 1e-12, s, 1.0) * theta, 2.0 * v)


def angle(q: np.ndarray) -> np.ndarray:
    """The rotation angle of unit quaternions, in degrees (0–180)."""
    w = np.abs(np.asarray(q, dtype=np.float64)[..., 3])
    return np.degrees(2.0 * np.arccos(np.clip(w, 0.0, 1.0)))


def angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The angle in degrees of the rotation that takes orientation `a` to `b` (sign-free)."""
    dot = np.abs(np.sum(unit(a) * unit(b), axis=-1))
    return np.degrees(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))


def heading(q: np.ndarray, axis: str = "z") -> np.ndarray:
    """The yaw of orientations about an axis of their reference frame, in radians: the angle φ whose rotation
    about the axis, undone, brings the orientation nearest the identity (`about(axis, −φ) · q`); for a cube
    upright about that axis, its turn about it. Undefined (0) for an orientation upside down about it."""
    k = axis_index(axis)
    i, j = (k + 1) % 3, (k + 2) % 3
    m = matrices(q)
    return np.arctan2(m[..., j, i] - m[..., i, j], m[..., i, i] + m[..., j, j])


def chordal_mean(q: np.ndarray) -> np.ndarray:
    """The mean orientation of unit quaternions (n × 4), sign-free: the principal eigenvector of Σ q qᵀ,
    w ≥ 0."""
    q = unit(np.asarray(q, dtype=np.float64).reshape(-1, 4))
    _, vectors = np.linalg.eigh(q.T @ q)
    return hemisphere(vectors[:, -1])


def cube_rotations() -> np.ndarray:
    """The 24 rotations of the cube's symmetry group (24 × 4, w ≥ 0): the signed permutation matrices of
    determinant 1, the identity first."""
    found = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            m = np.zeros((3, 3))
            for row, col in enumerate(perm):
                m[row, col] = signs[row]
            if np.linalg.det(m) > 0:
                found.append(m)
    q = from_matrices(np.stack(found))
    order = np.argsort(-q[:, 3], kind="stable")  # the identity (w = 1) first
    return q[order]


def yaw_grid(axis: str = "z", steps: int = YAW_STEPS) -> np.ndarray:
    """`steps` rotations about the axis, evenly spaced from 0 (steps × 4)."""
    return about(axis, 2.0 * math.pi * np.arange(steps) / steps)


def distinct(q: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
    """The rotations of `q` (n × 4) without repeats (q and −q are one), in their first order."""
    kept: list[np.ndarray] = []
    for row in hemisphere(unit(q)):
        if not any(abs(float(np.dot(row, other))) > 1.0 - tolerance for other in kept):
            kept.append(row)
    return np.stack(kept)


def rotation_grid(axis: str = "z", steps: int = YAW_STEPS) -> np.ndarray:
    """The estimation's grid over every rotation: each yaw of `yaw_grid` composed with each of the cube's 24
    symmetries (`yaw · symmetry`), repeats dropped: which of the six axis directions the rotation takes to
    the gravity axis, times the yaws about it (144 rotations for 15° steps)."""
    yaws, cube = yaw_grid(axis, steps), cube_rotations()
    return distinct(quaternion_product(yaws[:, None, :], cube[None, :, :]).reshape(-1, 4))


def calibration_grid(dof: str, axis: str = "z", steps: int = YAW_STEPS) -> np.ndarray:
    """The candidates the estimation starts from: the yaws about the gravity axis (`yaw`), or every
    rotation of `rotation_grid` (`rotation`)."""
    if dof not in DOFS:
        raise ValueError(f"calibration dof {dof!r}: one of {', '.join(DOFS)}")
    return yaw_grid(axis, steps) if dof == "yaw" else rotation_grid(axis, steps)


def body_rotations(q: np.ndarray) -> np.ndarray:
    """The cube's own change between consecutive rows of `q` (n × 4, unit, NaN where none): `conj(q_{t−1}) ·
    q_t`, the rotation in the cube's frame from the orientation at t − 1 to the one at t, in the hemisphere
    w ≥ 0; the identity for the first row and wherever either row has none. No change of the reference frame
    (`c · q`) changes it."""
    q = np.asarray(q, dtype=np.float64)
    out = np.tile(IDENTITY, (len(q), 1))
    if len(q) < 2:
        return out
    present = np.isfinite(q).all(axis=1)
    both = np.flatnonzero(present[1:] & present[:-1]) + 1
    out[both] = hemisphere(quaternion_product(conjugate(q[both - 1]), q[both]))
    return out


def calibrated_channels(raw: np.ndarray, c: np.ndarray, orientation: str = "matrix") -> np.ndarray:
    """The calibrated gyro channels of frames (NumPy; the models compute the same in PyTorch) from their raw
    block `raw` (n × 9: the orientation q in w ≥ 0, zeros where none; the cube's own change; the presence
    flag) and one calibration `c` (4): the orientation in the camera's frame `c · q` as its rotation
    matrix's 9 entries, row by row (`matrix`, sign-free), or as the quaternion in w ≥ 0 (`quat`), zeros where
    the frame has none; then the change and the flag as they are (n × 14, or n × 9)."""
    raw = np.asarray(raw, dtype=np.float64)
    q, change, flag = raw[:, :4], raw[:, 4:8], raw[:, 8:9]
    cam = quaternion_product(np.asarray(c, dtype=np.float64)[None, :], q)
    if orientation == "matrix":
        rotated = matrices(cam).reshape(len(raw), 9) * flag
    elif orientation == "quat":
        rotated = hemisphere(cam) * flag
    else:
        raise ValueError(f"orientation {orientation!r}: one of {', '.join(ORIENTATIONS)}")
    return np.concatenate([rotated, change, flag], axis=1).astype(np.float32)


def channel_names(orientation: str = "matrix") -> tuple[str, ...]:
    """The calibrated channels' names: the matrix's entries `r00`…`r22` (row, column) or `cqx`…`cqw`, then the
    cube's change `bqx`…`bqw` and the flag `gyro`."""
    if orientation == "matrix":
        rotated = tuple(f"r{i}{j}" for i in range(3) for j in range(3))
    elif orientation == "quat":
        rotated = ("cqx", "cqy", "cqz", "cqw")
    else:
        raise ValueError(f"orientation {orientation!r}: one of {', '.join(ORIENTATIONS)}")
    return (*rotated, "bqx", "bqy", "bqz", "bqw", "gyro")


# The scramble's pose.


def scramble_pose(
    attempt: dict[str, Any], gyro: dict[str, Any] | None, time_base: str = "fit"
) -> np.ndarray | None:
    """The cube's mean orientation (the chordal mean, w ≥ 0) over the gyro's samples inside the attempt's
    scramble window (`scrambleStart` to `scrambleDone` on the time base); None without gyro.json or with
    fewer than MIN_POSE_SAMPLES samples there. The app prescribes the scramble in the cube's own frame, so
    the solver holds the cube in one way while applying it: its pose there is the attempt's reference."""
    if gyro is None:
        return None
    sample_ms, quats = gyro_samples(gyro)
    times, _ = move_times(attempt, time_base)
    low, high = segment_window(event_times(attempt, times), "scramble")
    if not (math.isfinite(low) and math.isfinite(high)):
        return None
    inside = (sample_ms >= low) & (sample_ms <= high)
    if inside.sum() < MIN_POSE_SAMPLES:
        return None
    return chordal_mean(quats[inside])


def pose_calibration(pose: np.ndarray, dof: str = "yaw", axis: str = "z") -> np.ndarray:
    """The calibration that takes a scramble pose to the reference: the pose undone (`conj(pose)`, so that
    `c · pose` is the identity) for `rotation`; for `yaw`, only its heading about the gravity axis undone
    (`about(axis, −heading)`)."""
    if dof == "rotation":
        return hemisphere(conjugate(unit(pose)))
    if dof == "yaw":
        return about(axis, -float(heading(unit(pose), axis)))
    raise ValueError(f"calibration dof {dof!r}: one of {', '.join(DOFS)}")


def project(c: np.ndarray, dof: str = "yaw", axis: str = "z") -> np.ndarray:
    """A rotation as the calibration's degrees of freedom allow: itself (`rotation`), or the rotation about
    the gravity axis nearest it (`yaw`: its heading)."""
    if dof == "rotation":
        return hemisphere(unit(c))
    return about(axis, heading(unit(c), axis))
