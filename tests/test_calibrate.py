"""The calibration in PyTorch (skipped without the features extra): the calibrated inputs against the NumPy
channels, the learnt rotations (their gradients, unit length, start), the session tie, the label-free guess,
and the estimation recovering a known rotation from labels alone with a model that knows the faces."""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cubetrace_ml.align import conjugate, quaternion_product  # noqa: E402
from cubetrace_ml.calibrate import (  # noqa: E402
    Calibrations,
    Guess,
    Scorer,
    calibrated_inputs,
    exp_map,
    fit_calibrations,
    guess_of,
    key_groups,
    session_tie,
    stack_clips,
)
from cubetrace_ml.dataset import ClipRef  # noqa: E402
from cubetrace_ml.labels import ClipLabels  # noqa: E402
from cubetrace_ml.moves import INDEX  # noqa: E402
from cubetrace_ml.orientation import (  # noqa: E402
    IDENTITY,
    about,
    angle_between,
    calibrated_channels,
    cube_rotations,
    heading,
    hemisphere,
    unit,
)

RNG = np.random.default_rng(5)


def rotations(n: int) -> np.ndarray:
    return hemisphere(unit(RNG.normal(size=(n, 4))))


def raw_block(q: np.ndarray, present: np.ndarray) -> np.ndarray:
    raw = np.zeros((len(q), 9))
    raw[:, :4] = np.where(present[:, None], hemisphere(q), 0.0)
    raw[:, 4:8] = rotations(len(q))
    raw[:, 8] = present
    return raw


@pytest.mark.parametrize("orientation", ["matrix", "quat"])
def test_the_calibrated_inputs_are_the_numpy_channels(orientation: str) -> None:
    features = RNG.normal(size=(2, 7, 5))
    raws = [raw_block(rotations(7), RNG.random(7) > 0.3) for _ in range(2)]
    c = rotations(2)
    x = torch.as_tensor(np.concatenate([features, np.stack(raws)], axis=-1), dtype=torch.float64)
    out = calibrated_inputs(x, torch.as_tensor(c), 5, orientation).numpy()
    for i in range(2):
        np.testing.assert_allclose(out[i, :, :5], features[i])
        np.testing.assert_allclose(out[i, :, 5:], calibrated_channels(raws[i], c[i], orientation), atol=1e-6)


@pytest.mark.parametrize("dof", ["yaw", "rotation"])
def test_the_learnt_rotations(dof: str) -> None:
    start = np.stack([about("z", 0.4), about("z", -2.0), IDENTITY]) if dof == "yaw" else rotations(3)
    calibrations = Calibrations(["a/1", "a/2", "b/1"], dof, "z", start)
    np.testing.assert_allclose(calibrations.quaternions(), hemisphere(start), atol=1e-6)  # the start kept
    q = calibrations(torch.tensor([0, 2, 2]))
    assert q.shape == (3, 4)
    torch.testing.assert_close(q.norm(dim=-1), torch.ones(3))
    # Gradients reach the rotation of each key used, and only theirs.
    target = torch.as_tensor(rotations(3), dtype=torch.float32)
    (1.0 - (q * target).sum(dim=-1) ** 2).sum().backward()
    grad = calibrations.values.grad
    assert grad is not None and grad[0].abs().sum() > 0 and grad[2].abs().sum() > 0
    assert grad[1].abs().sum() == 0
    # A step keeps every rotation unit; the yaw stays a yaw.
    optimizer = torch.optim.Adam(calibrations.parameters(), lr=0.5)
    optimizer.step()
    after = calibrations.quaternions()
    np.testing.assert_allclose(np.linalg.norm(after, axis=1), 1.0, atol=1e-6)
    if dof == "yaw":
        np.testing.assert_allclose(after[:, :2], 0.0, atol=1e-7)
    # The identity, from nothing, has a gradient too (the exponential map is smooth there).
    zero = Calibrations(["k"], "rotation", "z")
    (zero(torch.tensor([0])) * torch.tensor([[0.3, -0.2, 0.1, 0.0]])).sum().backward()
    assert torch.isfinite(zero.values.grad).all() and zero.values.grad.abs().sum() > 0


def test_the_session_tie() -> None:
    same = Calibrations(["a/1", "a/2", "b/1"], "yaw", "z", np.stack([about("z", 0.5)] * 2 + [IDENTITY]))
    sessions = torch.tensor([0, 0, 1])
    assert session_tie(same, sessions).item() == pytest.approx(0.0, abs=1e-10)
    apart = Calibrations(["a/1", "a/2", "b/1"], "yaw", "z", np.stack([about("z", 0.5), IDENTITY, IDENTITY]))
    loss = session_tie(apart, sessions)
    loss.backward()
    grad = apart.values.grad
    assert loss.item() > 0 and grad[0] > 0 > grad[1] and grad[2] == 0  # the session's two pulled together


def test_the_label_free_guess() -> None:
    poses = rotations(6)
    starts = hemisphere(conjugate(poses))
    correction = about("z", math.radians(30))
    learnt = hemisphere(quaternion_product(np.tile(correction, (6, 1)), starts))
    guess = guess_of("attempt", "rotation", "z", learnt, starts)
    assert guess.uses_pose and guess.agreement == 1.0
    np.testing.assert_allclose(guess.gauge, hemisphere(correction), atol=1e-9)
    pose = rotations(1)[0]
    rotation, source = guess(pose)
    assert source == "pose"
    # The guess undoes the pose, then applies the training's correction.
    assert float(angle_between(quaternion_product(rotation, pose), correction)) == pytest.approx(
        0.0, abs=1e-6
    )
    assert guess(None)[1] == "mean"
    # Poses that do not predict the learnt rotations are left out: the learnt rotations' mean instead.
    scattered = guess_of("attempt", "rotation", "z", rotations(6), starts)
    assert not scattered.uses_pose and scattered(pose)[1] == "mean"
    # A fixed calibration's own rules; the record round trip.
    np.testing.assert_allclose(guess_of("identity", "yaw", "z", None, None)(pose)[0], IDENTITY)
    fixed = guess_of("pose", "yaw", "z", None, None)
    assert float(heading(quaternion_product(fixed(pose)[0], pose), "z")) == pytest.approx(0.0, abs=1e-9)
    info = {"kind": "attempt", "dof": "rotation", "axis": "z", "guess": guess.to_json()}
    again = Guess.from_record(info)
    assert again.uses_pose and np.allclose(again(pose)[0], rotation)


# The estimation with a model that knows the faces: the features say which side of the cube the camera saw
# turn; the model scores each face by how close its direction in the camera's frame is to that side.

SIDES = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float64)
FACES = {"R": (1, 0, 0), "L": (-1, 0, 0), "B": (0, 1, 0), "F": (0, -1, 0), "U": (0, 0, 1), "D": (0, 0, -1)}


class KnowsTheFaces(torch.nn.Module):
    def __init__(self, sharpness: float = 4.0) -> None:
        super().__init__()
        self.sharpness = sharpness
        self.unused = torch.nn.Parameter(torch.zeros(1))  # a parameter, frozen like a real model's

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        side, onset = x[..., :6], x[..., 6]
        matrix = x[..., 7:16].reshape(*x.shape[:2], 3, 3)
        seen = side @ torch.as_tensor(SIDES, dtype=x.dtype)
        logits = torch.full((*x.shape[:2], 25), -20.0, dtype=x.dtype)
        logits[..., 0] = 8.0 * (1.0 - 2.0 * onset)
        for face, axis in FACES.items():
            direction = matrix @ torch.as_tensor(axis, dtype=x.dtype)
            logits[..., INDEX[face] + 1] = self.sharpness * (seen * direction).sum(dim=-1) + 8.0 * onset - 8.0
        return logits + 0.0 * self.unused


def faces_clip(truth: np.ndarray, segment: str = "scramble", seed: int = 0) -> ClipLabels:
    """A clip of 30 onsets (a random face turn each, the cube at a random symmetry each time) seen by a camera
    whose frame is the gyro's turned by `truth`: the camera saw each face on the side its direction points
    to; the gyro recorded `conj(truth) · q`."""
    rng = np.random.default_rng(seed)
    frames = 4 * 30 + 4
    x = np.zeros((frames, 7))
    raw = np.zeros((frames, 9))
    raw[:, 4:8] = IDENTITY
    raw[:, 8] = 1.0
    target = np.zeros(frames, dtype=np.int64)
    cube = cube_rotations()
    symbols = []
    for k in range(30):
        q = cube[rng.integers(24)]
        face = str(rng.choice(list(FACES)))
        frame = 2 + 4 * k
        direction = (unit(q[None])[0], np.asarray(FACES[face], np.float64))
        camera = quaternion_product(quaternion_product(q, np.array([*direction[1], 0.0])), conjugate(q))[:3]
        x[frame, int(np.argmax(SIDES @ camera))] = 1.0
        x[frame, 6] = 1.0
        raw[frame - 2 : frame + 2, :4] = hemisphere(quaternion_product(conjugate(truth), q))
        target[frame] = INDEX[face] + 1
        symbols.append(INDEX[face])
    t = 1000.0 + 33.3 * np.arange(frames)
    clip = ClipLabels(
        ref=ClipRef("00000001-0000-4000-8000-000000000000", 1, "laptop", segment),
        frames=np.arange(frames),
        t_ms=t,
        symbols=np.array(symbols),
        onsets_ms=t[target > 0],
        target=target,
        near_class=target,
        near_weight=(target > 0).astype(np.float32),
        lag_ms=None,
        tps=None,
        facelets=None,
    )
    clip.x = x.astype(np.float16)
    clip.gyro = raw.astype(np.float32)
    return clip


@pytest.mark.parametrize(("dof", "truth"), [("yaw", about("z", math.radians(100))), ("rotation", None)])
def test_the_estimation_recovers_a_known_rotation(dof: str, truth: np.ndarray | None) -> None:
    truth = rotations(1)[0] if truth is None else truth
    clip = faces_clip(truth)
    fits = fit_calibrations(
        KnowsTheFaces(),
        {"k": [clip], "none": []},
        {"k": IDENTITY, "none": about("z", 0.3)},
        dim=7,
        orientation="matrix",
        head="perframe",
        weights=torch.ones(25),
        dof=dof,
        axis="z",
        device=torch.device("cpu"),
    )
    fit = fits["k"]
    assert float(angle_between(fit.rotation, truth)) < 3.0
    assert fit.loss < fit.identity_loss and fit.source == "grid" and (fit.clips, fit.onsets) == (1, 30)
    # A key without a clip keeps its guess.
    assert fits["none"].source == "guess (no clip)" and np.allclose(fits["none"].rotation, about("z", 0.3))


def test_the_scorer_s_losses() -> None:
    truth = about("z", math.radians(-60))
    clips = [faces_clip(truth, seed=1), faces_clip(truth, "solve", seed=2)]
    candidates = torch.as_tensor(np.stack([IDENTITY, truth]), dtype=torch.float32)
    for head in ("perframe", "ctc"):
        scorer = Scorer(
            KnowsTheFaces(),
            stack_clips(clips, torch.device("cpu")),
            dim=7,
            orientation="matrix",
            head=head,
            weights=torch.ones(25),
        )
        losses = scorer.losses(candidates).detach()
        assert losses.shape == (2,) and float(losses[1]) < float(losses[0])
        # The candidates in chunks of one give the same losses.
        import cubetrace_ml.calibrate as calibrate

        chunk, calibrate.CHUNK_VALUES = calibrate.CHUNK_VALUES, 1
        try:
            torch.testing.assert_close(scorer.losses(candidates).detach(), losses)
        finally:
            calibrate.CHUNK_VALUES = chunk
    groups = key_groups(clips, "attempt", ("scramble",))
    assert list(groups) == [f"{clips[0].ref.session}/1"] and len(groups[f"{clips[0].ref.session}/1"]) == 1
    assert exp_map(torch.zeros(3)).tolist() == pytest.approx([0.0, 0.0, 0.0, 1.0])
