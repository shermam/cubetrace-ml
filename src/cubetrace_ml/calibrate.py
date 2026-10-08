"""The orientation's calibration into the camera's frame, in PyTorch (the `features` extra): the calibrated
inputs of a batch, the rotations learnt per training attempt (or attempt and camera), the label-free guess
for an unseen attempt (its scramble pose, through the training's mean correction), and the estimation of an
attempt's rotation from its labels with the network frozen.

A calibrated run's input per frame is its features, then the gyro's raw block (9: the orientation q in
w ≥ 0, zeros where none; the cube's own change; the flag), which `calibrated_inputs` turns, with the clip's
rotation c, into the orientation in the camera's frame `c · q` (the rotation matrix's 9 entries, sign-free,
or the quaternion in w ≥ 0), the change and the flag: 14 (or 9) channels after the features. The cube's own
change and the flag do not depend on c.

The estimation (`fit_calibrations`): for each key, its clips' per-frame loss (the training's: the weighted
cross-entropy of the per-frame head, CTC's loss for CTC) at every candidate of a grid (the yaws about the
gravity axis in 15° steps; for a whole rotation, those yaws composed with the cube's 24 symmetries: 144),
plus the label-free guess; the best three refined, the network frozen: by a compass search (the default:
forward passes only, the step halved from half the grid's spacing to under a degree) or by a dozen Adam
steps on the rotation (a yaw angle or a rotation vector; on a CPU a backward pass through the BiGRU costs
about ten forward ones). Fit on a test attempt's scramble clips alone it is honest (the app
prescribes the scramble: its labels are known before the solve); on all its clips it is the oracle.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .align import conjugate as conjugate_np
from .align import quaternion_product as product_np
from .labels import ClipLabels, calibration_key, model_input
from .orientation import (
    AXES,
    IDENTITY,
    angle_between,
    calibration_grid,
    chordal_mean,
    heading,
    hemisphere,
    log_map,
    pose_calibration,
    project,
    unit,
)
from .orientation import about as about_np

REFINE_TOP = 3  # the grid's best candidates refined
REFINES = ("search", "adam")
REFINE_STEPS = 12  # Adam's steps
# Adam's first learning rate (radians), decayed tenfold over its steps: it reaches about 13° for a yaw, about
# 40° for a rotation (whose grid is coarser in its tilt).
REFINE_LR = {"yaw": 0.05, "rotation": 0.15}
# The search's first step (half the grid's spacing for a yaw; a rotation's grid is coarser in its tilt) and
# its last: the step halves until it is under SEARCH_LAST.
SEARCH_FIRST = {"yaw": math.radians(7.5), "rotation": math.radians(22.5)}
SEARCH_LAST = math.radians(1.0)
SEARCH_MOVES = 2  # moves at one step before it halves
CHUNK_VALUES = 30_000_000  # the floats of one batch of candidates' inputs (about 120 MB)


# Rotations in PyTorch (x, y, z, w; the Hamilton product).


def quaternion_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dim=-1,
    )


def matrices(q: torch.Tensor) -> torch.Tensor:
    """The rotation matrices (…, 3, 3) of unit quaternions (…, 4)."""
    x, y, z, w = q.unbind(-1)
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], dim=-1),
            torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], dim=-1),
            torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], dim=-1),
        ],
        dim=-2,
    )


def exp_map(r: torch.Tensor) -> torch.Tensor:
    """Unit quaternions of rotation vectors (…, 3), smooth through 0 (the identity)."""
    theta = torch.sqrt((r * r).sum(dim=-1, keepdim=True) + 1e-12)
    return torch.cat([torch.sin(theta / 2) / theta * r, torch.cos(theta / 2)], dim=-1)


def about(axis: str, angles: torch.Tensor) -> torch.Tensor:
    """Rotations (…, 4) by `angles` (radians) about a coordinate axis."""
    vector = torch.eye(3, dtype=angles.dtype, device=angles.device)[AXES.index(axis)]
    half = angles[..., None] / 2
    return torch.cat([torch.sin(half) * vector, torch.cos(half)], dim=-1)


def calibrated_inputs(
    x: torch.Tensor, c: torch.Tensor, dim: int, orientation: str = "matrix"
) -> torch.Tensor:
    """A batch's model inputs from its raw inputs `x` (B × T × (dim + 9): the features, then q, the cube's
    change and the flag) and one rotation per clip `c` (B × 4): the features, the orientation in the camera's
    frame (the matrix's 9 entries row by row, or the quaternion in w ≥ 0; zeros where the flag is 0), the
    change and the flag."""
    features, q, change, flag = x[..., :dim], x[..., dim : dim + 4], x[..., dim + 4 : dim + 8], x[..., -1:]
    cam = quaternion_product(c[:, None, :].to(x.dtype), q)
    if orientation == "matrix":
        rotated = matrices(cam).flatten(-2) * flag
    else:
        rotated = torch.where(cam[..., 3:] < 0, -cam, cam) * flag
    return torch.cat([features, rotated, change, flag], dim=-1)


class Calibrations(nn.Module):
    """One rotation per key, learnt: a yaw angle about the gravity axis (`yaw`) or a rotation vector
    (`rotation`, through the exponential map), so that every value is a unit quaternion; started at
    `initial` (K × 4: the identity, or each key's scramble pose undone)."""

    def __init__(self, keys: Sequence[str], dof: str, axis: str, initial: np.ndarray | None = None) -> None:
        super().__init__()
        self.keys = list(keys)
        self.index = {key: i for i, key in enumerate(self.keys)}
        self.dof, self.axis = dof, axis
        start = np.tile(IDENTITY, (len(self.keys), 1)) if initial is None else np.asarray(initial, np.float64)
        if dof == "yaw":
            values = heading(start, axis) if len(start) else np.zeros(0)
        else:
            values = log_map(start) if len(start) else np.zeros((0, 3))
        self.values = nn.Parameter(torch.as_tensor(values, dtype=torch.float32))

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """The keys' rotations (B × 4) for their indices (B)."""
        values = self.values[indices]
        return about(self.axis, values) if self.dof == "yaw" else exp_map(values)

    def quaternions(self) -> np.ndarray:
        """Every key's rotation (K × 4, w ≥ 0)."""
        with torch.no_grad():
            q = self(torch.arange(len(self.keys))).double().cpu().numpy()
        return hemisphere(q)


def session_tie(calibrations: Calibrations, sessions: torch.Tensor) -> torch.Tensor:
    """The pull of each session's rotations toward their mean: the mean squared distance (Frobenius) of each
    key's rotation matrix from its session's mean matrix (`sessions`: each key's session as an index)."""
    m = matrices(calibrations(torch.arange(len(calibrations.keys)))).flatten(-2)  # K × 9
    count = int(sessions.max()) + 1 if len(sessions) else 0
    sums = torch.zeros(count, 9, dtype=m.dtype).index_add(0, sessions, m)
    sizes = torch.bincount(sessions, minlength=count).clamp_min(1).to(m.dtype)
    means = sums / sizes[:, None]
    return ((m - means[sessions]) ** 2).sum(dim=-1).mean()


# The rotations of the clips the training never saw.


@dataclass
class Guess:
    """How a run guesses an unseen key's rotation without labels. A fixed calibration's own rule: the identity
    (`identity`), or the attempt's scramble pose undone (`pose`). A learnt one's: the scramble pose undone and
    then the training's mean correction `gauge` (the chordal mean of each learnt rotation times its own start
    undone) when the poses predict the learnt rotations (`uses_pose`: at least POSE_AGREEMENT of the training
    keys' corrections within POSE_REACH degrees of the gauge), else the learnt rotations' mean."""

    kind: str  # the run's data.calibration
    dof: str
    axis: str
    gauge: np.ndarray
    mean: np.ndarray
    uses_pose: bool = True
    agreement: float = math.nan  # the share of the training keys' corrections near the gauge

    def __call__(self, pose: np.ndarray | None) -> tuple[np.ndarray, str]:
        """A clip's rotation and where it came from (`identity`, `pose` or `mean`)."""
        if self.kind == "identity":
            return IDENTITY.copy(), "identity"
        if pose is not None and (self.kind == "pose" or self.uses_pose):
            start = pose_calibration(pose, self.dof, self.axis)
            return project(product_np(self.gauge, start), self.dof, self.axis), "pose"
        return self.mean.copy(), "mean"

    def to_json(self) -> dict[str, Any]:
        return {
            "gauge": self.gauge.tolist(),
            "mean": self.mean.tolist(),
            "usesPose": self.uses_pose,
            "agreement": None if math.isnan(self.agreement) else self.agreement,
        }

    @staticmethod
    def from_record(info: dict[str, Any]) -> Guess:
        """The guess a run's checkpoint recorded (`calibration`)."""
        guess = info["guess"]
        agreement = guess.get("agreement")
        return Guess(
            info["kind"],
            info["dof"],
            info["axis"],
            np.asarray(guess["gauge"], dtype=np.float64),
            np.asarray(guess["mean"], dtype=np.float64),
            bool(guess.get("usesPose", True)),
            math.nan if agreement is None else float(agreement),
        )


POSE_REACH = 45.0
POSE_AGREEMENT = 0.8


def guess_of(kind: str, dof: str, axis: str, learnt: np.ndarray | None, starts: np.ndarray | None) -> Guess:
    """The run's label-free guess: for a learnt calibration, from its keys' rotations (`learnt`, K × 4) and
    their starts (`starts`, K × 4, NaN rows where the key had no pose); for a fixed one, its own rule."""
    if kind in ("attempt", "camera") and learnt is not None and len(learnt):
        mean = project(chordal_mean(learnt), dof, axis)
        known = np.isfinite(starts).all(axis=1) if starts is not None else np.zeros(len(learnt), bool)
        if not known.any():
            return Guess(kind, dof, axis, IDENTITY.copy(), mean, uses_pose=False)
        corrections = product_np(learnt[known], conjugate_np(starts[known]))
        gauge = project(chordal_mean(corrections), dof, axis)
        agreement = float(
            np.mean(angle_between(corrections, np.tile(gauge, (len(corrections), 1))) <= POSE_REACH)
        )
        return Guess(kind, dof, axis, gauge, mean, agreement >= POSE_AGREEMENT, agreement)
    return Guess(kind, dof, axis, IDENTITY.copy(), IDENTITY.copy())


# The estimation from labels.


def frame_losses(
    logits: torch.Tensor,
    near_class: torch.Tensor,
    near_weight: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
    targets: Sequence[torch.Tensor],
    head: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per batch row, the loss's sum and its normaliser: the per-frame head's weighted cross-entropy terms and
    their weights (the training's loss is their quotient over a batch), or CTC's loss over the reference's
    length and 1."""
    if head == "perframe":
        logp = F.log_softmax(logits, dim=-1)
        on = near_weight * weights[near_class]
        off = (1.0 - near_weight) * weights[0]
        nll = -(off * logp[..., 0] + on * logp.gather(-1, near_class[..., None]).squeeze(-1))
        m = mask.to(logits.dtype)
        return (nll * m).sum(dim=1), ((on + off) * m).sum(dim=1)
    logp = F.log_softmax(logits, dim=-1).transpose(0, 1)
    lengths = mask.sum(dim=1)
    flat = torch.cat([t.to(logits.device) for t in targets]).long()
    target_lengths = torch.tensor([len(t) for t in targets], device=logits.device)
    loss = F.ctc_loss(logp, flat, lengths, target_lengths, blank=0, reduction="none", zero_infinity=True)
    return loss / target_lengths.clamp_min(1).to(loss.dtype), torch.ones_like(loss)


@dataclass
class Fit:
    """One key's estimated rotation."""

    key: str
    rotation: np.ndarray  # 4
    loss: float  # the fitted clips' loss there
    identity_loss: float  # and at the identity
    guess_loss: float  # and at the label-free guess
    clips: int
    frames: int
    onsets: int
    source: str  # "grid", "guess", or "guess (no clip)" when the key had no clip to fit on


@dataclass
class Stack:
    """A key's clips collated once: raw inputs, mask and targets."""

    x: torch.Tensor
    mask: torch.Tensor
    near_class: torch.Tensor
    near_weight: torch.Tensor
    targets: list[torch.Tensor]


def stack_clips(clips: Sequence[ClipLabels], device: torch.device) -> Stack:
    arrays = [model_input(c, "features+gyro") for c in clips]
    longest, width = max(len(a) for a in arrays), arrays[0].shape[1]
    x = torch.zeros(len(clips), longest, width)
    mask = torch.zeros(len(clips), longest, dtype=torch.bool)
    near_class = torch.zeros(len(clips), longest, dtype=torch.long)
    near_weight = torch.zeros(len(clips), longest)
    for i, (clip, a) in enumerate(zip(clips, arrays, strict=True)):
        n = len(a)
        x[i, :n] = torch.from_numpy(a)
        mask[i, :n] = True
        near_class[i, :n] = torch.from_numpy(clip.near_class)
        near_weight[i, :n] = torch.from_numpy(clip.near_weight.astype(np.float32))
    return Stack(
        x.to(device),
        mask.to(device),
        near_class.to(device),
        near_weight.to(device),
        [torch.from_numpy(c.symbols + 1) for c in clips],
    )


class Scorer:
    """A frozen model's loss on one key's clips for candidate rotations."""

    def __init__(
        self,
        model: nn.Module,
        stack: Stack,
        *,
        dim: int,
        orientation: str,
        head: str,
        weights: torch.Tensor,
    ) -> None:
        self.model, self.stack, self.dim = model, stack, dim
        self.orientation, self.head, self.weights = orientation, head, weights

    def losses(self, candidates: torch.Tensor) -> torch.Tensor:
        """The loss (the clips' terms summed over their normalisers summed) at each candidate (G × 4)."""
        s = self.stack
        clips, length, width = s.x.shape
        per_chunk = max(1, CHUNK_VALUES // max(1, clips * length * (width + 5)))
        out = []
        for k in range(0, len(candidates), per_chunk):
            chunk = candidates[k : k + per_chunk]
            out.append(self._losses(chunk))
        return torch.cat(out)

    def _losses(self, candidates: torch.Tensor) -> torch.Tensor:
        s = self.stack
        g, m = len(candidates), s.x.shape[0]
        x = s.x.repeat(g, 1, 1)  # rows: candidate-major
        c = candidates.repeat_interleave(m, dim=0)
        logits = self.model(calibrated_inputs(x, c, self.dim, self.orientation), s.mask.repeat(g, 1))
        total, norm = frame_losses(
            logits,
            s.near_class.repeat(g, 1),
            s.near_weight.repeat(g, 1),
            s.mask.repeat(g, 1),
            self.weights,
            s.targets * g,
            self.head,
        )
        return total.view(g, m).sum(dim=1) / norm.view(g, m).sum(dim=1).clamp_min(1e-8)


def search(
    scorer: Scorer, starts: np.ndarray, losses: np.ndarray, dof: str, axis: str
) -> tuple[np.ndarray, np.ndarray]:
    """A compass search from each of `starts` (k × 4, their `losses`), forward passes only: each start moves
    to the best of its neighbours (rotated by ± the step about the gravity axis for a yaw, about each axis for
    a rotation, on the left) when one is better, up to SEARCH_MOVES times a step, and the step halves from
    SEARCH_FIRST until under SEARCH_LAST; every start's neighbours are scored in one batch."""
    current = hemisphere(unit(np.asarray(starts, np.float64).reshape(-1, 4)))
    best = np.asarray(losses, np.float64).copy()
    axes = (axis,) if dof == "yaw" else ("x", "y", "z")
    step = SEARCH_FIRST[dof]
    device = scorer.stack.x.device
    rows = np.arange(len(current))
    while step >= SEARCH_LAST:
        moves = np.stack([about_np(a, sign * step) for a in axes for sign in (1.0, -1.0)])
        for _ in range(SEARCH_MOVES):
            neighbours = product_np(moves[None, :, :], current[:, None, :])  # k × moves × 4
            with torch.no_grad():
                flat = torch.as_tensor(neighbours.reshape(-1, 4), dtype=torch.float32, device=device)
                scored = scorer.losses(flat).double().cpu().numpy().reshape(len(current), len(moves))
            choice = scored.argmin(axis=1)
            better = scored[rows, choice] < best
            if not better.any():
                break
            current[better] = hemisphere(neighbours[rows, choice][better])
            best[better] = scored[rows, choice][better]
        step /= 2
    return current, best


def refine(
    scorer: Scorer,
    starts: np.ndarray,
    dof: str,
    axis: str,
    steps: int = REFINE_STEPS,
    lr: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Adam on the rotations from each of `starts` (k × 4) at once (a yaw angle each, or a rotation vector),
    the network frozen, the learning rate decayed tenfold over the steps: each start's best rotation met
    (k × 4) and its loss (k). The losses of the starts are independent, so one backward pass of their sum
    gives each its own gradient."""
    starts = unit(np.asarray(starts, np.float64).reshape(-1, 4))
    if dof == "yaw":
        values = torch.tensor(heading(starts, axis), dtype=torch.float32, requires_grad=True)
    else:
        values = torch.tensor(log_map(starts), dtype=torch.float32, requires_grad=True)
    device = scorer.stack.x.device
    lr = REFINE_LR[dof] if lr is None else lr
    optimizer = torch.optim.Adam([values], lr=lr)
    best_q, best_loss = starts.copy(), np.full(len(starts), math.inf)
    for step in range(steps + 1):
        q = about(axis, values) if dof == "yaw" else exp_map(values)
        losses = scorer.losses(q.to(device))
        current = losses.detach().double().cpu().numpy()
        better = current < best_loss
        best_loss[better] = current[better]
        best_q[better] = q.detach().double().cpu().numpy()[better]
        if step == steps:
            break
        for group in optimizer.param_groups:
            group["lr"] = lr * 0.1 ** (step / max(1, steps - 1))
        optimizer.zero_grad()
        losses.sum().backward()
        optimizer.step()
    return hemisphere(unit(best_q)), best_loss


def fit_calibrations(
    model: nn.Module,
    groups: dict[str, list[ClipLabels]],
    guesses: dict[str, np.ndarray],
    *,
    dim: int,
    orientation: str,
    head: str,
    weights: torch.Tensor,
    dof: str,
    axis: str,
    device: torch.device,
    top: int = REFINE_TOP,
    steps: int = REFINE_STEPS,
    method: str = "search",
    log: Callable[[str], None] | None = None,
) -> dict[str, Fit]:
    """Each key's rotation from its clips' labels (`groups`: key → the clips to fit on, possibly none), the
    network frozen: the grid and the key's guess scored, the best `top` refined (`search`: the compass
    search, forward passes only; `adam`: `steps` Adam steps); a key without a clip keeps its guess."""
    if method not in REFINES:
        raise ValueError(f"refine {method!r}: one of {', '.join(REFINES)}")
    grid = torch.as_tensor(calibration_grid(dof, axis), dtype=torch.float32)
    identity = torch.as_tensor(IDENTITY, dtype=torch.float32)[None, :]
    was_training = model.training
    model.eval()
    requires = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    fits: dict[str, Fit] = {}
    try:
        for n, (key, clips) in enumerate(groups.items()):
            guess = unit(np.asarray(guesses[key], np.float64))
            if not clips:
                fits[key] = Fit(
                    key, hemisphere(guess), math.nan, math.nan, math.nan, 0, 0, 0, "guess (no clip)"
                )
                continue
            scorer = Scorer(
                model,
                stack_clips(clips, device),
                dim=dim,
                orientation=orientation,
                head=head,
                weights=weights,
            )
            candidates = torch.cat([grid, torch.as_tensor(guess, dtype=torch.float32)[None, :]]).to(device)
            with torch.no_grad():
                losses = scorer.losses(candidates).cpu().numpy().astype(np.float64)
                identity_loss = float(scorer.losses(identity.to(device))[0])
            guess_loss = float(losses[-1])
            best_rotation, best_loss, source = hemisphere(guess), guess_loss, "guess"
            chosen = np.argsort(losses, kind="stable")[:top]
            starts = candidates[chosen].double().cpu().numpy()
            if method == "search":
                refined, refined_loss = search(scorer, starts, losses[chosen], dof, axis)
            else:
                refined, refined_loss = refine(scorer, starts, dof, axis, steps)
            k = int(np.argmin(refined_loss))
            if refined_loss[k] < best_loss:
                best_rotation, best_loss = refined[k], float(refined_loss[k])
                source = "guess" if chosen[k] == len(losses) - 1 else "grid"
            fits[key] = Fit(
                key,
                best_rotation,
                best_loss,
                identity_loss,
                guess_loss,
                len(clips),
                int(sum(len(c) for c in clips)),
                int(sum(len(c.symbols) for c in clips)),
                source,
            )
            if log is not None and (n + 1) % 25 == 0:
                log(f"  {n + 1} of {len(groups)} keys fitted")
    finally:
        for p, flag in zip(model.parameters(), requires, strict=True):
            p.requires_grad_(flag)
        model.train(was_training)
    return fits


def key_groups(clips: Sequence[ClipLabels], by: str, segments: Sequence[str]) -> dict[str, list[ClipLabels]]:
    """The clips' keys (in their order) with each key's clips of the given segments."""
    groups: dict[str, list[ClipLabels]] = {}
    for clip in clips:
        members = groups.setdefault(calibration_key(clip.ref, by), [])
        if clip.segment in segments:
            members.append(clip)
    return groups
