"""A 3x3 cube as its 54 facelets in Kociemba order, for the replay metric: does a move sequence take a state
(the record's `scrambledFacelets`) to solved?

Every facelet is a sticker position: its cubie's position and its normal in integer coordinates, the
right-handed frame U = +y, D = −y, R = +x, L = −x, F = +z, B = −z of the owner's simulator
(`ferramentas/cubo.py`). Kociemba's order lists U1–U9, R1–R9, F1–F9, D1–D9, L1–L9 and B1–B9, each face
read row by row as the usual net draws it (U above F, L R B beside it, D below). A clockwise quarter turn
of the face with normal n (seen from outside) maps every position and normal v of its layer to
(n·v)n − n×v, the simulator's rule; the facelet permutations follow from it.

The moves are face turns (`R`, `R'`, `R2`); a slice is what the cube reports of it, two opposite face
turns (`M` = `R` + `L'`, the `SLICES` table of `moves`), so the centres never move and a state is solved
when every face is one colour, whatever colour that is.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from functools import cache

import numpy as np

from .moves import FACES, OPPOSITE, SLICES

SOLVED = "".join(face * 9 for face in "URFDLB")
AXIS = {"U": (0, 1, 0), "R": (1, 0, 0), "F": (0, 0, 1), "D": (0, -1, 0), "L": (-1, 0, 0), "B": (0, 0, -1)}
# A slice symbol's primitive pair, from the normalization's table: M is R then L', and so on.
SLICE_PAIRS = {
    symbol: ((face, turns), (OPPOSITE[face], 4 - turns)) for (face, turns), symbol in SLICES.items()
}
TOKEN = re.compile(r"^([URFDLBMSE])(2'?|'2?)?$")
Vector = tuple[int, int, int]


def _position(face: str, row: int, col: int) -> Vector:
    """The cubie position of facelet (row, col) of a face in Kociemba's net."""
    return {
        "U": (col - 1, 1, row - 1),
        "R": (1, 1 - row, 1 - col),
        "F": (col - 1, 1 - row, 1),
        "D": (col - 1, -1, 1 - row),
        "L": (-1, 1 - row, col - 1),
        "B": (1 - col, 1 - row, -1),
    }[face]


def facelets() -> list[tuple[Vector, Vector]]:
    """The 54 facelets in Kociemba order, each as (position, normal)."""
    return [
        (_position(face, row, col), AXIS[face]) for face in "URFDLB" for row in range(3) for col in range(3)
    ]


def _turn(v: Vector, n: Vector) -> Vector:
    """A clockwise quarter turn about the outward normal n: (n·v)n − n×v."""
    d = n[0] * v[0] + n[1] * v[1] + n[2] * v[2]
    c = (n[1] * v[2] - n[2] * v[1], n[2] * v[0] - n[0] * v[2], n[0] * v[1] - n[1] * v[0])
    return (d * n[0] - c[0], d * n[1] - c[1], d * n[2] - c[2])


@cache
def _gathers() -> dict[tuple[str, int], np.ndarray]:
    """For each face and number of clockwise quarter turns (1–3), the index array `g` such that the state
    after the turn is `state[g]`."""
    cells = facelets()
    where = {cell: i for i, cell in enumerate(cells)}
    out: dict[tuple[str, int], np.ndarray] = {}
    for face in FACES:
        n = AXIS[face]
        moved = np.arange(54)  # moved[i]: where the sticker at i goes
        for i, (p, normal) in enumerate(cells):
            if p[0] * n[0] + p[1] * n[1] + p[2] * n[2] == 1:
                moved[i] = where[(_turn(p, n), _turn(normal, n))]
        gather = np.empty(54, dtype=np.int64)
        gather[moved] = np.arange(54)  # the sticker that lands on j came from gather[j]
        out[(face, 1)] = gather
        out[(face, 2)] = gather[gather]
        out[(face, 3)] = gather[gather][gather]
    return out


def primitives(move: str) -> list[tuple[str, int]]:
    """A move as face turns, (face, clockwise quarter turns): `R2` → [(R, 2)], `M` → [(R, 1), (L, 3)]."""
    match = TOKEN.match(move)
    if not match:
        raise ValueError(f"not a move: {move!r}")
    letter, suffix = match.group(1), match.group(2) or ""
    turns = 2 if "2" in suffix else 3 if "'" in suffix else 1
    if letter in AXIS:
        return [(letter, turns)]
    base = SLICE_PAIRS[letter if turns != 3 else letter + "'"]
    return list(base) * (2 if turns == 2 else 1)


def parse(moves: str | Iterable[str]) -> list[str]:
    """A sequence of moves from a string (separated by spaces) or an iterable of symbols."""
    return moves.split() if isinstance(moves, str) else list(moves)


def apply(state: str, moves: str | Iterable[str]) -> str:
    """The facelets (54 letters in Kociemba order) after the moves."""
    if len(state) != 54:
        raise ValueError(f"a cube state has 54 facelets, not {len(state)}")
    codes = np.frombuffer(state.encode("ascii"), dtype=np.uint8).copy()
    gathers = _gathers()
    for move in parse(moves):
        for face, turns in primitives(move):
            codes = codes[gathers[(face, turns)]]
    return codes.tobytes().decode("ascii")


def is_solved(state: str) -> bool:
    """Every face one colour (whichever colour it is)."""
    return len(state) == 54 and all(len(set(state[k : k + 9])) == 1 for k in range(0, 54, 9))


def replay(state: str, moves: str | Iterable[str]) -> bool:
    """Whether the moves take `state` to solved."""
    return is_solved(apply(state, moves))


def scrambled(scramble: str | Iterable[str]) -> str:
    """The facelets of a solved cube after the scramble."""
    return apply(SOLVED, scramble)


def inverse(moves: str | Sequence[str]) -> list[str]:
    """The moves that undo `moves`: reversed, each inverted (`R` ↔ `R'`, `R2` and `M2` stay)."""
    out = []
    for move in reversed(parse(moves)):
        if move.endswith("2") or move.endswith("2'"):
            out.append(move[0] + "2")
        else:
            out.append(move[0] if move.endswith("'") else move[0] + "'")
    return out
