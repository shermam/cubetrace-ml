"""The moves: their times on the host clock and the 24-symbol alphabet the quarter turns are merged into.

The cube reports quarter turns only: a double turn arrives as two quarter turns of one face, a slice as
two opposite faces turning the same physical way a few milliseconds apart. `normalize` merges them the
way the owner's simulator does (`normalizar()` in `ferramentas/cubo.py`): a greedy scan from the left,
a slice first, then a double, else the quarter turn itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

FACES = ("U", "R", "F", "D", "L", "B")
OPPOSITE = {"U": "D", "D": "U", "R": "L", "L": "R", "F": "B", "B": "F"}
# A slice: (face, turns) with the opposite face's opposite notation direction (the same physical way),
# looked up with either turn of the pair, in the usual notation: M turns as L (R + L'), S as F (F' + B)
# and E as D (U + D'). (The reference's table had E and E' swapped; it was corrected to match.)
SLICES = {("R", 1): "M", ("L", 1): "M'", ("F", 3): "S", ("B", 3): "S'", ("U", 1): "E", ("D", 1): "E'"}
SLICE_MS = 20.0
DOUBLE_MS = 200.0
SUFFIX = {1: "", 2: "2", 3: "'"}

TIME_BASES = ("fit", "arrival")
# The attempt's clock fit places a move when it is sane and the move is on its cube clock (within this
# much of the move's arrival); otherwise the move's hostMs does.
FIT_TOLERANCE_MS = 250.0
FIT_SLOPE_RANGE = (0.9, 1.1)


def alphabet() -> tuple[str, ...]:
    """The 24 symbols in their stable order: per face (U R F D L B) the clockwise, counter-clockwise and
    double turn, then M M' S S' E E'. A symbol's index in this tuple is its class id."""
    return (*(face + suffix for face in FACES for suffix in ("", "'", "2")), "M", "M'", "S", "S'", "E", "E'")


SYMBOLS = alphabet()
INDEX = {symbol: i for i, symbol in enumerate(SYMBOLS)}


def symbol_index(symbol: str) -> int:
    return INDEX[symbol]


def parse_move(m: str) -> tuple[str, int]:
    """`R` → (R, 1), `R2` → (R, 2), `R'` → (R, 3): the face and its clockwise quarter turns."""
    if len(m) not in (1, 2) or m[0] not in OPPOSITE or (len(m) == 2 and m[1] not in "2'"):
        raise ValueError(f"not a face turn: {m!r}")
    return m[0], 2 if m.endswith("2") else 3 if m.endswith("'") else 1


@dataclass(frozen=True)
class Symbol:
    """One normalized move: its symbol, the time of its first turn (the onset) and of its last turn (the
    end; the onset for a single turn), its phase, and which raw moves it merged (`first`, `turns`)."""

    symbol: str
    onset_ms: float
    end_ms: float
    phase: str
    first: int
    turns: int

    @property
    def index(self) -> int:
        return INDEX[self.symbol]


def normalize(
    names: Sequence[str],
    times: Sequence[float],
    phases: Sequence[str] | None = None,
    *,
    slice_ms: float = SLICE_MS,
    double_ms: float = DOUBLE_MS,
) -> list[Symbol]:
    """Merges a stream of face turns into symbols, greedily from the left: two opposite faces turning the
    same physical way (`R` then `L'`, in either order) less than `slice_ms` apart are a slice; two equal
    quarter turns of one face less than `double_ms` apart are a double (`X2`); anything else is itself.
    Two turns of different phases never merge (the reference normalizes one solve's stream)."""
    if len(names) != len(times) or (phases is not None and len(phases) != len(names)):
        raise ValueError("names, times and phases must have the same length")
    parsed = [parse_move(m) for m in names]
    out: list[Symbol] = []
    i = 0
    while i < len(parsed):
        face, turns = parsed[i]
        t = float(times[i])
        phase = phases[i] if phases is not None else ""
        if i + 1 < len(parsed) and (phases is None or phases[i + 1] == phase):
            other, other_turns = parsed[i + 1]
            u = float(times[i + 1])
            gap = u - t
            if other == OPPOSITE[face] and other_turns == 4 - turns and turns != 2 and gap < slice_ms:
                symbol = SLICES.get((face, turns)) or SLICES[(other, other_turns)]
                out.append(Symbol(symbol, t, u, phase, i, 2))
                i += 2
                continue
            if other == face and other_turns == turns and turns != 2 and gap < double_ms:
                out.append(Symbol(face + "2", t, u, phase, i, 2))
                i += 2
                continue
        out.append(Symbol(face + SUFFIX[turns], t, t, phase, i, 1))
        i += 1
    return out


def fit_usable(clock: dict[str, Any] | None) -> bool:
    """The attempt's clock fit is a line through one cube clock (not one through two: T2.9's old records)."""
    if not clock:
        return False
    low, high = FIT_SLOPE_RANGE
    return low < clock["a"] < high and clock["residualP95Ms"] <= FIT_TOLERANCE_MS


def move_times(attempt: dict[str, Any], time_base: str = "fit") -> tuple[np.ndarray, np.ndarray]:
    """Each move's time on the host clock and whether the attempt's fit placed it.

    `fit`: `a·cubeMs + b` of the attempt's `clock`, the cube's own timing without the Bluetooth jitter of
    the packet arrivals, for the moves on the fit's cube clock; `hostMs` (the arrival) for the others and
    when the fit is missing or not a line through one clock. `arrival`: `hostMs` for every move.
    """
    if time_base not in TIME_BASES:
        raise ValueError(f"time base {time_base!r}: one of {', '.join(TIME_BASES)}")
    moves = attempt["moves"]
    host = np.array([m["hostMs"] for m in moves], dtype=np.float64)
    clock = attempt.get("clock")
    if time_base == "arrival" or not fit_usable(clock):
        return host, np.zeros(len(moves), dtype=bool)
    cube = np.array([m["cubeMs"] for m in moves], dtype=np.float64)
    fitted = clock["a"] * cube + clock["b"]
    on_fit = np.abs(fitted - host) <= FIT_TOLERANCE_MS
    return np.where(on_fit, fitted, host), on_fit


def attempt_symbols(
    attempt: dict[str, Any],
    *,
    time_base: str = "fit",
    slice_ms: float = SLICE_MS,
    double_ms: float = DOUBLE_MS,
) -> list[Symbol]:
    """The attempt's moves (all phases) normalized, on the chosen time base."""
    times, _ = move_times(attempt, time_base)
    moves = attempt["moves"]
    return normalize(
        [m["m"] for m in moves],
        times,
        [m["phase"] for m in moves],
        slice_ms=slice_ms,
        double_ms=double_ms,
    )
