"""The facelet simulator: the owner's simulator's own cases (`selftest()` in `ferramentas/cubo.py`), the
Kociemba layout, the slices and the replay."""

import pytest

from cubetrace_ml.cube import SOLVED, apply, facelets, inverse, is_solved, primitives, replay, scrambled

SEXY = "R U R' U' "
TPERM = "R U R' U' R' F R2 U' R' U' R U R' F'"
SCRAMBLE = "U2 R2 L F2 D' B2 L2 B' U' F L' B2 R2 L2 U D F2 L2 D' F2 D' B2"


def test_the_owner_s_selftest_cases() -> None:
    assert is_solved(apply(SOLVED, SEXY * 6)) and not is_solved(apply(SOLVED, SEXY))
    assert not is_solved(apply(SOLVED, TPERM)) and is_solved(apply(SOLVED, TPERM + " " + TPERM))
    assert is_solved(apply(SOLVED, "F R U R' U' F' " * 6)) and is_solved(apply(SOLVED, "R R R R"))
    state = scrambled(SCRAMBLE)
    assert replay(state, inverse(SCRAMBLE)) and not replay(state, "R")


def test_the_layout_is_kociemba_s() -> None:
    cells = facelets()
    assert len(cells) == 54 and len(set(cells)) == 54
    where = {cell: i for i, cell in enumerate(cells)}
    position = {i: cell[0] for cell, i in where.items()}
    # Kociemba's corners and edges: the facelets of one cubie share its position.
    corners = ["U9 R1 F3", "U7 F1 L3", "U1 L1 B3", "U3 B1 R3", "D3 F9 R7", "D1 L9 F7", "D7 B9 L7", "D9 R9 B7"]
    edges = [
        "U6 R2",
        "U8 F2",
        "U4 L2",
        "U2 B2",
        "D6 R8",
        "D2 F8",
        "D4 L8",
        "D8 B8",
        "F6 R4",
        "F4 L6",
        "B6 L4",
        "B4 R6",
    ]
    for cubie in corners + edges:
        indices = ["URFDLB".index(name[0]) * 9 + int(name[1]) - 1 for name in cubie.split()]
        assert len({position[i] for i in indices}) == 1, cubie
    # A quarter turn of R on a solved cube, as Kociemba's notation writes it.
    assert apply(SOLVED, "R") == "UUFUUFUUFRRRRRRRRRFFDFFDFFDDDBDDBDDBLLLLLLLLLUBBUBBUBB"
    assert apply(SOLVED, "U") == "UUUUUUUUUBBBRRRRRRRRRFFFFFFDDDDDDDDDFFFLLLLLLLLLBBBBBB"


def test_the_turns_compose() -> None:
    for face in "URFDLB":
        assert apply(SOLVED, f"{face} {face}") == apply(SOLVED, f"{face}2")
        assert apply(SOLVED, f"{face} {face} {face}") == apply(SOLVED, f"{face}'")
        assert apply(apply(SOLVED, face), f"{face}'") == SOLVED
    assert apply(SOLVED, ["R", "U2", "F'"]) == apply(SOLVED, "R U2 F'")
    assert inverse("R U2 F' M") == ["M'", "F", "U2", "R'"]


def test_a_slice_is_its_primitive_pair() -> None:
    assert primitives("M") == [("R", 1), ("L", 3)]
    assert primitives("M'") == [("L", 1), ("R", 3)]
    assert primitives("S") == [("F", 3), ("B", 1)]
    assert primitives("E") == [("U", 1), ("D", 3)]
    assert primitives("E'") == [("D", 1), ("U", 3)]
    assert primitives("M2") == [("R", 1), ("L", 3), ("R", 1), ("L", 3)]
    assert primitives("R2'") == [("R", 2)]
    for symbol in ("M", "S", "E"):
        pair = " ".join(f"{f}{'' if t == 1 else chr(39)}" for f, t in primitives(symbol))
        assert apply(SOLVED, symbol) == apply(SOLVED, pair)
        assert not is_solved(apply(SOLVED, symbol))
        assert is_solved(apply(SOLVED, f"{symbol} {symbol}'")) and is_solved(apply(SOLVED, [symbol] * 4))
    with pytest.raises(ValueError, match="not a move"):
        apply(SOLVED, "x")


def test_solved_is_every_face_one_colour_whatever_the_colours() -> None:
    assert is_solved(SOLVED)
    relabelled = SOLVED.translate(str.maketrans("URFDLB", "FRDBLU"))  # the faces' colours permuted
    assert is_solved(relabelled) and not is_solved(apply(relabelled, "R"))
    assert not is_solved(SOLVED[:-1]) and not is_solved(SOLVED[:-2] + "UB")
    with pytest.raises(ValueError, match="54 facelets"):
        apply(SOLVED[:53], "R")
