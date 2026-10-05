import numpy as np
import pytest

from cubetrace_ml.moves import (
    INDEX,
    SYMBOLS,
    alphabet,
    attempt_symbols,
    move_times,
    normalize,
    parse_move,
    symbol_index,
)
from factory import T0, attempt_record, session_id


def symbols(stream: str, **kwargs) -> list[str]:
    """`normalize` of a stream written as the reference's tests write it: `R[0] L'[4] …`."""
    names, times = [], []
    for token in stream.split():
        name, _, ms = token.partition("[")
        names.append(name)
        times.append(float(ms.rstrip("]")))
    return [s.symbol for s in normalize(names, times, **kwargs)]


def test_the_alphabet_is_24_symbols_in_a_stable_order() -> None:
    assert alphabet() == SYMBOLS
    assert (
        *("U", "U'", "U2", "R", "R'", "R2", "F", "F'", "F2"),
        *("D", "D'", "D2", "L", "L'", "L2", "B", "B'", "B2"),
        *("M", "M'", "S", "S'", "E", "E'"),
    ) == SYMBOLS
    assert len(set(SYMBOLS)) == 24
    assert [symbol_index(s) for s in SYMBOLS] == list(range(24))
    assert INDEX["U"] == 0 and INDEX["B2"] == 17 and INDEX["M"] == 18 and INDEX["E'"] == 23


def test_parse_move() -> None:
    assert parse_move("R") == ("R", 1)
    assert parse_move("R2") == ("R", 2)
    assert parse_move("R'") == ("R", 3)
    for bad in ("", "M", "R3", "r", "R''"):
        with pytest.raises(ValueError):
            parse_move(bad)


def test_quarter_turns_stay_themselves() -> None:
    assert symbols("R[0] U'[300] F[600] D[900] L'[1200] B'[1500]") == ["R", "U'", "F", "D", "L'", "B'"]


def test_a_double_is_two_equal_quarter_turns_within_the_threshold() -> None:
    assert symbols("U[300] U[390]") == ["U2"]
    assert symbols("U'[300] U'[490]") == ["U2"]  # both ways make the same half turn
    assert symbols("U[0] U[200]") == ["U", "U"]  # the threshold is strict
    assert symbols("U[0] U[250]") == ["U", "U"]  # two deliberate turns
    assert symbols("U[0] U'[50]") == ["U", "U'"]  # a turn undone is no double
    assert symbols("U[0] U[120]", double_ms=100) == ["U", "U"]  # the threshold is a parameter
    assert symbols("U[0] U[90] U[180]") == ["U2", "U"]  # greedy, from the left


def test_a_slice_is_two_opposite_faces_the_same_physical_way_within_the_threshold() -> None:
    assert symbols("R[0] L'[4]") == ["M"]
    assert symbols("L'[0] R[4]") == ["M"]  # either order
    assert symbols("L[0] R'[4]") == ["M'"]
    assert symbols("F'[0] B[2]") == ["S"]
    assert symbols("B'[0] F[2]") == ["S'"]
    assert symbols("U[0] D'[3]") == ["E"]  # E turns as D: the usual notation
    assert symbols("D'[0] U[3]") == ["E"]
    assert symbols("U'[0] D[3]") == ["E'"]
    assert symbols("D[0] U'[3]") == ["E'"]
    assert symbols("R[0] L[4]") == ["R", "L"]  # opposite physical ways: a rotation's worth, not a slice
    assert symbols("R[0] L'[20]") == ["R", "L'"]  # the threshold is strict
    assert symbols("R[0] L'[25]", slice_ms=30) == ["M"]


def test_the_references_own_cases() -> None:
    assert symbols("R[0] L'[4] U[300] U[390] U[600] U[1200] F'[1300] B[1302]") == ["M", "U2", "U", "U", "S"]
    assert symbols("U'[0] D[3] R[100] R'[150]") == ["E'", "R", "R'"]


def test_a_merged_symbol_starts_at_its_first_turn_and_ends_at_its_second() -> None:
    out = normalize(["R", "R", "L", "R'", "U"], [1000.0, 1090.0, 1300.0, 1305.0, 1600.0])
    assert [(s.symbol, s.onset_ms, s.end_ms, s.first, s.turns) for s in out] == [
        ("R2", 1000.0, 1090.0, 0, 2),
        ("M'", 1300.0, 1305.0, 2, 2),
        ("U", 1600.0, 1600.0, 4, 1),
    ]
    assert [s.index for s in out] == [INDEX["R2"], INDEX["M'"], INDEX["U"]]


def test_half_turns_in_the_input_are_kept_and_never_merged() -> None:
    assert symbols("R2[0] R2[50]") == ["R2", "R2"]
    assert symbols("R2[0] L2[5]") == ["R2", "L2"]


def test_turns_of_two_phases_never_merge() -> None:
    out = normalize(["U", "U"], [0.0, 50.0], ["scramble", "solve"])
    assert [(s.symbol, s.phase) for s in out] == [("U", "scramble"), ("U", "solve")]
    with pytest.raises(ValueError):
        normalize(["U"], [0.0, 1.0])


def test_move_times_follow_the_attempts_fit() -> None:
    sid = session_id(1)
    host = [T0 + 1000.0, T0 + 1250.0, T0 + 1500.0]
    jitter = [8.0, -12.0, 4.0]  # the arrivals' Bluetooth jitter around the cube's clock
    attempt = attempt_record(
        sid,
        1,
        [("R", h + j) for h, j in zip(host, jitter, strict=True)],
        [],
        cube_ms=[h - T0 for h in host],
        clock={"a": 1.0, "b": T0, "residualP95Ms": 12.0, "samples": 3},
    )
    fit, on_fit = move_times(attempt, "fit")
    assert np.allclose(fit, host) and on_fit.all()
    arrival, on_fit = move_times(attempt, "arrival")
    assert np.allclose(arrival, [h + j for h, j in zip(host, jitter, strict=True)]) and not on_fit.any()
    with pytest.raises(ValueError):
        move_times(attempt, "cube")


def test_a_move_off_the_fits_clock_keeps_its_arrival() -> None:
    sid = session_id(1)
    attempt = attempt_record(
        sid,
        1,
        [("R", T0 + 1000.0), ("U", T0 + 2000.0)],
        [],
        cube_ms=[5_000.0, 1000.0],  # the first move was on the cube's previous clock
        clock={"a": 1.0, "b": T0 + 1000.0, "residualP95Ms": 0.0, "samples": 2},
    )
    times, on_fit = move_times(attempt)
    assert list(times) == [T0 + 1000.0, T0 + 2000.0]
    assert list(on_fit) == [False, True]


def test_a_fit_through_two_clocks_or_none_is_not_used() -> None:
    sid = session_id(1)
    for clock in (
        {"a": -0.81, "b": T0, "residualP95Ms": 48_000.0, "samples": 40},
        None,
    ):
        attempt = attempt_record(sid, 1, [("R", T0 + 1000.0)], [], cube_ms=[1000.0], clock=clock)
        times, on_fit = move_times(attempt)
        assert list(times) == [T0 + 1000.0] and not on_fit.any()


def test_attempt_symbols_normalize_each_phase() -> None:
    sid = session_id(1)
    attempt = attempt_record(
        sid,
        1,
        [("F", T0 + 100.0), ("F", T0 + 150.0)],
        [("F", T0 + 200.0), ("R", T0 + 1000.0), ("L'", T0 + 1010.0)],
    )
    out = attempt_symbols(attempt)
    assert [(s.symbol, s.phase) for s in out] == [("F2", "scramble"), ("F", "solve"), ("M", "solve")]
