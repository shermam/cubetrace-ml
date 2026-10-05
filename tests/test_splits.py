import pytest

from cubetrace_ml.splits import assign_splits
from factory import session_id

DAYS = {session_id(n): f"2026-10-{1 + n % 4:02d}" for n in range(1, 21)}  # 20 sessions over 4 days


def test_the_same_seed_gives_the_same_assignment() -> None:
    first = assign_splits(DAYS, seed=7)
    assert assign_splits(DAYS, seed=7) == first
    assert assign_splits(dict(reversed(list(DAYS.items()))), seed=7) == first  # whatever the listing order
    assert any(assign_splits(DAYS, seed=s) != first for s in range(1, 6))  # the seed matters


def test_the_held_out_day_is_the_test_split_and_only_it() -> None:
    split = assign_splits(DAYS, seed=0)
    latest = max(DAYS.values())
    assert {s for s, x in split.items() if x == "test"} == {s for s, d in DAYS.items() if d == latest}
    other = assign_splits(DAYS, seed=0, held_out_day="2026-10-02")
    assert {s for s, x in other.items() if x == "test"} == {s for s, d in DAYS.items() if d == "2026-10-02"}
    with pytest.raises(ValueError, match="no session on 2027-01-01"):
        assign_splits(DAYS, held_out_day="2027-01-01")


def test_validation_takes_its_fraction_of_the_other_sessions() -> None:
    split = assign_splits(DAYS, seed=0, val_fraction=0.2)
    rest = [s for s in DAYS if split[s] != "test"]
    assert sum(split[s] == "val" for s in rest) == round(0.2 * len(rest))
    assert set(split.values()) == {"train", "val", "test"}
    two = {session_id(1): "2026-10-01", session_id(2): "2026-10-01", session_id(3): "2026-10-02"}
    assert sorted(assign_splits(two, val_fraction=0.2).values()) == ["test", "train", "val"]  # at least one
    assert sorted(assign_splits(two, val_fraction=0.0).values()) == ["test", "train", "train"]


def test_one_day_and_unknown_days() -> None:
    one_day = {session_id(1): "2026-10-01", session_id(2): "2026-10-01"}
    assert set(assign_splits(one_day).values()) == {"test"}
    unknown = {session_id(1): None, session_id(2): "2026-10-01", session_id(3): "2026-10-01"}
    split = assign_splits(unknown)
    assert split[session_id(1)] == "train" and split[session_id(2)] == "test"
