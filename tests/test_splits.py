import pytest

from cubetrace_ml.splits import assign_splits, clips_by_day, pick_test_day
from factory import session_id

# 20 sessions over 4 days, 2 to 40 clips each; session 20 (day 2026-10-01) has none.
DAYS = {session_id(n): f"2026-10-{1 + n % 4:02d}" for n in range(1, 21)}
CLIPS = {session_id(n): (2 * n) % 41 if n != 20 else 0 for n in range(1, 21)}


def clips_in(split: dict[str, str], name: str) -> int:
    return sum(CLIPS[s] for s, x in split.items() if x == name)


def test_the_same_counts_and_seed_give_the_same_assignment() -> None:
    first = assign_splits(DAYS, CLIPS, seed=7)
    assert assign_splits(DAYS, CLIPS, seed=7) == first
    reordered = dict(reversed(list(DAYS.items())))
    assert assign_splits(reordered, dict(reversed(list(CLIPS.items()))), seed=7) == first
    assert any(assign_splits(DAYS, CLIPS, seed=s) != first for s in range(1, 6))  # the seed matters


def test_a_session_without_clips_joins_no_split() -> None:
    split = assign_splits(DAYS, CLIPS)
    assert split[session_id(20)] == "none"
    assert [s for s, x in split.items() if x == "none"] == [session_id(20)]
    # A session missing from the counts has no clips either; a day of such sessions is never the test day.
    days = {**DAYS, session_id(99): "2026-12-31"}
    assert assign_splits(days, CLIPS)[session_id(99)] == "none"
    assert pick_test_day(days, CLIPS, "latest") == "2026-10-04"
    with pytest.raises(ValueError, match="no session on 2026-12-31 has clips"):
        assign_splits(days, CLIPS, held_out_day="2026-12-31")


def test_the_test_day_is_the_one_closest_to_a_fifth_of_the_clips() -> None:
    per_day = clips_by_day(DAYS, CLIPS)
    total = sum(CLIPS.values())
    assert sum(per_day.values()) == total
    day = pick_test_day(DAYS, CLIPS)
    assert all(abs(per_day[day] - total / 5) <= abs(n - total / 5) for n in per_day.values())
    split = assign_splits(DAYS, CLIPS)
    test = {s for s, d in DAYS.items() if d == day and CLIPS[s]}
    assert {s for s, x in split.items() if x == "test"} == test

    four = {session_id(n): f"2026-10-0{n}" for n in range(1, 5)}
    assert pick_test_day(four, dict(zip(four, [10, 30, 20, 40], strict=True))) == "2026-10-03"  # 20 of 100
    tie = dict(zip(four, [10, 50, 30, 10], strict=True))  # 10-01, 10-03 and 10-04 are 10 away from 20
    assert pick_test_day(four, tie) == "2026-10-04"  # the later
    assert pick_test_day(four, tie, "latest") == "2026-10-04"
    assert pick_test_day(four, tie, "2026-10-02") == "2026-10-02"


def test_validation_takes_its_share_of_the_clips_in_whole_sessions() -> None:
    total = sum(CLIPS.values())
    for seed in range(10):
        split = assign_splits(DAYS, CLIPS, seed=seed, val_fraction=0.15)
        biggest = max(CLIPS.values())
        assert abs(clips_in(split, "val") - 0.15 * total) <= biggest / 2  # as close as whole sessions allow
        assert set(split.values()) == {"train", "val", "test", "none"}
    split = assign_splits(DAYS, CLIPS, val_fraction=0.0)
    assert "val" not in split.values()


def test_validation_keeps_at_least_one_session_and_never_all() -> None:
    days = {session_id(n): "2026-10-01" for n in (1, 2)} | {session_id(3): "2026-10-02"}
    big = {session_id(1): 100, session_id(2): 100, session_id(3): 40}  # 15% is 36: no session comes closer
    split = assign_splits(days, big, held_out_day="2026-10-02")
    assert sorted(split.values()) == ["test", "train", "val"]
    tiny = {session_id(1): 1, session_id(2): 1, session_id(3): 98}  # 15% is 15: both would come closer
    split = assign_splits(days, tiny, held_out_day="2026-10-02")
    assert sorted(split.values()) == ["test", "train", "val"]


def test_one_day_and_unknown_days() -> None:
    one_day = {session_id(1): "2026-10-01", session_id(2): "2026-10-01"}
    assert set(assign_splits(one_day, dict.fromkeys(one_day, 5)).values()) == {"test"}
    unknown = {session_id(1): None, session_id(2): "2026-10-01", session_id(3): "2026-10-01"}
    split = assign_splits(unknown, dict.fromkeys(unknown, 5))
    assert split[session_id(1)] == "train" and split[session_id(2)] == "test"
    assert set(assign_splits(unknown, {}).values()) == {"none"}
    assert pick_test_day(unknown, {}) is None
