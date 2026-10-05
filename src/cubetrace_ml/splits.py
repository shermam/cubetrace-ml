"""Splits by session, never by attempt, weighed by clips: a session without clips joins no split (`none`);
the held-out day's sessions are the test split; the other sessions, shuffled with a seed, give validation
its share of the clips; the rest is training."""

from __future__ import annotations

import random
from collections.abc import Mapping

SPLITS = ("train", "val", "test", "none")
VAL_FRACTION = 0.15
TEST_FRACTION = 0.2
LATEST = "latest"


def clips_by_day(days: Mapping[str, str | None], clips: Mapping[str, int]) -> dict[str, int]:
    """The clips of each day that has sessions with clips."""
    out: dict[str, int] = {}
    for session, day in days.items():
        if day and clips.get(session, 0) > 0:
            out[day] = out.get(day, 0) + clips[session]
    return dict(sorted(out.items()))


def pick_test_day(
    days: Mapping[str, str | None],
    clips: Mapping[str, int],
    held_out_day: str | None = None,
    test_fraction: float = TEST_FRACTION,
) -> str | None:
    """The held-out day: `held_out_day` when it names one (`latest`: the latest day with clips); by default
    the day whose clips are closest to `test_fraction` of all the clips (the later day on a tie). None when
    no session with a known day has clips."""
    per_day = clips_by_day(days, clips)
    if held_out_day == LATEST:
        return max(per_day) if per_day else None
    if held_out_day is not None:
        if held_out_day not in per_day:
            known = ", ".join(per_day) or "none"
            raise ValueError(f"no session on {held_out_day} has clips; the days with clips are {known}")
        return held_out_day
    if not per_day:
        return None
    target = test_fraction * sum(n for n in clips.values() if n > 0)
    return max(per_day, key=lambda day: (-abs(per_day[day] - target), day))


def assign_splits(
    days: Mapping[str, str | None],
    clips: Mapping[str, int],
    *,
    seed: int = 0,
    held_out_day: str | None = None,
    val_fraction: float = VAL_FRACTION,
    test_fraction: float = TEST_FRACTION,
) -> dict[str, str]:
    """Each session's split, from the sessions' recording days (`YYYY-MM-DD`, None when unknown) and their
    clip counts (a session missing from `clips` has none).

    A session without clips is `none`. The held-out day (`pick_test_day`) goes to `test`, whole. The other
    sessions with clips, sorted by id and shuffled with `random.Random(seed)`, are taken in that order into
    `val` whenever one brings val's clips closer to `val_fraction` of all the clips (at least one session,
    the closest to that share, and never all of them, when there are two or more); the rest is `train`. The
    same sessions, counts and seed give the same assignment.
    """
    day = pick_test_day(days, clips, held_out_day, test_fraction)
    with_clips = sorted(s for s in days if clips.get(s, 0) > 0)
    test = {s for s in with_clips if day is not None and days[s] == day}
    pool = [s for s in with_clips if s not in test]
    random.Random(seed).shuffle(pool)
    target = val_fraction * sum(clips[s] for s in with_clips)
    val: list[str] = []
    if len(pool) >= 2 and val_fraction > 0:
        taken = 0
        for session in pool:
            if len(val) == len(pool) - 1:
                break
            if abs(taken + clips[session] - target) < abs(taken - target):
                val.append(session)
                taken += clips[session]
        if not val:
            val.append(min(pool, key=lambda s: abs(clips[s] - target)))
    out = {s: "none" for s in days}
    out.update({s: "train" for s in pool})
    out.update({s: "val" for s in val})
    out.update({s: "test" for s in test})
    return {s: out[s] for s in sorted(out)}
