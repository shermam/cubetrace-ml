"""Splits by session, never by attempt: the held-out day's sessions are the test split; the others are
shuffled with a seed into validation and training."""

from __future__ import annotations

import random
from collections.abc import Mapping

SPLITS = ("train", "val", "test")
VAL_FRACTION = 0.2


def assign_splits(
    days: Mapping[str, str | None],
    *,
    seed: int = 0,
    held_out_day: str | None = None,
    val_fraction: float = VAL_FRACTION,
) -> dict[str, str]:
    """Each session's split, from the sessions' recording days (`YYYY-MM-DD`, None when unknown).

    The held-out day (by default the latest day) goes to `test`, whole. The other sessions, sorted by id
    and shuffled with `random.Random(seed)`, give `val` the first `round(val_fraction · n)` of them (at
    least one, and never all, when there are two or more) and `train` the rest. The same sessions and the
    same seed give the same assignment.
    """
    known = sorted({day for day in days.values() if day})
    if held_out_day is None:
        held_out_day = known[-1] if known else None
    elif held_out_day not in known:
        raise ValueError(f"no session on {held_out_day}; the days are {', '.join(known) or 'none'}")
    test = sorted(s for s, day in days.items() if held_out_day is not None and day == held_out_day)
    rest = sorted(s for s in days if s not in set(test))
    random.Random(seed).shuffle(rest)
    n_val = 0
    if len(rest) >= 2 and val_fraction > 0:
        n_val = min(len(rest) - 1, max(1, round(val_fraction * len(rest))))
    out = {s: "test" for s in test}
    out.update({s: "val" for s in rest[:n_val]})
    out.update({s: "train" for s in rest[n_val:]})
    return {s: out[s] for s in sorted(out)}
