"""A dataset root: `sessions/<sessionId>/session.json` and `sessions/<sessionId>/attempts/<nnnn>/` with
`attempt.json`, `gyro.json` and, per clip, `<camera>.<segment>.mp4` and `<camera>.<segment>.frames.json`.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import records
from .store import Store, open_store

ROOT_ENV = "CUBETRACE_DATA"
SEGMENTS = ("scramble", "solve")


@dataclass(frozen=True, order=True)
class ClipRef:
    """One clip: a camera's recording of one segment of one attempt."""

    session: str
    attempt: int
    camera: str
    segment: str

    def __str__(self) -> str:
        return f"{self.session}/{self.attempt:04d}/{self.camera}.{self.segment}"


def utc_day(ms: float) -> str:
    """The UTC date of a host time, `YYYY-MM-DD`."""
    return dt.datetime.fromtimestamp(ms / 1000, dt.UTC).strftime("%Y-%m-%d")


class Dataset:
    """The records under a root, read once each and validated against `schemas/` unless `validate` is off."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        cache_dir: str | Path | None = None,
        client: Any = None,
        validate: bool = True,
        store: Store | None = None,
    ) -> None:
        if store is None:
            root = root if root is not None else os.environ.get(ROOT_ENV)
            if not root:
                raise ValueError(f"no dataset root: pass one or set {ROOT_ENV}")
            store = open_store(root, cache_dir=cache_dir, client=client)
        self.store = store
        self.validate = validate
        self._json: dict[str, Any] = {}
        self._attempt_dirs: dict[str, dict[int, str]] = {}

    @property
    def root(self) -> str:
        return self.store.display

    # Listing.

    def sessions(self) -> list[str]:
        dirs, _ = self.store.listdir("sessions")
        return dirs

    def _attempts(self, session: str) -> dict[int, str]:
        if session not in self._attempt_dirs:
            found = {}
            for name in self.store.listdir(f"sessions/{session}/attempts")[0]:
                if (
                    name.isdigit()
                    and "attempt.json" in self.store.listdir(self.attempt_dir(session, name))[1]
                ):
                    found[int(name)] = name
            self._attempt_dirs[session] = dict(sorted(found.items()))
        return self._attempt_dirs[session]

    def attempts(self, session: str) -> list[int]:
        """The indices of the session's attempt folders that hold an `attempt.json`."""
        return list(self._attempts(session))

    @staticmethod
    def attempt_dir(session: str, name: str | int) -> str:
        folder = f"{name:04d}" if isinstance(name, int) else name
        return f"sessions/{session}/attempts/{folder}"

    def _dir(self, session: str, index: int) -> str:
        return self.attempt_dir(session, self._attempts(session).get(index, index))

    def files(self, session: str, index: int) -> list[str]:
        """The names of the files in the attempt's folder."""
        return self.store.listdir(self._dir(session, index))[1]

    def clips(self, session: str | None = None) -> Iterator[ClipRef]:
        """Every clip the attempts' `video[]` lists, in session, attempt, segment, camera order."""
        for sid in [session] if session else self.sessions():
            for index in self.attempts(sid):
                video = self.attempt(sid, index)["video"]
                for entry in sorted(video, key=lambda e: (SEGMENTS.index(e["segment"]), e["camera"])):
                    yield ClipRef(sid, index, entry["camera"], entry["segment"])

    # Records.

    def _read(self, kind: str, rel: str) -> Any:
        if rel not in self._json:
            doc = json.loads(self.store.read_bytes(rel))
            if self.validate:
                records.check(kind, doc, rel)
            self._json[rel] = doc
        return self._json[rel]

    def has_session_record(self, session: str) -> bool:
        return "session.json" in self.store.listdir(f"sessions/{session}")[1]

    def session(self, session: str) -> dict[str, Any] | None:
        """`session.json`; None when the session's folder has none."""
        if not self.has_session_record(session):
            return None
        return self._read("session", f"sessions/{session}/session.json")

    def attempt(self, session: str, index: int) -> dict[str, Any]:
        return self._read("attempt", f"{self._dir(session, index)}/attempt.json")

    def clip_entry(self, clip: ClipRef) -> dict[str, Any]:
        """The clip's entry in its attempt's `video[]`."""
        for entry in self.attempt(clip.session, clip.attempt)["video"]:
            if entry["camera"] == clip.camera and entry["segment"] == clip.segment:
                return entry
        raise KeyError(f"no clip {clip}")

    def frames(self, clip: ClipRef) -> dict[str, Any]:
        entry = self.clip_entry(clip)
        return self._read("frames", f"{self._dir(clip.session, clip.attempt)}/{entry['framesFile']}")

    def gyro(self, session: str, index: int) -> dict[str, Any] | None:
        """`gyro.json` when the attempt's record names one and the file is there; None otherwise."""
        summary = self.attempt(session, index).get("gyro")
        if not summary or summary["file"] not in self.files(session, index):
            return None
        return self._read("gyro", f"{self._dir(session, index)}/{summary['file']}")

    def video_rel(self, clip: ClipRef) -> str:
        return f"{self._dir(clip.session, clip.attempt)}/{self.clip_entry(clip)['file']}"

    def video_size(self, clip: ClipRef) -> int | None:
        """The MP4's size in bytes as the store lists it; None when it is missing."""
        return self.store.size(self.video_rel(clip))

    def video_path(self, clip: ClipRef) -> Path:
        """A local path to the clip's MP4 (downloaded into the cache for a bucket root)."""
        return self.store.local_path(self.video_rel(clip))

    def session_day(self, session: str) -> tuple[str | None, bool]:
        """The session's recording day (the UTC date of `createdMs`) and whether it came from `session.json`:
        without one, the day of its earliest `scrambleShown`."""
        record = self.session(session)
        if record is not None:
            return utc_day(record["createdMs"]), True
        shown = [self.attempt(session, index)["events"]["scrambleShown"] for index in self.attempts(session)]
        return (utc_day(min(shown)) if shown else None), False
