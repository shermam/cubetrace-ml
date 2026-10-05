"""The bucket roots, through a stand-in for google-cloud-storage's client: no network."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest

from cubetrace_ml.dataset import ClipRef, Dataset
from cubetrace_ml.store import GcsStore, LocalStore, open_store, parse_gs_url
from factory import T0, session_id, short_attempt


@dataclass
class FakeBlob:
    name: str
    data: bytes
    generation: int
    client: FakeClient

    @property
    def size(self) -> int:
        return len(self.data)

    def download_to_filename(self, filename: str) -> None:
        self.client.downloads[self.name] += 1
        Path(filename).write_bytes(self.data)


class FakeBucket:
    def __init__(self, client: FakeClient, name: str) -> None:
        self.client, self.name = client, name

    def blob(self, name: str, generation: int | None = None) -> FakeBlob:
        blob = self.client.objects[self.name][name]
        assert generation is None or generation == blob.generation
        return blob


class FakeClient:
    """`list_blobs`, `bucket(…).blob(…)` and `download_to_filename`, over objects in memory."""

    def __init__(self) -> None:
        self.objects: dict[str, dict[str, FakeBlob]] = {}
        self.downloads: Counter[str] = Counter()
        self.listings = 0

    def put(self, bucket: str, name: str, data: bytes) -> None:
        old = self.objects.setdefault(bucket, {}).get(name)
        self.objects[bucket][name] = FakeBlob(name, data, (old.generation + 1) if old else 1, self)

    def list_blobs(self, bucket: str, prefix: str = "") -> list[FakeBlob]:
        self.listings += 1
        return [b for n, b in sorted(self.objects.get(bucket, {}).items()) if n.startswith(prefix)]

    def bucket(self, name: str) -> FakeBucket:
        return FakeBucket(self, name)


def test_parse_gs_url() -> None:
    assert parse_gs_url("gs://cubetrace-data/users/abc") == ("cubetrace-data", "users/abc/")
    assert parse_gs_url("gs://cubetrace-data/users/abc/") == ("cubetrace-data", "users/abc/")
    assert parse_gs_url("gs://cubetrace-data") == ("cubetrace-data", "")
    assert parse_gs_url("gs://cubetrace-data/") == ("cubetrace-data", "")
    assert parse_gs_url("gs://b//a//b/") == ("b", "a/b/")
    for bad in ("gs://", "gs:///x", "s3://b/x", "/local/path"):
        with pytest.raises(ValueError):
            parse_gs_url(bad)


def test_open_store_picks_by_root(tmp_path: Path) -> None:
    assert isinstance(open_store(tmp_path), LocalStore)
    assert isinstance(open_store("gs://b/p", client=FakeClient(), cache_dir=tmp_path), GcsStore)
    with pytest.raises(FileNotFoundError):
        open_store(tmp_path / "nowhere")


def bucket_with_one_attempt(client: FakeClient, prefix: str) -> tuple[str, dict]:
    sid = session_id(5)
    attempt, frames, gyro = short_attempt(sid, 2, T0, {"laptop": 40.0})
    base = f"{prefix}sessions/{sid}/attempts/0002/"
    for record in frames:
        client.put(
            "data", base + f"{record['camera']}.{record['segment']}.frames.json", json.dumps(record).encode()
        )
    for entry in attempt["video"]:
        entry["bytes"] = 4
        client.put("data", base + entry["file"], b"mp4!")
    client.put("data", base + "gyro.json", json.dumps(gyro).encode())
    client.put("data", base + "attempt.json", json.dumps(attempt).encode())
    client.put("data", "users/someone-else/sessions/x/session.json", b"{}")  # outside the root
    return sid, attempt


def test_a_bucket_root_lists_and_reads_through_the_cache(tmp_path: Path) -> None:
    client = FakeClient()
    sid, attempt = bucket_with_one_attempt(client, "users/u1/")
    dataset = Dataset("gs://data/users/u1", client=client, cache_dir=tmp_path / "cache")
    assert dataset.root == "gs://data/users/u1/"
    assert dataset.sessions() == [sid]
    assert dataset.attempts(sid) == [2]
    assert dataset.attempt(sid, 2) == attempt
    clips = list(dataset.clips())
    assert clips == [ClipRef(sid, 2, "laptop", "scramble"), ClipRef(sid, 2, "laptop", "solve")]
    assert dataset.frames(clips[0])["camera"] == "laptop"
    assert dataset.video_size(clips[0]) == 4
    assert client.listings == 1
    assert not any(name.endswith(".mp4") for name in client.downloads)  # no video until one is needed
    cached = tmp_path / "cache" / "gcs" / "data" / f"users/u1/sessions/{sid}/attempts/0002/attempt.json"
    assert json.loads(cached.read_text()) == attempt

    again = Dataset("gs://data/users/u1", client=client, cache_dir=tmp_path / "cache")
    again.attempt(sid, 2)
    assert client.downloads[f"users/u1/sessions/{sid}/attempts/0002/attempt.json"] == 1  # read from the cache

    path = again.video_path(clips[1])
    assert path.read_bytes() == b"mp4!" and path.is_relative_to(tmp_path / "cache")
    assert sum(n for name, n in client.downloads.items() if name.endswith(".mp4")) == 1


def test_a_new_generation_is_downloaded_again(tmp_path: Path) -> None:
    client = FakeClient()
    sid, attempt = bucket_with_one_attempt(client, "")
    name = f"sessions/{sid}/attempts/0002/attempt.json"
    Dataset("gs://data", client=client, cache_dir=tmp_path).attempt(sid, 2)
    changed = dict(attempt, scramble="U")
    client.put("data", name, json.dumps(changed).encode())
    assert Dataset("gs://data", client=client, cache_dir=tmp_path).attempt(sid, 2)["scramble"] == "U"
    assert client.downloads[name] == 2


def test_a_missing_object_is_not_found(tmp_path: Path) -> None:
    store = GcsStore("gs://data/p", client=FakeClient(), cache_dir=tmp_path)
    assert store.listdir("sessions") == ([], [])
    assert store.size("sessions/x/session.json") is None
    with pytest.raises(FileNotFoundError):
        store.read_bytes("sessions/x/session.json")
