"""Where a dataset root's files are: a local folder, or a `gs://` prefix read through an on-disk cache.

Paths inside a store are relative to the root and use `/`: `sessions/<id>/attempts/0003/attempt.json`.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Protocol


class Store(Protocol):
    """The few operations the dataset needs from where its files are."""

    @property
    def display(self) -> str: ...

    def listdir(self, rel: str) -> tuple[list[str], list[str]]:
        """The folders and the files directly in `rel`, each sorted; empty when `rel` is not there."""
        ...

    def read_bytes(self, rel: str) -> bytes: ...

    def local_path(self, rel: str) -> Path:
        """A local file with the content of `rel` (for the bucket: downloaded into the cache)."""
        ...

    def size(self, rel: str) -> int | None:
        """The file's size in bytes; None when it is not there."""
        ...


class LocalStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()
        if not self.root.is_dir():
            raise FileNotFoundError(f"dataset root is not a folder: {self.root}")

    @property
    def display(self) -> str:
        return str(self.root)

    def _path(self, rel: str) -> Path:
        return self.root.joinpath(*[part for part in rel.split("/") if part])

    def listdir(self, rel: str) -> tuple[list[str], list[str]]:
        path = self._path(rel)
        if not path.is_dir():
            return [], []
        dirs, files = [], []
        for entry in os.scandir(path):
            (dirs if entry.is_dir() else files).append(entry.name)
        return sorted(dirs), sorted(files)

    def read_bytes(self, rel: str) -> bytes:
        return self._path(rel).read_bytes()

    def local_path(self, rel: str) -> Path:
        path = self._path(rel)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def size(self, rel: str) -> int | None:
        path = self._path(rel)
        return path.stat().st_size if path.is_file() else None


def parse_gs_url(url: str) -> tuple[str, str]:
    """`gs://bucket/some/prefix` → (`bucket`, `some/prefix/`); the prefix is empty or ends with `/`."""
    if not url.startswith("gs://"):
        raise ValueError(f"not a gs:// URL: {url!r}")
    bucket, _, prefix = url[len("gs://") :].partition("/")
    if not bucket:
        raise ValueError(f"no bucket in {url!r}")
    prefix = "/".join(part for part in prefix.split("/") if part)
    return bucket, prefix + "/" if prefix else ""


def default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "cubetrace-ml"


class GcsStore:
    """A `gs://` root: one listing of `<prefix>sessions/` (names, sizes, generations) per store, and files
    read through `<cache>/gcs/<bucket>/<object name>`, kept while their generation is the listed one.

    `client` is a `google.cloud.storage.Client` (Application Default Credentials) or a stand-in with the
    same three calls: `list_blobs(bucket, prefix=…)`, `bucket(name).blob(name, generation=…)` and the
    blob's `download_to_filename(path)`.
    """

    def __init__(self, url: str, *, cache_dir: str | Path | None = None, client: Any = None) -> None:
        self.bucket, self.prefix = parse_gs_url(url)
        self.cache = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
        if client is None:
            try:
                from google.cloud import storage
            except ImportError as error:  # pragma: no cover - depends on the installed extras
                raise ImportError(
                    "gs:// roots need google-cloud-storage: uv sync --extra gcs "
                    "(or pip install 'cubetrace-ml[gcs]')"
                ) from error
            client = storage.Client()
        self.client = client
        self._blobs: dict[str, tuple[int, int]] | None = None
        self._children: dict[str, tuple[set[str], set[str]]] = {}

    @property
    def display(self) -> str:
        return f"gs://{self.bucket}/{self.prefix}"

    def _index(self) -> dict[str, tuple[int, int]]:
        if self._blobs is None:
            blobs: dict[str, tuple[int, int]] = {}
            children: dict[str, tuple[set[str], set[str]]] = {}
            for blob in self.client.list_blobs(self.bucket, prefix=self.prefix + "sessions/"):
                rel = blob.name[len(self.prefix) :]
                if not rel or rel.endswith("/"):
                    continue
                blobs[rel] = (int(blob.size or 0), int(blob.generation or 0))
                parts = rel.split("/")
                for depth in range(len(parts)):
                    parent = "/".join(parts[:depth])
                    dirs, files = children.setdefault(parent, (set(), set()))
                    (files if depth == len(parts) - 1 else dirs).add(parts[depth])
            self._blobs, self._children = blobs, children
        return self._blobs

    def listdir(self, rel: str) -> tuple[list[str], list[str]]:
        self._index()
        dirs, files = self._children.get(rel.strip("/"), (set(), set()))
        return sorted(dirs), sorted(files)

    def size(self, rel: str) -> int | None:
        entry = self._index().get(rel.strip("/"))
        return entry[0] if entry else None

    def local_path(self, rel: str) -> Path:
        rel = rel.strip("/")
        entry = self._index().get(rel)
        if entry is None:
            raise FileNotFoundError(f"{self.display}{rel}")
        _, generation = entry
        name = self.prefix + rel
        path = self.cache / "gcs" / self.bucket / name
        stamp = path.with_name(path.name + ".generation")
        if path.is_file() and stamp.is_file() and stamp.read_text().strip() == str(generation):
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.part")
        try:
            self.client.bucket(self.bucket).blob(name, generation=generation).download_to_filename(
                str(partial)
            )
            os.replace(partial, path)
        finally:
            partial.unlink(missing_ok=True)
        stamp.write_text(f"{generation}\n")
        return path

    def read_bytes(self, rel: str) -> bytes:
        return self.local_path(rel).read_bytes()


def open_store(root: str | Path, *, cache_dir: str | Path | None = None, client: Any = None) -> Store:
    """The store of a dataset root: a `gs://bucket/prefix` URL or a local folder."""
    text = str(root)
    if text.startswith("gs://"):
        return GcsStore(text, cache_dir=cache_dir, client=client)
    return LocalStore(text)
