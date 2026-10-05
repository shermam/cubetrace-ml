"""The records' JSON Schemas (vendored in `schemas/`) and their validation."""

from __future__ import annotations

import json
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

from jsonschema.validators import validator_for

KINDS = ("attempt", "frames", "gyro", "session")
MAX_MESSAGE = 160


class RecordError(ValueError):
    """A record that does not match its schema; `errors` lists each mismatch."""

    def __init__(self, kind: str, where: str, errors: list[str]) -> None:
        self.kind = kind
        self.where = where
        self.errors = errors
        more = f" (and {len(errors) - 1} more)" if len(errors) > 1 else ""
        super().__init__(f"{where}: not a valid {kind} record: {errors[0]}{more}")


def schema_dir() -> Path:
    """The schemas: the package's copy in a built wheel, else the repository's `schemas/`."""
    packaged = resources.files("cubetrace_ml") / "schemas"
    if packaged.is_dir():
        return Path(str(packaged))
    return Path(__file__).resolve().parents[2] / "schemas"


@cache
def schema(kind: str) -> dict[str, Any]:
    if kind not in KINDS:
        raise ValueError(f"unknown record kind {kind!r}; one of {', '.join(KINDS)}")
    return json.loads((schema_dir() / f"{kind}.schema.json").read_text())


@cache
def _validator(kind: str) -> Any:
    doc = schema(kind)
    cls = validator_for(doc)
    cls.check_schema(doc)
    return cls(doc)


def errors(kind: str, doc: Any) -> list[str]:
    """Every mismatch of `doc` with the schema of `kind`, as `<JSON pointer>: <message>`, in path order."""
    found = []
    for error in _validator(kind).iter_errors(doc):
        pointer = "/" + "/".join(str(part) for part in error.absolute_path)
        message = error.message
        if len(message) > MAX_MESSAGE:
            message = message[: MAX_MESSAGE - 1] + "…"
        found.append((list(map(str, error.absolute_path)), f"{pointer}: {message}"))
    return [text for _, text in sorted(found)]


def check(kind: str, doc: Any, where: str = "") -> Any:
    """`doc` itself when it matches the schema of `kind`; a `RecordError` otherwise."""
    found = errors(kind, doc)
    if found:
        raise RecordError(kind, where or kind, found)
    return doc
