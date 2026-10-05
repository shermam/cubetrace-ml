import json
from pathlib import Path

import pytest

from cubetrace_ml import records
from cubetrace_ml.dataset import Dataset
from factory import (
    T0,
    camera,
    frames_record,
    gyro_record,
    session_id,
    session_record,
    short_attempt,
)


def test_the_schemas_load_and_are_valid() -> None:
    assert (records.schema_dir() / "attempt.schema.json").is_file()
    for kind in records.KINDS:
        assert records.schema(kind)["$schema"].endswith("2020-12/schema")
        assert records.errors(kind, {}) != []  # an empty object is never a record


def test_the_factory_records_match_the_schemas() -> None:
    sid = session_id(7)
    attempt, frames, gyro = short_attempt(sid, 3, T0, {"laptop": 40.0, "phone-rear": None})
    assert records.errors("attempt", attempt) == []
    for record in frames:
        assert records.errors("frames", record) == []
    assert records.errors("gyro", gyro) == []
    remote = frames_record("phone-rear", "solve", T0, [0, 33.3], remote=True)
    assert records.errors("frames", remote) == []
    session = session_record(sid, T0, [camera("laptop"), camera("phone-rear", local=False)], {"laptop": 40.0})
    assert records.errors("session", session) == []
    dnf, _, _ = short_attempt(sid, 4, T0, {"laptop": 40.0}, status="dnf")
    assert records.errors("attempt", dnf) == []
    assert records.errors("gyro", gyro_record(sid, 1, T0, [0.0], [(0, 0, 0, 1)])) == []


def test_a_mismatch_names_its_place() -> None:
    attempt, _, _ = short_attempt(session_id(1), 1, T0, {"laptop": 40.0})
    attempt["moves"][2]["m"] = "M"
    attempt["result"]["status"] = "fine"
    found = records.errors("attempt", attempt)
    assert any(line.startswith("/moves/2/m: ") for line in found)
    assert any(line.startswith("/result/status: ") for line in found)
    with pytest.raises(records.RecordError) as error:
        records.check("attempt", attempt, "somewhere/attempt.json")
    assert error.value.kind == "attempt" and len(error.value.errors) == 2
    assert "somewhere/attempt.json" in str(error.value)


def test_the_dataset_validates_what_it_reads(tmp_path: Path) -> None:
    sid = session_id(1)
    attempt, _, _ = short_attempt(sid, 1, T0, {"laptop": 40.0})
    attempt["video"] = []
    attempt["index"] = 0  # the schema wants 1 or more
    folder = tmp_path / "sessions" / sid / "attempts" / "0001"
    folder.mkdir(parents=True)
    (folder / "attempt.json").write_text(json.dumps(attempt))
    with pytest.raises(records.RecordError):
        Dataset(tmp_path).attempt(sid, 1)
    assert Dataset(tmp_path, validate=False).attempt(sid, 1)["index"] == 0
