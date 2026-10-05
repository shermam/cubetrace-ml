from __future__ import annotations

from pathlib import Path

import pytest

from cubetrace_ml.dataset import ROOT_ENV
from factory import build_dataset


@pytest.fixture(autouse=True)
def no_real_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tests never read the owner's recordings: the root always comes from the test itself."""
    monkeypatch.delenv(ROOT_ENV, raising=False)


@pytest.fixture(scope="session")
def dataset_root(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, str]]:
    """The factory's four-session dataset with its videos, written once (read-only for the tests)."""
    root = tmp_path_factory.mktemp("dataset")
    return root, build_dataset(root)
