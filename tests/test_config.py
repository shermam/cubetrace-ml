"""A run's configuration: the committed TOML files, the overrides, the checks and the JSON round trip."""

from pathlib import Path

import pytest

from cubetrace_ml.config import RunConfig, load_config, parse_override, read_config, write_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_the_committed_configs() -> None:
    found = {path.stem: load_config(path) for path in sorted(CONFIGS.glob("*.toml"))}
    assert set(found) == {"perframe-bigru", "ctc-bigru", "perframe-transformer"}
    assert {name: (c.model.head, c.model.body) for name, c in found.items()} == {
        "perframe-bigru": ("perframe", "bigru"),
        "ctc-bigru": ("ctc", "bigru"),
        "perframe-transformer": ("perframe", "transformer"),
    }
    for name, config in found.items():
        assert config.name == name
        assert (config.train.batch, config.train.lr, config.train.weight_decay) == (8, 1e-3, 0.01)
        assert (config.train.epochs, config.train.patience, config.model.dropout) == (30, 8, 0.2)
        assert (config.data.margin, config.data.fps, config.model.width) == (15, 0.0, 256)
    # Every key of perframe-bigru.toml is a default.
    defaults = RunConfig(name="perframe-bigru").to_json()
    assert found["perframe-bigru"].to_json() == defaults


def test_overrides_are_read_as_toml(tmp_path: Path) -> None:
    config = load_config(
        CONFIGS / "ctc-bigru.toml",
        ["train.epochs=3", "train.lr=3e-4", "data.fps=15", 'train.device="cpu"', "model.body=transformer"],
    )
    assert (config.train.epochs, config.train.lr, config.data.fps) == (3, 3e-4, 15.0)
    assert config.train.device == "cpu" and config.model.body == "transformer"  # a bare word is a string
    assert parse_override("name = x") == ("name", "x")
    assert load_config(None, ["name=mine"]).name == "mine"
    path = tmp_path / "config.json"
    write_config(config, path)
    assert read_config(path) == config


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("train.epoch=3", "unknown key 'train.epoch'"),
        ("training.epochs=3", "unknown configuration section 'training'"),
        ("train.epochs=3.5", "expected int"),
        ("model.head=crf", "model.head 'crf'"),
        ("model.body=lstm", "model.body 'lstm'"),
        ("train.batch=0", "at least 1"),
        ("data.fps=-1", "cannot be negative"),
        ("train.epochs", "expected section.key=value"),
    ],
)
def test_bad_overrides(override: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(None, [override])


def test_a_toml_file_s_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "x.toml"
    path.write_text('[model]\nhead = "ctc"\nlayers = 3\n')
    with pytest.raises(ValueError, match=r"unknown key 'model\.layers'"):
        load_config(path)
    path.write_text('model = "ctc"\n')
    with pytest.raises(ValueError, match="model must be a table"):
        load_config(path)
