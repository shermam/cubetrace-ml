"""A training run's configuration: a TOML file (`configs/*.toml`) of sections `data`, `model`, `train`,
`decode` and `paths`, every key optional (the defaults below), with `--set section.key=value` overrides
(the value read as TOML, else as a string). The resolved configuration is the run's `config.json`."""

from __future__ import annotations

import dataclasses
import json
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .decode import MIN_DISTANCE, NEIGHBOURS
from .labels import INPUTS, MARGIN, LabelConfig

HEADS = ("perframe", "ctc")
BODIES = ("bigru", "transformer")


@dataclass
class DataConfig:
    """The labels (`labels.LabelConfig`): the margin, the frame rate, the soft target and the time base; the
    model's inputs (`features`, or `features+gyro`: the gyro's 9 channels after the features) and whether the
    clips without a gyro are skipped (`require_gyro`), so that two runs can train on the same clips."""

    margin: int = MARGIN
    fps: float = 0.0
    label_frames: int = 0
    soft_decay: float = 0.5
    time_base: str = "fit"
    inputs: str = "features"
    require_gyro: bool = False

    def labels(self) -> LabelConfig:
        return LabelConfig(
            margin=self.margin,
            fps=self.fps,
            label_frames=self.label_frames,
            soft_decay=self.soft_decay,
            time_base=self.time_base,
            with_gyro=self.inputs == "features+gyro",
            require_gyro=self.require_gyro,
        )


@dataclass
class ModelConfig:
    """The features through a projection, the convolutions and the body (a BiGRU or a transformer encoder)
    into the head (`perframe`: 25 classes per frame; `ctc`: the blank and the 24 symbols)."""

    head: str = "perframe"
    body: str = "bigru"
    width: int = 256
    conv_layers: int = 2
    conv_kernel: int = 5
    gru_hidden: int = 128
    gru_layers: int = 2
    transformer_layers: int = 4
    transformer_heads: int = 4
    transformer_ff: int = 1024
    dropout: float = 0.2


@dataclass
class TrainConfig:
    seed: int = 0
    batch: int = 8  # whole clips per batch
    epochs: int = 30  # the cosine schedule's length
    lr: float = 1e-3
    weight_decay: float = 0.01
    clip_grad: float = 1.0
    patience: int = 8  # epochs without a better val metric before stopping
    min_epochs: int = 10  # epochs that always run before early stopping may stop (CTC's plateau)
    time_masks: int = 3  # spans of the input zeroed per clip (the augmentation)
    time_mask_frames: int = 10  # the longest span
    device: str = "auto"


@dataclass
class DecodeConfig:
    min_distance: int = MIN_DISTANCE
    neighbours: int = NEIGHBOURS
    threshold: float = 0.5  # the per-frame head's threshold until val chooses one


@dataclass
class PathsConfig:
    features: str = ""  # the features root (`$CUBETRACE_FEATURES`)
    encoder: str = ""
    root: str = ""  # the dataset root (`$CUBETRACE_DATA`)
    manifest: str = ""  # empty: built from the root


@dataclass
class RunConfig:
    name: str = "run"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @staticmethod
    def from_json(doc: dict[str, Any]) -> RunConfig:
        config = RunConfig()
        for key, value in doc.items():
            _set(config, key, value)
        config.check()
        return config

    def check(self) -> None:
        if self.model.head not in HEADS:
            raise ValueError(f"model.head {self.model.head!r}: one of {', '.join(HEADS)}")
        if self.model.body not in BODIES:
            raise ValueError(f"model.body {self.model.body!r}: one of {', '.join(BODIES)}")
        if self.data.inputs not in INPUTS:
            raise ValueError(f"data.inputs {self.data.inputs!r}: one of {', '.join(INPUTS)}")
        if self.data.fps < 0 or self.data.margin < 0 or self.data.label_frames < 0:
            raise ValueError("data.fps, data.margin and data.label_frames cannot be negative")
        if self.train.batch < 1 or self.train.epochs < 1:
            raise ValueError("train.batch and train.epochs must be at least 1")


def _set(config: RunConfig, key: str, value: Any) -> None:
    """Sets `section.key` (or a whole section from a table, or `name`), checking the key and the type."""
    section_name, _, name = key.partition(".")
    if section_name == "name" and not name:
        config.name = str(value)
        return
    sections = {f.name for f in dataclasses.fields(config)} - {"name"}
    if section_name not in sections:
        raise ValueError(
            f"unknown configuration section {section_name!r}; the sections are {sorted(sections)}"
        )
    section = getattr(config, section_name)
    if not name:
        if not isinstance(value, dict):
            raise ValueError(f"{section_name} must be a table")
        for k, v in value.items():
            _set(config, f"{section_name}.{k}", v)
        return
    fields = {f.name: f for f in dataclasses.fields(section)}
    if name not in fields:
        raise ValueError(f"unknown key {key!r}; {section_name} has {', '.join(fields)}")
    current = getattr(section, name)
    if not isinstance(current, bool | int | float | str):
        raise ValueError(f"{key}: unsupported type")
    if isinstance(current, float) and isinstance(value, int) and not isinstance(value, bool):
        value = float(value)
    if type(value) is not type(current):
        raise ValueError(f"{key} = {value!r}: expected {type(current).__name__}")
    setattr(section, name, value)


def parse_override(text: str) -> tuple[str, Any]:
    """`section.key=value`, the value read as TOML (`3`, `1e-3`, `"x"`) or else as a plain string."""
    key, sep, raw = text.partition("=")
    if not sep or not key.strip():
        raise ValueError(f"--set {text!r}: expected section.key=value")
    try:
        value = tomllib.loads(f"v = {raw.strip()}")["v"]
    except tomllib.TOMLDecodeError:
        value = raw.strip()
    return key.strip(), value


def load_config(path: str | Path | None, overrides: Sequence[str] = ()) -> RunConfig:
    """The configuration of a TOML file (the defaults without one), with the overrides applied; its name is
    the file's stem unless the file says otherwise."""
    config = RunConfig()
    if path is not None:
        path = Path(path)
        config.name = path.stem
        doc = tomllib.loads(path.read_text())
        for key, value in doc.items():
            _set(config, key, value)
    for text in overrides:
        _set(config, *parse_override(text))
    config.check()
    return config


def write_config(config: RunConfig, path: str | Path) -> None:
    Path(path).write_text(json.dumps(config.to_json(), indent=2) + "\n")


def read_config(path: str | Path) -> RunConfig:
    return RunConfig.from_json(json.loads(Path(path).read_text()))
