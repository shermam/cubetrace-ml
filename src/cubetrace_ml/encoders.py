"""The frozen frame encoders, by name. Each takes RGB frames at its square input size (an n × S × S × 3 uint8
array) and gives one feature vector per frame (n × dim, float32).

- `stub`: no weights and no PyTorch, a seeded random projection of the frame in gray: what the tests use.
- `resnet18`: torchvision's ResNet-18 with its ImageNet weights, the 512-dim pooled output.
- `dinov2-vits14`: DINOv2's ViT-S/14 through timm (`vit_small_patch14_dinov2.lvd142m`), at 224 (16 × 16
  patches, the position embeddings resampled from its 518), its 384-dim CLS token and the mean of its
  patch tokens concatenated: 768.

The PyTorch ones need the `features` extra (or `cu128` on a GPU machine); their weights download once, on
first use, into torch's and Hugging Face's caches.
"""

from __future__ import annotations

import importlib
import os
import platform
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEVICES = ("auto", "cpu", "cuda")
PRECISIONS = ("auto", "fp32", "fp16")
TIMM_DINOV2 = "vit_small_patch14_dinov2.lvd142m"
INSTALL = "uv sync --extra features (a GPU machine: --extra cu128)"


class EncoderError(RuntimeError):
    """An encoder that cannot be loaded here: its extra is not installed, or its weights did not download."""


@dataclass(frozen=True)
class EncoderInfo:
    """What an encoder takes and gives, and where its weights come from."""

    name: str
    input_size: int
    dim: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    extra: str | None  # the optional dependencies it needs
    weights: str
    output: str

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "inputSize": self.input_size,
            "dim": self.dim,
            "mean": list(self.mean),
            "std": list(self.std),
            "weights": self.weights,
            "output": self.output,
        }


ENCODERS: dict[str, EncoderInfo] = {
    info.name: info
    for info in (
        EncoderInfo(
            "stub",
            32,
            64,
            (0.5, 0.5, 0.5),
            (1.0, 1.0, 1.0),
            None,
            "none: a random projection, seed 0",
            "the 32x32 gray frame (BT.601 luma, 0-1, less 0.5) times a fixed Gaussian 1024x64 matrix",
        ),
        EncoderInfo(
            "resnet18",
            224,
            512,
            IMAGENET_MEAN,
            IMAGENET_STD,
            "features",
            "torchvision ResNet18_Weights.IMAGENET1K_V1 (download.pytorch.org)",
            "the global average pool before the classifier",
        ),
        EncoderInfo(
            "dinov2-vits14",
            224,
            768,
            IMAGENET_MEAN,
            IMAGENET_STD,
            "features",
            f"timm {TIMM_DINOV2} (huggingface.co)",
            "the final norm's CLS token (384) and the mean of its 256 patch tokens (384), concatenated",
        ),
    )
}


def encoder_info(name: str) -> EncoderInfo:
    if name not in ENCODERS:
        raise ValueError(f"no encoder {name!r}; the encoders are {', '.join(ENCODERS)}")
    return ENCODERS[name]


@cache
def cpu_name() -> str:
    """The CPU's model name (Linux's /proc/cpuinfo) and its logical cores."""
    model = ""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    return f"{model or platform.processor() or platform.machine()}, {os.cpu_count()} logical cores"


class Encoder:
    """A loaded encoder: `encode(frames)` maps n × S × S × 3 uint8 RGB frames to n × dim float32 features."""

    info: EncoderInfo
    device: str = "cpu"
    device_name: str = ""
    precision: str = "fp32"
    weights: str = "pretrained"  # `pretrained`, `random` (an architecture alone) or the stub's `seed <n>`

    def encode(self, frames: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        """The host and the run's settings, for the features' meta."""
        return {
            "device": self.device,
            "deviceName": self.device_name,
            "precision": self.precision,
            "torch": None,
        }


LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


class StubEncoder(Encoder):
    """A deterministic random projection of the gray frame: the same seed gives the same features."""

    def __init__(self, seed: int = 0) -> None:
        self.info = ENCODERS["stub"]
        self.weights = f"seed {seed}"
        size = self.info.input_size
        rng = np.random.default_rng(seed)
        self.projection = (rng.standard_normal((size * size, self.info.dim)) / size).astype(np.float32)
        self.device_name = cpu_name()

    def encode(self, frames: np.ndarray) -> np.ndarray:
        frames = np.asarray(frames)
        size = self.info.input_size
        if frames.ndim != 4 or frames.shape[1:] != (size, size, 3):
            raise ValueError(f"stub takes n x {size} x {size} x 3 frames, not {frames.shape}")
        gray = frames.astype(np.float32) @ LUMA / 255.0 - 0.5
        return gray.reshape(len(frames), size * size) @ self.projection


class TorchEncoder(Encoder):
    """A frozen PyTorch module: the frames normalized with the encoder's mean and std, on its device, under
    `torch.inference_mode` (and fp16 autocast with `precision` fp16)."""

    def __init__(self, info: EncoderInfo, module: Any, forward: Any, device: str, precision: str) -> None:
        import torch

        self.torch = torch
        self.info = info
        self.device = device
        self.precision = precision
        self.module = module.eval().to(device)
        self.forward = forward
        self.mean = torch.tensor(info.mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(info.std, device=device).view(1, 3, 1, 1)
        self.device_name = torch.cuda.get_device_name(device) if device == "cuda" else cpu_name()

    def encode(self, frames: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(frames)).to(self.device)
            x = (x.permute(0, 3, 1, 2).float() / 255.0 - self.mean) / self.std
            with torch.autocast(self.device, dtype=torch.float16, enabled=self.precision == "fp16"):
                y = self.forward(self.module, x)
            return y.float().cpu().numpy()

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "torch": self.torch.__version__}


def _require(module: str, name: str) -> Any:
    """`module`, imported; an EncoderError that names the extra when it is not installed."""
    try:
        return importlib.import_module(module)
    except ImportError as error:
        raise EncoderError(
            f"the encoder {name} needs {module}: install the features extra, {INSTALL}"
        ) from error


def _device(torch: Any, device: str) -> str:
    if device not in DEVICES:
        raise ValueError(f"device {device!r}: one of {', '.join(DEVICES)}")
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise EncoderError("--device cuda, but PyTorch sees no CUDA device here")
    return device


def _precision(precision: str, device: str) -> str:
    if precision not in PRECISIONS:
        raise ValueError(f"precision {precision!r}: one of {', '.join(PRECISIONS)}")
    if precision == "auto":
        return "fp16" if device == "cuda" else "fp32"
    if precision == "fp16" and device != "cuda":
        raise ValueError("fp16 runs on a CUDA device only")
    return precision


def _resnet18(torch: Any, pretrained: bool) -> tuple[Any, Any]:
    models = _require("torchvision.models", "resnet18")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = torch.nn.Identity()
    return model, lambda module, x: module(x)


def _dinov2(torch: Any, pretrained: bool) -> tuple[Any, Any]:
    timm = _require("timm", "dinov2-vits14")
    model = timm.create_model(
        TIMM_DINOV2, pretrained=pretrained, img_size=224, dynamic_img_size=True, num_classes=0
    )

    def forward(module: Any, x: Any) -> Any:
        tokens = module.forward_features(x)
        cls = tokens[:, 0]
        patches = tokens[:, module.num_prefix_tokens :].mean(dim=1)
        return torch.cat([cls, patches], dim=1)

    return model, forward


LOADERS = {"resnet18": _resnet18, "dinov2-vits14": _dinov2}


def load_encoder(
    name: str,
    *,
    device: str = "auto",
    precision: str = "auto",
    pretrained: bool = True,
    seed: int = 0,
) -> Encoder:
    """The encoder `name`, ready to encode. `pretrained=False` gives a PyTorch encoder its architecture with
    random weights (no download): for measuring throughput only."""
    info = encoder_info(name)
    if name == "stub":
        if device == "cuda":
            raise ValueError("the stub encoder runs on the CPU (NumPy)")
        return StubEncoder(seed)
    torch = _require("torch", name)
    where = _device(torch, device)
    mode = _precision(precision, where)
    try:
        module, forward = LOADERS[name](torch, pretrained)
    except EncoderError:
        raise
    except Exception as error:  # the weights' download: an HTTP or hub error, an unreachable host
        raise EncoderError(f"could not load {name}'s weights ({info.weights}): {error}") from error
    encoder = TorchEncoder(info, module, forward, where, mode)
    encoder.weights = "pretrained" if pretrained else "random"
    return encoder
