"""The encoders' registry and the stub; the PyTorch encoders are built with random weights (no download) when
the features extra is installed, and must refuse to load without it."""

import sys

import numpy as np
import pytest

from cubetrace_ml.encoders import ENCODERS, EncoderError, StubEncoder, encoder_info, load_encoder


def frames(n: int, size: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (n, size, size, 3), dtype=np.uint8)


def test_the_registry_names_each_encoder_s_input_and_output() -> None:
    assert {name: (e.input_size, e.dim) for name, e in ENCODERS.items()} == {
        "stub": (32, 64),
        "resnet18": (224, 512),
        "dinov2-vits14": (224, 768),
    }
    assert ENCODERS["stub"].extra is None
    assert ENCODERS["resnet18"].extra == ENCODERS["dinov2-vits14"].extra == "features"
    with pytest.raises(ValueError, match="no encoder 'vgg'"):
        encoder_info("vgg")


def test_the_stub_is_a_deterministic_projection_of_the_gray_frame() -> None:
    x = frames(5, 32)
    features = load_encoder("stub").encode(x)
    assert features.shape == (5, 64) and features.dtype == np.float32
    np.testing.assert_array_equal(StubEncoder().encode(x), features)  # the same seed, the same features
    assert not np.allclose(StubEncoder(seed=1).encode(x), features)
    np.testing.assert_allclose(StubEncoder().encode(x[2:3])[0], features[2], rtol=1e-5, atol=1e-6)
    # Gray: two frames of the same luma give the same features, whatever their hue.
    red, gray = np.zeros((1, 32, 32, 3), np.uint8), np.zeros((1, 32, 32, 3), np.uint8)
    red[..., 0], gray[...] = 200, round(0.299 * 200)
    np.testing.assert_allclose(StubEncoder().encode(red), StubEncoder().encode(gray), atol=0.02)
    assert load_encoder("stub").weights == "seed 0"
    with pytest.raises(ValueError, match="32 x 32 x 3"):
        StubEncoder().encode(frames(1, 64))
    with pytest.raises(ValueError, match="runs on the CPU"):
        load_encoder("stub", device="cuda")


def test_the_pytorch_encoders_refuse_to_load_without_their_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)  # `import torch` fails as if it were not installed
    for name in ("resnet18", "dinov2-vits14"):
        with pytest.raises(EncoderError, match="needs torch: install the features extra"):
            load_encoder(name)


def test_the_pytorch_encoders_without_downloading_their_weights() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("timm")
    x = frames(2, 224)
    for name, dim in (("resnet18", 512), ("dinov2-vits14", 768)):
        encoder = load_encoder(name, device="cpu", pretrained=False)
        features = encoder.encode(x)
        assert features.shape == (2, dim) and features.dtype == np.float32 and np.isfinite(features).all()
        assert (encoder.device, encoder.precision, encoder.weights) == ("cpu", "fp32", "random")
        assert encoder.describe()["torch"]
        np.testing.assert_allclose(encoder.encode(x[1:]), features[1:], rtol=1e-3, atol=1e-4)
    with pytest.raises(ValueError, match="fp16 runs on a CUDA device only"):
        load_encoder("resnet18", device="cpu", precision="fp16", pretrained=False)


def test_the_normalization_is_the_weights_own() -> None:
    timm = pytest.importorskip("timm")
    config = timm.models.get_pretrained_cfg("vit_small_patch14_dinov2.lvd142m")
    assert (tuple(config.mean), tuple(config.std)) == (
        ENCODERS["dinov2-vits14"].mean,
        ENCODERS["dinov2-vits14"].std,
    )
    torchvision = pytest.importorskip("torchvision")
    preset = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()
    assert (tuple(preset.mean), tuple(preset.std)) == (ENCODERS["resnet18"].mean, ENCODERS["resnet18"].std)
