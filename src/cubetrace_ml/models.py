"""The temporal models on the cached features (PyTorch: the `features` extra).

A clip's features (T × D, the training set's mean and standard deviation subtracted and divided, both kept
in the model's state) go through a linear projection to `width` (256), `conv_layers` (2) residual 1D
convolutions (kernel 5), the body — a BiGRU (`gru_layers` 2, `gru_hidden` 128 per direction) or a
transformer encoder (`transformer_layers` 4, `width` 256, `transformer_heads` 4, sinusoidal positions)
— and a linear head of 25 outputs per frame: "no onset" and the 24 symbols (`perframe`), or the blank and
the 24 symbols (`ctc`). Batches are whole clips padded at the end, with a mask: the padding changes no
output of a real frame.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .config import ModelConfig
from .labels import CLASSES


def positions(length: int, width: int, device: torch.device | str) -> torch.Tensor:
    """Sinusoidal position encodings, length × width."""
    position = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    rate = torch.exp(torch.arange(0, width, 2, device=device, dtype=torch.float32) * (-math.log(1e4) / width))
    out = torch.zeros(length, width, device=device)
    out[:, 0::2] = torch.sin(position * rate)
    out[:, 1::2] = torch.cos(position * rate[: width // 2])
    return out


class MoveModel(nn.Module):
    """Features → per-frame logits over the 25 classes."""

    def __init__(self, dim: int, config: ModelConfig) -> None:
        super().__init__()
        self.dim = dim
        self.body = config.body
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        width = config.width
        self.proj = nn.Linear(dim, width)
        kernel = config.conv_kernel
        self.convs = nn.ModuleList(
            nn.Conv1d(width, width, kernel, padding=kernel // 2) for _ in range(config.conv_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(config.conv_layers))
        self.dropout = nn.Dropout(config.dropout)
        if config.body == "bigru":
            self.gru = nn.GRU(
                width,
                config.gru_hidden,
                num_layers=config.gru_layers,
                batch_first=True,
                bidirectional=True,
                dropout=config.dropout if config.gru_layers > 1 else 0.0,
            )
            out = 2 * config.gru_hidden
        elif config.body == "transformer":
            layer = nn.TransformerEncoderLayer(
                width,
                config.transformer_heads,
                config.transformer_ff,
                config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, config.transformer_layers, enable_nested_tensor=False)
            self.final = nn.LayerNorm(width)
            out = width
        else:
            raise ValueError(f"no body {config.body!r}")
        self.head = nn.Linear(out, CLASSES)

    def set_normalization(self, mean: Any, std: Any) -> None:
        self.mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.std.copy_(torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-6))

    def standardize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, time_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """`x` B × T × D (the features as stored), `mask` B × T (True on a clip's frames), `time_mask` B × T
        (True: the frame's standardized features zeroed); the logits, B × T × 25."""
        h = self.standardize(x)
        if time_mask is not None:
            h = h.masked_fill(time_mask[..., None], 0.0)
        keep = mask[..., None].to(h.dtype)
        h = self.dropout(F.gelu(self.proj(h))) * keep
        for conv, norm in zip(self.convs, self.norms, strict=True):
            h = norm(h + self.dropout(F.gelu(conv(h.transpose(1, 2)).transpose(1, 2)))) * keep
        if self.body == "bigru":
            lengths = mask.sum(dim=1).cpu()
            packed = pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
            out, _ = self.gru(packed)
            h, _ = pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        else:
            h = h + positions(h.shape[1], h.shape[2], h.device)
            h = self.final(self.encoder(h, src_key_padding_mask=~mask))
        return self.head(self.dropout(h))


def class_weights(targets: Sequence[Any]) -> torch.Tensor:
    """The classes' weights: 1 for "no onset" and one weight for every onset class, so that the onset frames
    weigh as much in all as the other frames (counted on the training clips' hard targets)."""
    negatives = sum(int((t == 0).sum()) for t in targets)
    positives = sum(int((t != 0).sum()) for t in targets)
    weight = negatives / positives if positives else 1.0
    return torch.tensor([1.0] + [weight] * (CLASSES - 1))


def perframe_loss(
    logits: torch.Tensor,
    near_class: torch.Tensor,
    near_weight: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """The weighted cross-entropy against the (soft) target: on each frame `near_weight` on its
    `near_class` and the rest on "no onset", each class's term times its weight, normalized by the frames'
    total weight (the hard targets' case is PyTorch's weighted mean)."""
    logp = F.log_softmax(logits, dim=-1)
    on = near_weight * weights[near_class]
    off = (1.0 - near_weight) * weights[0]
    nll = -(off * logp[..., 0] + on * logp.gather(-1, near_class[..., None]).squeeze(-1))
    m = mask.to(logits.dtype)
    return (nll * m).sum() / ((on + off) * m).sum().clamp_min(1e-8)


def ctc_loss(logits: torch.Tensor, mask: torch.Tensor, targets: Sequence[torch.Tensor]) -> torch.Tensor:
    """CTC with blank 0 and the symbols 1–24 against each clip's reference sequence (each loss over its
    reference's length, then the batch's mean; an impossible alignment counts 0)."""
    logp = F.log_softmax(logits, dim=-1).transpose(0, 1)
    lengths = mask.sum(dim=1)
    flat = torch.cat([t.to(logits.device) for t in targets]) if targets else torch.zeros(0)
    target_lengths = torch.tensor([len(t) for t in targets], device=logits.device)
    return F.ctc_loss(
        logp, flat.long(), lengths, target_lengths, blank=0, reduction="mean", zero_infinity=True
    )
