from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 600) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Vanilla sinusoidal encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.shape[1], :]
        return self.dropout(x)


def enc_dec_mask(
    T: int,
    S: int,
    frame_width: int = 2,
    expansion: int = 0,
    device: str = 'cuda',
) -> torch.Tensor:
    mask = torch.ones(T, S)
    for i in range(T):
        mask[i, max(0, (i - expansion) * frame_width) : (i + expansion + 1) * frame_width] = 0
    return (mask == 1).to(device=device)


def pad_audio(
    audio: torch.Tensor,
    audio_unit: int = 320,
    pad_threshold: int = 80,
) -> torch.Tensor:
    batch_size, audio_len = audio.shape
    n_units = audio_len // audio_unit
    side_len = math.ceil((audio_unit * n_units + pad_threshold - audio_len) / 2)
    if side_len >= 0:
        reflect_len = side_len // 2
        replicate_len = side_len % 2
        if reflect_len > 0:
            audio = F.pad(audio, (reflect_len, reflect_len), mode='reflect')
            audio = F.pad(audio, (reflect_len, reflect_len), mode='reflect')
        if replicate_len > 0:
            audio = F.pad(audio, (1, 1), mode='replicate')
    return audio
