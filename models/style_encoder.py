from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .common import PositionalEncoding


class StyleEncoder(nn.Module):
    def __init__(self, args: Any) -> None:
        super().__init__()

        self.motion_coef_dim = 50
        if args.rot_repr == 'aa':
            self.motion_coef_dim += 1 if args.no_head_pose else 4
        else:
            raise ValueError(f'Unknown rotation representation {args.rot_repr}!')

        self.feature_dim: int = args.feature_dim
        self.n_heads: int = args.n_heads
        self.n_layers: int = args.n_layers
        self.mlp_ratio: int = args.mlp_ratio

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=self.n_heads,
            dim_feedforward=self.mlp_ratio * self.feature_dim,
            activation='gelu',
            batch_first=True,
        )

        self.PE = PositionalEncoding(self.feature_dim)
        self.encoder = nn.ModuleDict({
            'motion_proj': nn.Linear(self.motion_coef_dim, self.feature_dim),
            'transformer': nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers),
        })

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, motion_coef: torch.Tensor) -> torch.Tensor:
        """
        Args:
            motion_coef: (batch_size, seq_len, motion_coef_dim)

        Returns:
            torch.Tensor: (batch_size, feature_dim)
        """
        motion_feat = self.encoder['motion_proj'](motion_coef)
        motion_feat = self.PE(motion_feat)
        feat = self.encoder['transformer'](motion_feat)  # (N, L, feat_dim)
        feat = feat.mean(dim=1)  # Pool to (N, feat_dim)
        return feat
