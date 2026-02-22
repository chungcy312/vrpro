from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class DSNPairOutput:
    shared_a: torch.Tensor
    private_a: torch.Tensor
    shared_b: torch.Tensor
    private_b: torch.Tensor
    recon_a: torch.Tensor
    recon_b: torch.Tensor
    logits_a: Optional[torch.Tensor]
    logits_b: Optional[torch.Tensor]


class TemporalProjector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x = self.temporal(x)
        return self.output(x)


class DSNFeatureExtractor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 384,
        shared_dim: int = 128,
        private_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_classes: int = 0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.shared_dim = shared_dim
        self.private_dim = private_dim

        self.shared_encoder = TemporalProjector(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=shared_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.private_encoder = TemporalProjector(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=private_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.decoder = TemporalProjector(
            input_dim=shared_dim + private_dim,
            hidden_dim=hidden_dim,
            out_dim=input_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.classifier = None
        if num_classes > 0:
            self.classifier = nn.Sequential(
                nn.LayerNorm(shared_dim),
                nn.Linear(shared_dim, num_classes),
            )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x3, _ = self._to_btd(x)
        shared = self.shared_encoder(x3)
        private = self.private_encoder(x3)
        return shared, private

    def reconstruct(self, shared: torch.Tensor, private: torch.Tensor, ref_shape: Tuple[int, ...]) -> torch.Tensor:
        x = torch.cat([shared, private], dim=-1)
        x = self.decoder(x)
        return self._restore_shape(x, ref_shape)

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> DSNPairOutput:
        xa, shape_a = self._to_btd(x_a)
        xb, shape_b = self._to_btd(x_b)

        shared_a = self.shared_encoder(xa)
        private_a = self.private_encoder(xa)
        shared_b = self.shared_encoder(xb)
        private_b = self.private_encoder(xb)

        recon_a = self.reconstruct(shared_a, private_a, shape_a)
        recon_b = self.reconstruct(shared_b, private_b, shape_b)

        logits_a = None
        logits_b = None
        if self.classifier is not None:
            logits_a = self.classifier(shared_a.mean(dim=1))
            logits_b = self.classifier(shared_b.mean(dim=1))

        return DSNPairOutput(
            shared_a=shared_a,
            private_a=private_a,
            shared_b=shared_b,
            private_b=private_b,
            recon_a=recon_a,
            recon_b=recon_b,
            logits_a=logits_a,
            logits_b=logits_b,
        )

    @staticmethod
    def _to_btd(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        if x.ndim == 3:
            return x, x.shape
        if x.ndim == 4:
            # [B, T, N, C] -> token mean [B, T, C]
            return x.mean(dim=2), x.shape
        if x.ndim == 5:
            # [B, T, H, W, C] -> spatial mean [B, T, C]
            return x.mean(dim=(2, 3)), x.shape
        raise ValueError(f"Expected [B,T,C], [B,T,N,C], or [B,T,H,W,C], got {tuple(x.shape)}")

    @staticmethod
    def _restore_shape(x: torch.Tensor, original_shape: Tuple[int, ...]) -> torch.Tensor:
        if len(original_shape) == 3:
            return x
        if len(original_shape) == 4:
            b, t, n, c = original_shape
            return x.unsqueeze(2).expand(b, t, n, c)
        if len(original_shape) == 5:
            b, t, h, w, c = original_shape
            return x.unsqueeze(2).unsqueeze(3).expand(b, t, h, w, c)
        raise ValueError(f"Unsupported original shape: {original_shape}")
