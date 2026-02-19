from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class MotionAEOutput:
    latent: torch.Tensor
    reconstruction: torch.Tensor
    motion_embedding: torch.Tensor


class TemporalConvBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
        self.pointwise = nn.Conv1d(dim, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [BN, T, D]
        residual = x
        x = x.transpose(1, 2)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x + residual


class MotionFeatureEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int = 384,
        latent_dim: int = 192,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_token: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.latent_dim = latent_dim
        self.use_token = use_token

        in_proj_dim = input_dim * 2 if use_token else input_dim
        self.input_proj = nn.Linear(in_proj_dim, model_dim)
        self.conv_block = TemporalConvBlock(model_dim, kernel_size=3, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.latent_proj = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, orig_shape = self._to_btnc(x)
        motion = self._temporal_difference(x)

        if self.use_token:
            x = torch.cat([x, motion], dim=-1)
        else:
            x = motion

        b, t, n, _ = x.shape
        x = self.input_proj(x)
        x = x.reshape(b * n, t, self.model_dim)
        x = self.conv_block(x)
        x = self.temporal_transformer(x)
        z = self.latent_proj(x)
        z = z.reshape(b, t, n, self.latent_dim)
        return self._restore_shape(z, orig_shape)

    @staticmethod
    def _temporal_difference(x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, C]
        delta = x[:, 1:] - x[:, :-1]
        first = torch.zeros_like(x[:, :1])
        return torch.cat([first, delta], dim=1)

    @staticmethod
    def _to_btnc(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        if x.ndim == 4:
            # [B, T, N, C]
            return x, x.shape
        if x.ndim == 5:
            # [B, T, H, W, C] -> [B, T, N, C]
            b, t, h, w, c = x.shape
            return x.reshape(b, t, h * w, c), x.shape
        raise ValueError(f"Expected [B,T,N,C] or [B,T,H,W,C], got shape={tuple(x.shape)}")

    @staticmethod
    def _restore_shape(x: torch.Tensor, orig_shape: Tuple[int, ...]) -> torch.Tensor:
        if len(orig_shape) == 4:
            return x
        b, t, h, w, _ = orig_shape
        return x.reshape(b, t, h, w, -1)


class MotionFeatureDecoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        model_dim: int = 384,
        latent_dim: int = 192,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.model_dim = model_dim

        self.input_proj = nn.Linear(latent_dim, model_dim)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.conv_block = TemporalConvBlock(model_dim, kernel_size=3, dropout=dropout)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, output_dim),
        )

    def forward(self, z: torch.Tensor, output_layout: Tuple[int, ...] | None = None) -> torch.Tensor:
        z, orig_shape = MotionFeatureEncoder._to_btnc(z)
        b, t, n, _ = z.shape

        x = self.input_proj(z)
        x = x.reshape(b * n, t, self.model_dim)
        x = self.temporal_transformer(x)
        x = self.conv_block(x)
        x = self.output_proj(x)
        x = x.reshape(b, t, n, self.output_dim)

        shape_ref = output_layout if output_layout is not None else orig_shape
        return MotionFeatureEncoder._restore_shape(x, shape_ref)


class MotionFeatureAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int = 384,
        latent_dim: int = 192,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_token: bool = True,
    ):
        super().__init__()
        self.encoder = MotionFeatureEncoder(
            input_dim=input_dim,
            model_dim=model_dim,
            latent_dim=latent_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            use_token=use_token,
        )
        self.decoder = MotionFeatureDecoder(
            output_dim=input_dim,
            model_dim=model_dim,
            latent_dim=latent_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> MotionAEOutput:
        z = self.encoder(x)
        x_hat = self.decoder(z, output_layout=x.shape)

        # Motion embedding for downstream tasks (compact and motion-aware)
        if z.ndim == 4:
            motion_embedding = z.mean(dim=2)
        else:
            motion_embedding = z.mean(dim=(2, 3))
        return MotionAEOutput(latent=z, reconstruction=x_hat, motion_embedding=motion_embedding)

    @property
    def compression_ratio(self) -> float:
        return self.encoder.latent_dim / self.encoder.input_dim
