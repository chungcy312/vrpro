from __future__ import annotations

import torch
import torch.nn.functional as F


def _temporal_diff(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        return x[:, 1:] - x[:, :-1]
    if x.ndim == 5:
        return x[:, 1:] - x[:, :-1]
    raise ValueError(f"Unsupported tensor ndim: {x.ndim}")


def motion_autoencoder_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    latent: torch.Tensor,
    recon_weight: float = 1.0,
    motion_weight: float = 1.0,
    smooth_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    recon = F.mse_loss(x_hat, x)

    x_d = _temporal_diff(x)
    x_hat_d = _temporal_diff(x_hat)
    motion = F.mse_loss(x_hat_d, x_d)

    # Encourage compact latent trajectory without killing dynamics.
    latent_d = _temporal_diff(latent)
    smooth = latent_d.abs().mean()

    total = recon_weight * recon + motion_weight * motion + smooth_weight * smooth
    logs = {
        "loss_total": total.detach(),
        "loss_recon": recon.detach(),
        "loss_motion": motion.detach(),
        "loss_smooth": smooth.detach(),
    }
    return total, logs
