from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _orthogonality_penalty(shared: torch.Tensor, private: torch.Tensor) -> torch.Tensor:
    # shared/private: [B, T, D]
    s = F.normalize(shared.reshape(shared.shape[0], -1), dim=-1)
    p = F.normalize(private.reshape(private.shape[0], -1), dim=-1)
    return (s * p).sum(dim=-1).abs().mean()


def dsn_pair_loss(
    x_a: torch.Tensor,
    x_b: torch.Tensor,
    recon_a: torch.Tensor,
    recon_b: torch.Tensor,
    shared_a: torch.Tensor,
    shared_b: torch.Tensor,
    private_a: torch.Tensor,
    private_b: torch.Tensor,
    logits_a: Optional[torch.Tensor] = None,
    logits_b: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    recon_weight: float = 1.0,
    shared_weight: float = 1.0,
    orth_weight: float = 0.1,
    private_margin_weight: float = 0.1,
    private_margin: float = 1.0,
    classifier_weight: float = 0.0,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    loss_recon = F.mse_loss(recon_a, x_a) + F.mse_loss(recon_b, x_b)
    loss_shared = F.mse_loss(shared_a, shared_b)

    loss_orth = _orthogonality_penalty(shared_a, private_a) + _orthogonality_penalty(shared_b, private_b)

    private_distance = (private_a - private_b).pow(2).mean(dim=(1, 2)).sqrt()
    loss_private_margin = F.relu(private_margin - private_distance).mean()

    total = (
        recon_weight * loss_recon
        + shared_weight * loss_shared
        + orth_weight * loss_orth
        + private_margin_weight * loss_private_margin
    )

    loss_classifier = torch.tensor(0.0, device=x_a.device)
    if (
        classifier_weight > 0
        and labels is not None
        and logits_a is not None
        and logits_b is not None
    ):
        loss_classifier = 0.5 * (
            F.cross_entropy(logits_a, labels) + F.cross_entropy(logits_b, labels)
        )
        total = total + classifier_weight * loss_classifier

    logs: Dict[str, torch.Tensor] = {
        "loss_total": total,
        "loss_recon": loss_recon,
        "loss_shared": loss_shared,
        "loss_orth": loss_orth,
        "loss_private_margin": loss_private_margin,
        "loss_classifier": loss_classifier,
    }
    return total, logs
