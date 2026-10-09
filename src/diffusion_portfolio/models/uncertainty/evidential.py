"""Evidential regime classification for proposal regime probe."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class EvidentialOutput:
    """Dirichlet evidential prediction."""

    evidence: Tensor
    alpha: Tensor
    probabilities: Tensor
    vacuity: Tensor


@dataclass(frozen=True)
class EvidentialLossOutput:
    """Components of the evidential classification objective."""

    loss: Tensor
    data_loss: Tensor
    kl_loss: Tensor


class EvidentialRegimeHead(nn.Module):
    """Map a CDE condition vector to Dirichlet regime evidence.

    Proposal formulation:

        evidence = Softplus(W h + b)
        alpha = evidence + 1
        p_k = alpha_k / sum(alpha)
        u = K / sum(alpha)

    We call ``u`` evidential vacuity rather than assuming it is
    necessarily a faithful estimate of epistemic uncertainty.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        n_classes: int = 3,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError(
                "input_dim must be positive"
            )

        if n_classes < 2:
            raise ValueError(
                "n_classes must be at least 2"
            )

        self.input_dim = input_dim
        self.n_classes = n_classes

        self.evidence_layer = nn.Linear(
            input_dim,
            n_classes,
        )

    def forward(
        self,
        features: Tensor,
    ) -> EvidentialOutput:
        if features.ndim != 2:
            raise ValueError(
                "features must have shape "
                "[batch, input_dim]"
            )

        if features.shape[1] != self.input_dim:
            raise ValueError(
                "feature dimension does not "
                "match input_dim"
            )

        evidence = F.softplus(
            self.evidence_layer(
                features
            )
        )

        alpha = evidence + 1.0

        strength = alpha.sum(
            dim=-1,
            keepdim=True,
        )

        probabilities = (
            alpha
            / strength
        )

        vacuity = (
            float(self.n_classes)
            / strength.squeeze(-1)
        )

        return EvidentialOutput(
            evidence=evidence,
            alpha=alpha,
            probabilities=probabilities,
            vacuity=vacuity,
        )


def dirichlet_kl_to_uniform(
    alpha: Tensor,
) -> Tensor:
    """KL[Dir(alpha) || Dir(1)] for each sample."""

    if alpha.ndim != 2:
        raise ValueError(
            "alpha must have shape "
            "[batch, classes]"
        )

    if torch.any(
        alpha <= 0
    ):
        raise ValueError(
            "Dirichlet parameters must be positive"
        )

    n_classes = alpha.shape[-1]

    strength = alpha.sum(
        dim=-1,
        keepdim=True,
    )

    log_normalizer = (
        torch.lgamma(
            strength
        )
        - torch.lgamma(
            alpha
        ).sum(
            dim=-1,
            keepdim=True,
        )
        - torch.lgamma(
            torch.tensor(
                float(n_classes),
                device=alpha.device,
                dtype=alpha.dtype,
            )
        )
    )

    expectation = (
        (
            alpha
            - 1.0
        )
        * (
            torch.digamma(
                alpha
            )
            - torch.digamma(
                strength
            )
        )
    ).sum(
        dim=-1,
        keepdim=True,
    )

    return (
        log_normalizer
        + expectation
    ).squeeze(-1)


def evidential_classification_loss(
    alpha: Tensor,
    targets: Tensor,
    *,
    kl_weight: float,
) -> EvidentialLossOutput:
    """Expected CE plus annealed KL regularization."""

    if alpha.ndim != 2:
        raise ValueError(
            "alpha must have shape "
            "[batch, classes]"
        )

    if targets.ndim != 1:
        raise ValueError(
            "targets must have shape [batch]"
        )

    if targets.shape[0] != alpha.shape[0]:
        raise ValueError(
            "target batch dimension mismatch"
        )

    if kl_weight < 0:
        raise ValueError(
            "kl_weight cannot be negative"
        )

    n_classes = alpha.shape[-1]

    if (
        torch.any(targets < 0)
        or torch.any(
            targets >= n_classes
        )
    ):
        raise ValueError(
            "targets are outside class range"
        )

    one_hot = F.one_hot(
        targets,
        num_classes=n_classes,
    ).to(
        dtype=alpha.dtype
    )

    strength = alpha.sum(
        dim=-1,
        keepdim=True,
    )

    data_loss = (
        one_hot
        * (
            torch.digamma(
                strength
            )
            - torch.digamma(
                alpha
            )
        )
    ).sum(
        dim=-1
    ).mean()

    # Remove evidence for the correct class before applying the
    # KL-to-uniform penalty, following the standard EDL objective.
    adjusted_alpha = (
        one_hot
        + (
            1.0
            - one_hot
        )
        * alpha
    )

    kl_loss = (
        dirichlet_kl_to_uniform(
            adjusted_alpha
        )
        .mean()
    )

    loss = (
        data_loss
        + float(kl_weight)
        * kl_loss
    )

    return EvidentialLossOutput(
        loss=loss,
        data_loss=data_loss,
        kl_loss=kl_loss,
    )