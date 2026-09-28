"""
Dirichlet Evidential Head for Epistemic Uncertainty Quantification.

Reference: Proposal §5.2.2 — Subjective Logic Epistemic Quantification.

Maps the Neural CDE hidden state H(t) to a Dirichlet distribution over
K market regimes, providing principled epistemic uncertainty:

    e(t) = Softplus(W_ev · H(t) + b_ev)     — evidence vector
    α_k(t) = e_k(t) + 1                      — Dirichlet concentration
    S(t) = Σ_k α_k(t)                        — Dirichlet strength
    p_k(t) = α_k(t) / S(t)                   — regime probabilities
    u(t) = K / S(t)                           — epistemic uncertainty

Key Properties:
    - u(t) ∈ (0, 1]:  u → 0 (high certainty), u → 1 (total ignorance)
    - p(t) ∈ Δ^{K-1}: lives on the probability simplex
    - As evidence e → 0: α → 1, u → 1 (uniform prior = "I don't know")
    - As evidence e → ∞: u → 0 (concentrated posterior = high confidence)

Evidential Loss (§6.2.1):
    L_ev = Σ_k (ψ(S) − ψ(α_k)) · (y_k − p_k)²
         + λ_KL · KL[Dir(α̃) || Dir(1)]

    where ψ is the digamma function and α̃ removes evidence from the
    correct class to penalise conflicting evidence on incorrect classes.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class DirichletHead(nn.Module):
    r"""Dirichlet evidential classification head.

    Transforms the CDE hidden state into a Dirichlet distribution
    over K market regimes, providing both regime probabilities and
    calibrated epistemic uncertainty.

    .. math::
        \mathbf{e}(t) = \text{Softplus}(W_{ev} H(t) + b_{ev})
        \alpha_k(t) = e_k(t) + 1
        u(t) = K / \sum_k \alpha_k(t)

    Args:
        hidden_dim: h — dimension of the CDE hidden state H(t).
        n_regimes: K — number of market regime classes.

    Example:
        >>> head = DirichletHead(hidden_dim=256, n_regimes=4)
        >>> H = torch.randn(32, 256)  # [B, h]
        >>> output = head(H)
        >>> output["uncertainty"].shape  # [32]
        >>> output["probabilities"].shape  # [32, 4]
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        n_regimes: int = 4,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_regimes = n_regimes

        # Evidence projection: H(t) → e(t)
        self.evidence_proj = nn.Linear(hidden_dim, n_regimes)

        # Initialise with small weights so initial evidence ≈ 0
        # → α ≈ 1 → u ≈ 1 (maximum uncertainty at initialisation)
        nn.init.xavier_uniform_(self.evidence_proj.weight, gain=0.01)
        nn.init.zeros_(self.evidence_proj.bias)

        logger.info(
            f"DirichletHead: hidden_dim={hidden_dim}, n_regimes={n_regimes}"
        )

    def forward(self, h: Tensor) -> Dict[str, Tensor]:
        """Compute Dirichlet parameters and epistemic uncertainty.

        Args:
            h: CDE hidden state, shape [..., h].

        Returns:
            Dictionary with:
                - ``evidence``: e(t), shape [..., K]. Non-negative.
                - ``alpha``: α(t) = e(t) + 1, shape [..., K]. All ≥ 1.
                - ``strength``: S(t) = Σ α_k, shape [...]. S ≥ K.
                - ``probabilities``: p(t) = α/S, shape [..., K]. On simplex.
                - ``uncertainty``: u(t) = K/S, shape [...]. In (0, 1].
        """
        # Evidence: Softplus ensures non-negativity
        evidence = F.softplus(self.evidence_proj(h))  # [..., K]

        # Dirichlet concentration parameters
        alpha = evidence + 1.0  # [..., K], all ≥ 1

        # Dirichlet strength
        strength = alpha.sum(dim=-1)  # [...]

        # Regime probabilities (expected value of the Dirichlet)
        probabilities = alpha / strength.unsqueeze(-1)  # [..., K]

        # Epistemic uncertainty
        uncertainty = self.n_regimes / strength  # [...]

        return {
            "evidence": evidence,
            "alpha": alpha,
            "strength": strength,
            "probabilities": probabilities,
            "uncertainty": uncertainty,
        }

    def compute_evidential_loss(
        self,
        alpha: Tensor,
        target_onehot: Tensor,
        lambda_kl: float = 0.01,
        epoch: int = 0,
        n_epochs: int = 200,
    ) -> Tensor:
        r"""Compute the evidential Dirichlet loss L_ev (§6.2.1).

        .. math::
            \mathcal{L}_{ev} = \sum_k (\psi(S) - \psi(\alpha_k))(y_k - p_k)^2
                             + \lambda_{KL} \cdot KL[Dir(\tilde{\alpha}) \| Dir(\mathbf{1})]

        The KL term uses an annealing coefficient that increases from 0 to
        λ_KL over training, preventing premature evidence shrinkage.

        Args:
            alpha: Dirichlet concentrations, shape [B, K].
            target_onehot: One-hot regime labels, shape [B, K].
            lambda_kl: KL divergence regularisation weight.
            epoch: Current training epoch (for annealing).
            n_epochs: Total epochs (for annealing schedule).

        Returns:
            Scalar loss value.
        """
        S = alpha.sum(dim=-1, keepdim=True)  # [B, 1]
        p = alpha / S  # [B, K]

        # --- Type II Maximum Likelihood loss ---
        # Uses digamma for the expected log-likelihood under the Dirichlet
        # ψ(S) − ψ(α_k) = E[log(1/π_k)] under Dir(α)
        digamma_S = torch.digamma(S)
        digamma_alpha = torch.digamma(alpha)

        # Squared error weighted by expected uncertainty
        squared_err = (target_onehot - p) ** 2

        # Variance of the Dirichlet: α_k(S−α_k) / (S²(S+1))
        variance = alpha * (S - alpha) / (S ** 2 * (S + 1.0))

        loss_mse = (squared_err + variance).sum(dim=-1).mean()

        # --- KL divergence regularisation ---
        # Remove evidence from the correct class to avoid penalising
        # correct predictions
        alpha_tilde = target_onehot + (1.0 - target_onehot) * (alpha - 1.0) + 1.0

        # KL[Dir(α̃) || Dir(1)]
        kl = self._kl_dirichlet_uniform(alpha_tilde)

        # Annealing: linearly increase KL weight from 0 to λ_KL
        annealing_coeff = min(1.0, epoch / max(1, n_epochs // 2))
        kl_loss = annealing_coeff * lambda_kl * kl.mean()

        return loss_mse + kl_loss

    @staticmethod
    def _kl_dirichlet_uniform(alpha: Tensor) -> Tensor:
        r"""KL divergence from Dir(α) to Dir(1) (uniform Dirichlet).

        .. math::
            KL[Dir(\alpha) \| Dir(\mathbf{1})] =
                \ln \frac{\Gamma(\sum \alpha_k)}{\Gamma(K) \prod \Gamma(\alpha_k)}
                + \sum_k (\alpha_k - 1)(\psi(\alpha_k) - \psi(\sum \alpha_j))

        Args:
            alpha: Dirichlet concentrations, shape [..., K].

        Returns:
            KL divergence, shape [...].
        """
        K = alpha.shape[-1]
        S = alpha.sum(dim=-1, keepdim=True)

        # Log-normalising constants
        log_B_alpha = (
            torch.lgamma(alpha).sum(dim=-1) - torch.lgamma(S.squeeze(-1))
        )
        log_B_uniform = K * torch.lgamma(torch.ones(1, device=alpha.device))

        # Digamma terms
        digamma_sum = torch.digamma(S)
        kl = (
            -log_B_alpha
            + log_B_uniform
            + ((alpha - 1.0) * (torch.digamma(alpha) - digamma_sum)).sum(dim=-1)
        )

        return kl
