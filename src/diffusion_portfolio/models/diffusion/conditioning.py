"""Simple conditioning encoder for the vanilla diffusion baseline."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class HistoryEncoder(nn.Module):
    """Encode a fixed historical return window into one condition vector.

    The vanilla baseline intentionally uses a simple MLP over the flattened
    history. More sophisticated temporal encoders belong to later ablations.

    Input
    -----
    history:
        [batch, lookback, assets]

    Output
    ------
    condition:
        [batch, condition_dim]
    """

    def __init__(
        self,
        *,
        lookback: int,
        n_assets: int,
        condition_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        if lookback <= 0:
            raise ValueError(
                "lookback must be positive"
            )

        if n_assets <= 0:
            raise ValueError(
                "n_assets must be positive"
            )

        if condition_dim <= 0:
            raise ValueError(
                "condition_dim must be positive"
            )

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be positive"
            )

        self.lookback = lookback
        self.n_assets = n_assets
        self.condition_dim = condition_dim

        input_dim = (
            lookback
            * n_assets
        )

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.LayerNorm(
                hidden_dim
            ),
            nn.SiLU(),
            nn.Linear(
                hidden_dim,
                condition_dim,
            ),
        )

    def forward(
        self,
        history: Tensor,
    ) -> Tensor:
        if history.ndim != 3:
            raise ValueError(
                "history must have shape "
                "[batch, lookback, assets]"
            )

        if history.shape[1] != self.lookback:
            raise ValueError(
                "history lookback dimension "
                "does not match encoder"
            )

        if history.shape[2] != self.n_assets:
            raise ValueError(
                "history asset dimension "
                "does not match encoder"
            )

        flattened = history.reshape(
            history.shape[0],
            -1,
        )

        return self.network(
            flattened
        )