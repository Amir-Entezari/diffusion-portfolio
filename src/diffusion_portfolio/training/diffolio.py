"""Step-based training utilities for the Diffolio reproduction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from diffusion_portfolio.models.diffolio import (
    DiffolioObjective,
)


@dataclass(frozen=True)
class DiffolioStepRecord:
    """Metrics recorded at one optimizer step."""

    step: int
    loss: float
    ddpm_loss: float
    correlation_loss: float
    learning_rate: float


@dataclass(frozen=True)
class DiffolioFitResult:
    """Summary of a fixed-step Diffolio training run."""

    history: tuple[DiffolioStepRecord, ...]
    final_step: int
    checkpoint_path: str | None


def diffolio_learning_rate(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    max_learning_rate: float,
) -> float:
    """Linear warmup followed by cosine decay to zero.

    ``step`` is one-based:

        1, 2, ..., total_steps
    """

    if total_steps <= 0:
        raise ValueError(
            "total_steps must be positive"
        )

    if not 1 <= step <= total_steps:
        raise ValueError(
            "step must lie in [1, total_steps]"
        )

    if warmup_steps < 0:
        raise ValueError(
            "warmup_steps cannot be negative"
        )

    if warmup_steps > total_steps:
        raise ValueError(
            "warmup_steps cannot exceed total_steps"
        )

    if max_learning_rate <= 0.0:
        raise ValueError(
            "max_learning_rate must be positive"
        )

    if (
        warmup_steps > 0
        and step <= warmup_steps
    ):
        return (
            max_learning_rate
            * step
            / warmup_steps
        )

    decay_steps = (
        total_steps
        - warmup_steps
    )

    if decay_steps == 0:
        return max_learning_rate

    # With no warmup, begin exactly at max LR.
    if warmup_steps == 0:
        if total_steps == 1:
            return max_learning_rate

        progress = (
            step - 1
        ) / (
            total_steps - 1
        )

    else:
        progress = (
            step - warmup_steps
        ) / decay_steps

    multiplier = 0.5 * (
        1.0
        + math.cos(
            math.pi * progress
        )
    )

    return (
        max_learning_rate
        * multiplier
    )


def _move_diffolio_batch(
    batch: dict,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Move the tensors required by the Diffolio objective."""

    target = batch[
        "target"
    ].to(
        device=device,
        non_blocking=True,
    )

    return_history = batch[
        "return_history"
    ].to(
        device=device,
        non_blocking=True,
    )

    return_history_raw = batch[
        "return_history_raw"
    ].to(
        device=device,
        non_blocking=True,
    )

    asset_covariates = batch[
        "asset_covariates"
    ].to(
        device=device,
        non_blocking=True,
    )

    systematic_covariates = batch[
        "systematic_covariates"
    ].to(
        device=device,
        non_blocking=True,
    )

    return (
        target,
        return_history,
        return_history_raw,
        asset_covariates,
        systematic_covariates,
    )


def _set_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> None:
    for group in optimizer.param_groups:
        group[
            "lr"
        ] = learning_rate


def _save_diffolio_checkpoint(
    path: str | Path,
    *,
    model: DiffolioObjective,
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
    warmup_steps: int,
    max_learning_rate: float,
    weight_decay: float,
) -> None:
    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "step": step,
            "model_state_dict": (
                model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "training": {
                "total_steps": total_steps,
                "warmup_steps": warmup_steps,
                "max_learning_rate": (
                    max_learning_rate
                ),
                "weight_decay": (
                    weight_decay
                ),
            },
        },
        path,
    )


def fit_diffolio_steps(
    model: DiffolioObjective,
    train_loader: DataLoader,
    *,
    total_steps: int,
    warmup_steps: int,
    max_learning_rate: float,
    weight_decay: float = 0.01,
    gradient_clip_norm: float | None = None,
    device: str | torch.device = "cpu",
    checkpoint_path: str | Path | None = None,
    record_every: int = 100,
    verbose: bool = True,
) -> DiffolioFitResult:
    """Train Diffolio for an exact number of optimizer steps.

    The dataloader is cycled as many times as necessary.

    This matches Diffolio's step-based training protocol
    rather than Stage 5's epoch-based MVP trainer.
    """

    if total_steps <= 0:
        raise ValueError(
            "total_steps must be positive"
        )

    if warmup_steps < 0:
        raise ValueError(
            "warmup_steps cannot be negative"
        )

    if warmup_steps > total_steps:
        raise ValueError(
            "warmup_steps cannot exceed total_steps"
        )

    if max_learning_rate <= 0.0:
        raise ValueError(
            "max_learning_rate must be positive"
        )

    if weight_decay < 0.0:
        raise ValueError(
            "weight_decay cannot be negative"
        )

    if (
        gradient_clip_norm is not None
        and gradient_clip_norm <= 0.0
    ):
        raise ValueError(
            "gradient_clip_norm must be positive "
            "when supplied"
        )

    if record_every <= 0:
        raise ValueError(
            "record_every must be positive"
        )

    if len(
        train_loader
    ) == 0:
        raise ValueError(
            "Training loader produced no batches"
        )

    device = torch.device(
        device
    )

    model.to(
        device
    )

    model.train()

    # The paper specifies AdamW but does not report
    # betas, epsilon, or weight decay.
    #
    # We use PyTorch's standard AdamW betas/epsilon
    # and make weight decay explicit/configurable.
    optimizer = AdamW(
        model.parameters(),
        lr=max_learning_rate,
        weight_decay=weight_decay,
    )

    iterator = iter(
        train_loader
    )

    records: list[
        DiffolioStepRecord
    ] = []

    for step in range(
        1,
        total_steps + 1,
    ):
        try:
            batch = next(
                iterator
            )

        except StopIteration:
            iterator = iter(
                train_loader
            )

            batch = next(
                iterator
            )

        learning_rate = (
            diffolio_learning_rate(
                step,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                max_learning_rate=(
                    max_learning_rate
                ),
            )
        )

        _set_learning_rate(
            optimizer,
            learning_rate,
        )

        (
            target,
            return_history,
            return_history_raw,
            asset_covariates,
            systematic_covariates,
        ) = _move_diffolio_batch(
            batch,
            device,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        output = model.training_loss(
            target,
            return_history,
            return_history_raw,
            asset_covariates,
            systematic_covariates,
        )

        if not torch.isfinite(
            output.loss
        ):
            raise RuntimeError(
                "Non-finite Diffolio loss encountered "
                f"at step {step}"
            )

        if not torch.isfinite(
            output.ddpm_loss
        ):
            raise RuntimeError(
                "Non-finite DDPM loss encountered "
                f"at step {step}"
            )

        if not torch.isfinite(
            output.correlation_loss
        ):
            raise RuntimeError(
                "Non-finite correlation loss encountered "
                f"at step {step}"
            )

        output.loss.backward()

        if gradient_clip_norm is not None:
            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=(
                    gradient_clip_norm
                ),
            )

        optimizer.step()

        should_record = (
            step == 1
            or step % record_every == 0
            or step == total_steps
        )

        if should_record:
            record = DiffolioStepRecord(
                step=step,
                loss=float(
                    output.loss
                    .detach()
                    .cpu()
                ),
                ddpm_loss=float(
                    output.ddpm_loss
                    .detach()
                    .cpu()
                ),
                correlation_loss=float(
                    output.correlation_loss
                    .detach()
                    .cpu()
                ),
                learning_rate=(
                    learning_rate
                ),
            )

            records.append(
                record
            )

            if verbose:
                print(
                    f"Step "
                    f"{step:06d}/"
                    f"{total_steps:06d} | "
                    f"loss={record.loss:.6f} | "
                    f"ddpm={record.ddpm_loss:.6f} | "
                    f"corr={record.correlation_loss:.6f} | "
                    f"lr={record.learning_rate:.3e}"
                )

    if checkpoint_path is not None:
        _save_diffolio_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            step=total_steps,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            max_learning_rate=(
                max_learning_rate
            ),
            weight_decay=weight_decay,
        )

    return DiffolioFitResult(
        history=tuple(
            records
        ),
        final_step=total_steps,
        checkpoint_path=(
            None
            if checkpoint_path is None
            else str(
                checkpoint_path
            )
        ),
    )