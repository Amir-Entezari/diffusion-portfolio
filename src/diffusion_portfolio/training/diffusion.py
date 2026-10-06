"""Training utilities for the vanilla conditional diffusion model."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from diffusion_portfolio.models.diffusion import (
    ConditionalDiffusionModel,
)


@dataclass(frozen=True)
class EpochRecord:
    """Losses recorded after one training epoch."""

    epoch: int
    train_loss: float
    val_loss: float


@dataclass(frozen=True)
class TrainingResult:
    """Summary of one diffusion training run."""

    history: tuple[EpochRecord, ...]
    best_epoch: int
    best_val_loss: float
    checkpoint_path: str | None


def _move_batch(
    batch: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move only the model-space tensors needed for DDPM training."""

    history = batch["history"].to(
        device=device,
        non_blocking=True,
    )

    target = batch["target"].to(
        device=device,
        non_blocking=True,
    )

    return history, target


def train_one_epoch(
    model: ConditionalDiffusionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    gradient_clip_norm: float,
) -> float:
    """Train for one epoch and return sample-weighted mean loss."""

    if gradient_clip_norm <= 0:
        raise ValueError(
            "gradient_clip_norm must be positive"
        )

    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        history, target = _move_batch(
            batch,
            device,
        )

        batch_size = history.shape[0]

        optimizer.zero_grad(
            set_to_none=True
        )

        output = model.training_loss(
            history,
            target,
        )

        loss = output.loss

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite training loss encountered"
            )

        loss.backward()

        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=gradient_clip_norm,
        )

        optimizer.step()

        total_loss += (
            float(loss.detach().cpu())
            * batch_size
        )

        total_samples += batch_size

    if total_samples == 0:
        raise ValueError(
            "Training loader produced no samples"
        )

    return (
        total_loss
        / total_samples
    )


@torch.no_grad()
def evaluate_diffusion_loss(
    model: ConditionalDiffusionModel,
    loader: DataLoader,
    *,
    device: torch.device,
    validation_seed: int,
) -> float:
    """Evaluate DDPM loss using fixed corruption randomness.

    A fresh CPU generator is initialized with the same seed on every call.
    Therefore the validation examples receive the same noise and diffusion
    timesteps at every epoch.

    This makes validation losses directly comparable across epochs.
    """

    if validation_seed < 0:
        raise ValueError(
            "validation_seed cannot be negative"
        )

    model.eval()

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        validation_seed
    )

    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        history, target = _move_batch(
            batch,
            device,
        )

        batch_size = history.shape[0]

        timesteps = torch.randint(
            low=0,
            high=model.diffusion_steps,
            size=(batch_size,),
            generator=generator,
            device="cpu",
            dtype=torch.long,
        ).to(device)

        noise = torch.randn(
            (
                batch_size,
                model.n_assets,
            ),
            generator=generator,
            device="cpu",
            dtype=target.dtype,
        ).to(device)

        output = model.training_loss(
            history,
            target,
            timesteps=timesteps,
            noise=noise,
        )

        loss = output.loss

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite validation loss encountered"
            )

        total_loss += (
            float(loss.cpu())
            * batch_size
        )

        total_samples += batch_size

    if total_samples == 0:
        raise ValueError(
            "Validation loader produced no samples"
        )

    return (
        total_loss
        / total_samples
    )


def _save_checkpoint(
    path: str | Path,
    *,
    model: ConditionalDiffusionModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
) -> None:
    """Save the best model state for later reproduction."""

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": (
                model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        path,
    )


def fit_diffusion(
    model: ConditionalDiffusionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    validation_seed: int,
    device: str | torch.device,
    checkpoint_path: str | Path | None = None,
    resume_path: str | Path | None = None,
    stop_after_epoch: int | None = None,
    verbose: bool = True,
) -> TrainingResult:
    """Train a conditional DDPM and restore its best validation state."""

    if epochs <= 0:
        raise ValueError(
            "epochs must be positive"
        )

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate must be positive"
        )

    if weight_decay < 0:
        raise ValueError(
            "weight_decay cannot be negative"
        )

    device = torch.device(
        device
    )

    model.to(
        device
    )

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    records: list[EpochRecord] = []

    best_epoch = -1
    best_val_loss = float("inf")
    best_state = None

    start_epoch = 1

    if resume_path is not None:
        resume_path = Path(
            resume_path
        )

        if resume_path.exists():
            saved = torch.load(
                resume_path,
                map_location="cpu",
                weights_only=False,
            )

            model.load_state_dict(
                saved[
                    "model_state_dict"
                ]
            )

            optimizer.load_state_dict(
                saved[
                    "optimizer_state_dict"
                ]
            )

            records = [
                EpochRecord(
                    epoch=int(
                        item["epoch"]
                    ),
                    train_loss=float(
                        item["train_loss"]
                    ),
                    val_loss=float(
                        item["val_loss"]
                    ),
                )
                for item
                in saved["history"]
            ]

            best_epoch = int(
                saved["best_epoch"]
            )

            best_val_loss = float(
                saved["best_val_loss"]
            )

            torch.set_rng_state(
                saved[
                    "torch_rng_state"
                ]
            )

            if (
                saved[
                    "cuda_rng_state"
                ]
                is not None
                and torch.cuda.is_available()
            ):
                torch.cuda.set_rng_state_all(
                    saved[
                        "cuda_rng_state"
                    ]
                )

            loader_generator = getattr(
                train_loader,
                "generator",
                None,
            )

            if (
                loader_generator
                is not None
                and saved[
                    "loader_rng_state"
                ]
                is not None
            ):
                loader_generator.set_state(
                    saved[
                        "loader_rng_state"
                    ]
                )

            start_epoch = (
                int(
                    saved["epoch"]
                )
                + 1
            )

            if verbose:
                print(
                    "Resuming from epoch "
                    f"{start_epoch - 1}"
                )

    end_epoch = epochs

    if stop_after_epoch is not None:
        end_epoch = min(
            epochs,
            stop_after_epoch,
        )

    for epoch in range(
        start_epoch,
        end_epoch + 1,
    ):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            gradient_clip_norm=(
                gradient_clip_norm
            ),
        )

        val_loss = evaluate_diffusion_loss(
            model,
            val_loader,
            device=device,
            validation_seed=(
                validation_seed
            ),
        )

        record = EpochRecord(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
        )

        records.append(
            record
        )

        if verbose:
            print(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"train={train_loss:.6f} | "
                f"val={val_loss:.6f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = (
                val_loss
            )

            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            if checkpoint_path is not None:
                _save_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                )
        if resume_path is not None:
            loader_generator = getattr(
                train_loader,
                "generator",
                None,
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                    "history": [
                        {
                            "epoch": record.epoch,
                            "train_loss": (
                                record.train_loss
                            ),
                            "val_loss": (
                                record.val_loss
                            ),
                        }
                        for record
                        in records
                    ],
                    "best_epoch": best_epoch,
                    "best_val_loss": (
                        best_val_loss
                    ),
                    "torch_rng_state": (
                        torch.get_rng_state()
                    ),
                    "cuda_rng_state": (
                        torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available()
                        else None
                    ),
                    "loader_rng_state": (
                        loader_generator.get_state()
                        if loader_generator
                        is not None
                        else None
                    ),
                },
                resume_path,
            )
    if best_state is None:
        if (
            checkpoint_path is None
            or not Path(
                checkpoint_path
            ).exists()
        ):
            raise RuntimeError(
                "Training finished without "
                "a valid model state"
            )

        best_checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        best_state = (
            best_checkpoint[
                "model_state_dict"
            ]
        )

    # All downstream evaluation should use the best validation model,
    # not blindly the final epoch.
    model.load_state_dict(
        best_state
    )

    return TrainingResult(
        history=tuple(records),
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        checkpoint_path=(
            None
            if checkpoint_path is None
            else str(
                checkpoint_path
            )
        ),
    )