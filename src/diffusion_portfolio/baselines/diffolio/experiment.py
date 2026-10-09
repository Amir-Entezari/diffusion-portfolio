"""Train the Diffolio reproduction on the real KF12 dataset.

Examples
--------
Smoke run:

    python scripts/train_diffolio.py \
        --config configs/diffolio.yaml \
        --steps 100

Full configured run:

    python scripts/train_diffolio.py \
        --config configs/diffolio.yaml
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

import json
import shutil
import subprocess

from diffusion_portfolio.data import CovariateStandardizer, TrainStandardizer
from diffusion_portfolio.baselines.diffolio.loaders import make_diffolio_dataloaders, make_diffolio_datasets
from diffusion_portfolio.baselines.diffolio import compute_training_covariance
from diffusion_portfolio.baselines.diffolio.training import fit_diffolio_steps


from diffusion_portfolio.utils.device import resolve_device
from diffusion_portfolio.baselines.diffolio.build import build_model
from diffusion_portfolio.baselines.diffolio.preparation import prepare_diffolio_data

def load_raw_config(
    path: Path,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Config does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = (
            yaml.safe_load(
                handle
            )
            or {}
        )

    for section in (
        "data",
        "model",
        "training",
        "inference",
        "evaluation",
    ):
        if section not in config:
            raise ValueError(
                f"Missing config section: "
                f"{section}"
            )

    return config


def seed_everything(
    seed: int,
) -> None:
    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def resolve_training_settings(
    config: dict,
    args: argparse.Namespace,
) -> tuple[
    int,
    int,
    int,
    Path,
]:
    training = config[
        "training"
    ]

    configured_steps = int(
        training[
            "total_steps"
        ]
    )

    total_steps = (
        configured_steps
        if args.steps is None
        else int(
            args.steps
        )
    )

    if total_steps <= 0:
        raise ValueError(
            "--steps must be positive"
        )

    configured_warmup = int(
        training[
            "warmup_steps"
        ]
    )

    if args.warmup_steps is not None:
        warmup_steps = int(
            args.warmup_steps
        )

    elif (
        args.steps is not None
        and configured_warmup
        > total_steps
    ):
        # A short smoke run cannot use the
        # full 1000-step paper warmup.
        #
        # Use 10% of the smoke run unless
        # explicitly overridden.
        warmup_steps = max(
            1,
            total_steps // 10,
        )

    else:
        warmup_steps = (
            configured_warmup
        )

    if warmup_steps > total_steps:
        raise ValueError(
            "Effective warmup exceeds "
            "effective total steps"
        )

    configured_record_every = int(
        training[
            "record_every"
        ]
    )

    if args.steps is not None:
        record_every = min(
            configured_record_every,
            max(
                1,
                total_steps // 10,
            ),
        )

    else:
        record_every = (
            configured_record_every
        )

    if args.checkpoint is not None:
        checkpoint = args.checkpoint

    else:
        configured_checkpoint = Path(
            training[
                "checkpoint_path"
            ]
        )

        if (
            args.steps is not None
            and total_steps
            != configured_steps
        ):
            # Never overwrite the future
            # full 100k checkpoint with a
            # diagnostic run.
            checkpoint = (
                configured_checkpoint
                .parent
                / (
                    "diffolio_smoke_"
                    f"{total_steps}.pt"
                )
            )

        else:
            checkpoint = (
                configured_checkpoint
            )

    return (
        total_steps,
        warmup_steps,
        record_every,
        checkpoint,
    )


def prepare_data(config):
    data = config["data"]
    if not Path(data["goyal_path"]).exists():
        raise FileNotFoundError(f"Goyal workbook not found: {data['goyal_path']}")
    splits, raw_returns, return_scaler, covariate_scaler = prepare_diffolio_data(data)
    datasets = make_diffolio_datasets(splits)
    loaders = make_diffolio_dataloaders(
        datasets, batch_size=int(config["training"]["batch_size"]),
        seed=int(config["seed"]), num_workers=int(config["training"]["num_workers"]),
    )
    train_mask = raw_returns.dates <= pd.Timestamp(data["train_end"])
    training_covariance = compute_training_covariance(
        torch.from_numpy(raw_returns.returns[train_mask]).float()
    )
    return loaders, training_covariance, return_scaler, covariate_scaler


def save_run_metadata(
    *,
    config: dict,
    total_steps: int,
    warmup_steps: int,
    record_every: int,
    checkpoint: Path,
    return_scaler: TrainStandardizer,
    covariate_scaler: CovariateStandardizer,
) -> tuple[Path, Path]:
    checkpoint.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Exact effective run configuration.
    # --------------------------------------------------------

    effective_config = copy.deepcopy(
        config
    )

    effective_config[
        "training"
    ][
        "total_steps"
    ] = total_steps

    effective_config[
        "training"
    ][
        "warmup_steps"
    ] = warmup_steps

    effective_config[
        "training"
    ][
        "record_every"
    ] = record_every

    effective_config[
        "training"
    ][
        "checkpoint_path"
    ] = str(
        checkpoint
    )

    config_path = (
        checkpoint.parent
        / (
            checkpoint.stem
            + "_config.yaml"
        )
    )

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        yaml.safe_dump(
            effective_config,
            handle,
            sort_keys=False,
        )

    # --------------------------------------------------------
    # Frozen preprocessing statistics needed to convert
    # generated samples back into raw decimal returns.
    # --------------------------------------------------------

    scaler_path = (
        checkpoint.parent
        / (
            checkpoint.stem
            + "_preprocessing.npz"
        )
    )

    np.savez(
        scaler_path,
        return_mean=(
            return_scaler.mean
        ),
        return_std=(
            return_scaler.std
        ),
        return_columns=np.asarray(
            return_scaler.columns,
            dtype=str,
        ),
        asset_covariate_mean=(
            covariate_scaler.asset_mean
        ),
        asset_covariate_std=(
            covariate_scaler.asset_std
        ),
        asset_covariate_columns=np.asarray(
            covariate_scaler.asset_columns,
            dtype=str,
        ),
        systematic_mean=(
            covariate_scaler.systematic_mean
        ),
        systematic_std=(
            covariate_scaler.systematic_std
        ),
        systematic_columns=np.asarray(
            covariate_scaler.systematic_columns,
            dtype=str,
        ),
    )

    print(
        "Run config:",
        config_path,
    )

    print(
        "Preprocessing stats:",
        scaler_path,
    )
    return (
        config_path,
        scaler_path,
    )


def make_kaggle_backup_callback(
    *,
    dataset_handle: str,
    every_steps: int,
    total_steps: int,
    run_config_path: Path,
    preprocessing_path: Path,
):
    """Create a callback that versions checkpoints on Kaggle.

    The latest training checkpoint is uploaded as a new
    Kaggle Dataset version every ``every_steps``.

    Previous Dataset versions remain available even if the
    current Kaggle notebook session disappears.
    """

    if every_steps <= 0:
        raise ValueError(
            "Kaggle backup interval must be positive"
        )

    # Import lazily so normal/local training does not require
    # KaggleHub to be installed.
    import kagglehub

    def backup(
        step: int,
        checkpoint_path: Path,
    ) -> None:
        # Do nothing on ordinary 5k local checkpoints.
        #
        # Always backup final step as well.
        if (
            step % every_steps != 0
            and step != total_steps
        ):
            return

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "Checkpoint does not exist: "
                f"{checkpoint_path}"
            )

        staging_dir = (
            checkpoint_path.parent
            / "_kaggle_backup"
        )

        if staging_dir.exists():
            shutil.rmtree(
                staging_dir
            )

        staging_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        uploaded_checkpoint = (
            staging_dir
            / "diffolio_100k.pt"
        )

        shutil.copy2(
            checkpoint_path,
            uploaded_checkpoint,
        )

        # ----------------------------------------------------
        # Exact effective config + preprocessing statistics.
        # ----------------------------------------------------

        shutil.copy2(
            run_config_path,
            staging_dir
            / "diffolio_config.yaml",
        )

        shutil.copy2(
            preprocessing_path,
            staging_dir
            / "diffolio_preprocessing.npz",
        )

        # ----------------------------------------------------
        # Record exactly which repository revision produced
        # this checkpoint.
        # ----------------------------------------------------

        git_result = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        git_commit = (
            git_result.stdout.strip()
            if git_result.returncode == 0
            else "unknown"
        )

        manifest = {
            "step": step,
            "total_steps": total_steps,
            "checkpoint_file": (
                uploaded_checkpoint.name
            ),
            "git_commit": git_commit,
            "dataset_handle": (
                dataset_handle
            ),
        }

        with (
            staging_dir
            / "manifest.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                manifest,
                handle,
                indent=2,
            )

        print()
        print(
            "=" * 72
        )

        print(
            "KAGGLE CHECKPOINT BACKUP"
        )

        print(
            "=" * 72
        )

        print(
            f"Step: {step:,}/"
            f"{total_steps:,}"
        )

        print(
            "Dataset:",
            dataset_handle,
        )

        print(
            "Git commit:",
            git_commit,
        )

        # Upload can occasionally fail due to a transient
        # network/API issue. Retry before aborting the run.
        delays = (
            15,
            30,
            60,
        )

        last_error = None

        for attempt in range(
            1,
            4,
        ):
            try:
                kagglehub.dataset_upload(
                    dataset_handle,
                    str(
                        staging_dir
                    ),
                    version_notes=(
                        "Diffolio training "
                        f"checkpoint: step "
                        f"{step:,}/"
                        f"{total_steps:,}; "
                        f"git={git_commit}"
                    ),
                )

                print(
                    "Kaggle backup complete."
                )

                print(
                    "=" * 72
                )

                print()

                return

            except Exception as exc:
                last_error = exc

                if attempt == 3:
                    break

                delay = delays[
                    attempt - 1
                ]

                print(
                    "Backup upload failed "
                    f"(attempt {attempt}/3): "
                    f"{exc}"
                )

                print(
                    f"Retrying in "
                    f"{delay} seconds..."
                )

                time.sleep(
                    delay
                )

        raise RuntimeError(
            "Kaggle checkpoint backup failed "
            "after 3 attempts"
        ) from last_error

    return backup

def run_training(args) -> None:

    config = load_raw_config(
        args.config
    )

    seed = int(
        config.get(
            "seed",
            42,
        )
    )

    seed_everything(
        seed
    )

    device = resolve_device(
        args.device
    )

    (
        total_steps,
        warmup_steps,
        record_every,
        checkpoint,
    ) = resolve_training_settings(
        config,
        args,
    )

    print(
        "=" * 72
    )
    print(
        "DIFFOLIO TRAINING"
    )
    print(
        "=" * 72
    )

    print(
        "Config:",
        args.config,
    )

    print(
        "Device:",
        device,
    )

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(
                device
            ),
        )

    print(
        "Total steps:",
        total_steps,
    )

    print(
        "Warmup steps:",
        warmup_steps,
    )

    print(
        "Batch size:",
        config[
            "training"
        ][
            "batch_size"
        ],
    )

    print(
        "Checkpoint:",
        checkpoint,
    )

    if (
        args.steps is not None
        and int(
            config[
                "training"
            ][
                "warmup_steps"
            ]
        )
        > total_steps
        and args.warmup_steps is None
    ):
        print(
            "NOTE: smoke-run warmup "
            f"automatically adjusted to "
            f"{warmup_steps} steps."
        )

    (
        loaders,
        training_covariance,
        return_scaler,
        covariate_scaler,
    ) = prepare_data(
        config
    )

    model = build_model(
        config,
        training_covariance,
    )

    n_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    n_trainable = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    print()
    print(
        "Model parameters:",
        f"{n_parameters:,}",
    )

    print(
        "Trainable parameters:",
        f"{n_trainable:,}",
    )

    training_cfg = config[
        "training"
    ]

    (
        run_config_path,
        preprocessing_path,
    ) = save_run_metadata(
        config=config,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        record_every=record_every,
        checkpoint=checkpoint,
        return_scaler=return_scaler,
        covariate_scaler=(
            covariate_scaler
        ),
    )

    kaggle_backup_cfg = (
        training_cfg.get(
            "kaggle_backup",
            {},
        )
    )

    checkpoint_callback = None

    configured_total_steps = int(
        training_cfg[
            "total_steps"
        ]
    )

    is_full_run = (
        total_steps
        == configured_total_steps
    )

    if (
        kaggle_backup_cfg.get(
            "enabled",
            False,
        )
        and is_full_run
    ):
        checkpoint_callback = (
            make_kaggle_backup_callback(
                dataset_handle=str(
                    kaggle_backup_cfg[
                        "dataset_handle"
                    ]
                ),
                every_steps=int(
                    kaggle_backup_cfg[
                        "every_steps"
                    ]
                ),
                total_steps=total_steps,
                run_config_path=(
                    run_config_path
                ),
                preprocessing_path=(
                    preprocessing_path
                ),
            )
        )

        print(
            "Kaggle backup:",
            kaggle_backup_cfg[
                "dataset_handle"
            ],
        )

        print(
            "Kaggle backup every:",
            f"{int(kaggle_backup_cfg['every_steps']):,}",
            "steps",
        )

    elif kaggle_backup_cfg.get(
        "enabled",
        False,
    ):
        print(
            "Kaggle backup disabled for "
            "this shortened smoke run."
        )


    gradient_clip = (
        training_cfg.get(
            "gradient_clip_norm"
        )
    )

    if gradient_clip is not None:
        gradient_clip = float(
            gradient_clip
        )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(
            device
        )


    print()
    print(
        "=" * 72
    )
    print(
        "STARTING OPTIMIZATION"
    )
    print(
        "=" * 72
    )

    start_time = time.perf_counter()

    checkpoint_every = (
        training_cfg.get(
            "checkpoint_every"
        )
    )

    if checkpoint_every is not None:
        checkpoint_every = int(
            checkpoint_every
        )

    if args.resume is not None:
        print(
            "Resume checkpoint:",
            args.resume,
        )

    result = fit_diffolio_steps(
        model,
        loaders.train,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        max_learning_rate=float(
            training_cfg[
                "max_learning_rate"
            ]
        ),
        weight_decay=float(
            training_cfg[
                "weight_decay"
            ]
        ),
        gradient_clip_norm=(
            gradient_clip
        ),
        device=device,
        checkpoint_path=checkpoint,
        checkpoint_every=(
            checkpoint_every
        ),
        checkpoint_callback=(
            checkpoint_callback
        ),
        resume_checkpoint=(
            args.resume
        ),
        record_every=record_every,
        verbose=True,
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    steps_per_second = (
        total_steps
        / elapsed
    )

    seconds_per_step = (
        elapsed
        / total_steps
    )

    print()
    print(
        "=" * 72
    )
    print(
        "TRAINING COMPLETE"
    )
    print(
        "=" * 72
    )

    print(
        "Final step:",
        result.final_step,
    )

    print(
        "Elapsed seconds:",
        f"{elapsed:.2f}",
    )

    print(
        "Steps / second:",
        f"{steps_per_second:.3f}",
    )

    print(
        "Seconds / step:",
        f"{seconds_per_step:.4f}",
    )

    if device.type == "cuda":
        peak_bytes = (
            torch.cuda.max_memory_allocated(
                device
            )
        )

        peak_gib = (
            peak_bytes
            / 1024**3
        )

        print(
            "Peak CUDA allocated:",
            f"{peak_gib:.3f} GiB",
        )

    print(
        "Checkpoint:",
        result.checkpoint_path,
    )
