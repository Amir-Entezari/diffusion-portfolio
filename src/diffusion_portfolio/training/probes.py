"""Train regime probe evidential and softmax probes on frozen CDE features."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from diffusion_portfolio.config import (
    load_config,
)
from diffusion_portfolio.data import collate_return_batch
from diffusion_portfolio.evaluation.regimes import (
    assign_regime_labels,
    cross_sectional_rms,
    fit_regime_thresholds,
    trailing_cross_sectional_rms,
)
from diffusion_portfolio.models.uncertainty.evidential import (
    EvidentialRegimeHead,
    evidential_classification_loss,
)
from diffusion_portfolio.utils.seed import (
    set_global_seed,
)


from diffusion_portfolio.utils.device import resolve_device
from diffusion_portfolio.evaluation.classification import classification_metrics

from diffusion_portfolio.data.preparation import prepare_datasets

def load_probe_config(
    path: Path,
) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(
            handle
        )

    if not isinstance(
        config,
        dict,
    ):
        raise ValueError(
            "probe config must be a mapping"
        )

    return config


@torch.no_grad()
def extract_features(
    encoder: nn.Module,
    dataset,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    np.ndarray,
    np.ndarray,
    pd.DatetimeIndex,
]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_return_batch,
    )

    feature_parts = []
    history_parts = []
    target_parts = []
    dates = []

    encoder.eval()

    for batch in loader:
        history = batch[
            "history"
        ].to(
            device
        )

        features = encoder(
            history
        )

        feature_parts.append(
            features.detach().cpu()
        )

        history_raw = batch[
            "history_raw"
        ]

        if history_raw.ndim != 3:
            raise RuntimeError(
                "regime probe expects history_raw "
                "with shape [batch, lookback, assets]"
            )

        history_parts.append(
            history_raw.numpy()
        )

        target_raw = batch[
            "target_raw"
        ]

        if (
            target_raw.ndim != 3
            or target_raw.shape[1] != 1
        ):
            raise RuntimeError(
                "regime probe expects horizon=1"
            )

        target_parts.append(
            target_raw[
                :,
                0,
                :,
            ].numpy()
        )

        dates.extend(
            batch[
                "target_date"
            ].tolist()
        )

    return (
        torch.cat(
            feature_parts,
            dim=0,
        ),
        np.concatenate(
            history_parts,
            axis=0,
        ),
        np.concatenate(
            target_parts,
            axis=0,
        ),
        pd.DatetimeIndex(
            dates
        ),
    )


def class_counts(
    labels: torch.Tensor,
    n_classes: int,
) -> list[int]:
    return [
        int(
            (
                labels
                == class_index
            ).sum()
        )
        for class_index
        in range(
            n_classes
        )
    ]


def make_embedding_loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(
        seed
    )

    return DataLoader(
        TensorDataset(
            features,
            labels,
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )


@torch.no_grad()
def evidential_probabilities(
    head: EvidentialRegimeHead,
    features: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    output = head(
        features.to(
            device
        )
    )

    return (
        output.probabilities.cpu(),
        output.vacuity.cpu(),
    )


@torch.no_grad()
def softmax_probabilities(
    head: nn.Linear,
    features: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    logits = head(
        features.to(
            device
        )
    )

    return torch.softmax(
        logits,
        dim=-1,
    ).cpu()


def train_evidential(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    *,
    input_dim: int,
    n_classes: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    kl_anneal_epochs: int,
    seed: int,
    device: torch.device,
) -> tuple[
    EvidentialRegimeHead,
    list[dict],
]:
    set_global_seed(
        seed,
        deterministic=False,
    )

    head = EvidentialRegimeHead(
        input_dim=input_dim,
        n_classes=n_classes,
    ).to(
        device
    )

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    loader = make_embedding_loader(
        train_features,
        train_labels,
        batch_size=batch_size,
        seed=seed,
    )

    best_state = None
    best_val_nll = float(
        "inf"
    )

    history = []

    for epoch in range(
        1,
        epochs + 1,
    ):
        head.train()

        total_loss = 0.0
        total_samples = 0

        kl_weight = min(
            1.0,
            epoch
            / max(
                kl_anneal_epochs,
                1,
            ),
        )

        for features, labels in loader:
            features = features.to(
                device
            )

            labels = labels.to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            output = head(
                features
            )

            loss_output = (
                evidential_classification_loss(
                    output.alpha,
                    labels,
                    kl_weight=kl_weight,
                )
            )

            loss_output.loss.backward()

            optimizer.step()

            batch_n = len(
                labels
            )

            total_loss += (
                float(
                    loss_output
                    .loss
                    .detach()
                    .cpu()
                )
                * batch_n
            )

            total_samples += (
                batch_n
            )

        head.eval()

        val_probabilities, _ = (
            evidential_probabilities(
                head,
                val_features,
                device=device,
            )
        )

        val_nll = float(
            F.nll_loss(
                torch.log(
                    val_probabilities.clamp_min(
                        1e-8
                    )
                ),
                val_labels,
            )
        )

        train_loss = (
            total_loss
            / total_samples
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_nll": val_nll,
                "kl_weight": kl_weight,
            }
        )

        print(
            f"Evidential "
            f"{epoch:03d}/{epochs:03d} | "
            f"train={train_loss:.6f} | "
            f"val_nll={val_nll:.6f} | "
            f"kl={kl_weight:.3f}"
        )

        if (
            val_nll
            < best_val_nll
        ):
            best_val_nll = (
                val_nll
            )

            best_state = copy.deepcopy(
                head.state_dict()
            )

    if best_state is None:
        raise RuntimeError(
            "No evidential best state"
        )

    head.load_state_dict(
        best_state
    )

    return (
        head,
        history,
    )


def train_softmax(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    *,
    input_dim: int,
    n_classes: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> tuple[
    nn.Linear,
    list[dict],
]:
    set_global_seed(
        seed,
        deterministic=False,
    )

    head = nn.Linear(
        input_dim,
        n_classes,
    ).to(
        device
    )

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    loader = make_embedding_loader(
        train_features,
        train_labels,
        batch_size=batch_size,
        seed=seed,
    )

    best_state = None
    best_val_nll = float(
        "inf"
    )

    history = []

    for epoch in range(
        1,
        epochs + 1,
    ):
        head.train()

        total_loss = 0.0
        total_samples = 0

        for features, labels in loader:
            features = features.to(
                device
            )

            labels = labels.to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = head(
                features
            )

            loss = F.cross_entropy(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            batch_n = len(
                labels
            )

            total_loss += (
                float(
                    loss
                    .detach()
                    .cpu()
                )
                * batch_n
            )

            total_samples += (
                batch_n
            )

        head.eval()

        val_probabilities = (
            softmax_probabilities(
                head,
                val_features,
                device=device,
            )
        )

        val_nll = float(
            F.nll_loss(
                torch.log(
                    val_probabilities.clamp_min(
                        1e-8
                    )
                ),
                val_labels,
            )
        )

        train_loss = (
            total_loss
            / total_samples
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_nll": val_nll,
            }
        )

        print(
            f"Softmax    "
            f"{epoch:03d}/{epochs:03d} | "
            f"train={train_loss:.6f} | "
            f"val_nll={val_nll:.6f}"
        )

        if (
            val_nll
            < best_val_nll
        ):
            best_val_nll = (
                val_nll
            )

            best_state = copy.deepcopy(
                head.state_dict()
            )

    if best_state is None:
        raise RuntimeError(
            "No softmax best state"
        )

    head.load_state_dict(
        best_state
    )

    return (
        head,
        history,
    )


def run_probe(args) -> None:

    cde_run_dir = Path(
        args.cde_run_dir
    )

    probe_config_path = Path(
        args.config
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = resolve_device(
        args.device
    )

    probe_cfg = load_probe_config(
        probe_config_path
    )

    seed = int(
        probe_cfg[
            "seed"
        ]
    )

    set_global_seed(
        seed,
        deterministic=False,
    )

    print("=" * 72)
    print(
        "REGIME PROBE — FROZEN CDE REGIME PROBES"
    )
    print("=" * 72)

    print(
        "Device:",
        device,
    )

    # ---------------------------------------------------------
    # Verify retained CDE checkpoint
    # ---------------------------------------------------------
    cde_cfg = load_config(
        cde_run_dir
        / "config.yaml"
    )

    checkpoint = torch.load(
        cde_run_dir
        / "best.pt",
        map_location="cpu",
        weights_only=False,
    )

    with (
        cde_run_dir
        / "summary.json"
    ).open(
        "r",
        encoding="utf-8",
    ) as handle:
        cde_summary = json.load(
            handle
        )

    if (
        checkpoint[
            "epoch"
        ]
        != cde_summary[
            "best_epoch"
        ]
    ):
        raise RuntimeError(
            "CDE checkpoint epoch does not "
            "match retained CDE summary"
        )

    if not np.isclose(
        checkpoint[
            "val_loss"
        ],
        cde_summary[
            "best_val_loss"
        ],
    ):
        raise RuntimeError(
            "CDE checkpoint validation loss "
            "does not match retained CDE summary"
        )

    print(
        "Frozen CDE checkpoint epoch:",
        checkpoint[
            "epoch"
        ],
    )

    print(
        "Frozen CDE validation loss:",
        checkpoint[
            "val_loss"
        ],
    )

    # ---------------------------------------------------------
    # Reconstruct frozen CDE
    # ---------------------------------------------------------
    diffusion_model = build_cde_model(
        cde_cfg
    )

    diffusion_model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    encoder = (
        diffusion_model
        .history_encoder
        .to(
            device
        )
    )

    encoder.eval()

    for parameter in (
        encoder.parameters()
    ):
        parameter.requires_grad_(
            False
        )

    if any(
        parameter.requires_grad
        for parameter
        in encoder.parameters()
    ):
        raise RuntimeError(
            "CDE encoder was not fully frozen"
        )

    # ---------------------------------------------------------
    # Reconstruct exact retained CDE data
    # ---------------------------------------------------------
    datasets = prepare_datasets(
        cde_cfg
    )

    extraction_batch_size = int(
        probe_cfg[
            "training"
        ][
            "batch_size"
        ]
    )

    print()
    print(
        "Extracting frozen CDE embeddings..."
    )

    (
        train_features,
        train_history_raw,
        train_target_raw,
        train_dates,
    ) = extract_features(
        encoder,
        datasets.train,
        batch_size=(
            extraction_batch_size
        ),
        device=device,
    )

    (
        val_features,
        val_history_raw,
        val_target_raw,
        val_dates,
    ) = extract_features(
        encoder,
        datasets.val,
        batch_size=(
            extraction_batch_size
        ),
        device=device,
    )

    print(
        "Train embeddings:",
        tuple(
            train_features.shape
        ),
    )

    print(
        "Validation embeddings:",
        tuple(
            val_features.shape
        ),
    )

    # ---------------------------------------------------------
    # Train-only regime construction
    # ---------------------------------------------------------
    regime_cfg = probe_cfg[
        "regimes"
    ]

    n_classes = int(
        regime_cfg[
            "n_classes"
        ]
    )

    if n_classes != 3:
        raise ValueError(
            "Current regime probe experiment "
            "requires exactly 3 regimes"
        )

    quantiles = tuple(
        float(value)
        for value
        in regime_cfg[
            "quantiles"
        ]
    )

    if len(
        quantiles
    ) != 2:
        raise ValueError(
            "Expected exactly two quantiles"
        )

    recent_days = None

    if args.regime_target == "next_day":
        score_name = (
            "cross_sectional_rms"
        )

        train_scores = (
            cross_sectional_rms(
                train_target_raw
            )
        )

        val_scores = (
            cross_sectional_rms(
                val_target_raw
            )
        )

    else:
        score_name = (
            "trailing_cross_sectional_rms"
        )

        recent_days = int(
            regime_cfg[
                "recent_days"
            ]
        )

        train_scores = (
            trailing_cross_sectional_rms(
                train_history_raw,
                recent_days=recent_days,
            )
        )

        val_scores = (
            trailing_cross_sectional_rms(
                val_history_raw,
                recent_days=recent_days,
            )
        )

    print(
        "Regime target:",
        args.regime_target,
    )

    print(
        "Regime score:",
        score_name,
    )

    thresholds = (
        fit_regime_thresholds(
            train_scores,
            quantiles=quantiles,
        )
    )

    train_labels_np = (
        assign_regime_labels(
            train_scores,
            thresholds=thresholds,
        )
    )

    val_labels_np = (
        assign_regime_labels(
            val_scores,
            thresholds=thresholds,
        )
    )

    train_labels = torch.from_numpy(
        train_labels_np
    ).long()

    val_labels = torch.from_numpy(
        val_labels_np
    ).long()

    print()
    print(
        "Train-only thresholds:",
        thresholds,
    )

    print(
        "Train class counts:",
        class_counts(
            train_labels,
            n_classes,
        ),
    )

    print(
        "Validation class counts:",
        class_counts(
            val_labels,
            n_classes,
        ),
    )

    # ---------------------------------------------------------
    # Train equal-capacity probes
    # ---------------------------------------------------------
    training_cfg = probe_cfg[
        "training"
    ]

    input_dim = int(
        train_features.shape[
            1
        ]
    )

    configured_input_dim = int(
        probe_cfg[
            "model"
        ][
            "input_dim"
        ]
    )

    if (
        input_dim
        != configured_input_dim
    ):
        raise RuntimeError(
            "CDE embedding dimension does not "
            "match regime probe config"
        )

    common_kwargs = {
        "input_dim": input_dim,
        "n_classes": n_classes,
        "epochs": int(
            training_cfg[
                "epochs"
            ]
        ),
        "batch_size": int(
            training_cfg[
                "batch_size"
            ]
        ),
        "learning_rate": float(
            training_cfg[
                "learning_rate"
            ]
        ),
        "weight_decay": float(
            training_cfg[
                "weight_decay"
            ]
        ),
        "seed": seed,
        "device": device,
    }

    print()
    print(
        "Training evidential probe..."
    )

    (
        evidential_head,
        evidential_history,
    ) = train_evidential(
        train_features,
        train_labels,
        val_features,
        val_labels,
        kl_anneal_epochs=int(
            training_cfg[
                "kl_anneal_epochs"
            ]
        ),
        **common_kwargs,
    )

    print()
    print(
        "Training softmax control..."
    )

    (
        softmax_head,
        softmax_history,
    ) = train_softmax(
        train_features,
        train_labels,
        val_features,
        val_labels,
        **common_kwargs,
    )

    # ---------------------------------------------------------
    # Validation-only evaluation
    # ---------------------------------------------------------
    evidential_head.eval()

    (
        evidential_probs,
        val_vacuity,
    ) = evidential_probabilities(
        evidential_head,
        val_features,
        device=device,
    )

    softmax_head.eval()

    softmax_probs = (
        softmax_probabilities(
            softmax_head,
            val_features,
            device=device,
        )
    )

    ece_bins = int(
        probe_cfg[
            "evaluation"
        ][
            "ece_bins"
        ]
    )

    evidential_metrics = (
        classification_metrics(
            evidential_probs,
            val_labels,
            n_classes=n_classes,
            ece_bins=ece_bins,
            vacuity=val_vacuity,
        )
    )

    softmax_metrics = (
        classification_metrics(
            softmax_probs,
            val_labels,
            n_classes=n_classes,
            ece_bins=ece_bins,
        )
    )

    # ---------------------------------------------------------
    # Save artifacts
    # ---------------------------------------------------------
    torch.save(
        {
            "model_state_dict": (
                evidential_head
                .state_dict()
            ),
            "input_dim": input_dim,
            "n_classes": n_classes,
        },
        output_dir
        / "evidential_best.pt",
    )

    torch.save(
        {
            "model_state_dict": (
                softmax_head
                .state_dict()
            ),
            "input_dim": input_dim,
            "n_classes": n_classes,
        },
        output_dir
        / "softmax_best.pt",
    )

    pd.DataFrame(
        evidential_history
    ).to_csv(
        output_dir
        / "evidential_history.csv",
        index=False,
    )

    pd.DataFrame(
        softmax_history
    ).to_csv(
        output_dir
        / "softmax_history.csv",
        index=False,
    )

    thresholds_payload = {
        "regime_target": (
            args.regime_target
        ),
        "score": score_name,
        "quantiles": list(
            quantiles
        ),
        "thresholds": {
            "calm_normal": (
                thresholds[0]
            ),
            "normal_stressed": (
                thresholds[1]
            ),
        },
        "train_class_counts": (
            class_counts(
                train_labels,
                n_classes,
            )
        ),
        "val_class_counts": (
            class_counts(
                val_labels,
                n_classes,
            )
        ),
    }
    if recent_days is not None:
        thresholds_payload[
            "recent_days"
        ] = recent_days
    with (
        output_dir
        / "regime_thresholds.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            thresholds_payload,
            handle,
            indent=2,
        )

    metrics = {
        "phase0a_checkpoint_epoch": (
            checkpoint[
                "epoch"
            ]
        ),
        "phase0a_validation_loss": (
            checkpoint[
                "val_loss"
            ]
        ),
        "evidential": (
            evidential_metrics
        ),
        "softmax": (
            softmax_metrics
        ),
    }

    with (
        output_dir
        / "validation_metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metrics,
            handle,
            indent=2,
        )

    np.savez(
        output_dir
        / "validation_predictions.npz",
        dates=val_dates.values,
        labels=val_labels_np,
        stress_scores=val_scores,
        evidential_probabilities=(
            evidential_probs.numpy()
        ),
        evidential_vacuity=(
            val_vacuity.numpy()
        ),
        softmax_probabilities=(
            softmax_probs.numpy()
        ),
    )

    print()
    print("=" * 72)
    print(
        "REGIME PROBE VALIDATION RESULTS"
    )
    print("=" * 72)

    print()
    print(
        "Evidential:"
    )

    for key, value in (
        evidential_metrics.items()
    ):
        print(
            f"  {key}: {value}"
        )

    print()
    print(
        "Softmax control:"
    )

    for key, value in (
        softmax_metrics.items()
    ):
        print(
            f"  {key}: {value}"
        )

    print()
    print(
        "Artifacts:",
        output_dir,
    )

    print()
    print(
        "TEST SPLIT WAS NOT EVALUATED."
    )
