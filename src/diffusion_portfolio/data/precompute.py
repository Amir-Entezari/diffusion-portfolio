"""Frozen-CDE topology and covariance feature caches for current ablations."""
from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from diffusion_portfolio.config import load_config
from diffusion_portfolio.data import collate_return_batch
from diffusion_portfolio.data.features import standardize_features
from diffusion_portfolio.data.preparation import prepare_return_data
from diffusion_portfolio.models.diffusion.build import build_diffusion
from diffusion_portfolio.models.topology import trajectory_geometry_features, trajectory_persistence_features
from diffusion_portfolio.models.spd import covariance_features, log_euclidean_spd_features, regularized_covariance
from diffusion_portfolio.utils.device import resolve_device


@torch.no_grad()
def extract_topology_split(
    encoder,
    dataset,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_return_batch,
    )

    conditions = []
    geometry = []
    topology = []
    dates = []

    encoder.eval()

    for batch in loader:
        history = batch[
            "history"
        ].to(
            device
        )

        condition = encoder(
            history
        )

        trajectory = (
            encoder.encode_trajectory(
                history
            )
        )

        geometry_features = (
            trajectory_geometry_features(
                trajectory
            )
        )

        topology_features = (
            trajectory_persistence_features(
                trajectory,
                max_homology_dim=1,
                top_k=3,
            )
        )

        conditions.append(
            condition.cpu().numpy()
        )

        geometry.append(
            geometry_features.cpu().numpy()
        )

        topology.append(
            topology_features.cpu().numpy()
        )

        dates.append(
            batch[
                "target_date"
            ].values
        )

    return {
        "dates": np.concatenate(
            dates
        ),
        "cde_condition": np.concatenate(
            conditions
        ),
        "geometry_raw": np.concatenate(
            geometry
        ),
        "tda_raw": np.concatenate(
            topology
        ),
    }


@torch.no_grad()
def extract_spd_split(
    encoder,
    dataset,
    *,
    batch_size: int,
    device: torch.device,
    eigenvalue_floor: float,
) -> dict[str, np.ndarray]:

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_return_batch,
    )

    conditions = []
    covariance_parts = []
    log_spd_parts = []

    min_eigenvalues = []
    max_eigenvalues = []
    condition_numbers = []

    dates = []

    encoder.eval()

    for batch in loader:
        history = batch[
            "history"
        ].to(
            device
        )

        history_raw = batch[
            "history_raw"
        ].to(
            device
        )

        condition = encoder(
            history
        )

        covariance = regularized_covariance(
            history_raw,
            relative_eigenvalue_floor=(
                eigenvalue_floor
            ),
        )

        eigenvalues = torch.linalg.eigvalsh(
            covariance
        )

        minimum = eigenvalues[
            :,
            0,
        ]

        maximum = eigenvalues[
            :,
            -1,
        ]

        condition_number = (
            maximum
            / minimum
        )

        covariance_vector = (
            covariance_features(
                history_raw,
                relative_eigenvalue_floor=(
                    eigenvalue_floor
                ),
            )
        )

        log_spd_vector = (
            log_euclidean_spd_features(
                history_raw,
                relative_eigenvalue_floor=(
                    eigenvalue_floor
                ),
            )
        )

        conditions.append(
            condition.cpu().numpy()
        )

        covariance_parts.append(
            covariance_vector
            .cpu()
            .numpy()
        )

        log_spd_parts.append(
            log_spd_vector
            .cpu()
            .numpy()
        )

        min_eigenvalues.append(
            minimum.cpu().numpy()
        )

        max_eigenvalues.append(
            maximum.cpu().numpy()
        )

        condition_numbers.append(
            condition_number
            .cpu()
            .numpy()
        )

        dates.append(
            batch[
                "target_date"
            ].values
        )

    return {
        "dates": np.concatenate(
            dates
        ),
        "cde_condition": np.concatenate(
            conditions
        ),
        "covariance_raw": np.concatenate(
            covariance_parts
        ),
        "log_spd_raw": np.concatenate(
            log_spd_parts
        ),
        "min_eigenvalue": np.concatenate(
            min_eigenvalues
        ),
        "max_eigenvalue": np.concatenate(
            max_eigenvalues
        ),
        "condition_number": np.concatenate(
            condition_numbers
        ),
    }


def spectral_summary(
    split: dict[str, np.ndarray],
) -> dict:
    minimum = split[
        "min_eigenvalue"
    ]

    maximum = split[
        "max_eigenvalue"
    ]

    condition = split[
        "condition_number"
    ]

    return {
        "min_eigenvalue_min": float(
            minimum.min()
        ),
        "min_eigenvalue_median": float(
            np.median(
                minimum
            )
        ),
        "max_eigenvalue_median": float(
            np.median(
                maximum
            )
        ),
        "condition_number_median": float(
            np.median(
                condition
            )
        ),
        "condition_number_p95": float(
            np.quantile(
                condition,
                0.95,
            )
        ),
        "condition_number_max": float(
            condition.max()
        ),
    }


def precompute_features(kind, cde_run_dir, output_dir, *, batch_size=None,
                        device="auto", eigenvalue_floor=1e-6):
    if kind not in ("topology", "spd"):
        raise ValueError("Unknown feature kind")
    batch_size = (32 if kind == "topology" else 128) if batch_size is None else batch_size
    if batch_size <= 0 or eigenvalue_floor <= 0:
        raise ValueError("Batch size and eigenvalue floor must be positive")
    cde_run_dir, output_dir = Path(cde_run_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(cde_run_dir / "config.yaml")
    device = resolve_device(device)
    checkpoint = torch.load(cde_run_dir / "best.pt", map_location="cpu", weights_only=False)
    datasets, scaler = prepare_return_data(
        cfg, standardizer_path=cde_run_dir / "standardizer.npz"
    )
    model = build_diffusion(cfg, n_assets=len(scaler.columns))
    model.load_state_dict(checkpoint["model_state_dict"])
    encoder = model.history_encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    if not hasattr(encoder, "encode_trajectory"):
        raise ValueError("Feature extraction requires the retained CDE encoder")
    kwargs = dict(batch_size=batch_size, device=device)
    if kind == "topology":
        extract = extract_topology_split
        names = ("geometry", "tda")
        metadata = {
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_val_loss": float(checkpoint["val_loss"]),
        }
    else:
        extract = extract_spd_split
        kwargs["eigenvalue_floor"] = eigenvalue_floor
        names = ("covariance", "log_spd")
        # Preserve the cache metadata schema consumed by archived experiments.
        metadata = {
            "phase0_checkpoint_epoch": int(checkpoint["epoch"]),
            "phase0_checkpoint_val_loss": float(checkpoint["val_loss"]),
            "relative_eigenvalue_floor": float(eigenvalue_floor),
        }
    train = extract(encoder, datasets.train, **kwargs)
    validation = extract(encoder, datasets.val, **kwargs)
    metadata["cde_condition_dim"] = int(train["cde_condition"].shape[1])
    train_cache = {key: train[key] for key in ("dates", "cde_condition")}
    val_cache = {key: validation[key] for key in ("dates", "cde_condition")}
    for name in names:
        raw_key = f"{name}_raw"
        train_scaled, val_scaled, mean, std, constant = standardize_features(
            train[raw_key], validation[raw_key], drop_constant=kind == "topology"
        )
        if not np.isfinite(train_scaled).all() or not np.isfinite(val_scaled).all():
            raise RuntimeError(f"Non-finite standardized {name} features")
        train_cache.update({name: train_scaled, raw_key: train[raw_key]})
        val_cache.update({name: val_scaled, raw_key: validation[raw_key]})
        metadata.update({
            f"{name}_dim": int(train_scaled.shape[1]),
            f"{name}_mean": mean.tolist(), f"{name}_std": std.tolist(),
        })
        if kind == "topology":
            metadata[f"{name}_raw_dim"] = int(len(constant))
            metadata[f"{name}_keep_indices"] = np.flatnonzero(~constant).tolist()
        else:
            metadata[f"{name}_constant_indices"] = np.flatnonzero(constant).tolist()
    if kind == "spd":
        if train_cache["covariance"].shape[1] != train_cache["log_spd"].shape[1]:
            raise RuntimeError("Covariance and Log-SPD dimensions must match")
        metadata.update(
            train_spectrum=spectral_summary(train),
            validation_spectrum=spectral_summary(validation),
        )
    np.savez_compressed(output_dir / "train.npz", **train_cache)
    np.savez_compressed(output_dir / "validation.npz", **val_cache)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print("Feature cache:", output_dir, "Dimensions:", [metadata[f"{n}_dim"] for n in names])
    return metadata
