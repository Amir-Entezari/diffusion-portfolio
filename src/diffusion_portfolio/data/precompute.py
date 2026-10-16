"""Frozen-CDE topology and covariance feature caches for current ablations."""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from diffusion_portfolio.config import load_config
from diffusion_portfolio.data import collate_return_batch
from diffusion_portfolio.data.features import standardize_features
from diffusion_portfolio.data.preparation import prepare_return_data
from diffusion_portfolio.models.diffusion.build import build_diffusion
from diffusion_portfolio.models.topology import trajectory_geometry_features, trajectory_persistence_features
from diffusion_portfolio.models.spd import covariance_features, log_euclidean_spd_features, regularized_covariance
from diffusion_portfolio.utils.device import resolve_device
from diffusion_portfolio.utils.research_checks import source_identity


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


def prepared_data_identity(datasets):
    """Bind a feature cache to exact retained train/validation returns and dates."""
    result = {}
    for name, dataset in (("train", datasets.train), ("validation", datasets.val)):
        digest = hashlib.sha256()
        windows = dataset.raw_windows
        for values in (windows.target_dates.values, windows.history, windows.target):
            digest.update(memoryview(np.ascontiguousarray(values).view(np.uint8)))
        result[name] = digest.hexdigest()
    return result


def precompute_features(kind, cde_run_dir, output_dir, *, batch_size=None,
                        device="auto", eigenvalue_floor=1e-6, smoke=False, reuse=False):
    if kind not in ("topology", "spd"):
        raise ValueError("Unknown feature kind")
    batch_size = (32 if kind == "topology" else 128) if batch_size is None else batch_size
    if batch_size <= 0 or eigenvalue_floor <= 0:
        raise ValueError("Batch size and eigenvalue floor must be positive")
    cde_run_dir, output_dir = Path(cde_run_dir), Path(output_dir)
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
    metadata["source"] = source_identity(cde_run_dir)
    metadata["kind"] = kind
    metadata["prepared_data"] = prepared_data_identity(datasets)
    if reuse and not smoke and output_dir.exists():
        for filename in ("train.npz", "validation.npz", "metadata.json"):
            if not (output_dir / filename).is_file():
                raise RuntimeError(f"Incomplete feature cache; choose a fresh output directory: {output_dir}")
        saved = json.loads((output_dir / "metadata.json").read_text())
        if (saved.get("source") != metadata["source"] or saved.get("kind") != kind
                or saved.get("prepared_data") != metadata["prepared_data"]
                or (kind == "spd" and saved.get("relative_eigenvalue_floor") != eigenvalue_floor)):
            raise RuntimeError("Feature cache belongs to another source/settings; it will not be overwritten")
        validate_feature_cache(output_dir, datasets, kind)
        print("Reusing feature cache:", output_dir)
        return saved
    if smoke:
        refitted, refit_scaler = prepare_return_data(cfg)
        if (refit_scaler.columns != scaler.columns
                or not np.array_equal(refit_scaler.mean, scaler.mean)
                or not np.array_equal(refit_scaler.std, scaler.std)):
            raise RuntimeError("Current source data no longer matches the saved training scaler")
        del refitted
        small = Subset(datasets.train, range(min(2, len(datasets.train))))
        values = extract(encoder, small, **kwargs)
        if not all(np.isfinite(value).all() for key, value in values.items() if key != "dates"):
            raise RuntimeError("Non-finite feasibility features")
        assets = len(scaler.columns)
        dimension = assets * (assets + 1) // 2 if kind == "spd" else 7 + 16
        if kind == "spd":
            if (values["covariance_raw"].shape[1] != dimension
                    or values["log_spd_raw"].shape[1] != dimension
                    or np.any(values["min_eigenvalue"] <= 0)):
                raise RuntimeError("SPD feasibility failed: dimension or positive-definiteness")
        # Raw and scaled extras plus conditions; peak extraction/training RAM is higher.
        extras = (4 if kind == "spd" else 2) * dimension
        cache_bytes = (len(datasets.train) + len(datasets.val)) * (extras + cfg.model.condition_dim) * 4
        report = {"assets": assets, "lookback": cfg.data.lookback,
                  "shapes": {key: list(value.shape) for key, value in values.items()},
                  "estimated_cache_GiB": cache_bytes / 1024**3,
                  "train_samples": len(datasets.train), "validation_samples": len(datasets.val)}
        if kind == "spd":
            report["spectrum"] = spectral_summary(values)
        if device.type == "cuda":
            report["gpu_free_GiB"] = torch.cuda.mem_get_info(device)[0] / 1024**3
            report["gpu_peak_allocated_GiB"] = torch.cuda.max_memory_allocated(device) / 1024**3
        print(json.dumps(report, indent=2))
        print("Feasibility only: two training windows; no full cache written. Peak RAM exceeds the cache estimate.")
        return report
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
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "train.npz", **train_cache)
    np.savez_compressed(output_dir / "validation.npz", **val_cache)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print("Feature cache:", output_dir, "Dimensions:", [metadata[f"{n}_dim"] for n in names])
    return metadata


def validate_feature_cache(feature_dir, datasets, kind):
    """Check cached dates, finite values and the recorded train-only transform."""
    names = ("geometry", "tda") if kind == "topology" else ("covariance", "log_spd")
    metadata = json.loads((Path(feature_dir) / "metadata.json").read_text())
    if metadata.get("prepared_data") != prepared_data_identity(datasets):
        raise RuntimeError("Feature cache dates/returns differ from current data preparation")
    with np.load(Path(feature_dir) / "train.npz") as train, np.load(Path(feature_dir) / "validation.npz") as val:
        for cache, dataset in ((train, datasets.train), (val, datasets.val)):
            if not np.array_equal(cache["dates"], dataset.model_windows.target_dates.values):
                raise RuntimeError("Feature cache dates do not match the canonical windows")
            count = len(dataset)
            if cache["cde_condition"].shape != (count, metadata["cde_condition_dim"]):
                raise RuntimeError("Cached CDE conditioning dimensions differ")
            for name in names:
                if cache[name].shape != (count, metadata[name + "_dim"]):
                    raise RuntimeError(f"Cached {name} dimensions differ")
            for key in ("cde_condition", *names, *[name + "_raw" for name in names]):
                if not np.isfinite(cache[key]).all():
                    raise RuntimeError(f"Non-finite cached feature: {key}")
        for name in names:
            train_scaled, val_scaled, mean, std, _ = standardize_features(
                train[name + "_raw"], val[name + "_raw"], drop_constant=kind == "topology"
            )
            if (not np.allclose(mean, metadata[name + "_mean"])
                    or not np.allclose(std, metadata[name + "_std"])
                    or not np.allclose(train_scaled, train[name])
                    or not np.allclose(val_scaled, val[name])):
                raise RuntimeError(f"Cache does not use the recorded train-only {name} transform")
    return metadata
