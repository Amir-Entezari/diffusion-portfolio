"""Scenario generation and serialization shared by diffusion evaluations."""
import numpy as np
import torch
from tqdm import tqdm


def seed_sampling(seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def generate_scenarios(model, loader, scaler, *, n_samples, n_scenarios, device):
    """Generate raw decimal returns, preserving batch order and sampler RNG."""
    forecast = np.empty(
        (n_samples, n_scenarios, len(scaler.columns)), dtype=np.float32
    )
    offset = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Sampling"):
            history = batch["history"].to(device, non_blocking=True)
            generated = model.sample(history, n_scenarios=n_scenarios)
            end = offset + history.shape[0]
            forecast[offset:end] = scaler.inverse_transform(generated.cpu().numpy())
            offset = end
    if offset != n_samples:
        raise RuntimeError("Did not generate every evaluation sample")
    if not np.isfinite(forecast).all():
        raise RuntimeError("Generated scenarios contain non-finite values")
    return forecast


def probabilistic_summary(metrics, columns=None):
    result = {
        "crps_mean": float(metrics.crps_mean),
        "crps_std": float(metrics.crps_std),
        "energy_score": float(metrics.energy_score),
        "calibration": [
            {"level": float(level), "picp": float(picp), "ace": float(ace)}
            for level, picp, ace in zip(
                metrics.calibration.levels,
                metrics.calibration.picp,
                metrics.calibration.ace,
            )
        ],
    }
    if columns is not None:
        result["crps_by_asset"] = {
            column: float(value) for column, value in zip(columns, metrics.crps_by_asset)
        }
    return result


def save_scenarios(path, scenarios, observed, dates, columns, *, compressed=False):
    save = np.savez_compressed if compressed else np.savez
    save(
        path, scenarios=scenarios, observed=observed,
        dates=dates.values.astype("datetime64[D]"), columns=np.asarray(columns),
    )
