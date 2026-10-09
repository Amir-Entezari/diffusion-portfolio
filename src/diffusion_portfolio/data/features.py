"""Cached conditioning datasets and train-only feature standardization."""
import numpy as np
import torch
from torch.utils.data import Dataset


FEATURE_VARIANTS = ("cde", "geometry", "tda", "covariance", "log_spd")


class FeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        if len(features) != len(targets):
            raise ValueError("feature/target counts differ")
        self.features = torch.from_numpy(features).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict:
        return {"history": self.features[index], "target": self.targets[index]}


def cached_features(cache, variant: str, dates):
    if variant not in FEATURE_VARIANTS:
        raise ValueError(f"Unknown feature variant: {variant}")
    if not np.array_equal(cache["dates"], dates.values):
        raise RuntimeError("Cached dates do not match canonical dataset")
    condition = cache["cde_condition"]
    if variant == "cde":
        return condition, 0
    extra = cache[variant]
    return np.concatenate([condition, extra], axis=1), int(extra.shape[1])


def standardize_features(train, validation, *, drop_constant=False):
    """Preserve topology's dropped constants and SPD's retained constants."""
    mean = train.mean(axis=0, dtype=np.float64)
    std = train.std(axis=0, dtype=np.float64)
    constant = std <= 1e-12
    if drop_constant:
        keep = ~constant
        train_scaled = (train[:, keep] - mean[keep]) / std[keep]
        validation_scaled = (validation[:, keep] - mean[keep]) / std[keep]
    else:
        safe_std = std.copy()
        safe_std[constant] = 1.0
        train_scaled = (train - mean) / safe_std
        validation_scaled = (validation - mean) / safe_std
    return (
        train_scaled.astype(np.float32), validation_scaled.astype(np.float32),
        mean, std, constant,
    )
