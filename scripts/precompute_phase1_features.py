"""Precompute frozen Phase-0 CDE trajectory features for Phase 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from diffusion_portfolio.config import (
    load_config,
)
from diffusion_portfolio.data import (
    collate_return_batch,
)
from diffusion_portfolio.models.topology import (
    trajectory_geometry_features,
    trajectory_persistence_features,
)

from train_evidential_regime import (
    build_cde_model,
    prepare_datasets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cde-run-dir",
        default="/kaggle/working/phase0_cde",
    )

    parser.add_argument(
        "--output-dir",
        default="/kaggle/working/phase1_features",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
    )

    return parser.parse_args()


def resolve_device(
    requested: str,
) -> torch.device:
    if requested == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if (
        requested == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable"
        )

    return torch.device(
        requested
    )


@torch.no_grad()
def extract_split(
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

        trajectory = (
            encoder.encode_trajectory(
                history
            )
        )

        condition = encoder.readout(
            trajectory[
                :,
                -1,
            ]
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


def fit_transform(
    train: np.ndarray,
    validation: np.ndarray,
):
    mean = train.mean(
        axis=0,
        dtype=np.float64,
    )

    std = train.std(
        axis=0,
        dtype=np.float64,
    )

    keep = (
        std > 1e-12
    )

    train_scaled = (
        train[
            :,
            keep,
        ]
        - mean[
            keep
        ]
    ) / std[
        keep
    ]

    validation_scaled = (
        validation[
            :,
            keep,
        ]
        - mean[
            keep
        ]
    ) / std[
        keep
    ]

    return (
        train_scaled.astype(
            np.float32
        ),
        validation_scaled.astype(
            np.float32
        ),
        keep,
        mean,
        std,
    )


def main() -> None:
    args = parse_args()

    cde_run_dir = Path(
        args.cde_run_dir
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

    cfg = load_config(
        cde_run_dir
        / "config.yaml"
    )

    checkpoint = torch.load(
        cde_run_dir
        / "best.pt",
        map_location="cpu",
        weights_only=False,
    )

    model = build_cde_model(
        cfg
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    encoder = (
        model.history_encoder
        .to(device)
        .eval()
    )

    datasets = prepare_datasets(
        cfg
    )

    print(
        "Device:",
        device,
    )

    print(
        "Checkpoint epoch:",
        checkpoint[
            "epoch"
        ],
    )

    print(
        "Extracting train features..."
    )

    train = extract_split(
        encoder,
        datasets.train,
        batch_size=args.batch_size,
        device=device,
    )

    print(
        "Extracting validation features..."
    )

    validation = extract_split(
        encoder,
        datasets.val,
        batch_size=args.batch_size,
        device=device,
    )

    (
        train_geometry,
        val_geometry,
        geometry_keep,
        geometry_mean,
        geometry_std,
    ) = fit_transform(
        train[
            "geometry_raw"
        ],
        validation[
            "geometry_raw"
        ],
    )

    (
        train_tda,
        val_tda,
        tda_keep,
        tda_mean,
        tda_std,
    ) = fit_transform(
        train[
            "tda_raw"
        ],
        validation[
            "tda_raw"
        ],
    )

    np.savez_compressed(
        output_dir
        / "train.npz",
        dates=train[
            "dates"
        ],
        cde_condition=train[
            "cde_condition"
        ],
        geometry=train_geometry,
        tda=train_tda,
        geometry_raw=train[
            "geometry_raw"
        ],
        tda_raw=train[
            "tda_raw"
        ],
    )

    np.savez_compressed(
        output_dir
        / "validation.npz",
        dates=validation[
            "dates"
        ],
        cde_condition=validation[
            "cde_condition"
        ],
        geometry=val_geometry,
        tda=val_tda,
        geometry_raw=validation[
            "geometry_raw"
        ],
        tda_raw=validation[
            "tda_raw"
        ],
    )

    metadata = {
        "checkpoint_epoch": int(
            checkpoint[
                "epoch"
            ]
        ),
        "checkpoint_val_loss": float(
            checkpoint[
                "val_loss"
            ]
        ),
        "cde_condition_dim": int(
            train[
                "cde_condition"
            ].shape[1]
        ),
        "geometry_raw_dim": int(
            len(
                geometry_keep
            )
        ),
        "geometry_dim": int(
            geometry_keep.sum()
        ),
        "geometry_keep_indices": (
            np.flatnonzero(
                geometry_keep
            ).tolist()
        ),
        "geometry_mean": (
            geometry_mean.tolist()
        ),
        "geometry_std": (
            geometry_std.tolist()
        ),
        "tda_raw_dim": int(
            len(
                tda_keep
            )
        ),
        "tda_dim": int(
            tda_keep.sum()
        ),
        "tda_keep_indices": (
            np.flatnonzero(
                tda_keep
            ).tolist()
        ),
        "tda_mean": (
            tda_mean.tolist()
        ),
        "tda_std": (
            tda_std.tolist()
        ),
    }

    with (
        output_dir
        / "metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
        )

    print()
    print(
        "Train:",
        train[
            "cde_condition"
        ].shape[0],
    )

    print(
        "Validation:",
        validation[
            "cde_condition"
        ].shape[0],
    )

    print(
        "CDE condition:",
        train[
            "cde_condition"
        ].shape[1],
    )

    print(
        "Geometry:",
        train_geometry.shape[1],
    )

    print(
        "TDA:",
        train_tda.shape[1],
    )

    print()
    print(
        "Artifacts:",
        output_dir,
    )

    print(
        "TEST SPLIT WAS NOT TOUCHED."
    )


if __name__ == "__main__":
    main()