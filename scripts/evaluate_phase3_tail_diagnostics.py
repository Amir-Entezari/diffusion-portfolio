"""Tail-fidelity diagnostics for the retained Phase-0 CDE diffusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from diffusion_portfolio.config import (
    load_config,
)
from diffusion_portfolio.evaluation import (
    brier_score,
    evaluate_tail_forecast,
)
from diffusion_portfolio.evaluation.regimes import (
    cross_sectional_rms,
)

from train_evidential_regime import (
    prepare_datasets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cde-run-dir",
        default="/kaggle/working/phase0_cde",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = Path(
        args.cde_run_dir
    )

    config_path = (
        run_dir
        / "config.yaml"
    )

    standardizer_path = (
        run_dir
        / "standardizer.npz"
    )

    scenario_path = (
        run_dir
        / "validation_scenarios.npz"
    )

    for path in (
        config_path,
        standardizer_path,
        scenario_path,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    cfg = load_config(
        config_path
    )

    datasets = prepare_datasets(
        cfg,
        standardizer_path=(
            standardizer_path
        ),
    )

    archive = np.load(
        scenario_path,
        allow_pickle=False,
    )

    scenarios = archive[
        "scenarios"
    ]

    observed = archive[
        "observed"
    ]

    saved_dates = archive[
        "dates"
    ].astype(
        "datetime64[D]"
    )

    canonical_dates = (
        datasets.val
        .raw_windows
        .target_dates
        .values
        .astype(
            "datetime64[D]"
        )
    )

    canonical_observed = (
        datasets.val
        .raw_windows
        .target[
            :,
            0,
            :,
        ]
        .astype(
            np.float32,
            copy=False,
        )
    )

    if not np.array_equal(
        saved_dates,
        canonical_dates,
    ):
        raise RuntimeError(
            "Saved scenario dates do not match "
            "the canonical validation split"
        )

    np.testing.assert_allclose(
        observed,
        canonical_observed,
        rtol=0.0,
        atol=1e-8,
    )

    train_targets = (
        datasets.train
        .raw_windows
        .target[
            :,
            0,
            :,
        ]
        .astype(
            np.float32,
            copy=False,
        )
    )

    metrics = evaluate_tail_forecast(
        scenarios,
        observed,
        train_targets,
        levels=(
            0.01,
            0.05,
            0.95,
            0.99,
        ),
        stress_quantile=0.95,
    )

    coverage_mean = (
        metrics
        .quantile_coverage_by_asset
        .mean(
            axis=1
        )
    )

    mean_abs_calibration = (
        np.abs(
            metrics
            .quantile_calibration_error_by_asset
        )
        .mean(
            axis=1
        )
    )

    pinball_mean = (
        metrics
        .pinball_loss_by_asset
        .mean(
            axis=1
        )
    )

    # --------------------------------------------------------
    # Stress-event climatology baseline.
    #
    # Threshold is fitted on TRAIN only.
    # A constant forecast equal to the training exceedance
    # frequency provides a simple Brier reference.
    # --------------------------------------------------------

    train_stress = (
        cross_sectional_rms(
            train_targets
        )
        > metrics.stress_threshold
    )

    validation_stress = (
        cross_sectional_rms(
            observed
        )
        > metrics.stress_threshold
    )

    train_stress_rate = float(
        train_stress.mean()
    )

    climatology_probability = np.full(
        len(
            validation_stress
        ),
        train_stress_rate,
        dtype=np.float64,
    )

    climatology_brier = brier_score(
        climatology_probability,
        validation_stress.astype(
            np.float64
        ),
    )

    if climatology_brier > 0:
        brier_skill_score = (
            1.0
            - metrics.stress_brier_score
            / climatology_brier
        )
    else:
        brier_skill_score = float(
            "nan"
        )

    result = {
        "split": "validation",
        "n_validation": int(
            len(
                observed
            )
        ),
        "n_scenarios": int(
            scenarios.shape[1]
        ),
        "levels": (
            metrics.levels.tolist()
        ),
        "quantile_coverage_mean_across_assets": (
            coverage_mean.tolist()
        ),
        "quantile_mean_abs_calibration_error": (
            mean_abs_calibration.tolist()
        ),
        "quantile_pinball_mean": (
            pinball_mean.tolist()
        ),
        "quantile_coverage_by_asset": (
            metrics
            .quantile_coverage_by_asset
            .tolist()
        ),
        "quantile_calibration_error_by_asset": (
            metrics
            .quantile_calibration_error_by_asset
            .tolist()
        ),
        "pinball_loss_by_asset": (
            metrics
            .pinball_loss_by_asset
            .tolist()
        ),
        "stress_threshold": float(
            metrics.stress_threshold
        ),
        "train_stress_rate": (
            train_stress_rate
        ),
        "validation_stress_rate": float(
            metrics.stress_base_rate
        ),
        "mean_predicted_stress_probability": float(
            metrics
            .mean_predicted_stress_probability
        ),
        "stress_brier_score": float(
            metrics.stress_brier_score
        ),
        "climatology_brier_score": float(
            climatology_brier
        ),
        "stress_brier_skill_score": float(
            brier_skill_score
        ),
        "n_stress_dates": int(
            metrics.n_stress_dates
        ),
        "stress_crps_mean": float(
            metrics.stress_crps_mean
        ),
        "stress_energy_score": float(
            metrics.stress_energy_score
        ),
    }

    output_path = (
        run_dir
        / "validation_tail_diagnostics.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            result,
            handle,
            indent=2,
        )

    print("=" * 72)
    print(
        "PHASE 3A TAIL-FIDELITY GATE"
    )
    print("=" * 72)

    print(
        "Validation samples:",
        result[
            "n_validation"
        ],
    )

    print(
        "Scenarios per date:",
        result[
            "n_scenarios"
        ],
    )

    print()
    print(
        "Marginal tail quantiles:"
    )

    print(
        f"{'q':>8}"
        f"{'coverage':>14}"
        f"{'mean |error|':>16}"
        f"{'pinball':>14}"
    )

    for (
        level,
        coverage,
        calibration,
        pinball,
    ) in zip(
        metrics.levels,
        coverage_mean,
        mean_abs_calibration,
        pinball_mean,
    ):
        print(
            f"{level:8.2f}"
            f"{coverage:14.6f}"
            f"{calibration:16.6f}"
            f"{pinball:14.8f}"
        )

    print()
    print(
        "Stress threshold:",
        result[
            "stress_threshold"
        ],
    )

    print(
        "Train stress rate:",
        result[
            "train_stress_rate"
        ],
    )

    print(
        "Validation stress rate:",
        result[
            "validation_stress_rate"
        ],
    )

    print(
        "Mean predicted stress probability:",
        result[
            "mean_predicted_stress_probability"
        ],
    )

    print(
        "Stress Brier:",
        result[
            "stress_brier_score"
        ],
    )

    print(
        "Climatology Brier:",
        result[
            "climatology_brier_score"
        ],
    )

    print(
        "Brier skill score:",
        result[
            "stress_brier_skill_score"
        ],
    )

    print()
    print(
        "Realized stress dates:",
        result[
            "n_stress_dates"
        ],
    )

    print(
        "Stress-date CRPS:",
        result[
            "stress_crps_mean"
        ],
    )

    print(
        "Stress-date Energy:",
        result[
            "stress_energy_score"
        ],
    )

    print()
    print(
        "Saved:",
        output_path,
    )

    print()
    print(
        "TEST SPLIT WAS NOT EVALUATED."
    )


if __name__ == "__main__":
    main()