"""Diagnose pathological tails in frozen Diffolio scenario forecasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


QUANTILE_LEVELS = (
    0.00001,
    0.0001,
    0.001,
    0.01,
    0.50,
    0.99,
    0.999,
    0.9999,
    0.99999,
)

ABS_THRESHOLDS = (
    0.10,
    0.20,
    0.50,
    1.00,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path(
            "artifacts/diffolio/val_scenarios.npz"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
    )

    return parser.parse_args()


def clean_float(value) -> float:
    return float(
        np.asarray(value).item()
    )


def main() -> None:
    args = parse_args()

    if not args.scenarios.exists():
        raise FileNotFoundError(
            args.scenarios
        )

    if args.top_k <= 0:
        raise ValueError(
            "top-k must be positive"
        )

    archive = np.load(
        args.scenarios,
        allow_pickle=False,
    )

    scenarios = archive[
        "scenarios"
    ].astype(
        np.float64
    )

    observed = archive[
        "observed"
    ].astype(
        np.float64
    )

    dates = pd.DatetimeIndex(
        archive[
            "dates"
        ]
    )

    columns = tuple(
        str(value)
        for value
        in archive[
            "columns"
        ]
    )

    if scenarios.ndim != 3:
        raise ValueError(
            "scenarios must have shape "
            "[dates, scenarios, assets]"
        )

    n_dates, n_scenarios, n_assets = (
        scenarios.shape
    )

    if observed.shape != (
        n_dates,
        n_assets,
    ):
        raise ValueError(
            "observed shape does not match scenarios"
        )

    if len(dates) != n_dates:
        raise ValueError(
            "date count does not match scenarios"
        )

    if len(columns) != n_assets:
        raise ValueError(
            "asset count does not match scenarios"
        )

    if not np.isfinite(
        scenarios
    ).all():
        raise ValueError(
            "scenarios contain non-finite values"
        )

    if not np.isfinite(
        observed
    ).all():
        raise ValueError(
            "observed contains non-finite values"
        )

    flat = scenarios.reshape(
        -1
    )

    abs_flat = np.abs(
        flat
    )

    output_dir = (
        args.scenarios.parent
    )

    stem = (
        args.scenarios.stem
        .replace(
            "_scenarios",
            "",
        )
    )

    print(
        "=" * 76
    )
    print(
        "DIFFOLIO SCENARIO DIAGNOSTICS"
    )
    print(
        "=" * 76
    )

    print(
        "Scenario file:",
        args.scenarios,
    )

    print(
        "Scenario cube:",
        scenarios.shape,
    )

    print(
        "Dates:",
        dates[0],
        "to",
        dates[-1],
    )

    print(
        "Assets:",
        n_assets,
    )

    print()

    print(
        "=" * 76
    )
    print(
        "GLOBAL DISTRIBUTION"
    )
    print(
        "=" * 76
    )

    global_stats = {
        "scenario_mean": clean_float(
            flat.mean()
        ),
        "scenario_median": clean_float(
            np.median(
                flat
            )
        ),
        "scenario_std": clean_float(
            flat.std(
                ddof=0
            )
        ),
        "scenario_min": clean_float(
            flat.min()
        ),
        "scenario_max": clean_float(
            flat.max()
        ),
        "scenario_max_abs": clean_float(
            abs_flat.max()
        ),
        "observed_mean": clean_float(
            observed.mean()
        ),
        "observed_std": clean_float(
            observed.std(
                ddof=0
            )
        ),
        "observed_min": clean_float(
            observed.min()
        ),
        "observed_max": clean_float(
            observed.max()
        ),
        "observed_max_abs": clean_float(
            np.abs(
                observed
            ).max()
        ),
    }

    for key, value in global_stats.items():
        print(
            f"{key:24s}: "
            f"{value: .8f}"
        )

    print()

    print(
        "=" * 76
    )
    print(
        "SCENARIO QUANTILES"
    )
    print(
        "=" * 76
    )

    quantile_values = np.quantile(
        flat,
        QUANTILE_LEVELS,
    )

    quantile_dict = {}

    for level, value in zip(
        QUANTILE_LEVELS,
        quantile_values,
    ):
        quantile_dict[
            str(
                level
            )
        ] = clean_float(
            value
        )

        print(
            f"{100.0 * level:10.5f}%"
            f" : {value: .8f}"
        )

    print()

    print(
        "=" * 76
    )
    print(
        "EXTREME-RETURN FREQUENCY"
    )
    print(
        "=" * 76
    )

    date_max_abs = np.max(
        np.abs(
            scenarios
        ),
        axis=(1, 2),
    )

    scenario_vector_max_abs = np.max(
        np.abs(
            scenarios
        ),
        axis=2,
    )

    threshold_stats = {}

    for threshold in ABS_THRESHOLDS:
        element_fraction = np.mean(
            abs_flat
            > threshold
        )

        date_fraction = np.mean(
            date_max_abs
            > threshold
        )

        vector_fraction = np.mean(
            scenario_vector_max_abs
            > threshold
        )

        threshold_stats[
            str(
                threshold
            )
        ] = {
            "element_fraction": (
                clean_float(
                    element_fraction
                )
            ),
            "scenario_vector_fraction": (
                clean_float(
                    vector_fraction
                )
            ),
            "date_fraction": (
                clean_float(
                    date_fraction
                )
            ),
        }

        print(
            f"|r| > {threshold:4.0%}: "
            f"elements={element_fraction:10.6%}  "
            f"scenario-vectors="
            f"{vector_fraction:10.6%}  "
            f"dates={date_fraction:10.6%}"
        )

    print()

    print(
        "=" * 76
    )
    print(
        "WHAT HAPPENS IF EXTREME VALUES ARE EXCLUDED?"
    )
    print(
        "=" * 76
    )

    filtered_stats = {}

    for threshold in (
        0.10,
        0.20,
        0.50,
    ):
        mask = (
            abs_flat
            <= threshold
        )

        filtered = flat[
            mask
        ]

        retained = (
            filtered.size
            / flat.size
        )

        stats = {
            "retained_fraction": (
                clean_float(
                    retained
                )
            ),
            "mean": clean_float(
                filtered.mean()
            ),
            "std": clean_float(
                filtered.std(
                    ddof=0
                )
            ),
        }

        filtered_stats[
            str(
                threshold
            )
        ] = stats

        print(
            f"|r| <= {threshold:4.0%}: "
            f"retained={retained:10.6%}  "
            f"mean={stats['mean']: .8f}  "
            f"std={stats['std']: .8f}"
        )

    # ---------------------------------------------------------
    # Per-date diagnostics
    # ---------------------------------------------------------
    date_generated_mean = scenarios.mean(
        axis=(1, 2)
    )

    date_generated_std = scenarios.std(
        axis=(1, 2),
        ddof=0,
    )

    date_observed_mean = observed.mean(
        axis=1
    )

    date_observed_std = observed.std(
        axis=1,
        ddof=0,
    )

    date_frame = pd.DataFrame(
        {
            "date": dates,
            "generated_mean": (
                date_generated_mean
            ),
            "generated_std": (
                date_generated_std
            ),
            "generated_max_abs": (
                date_max_abs
            ),
            "observed_mean": (
                date_observed_mean
            ),
            "observed_std": (
                date_observed_std
            ),
            "observed_max_abs": (
                np.max(
                    np.abs(
                        observed
                    ),
                    axis=1,
                )
            ),
        }
    )

    extreme_dates = (
        date_frame
        .sort_values(
            "generated_max_abs",
            ascending=False,
        )
        .head(
            args.top_k
        )
        .reset_index(
            drop=True
        )
    )

    extreme_dates_path = (
        output_dir
        / f"{stem}_extreme_dates.csv"
    )

    date_frame.to_csv(
        output_dir
        / f"{stem}_date_diagnostics.csv",
        index=False,
    )

    extreme_dates.to_csv(
        extreme_dates_path,
        index=False,
    )

    print()

    print(
        "=" * 76
    )
    print(
        f"TOP {args.top_k} DATES BY GENERATED |RETURN|"
    )
    print(
        "=" * 76
    )

    print(
        extreme_dates.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # Per-asset diagnostics
    # ---------------------------------------------------------
    asset_rows = []

    for asset_index, asset in enumerate(
        columns
    ):
        asset_values = scenarios[
            :,
            :,
            asset_index,
        ]

        asset_observed = observed[
            :,
            asset_index,
        ]

        row = {
            "asset": asset,
            "generated_mean": clean_float(
                asset_values.mean()
            ),
            "generated_std": clean_float(
                asset_values.std(
                    ddof=0
                )
            ),
            "generated_min": clean_float(
                asset_values.min()
            ),
            "generated_max": clean_float(
                asset_values.max()
            ),
            "generated_max_abs": clean_float(
                np.abs(
                    asset_values
                ).max()
            ),
            "observed_mean": clean_float(
                asset_observed.mean()
            ),
            "observed_std": clean_float(
                asset_observed.std(
                    ddof=0
                )
            ),
            "fraction_abs_gt_10pct": clean_float(
                np.mean(
                    np.abs(
                        asset_values
                    )
                    > 0.10
                )
            ),
            "fraction_abs_gt_20pct": clean_float(
                np.mean(
                    np.abs(
                        asset_values
                    )
                    > 0.20
                )
            ),
            "fraction_abs_gt_50pct": clean_float(
                np.mean(
                    np.abs(
                        asset_values
                    )
                    > 0.50
                )
            ),
            "fraction_abs_gt_100pct": clean_float(
                np.mean(
                    np.abs(
                        asset_values
                    )
                    > 1.00
                )
            ),
        }

        asset_rows.append(
            row
        )

    asset_frame = (
        pd.DataFrame(
            asset_rows
        )
        .sort_values(
            "generated_max_abs",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    asset_path = (
        output_dir
        / f"{stem}_asset_diagnostics.csv"
    )

    asset_frame.to_csv(
        asset_path,
        index=False,
    )

    print()

    print(
        "=" * 76
    )
    print(
        "ASSET DIAGNOSTICS"
    )
    print(
        "=" * 76
    )

    print(
        asset_frame.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # Individual most-extreme generated values
    # ---------------------------------------------------------
    top_k = min(
        args.top_k,
        flat.size,
    )

    flat_indices = np.argpartition(
        abs_flat,
        -top_k,
    )[
        -top_k:
    ]

    flat_indices = flat_indices[
        np.argsort(
            abs_flat[
                flat_indices
            ]
        )[
            ::-1
        ]
    ]

    unravelled = np.unravel_index(
        flat_indices,
        scenarios.shape,
    )

    extreme_rows = []

    for (
        date_index,
        scenario_index,
        asset_index,
    ) in zip(
        *unravelled
    ):
        value = scenarios[
            date_index,
            scenario_index,
            asset_index,
        ]

        extreme_rows.append(
            {
                "date": (
                    dates[
                        date_index
                    ]
                ),
                "scenario_index": int(
                    scenario_index
                ),
                "asset": columns[
                    asset_index
                ],
                "return": clean_float(
                    value
                ),
                "abs_return": clean_float(
                    abs(
                        value
                    )
                ),
                "observed_return": clean_float(
                    observed[
                        date_index,
                        asset_index,
                    ]
                ),
            }
        )

    extreme_samples = pd.DataFrame(
        extreme_rows
    )

    extreme_samples_path = (
        output_dir
        / f"{stem}_extreme_samples.csv"
    )

    extreme_samples.to_csv(
        extreme_samples_path,
        index=False,
    )

    print()

    print(
        "=" * 76
    )
    print(
        f"TOP {top_k} INDIVIDUAL GENERATED VALUES"
    )
    print(
        "=" * 76
    )

    print(
        extreme_samples.to_string(
            index=False
        )
    )

    summary = {
        "scenario_file": str(
            args.scenarios
        ),
        "shape": {
            "dates": n_dates,
            "scenarios_per_date": (
                n_scenarios
            ),
            "assets": n_assets,
        },
        "date_start": str(
            dates[
                0
            ].date()
        ),
        "date_end": str(
            dates[
                -1
            ].date()
        ),
        "global": global_stats,
        "quantiles": quantile_dict,
        "thresholds": threshold_stats,
        "filtered": filtered_stats,
    }

    summary_path = (
        output_dir
        / f"{stem}_scenario_diagnostics.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    print()

    print(
        "=" * 76
    )
    print(
        "SAVED"
    )
    print(
        "=" * 76
    )

    print(
        "Summary:",
        summary_path,
    )

    print(
        "Date diagnostics:",
        output_dir
        / f"{stem}_date_diagnostics.csv",
    )

    print(
        "Extreme dates:",
        extreme_dates_path,
    )

    print(
        "Asset diagnostics:",
        asset_path,
    )

    print(
        "Extreme samples:",
        extreme_samples_path,
    )


if __name__ == "__main__":
    main()