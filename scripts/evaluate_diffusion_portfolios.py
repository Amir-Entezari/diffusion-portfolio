"""Construct portfolios from frozen diffusion test scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from diffusion_portfolio.config import load_config
from diffusion_portfolio.data import load_daily_risk_free
from diffusion_portfolio.evaluation import (
    evaluate_portfolio,
    evaluate_rebalanced_portfolio,
)
from diffusion_portfolio.portfolio import (
    solve_long_only_mean_cvar,
    solve_long_only_minimum_variance,
    solve_long_only_tangency,
)


METHODS = (
    "diffusion_min_variance",
    "diffusion_tangency",
    "diffusion_mean_cvar",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-dir",
        required=True,
    )

    return parser.parse_args()


def metric_dict(metrics) -> dict:
    return {
        "annualized_return": float(
            metrics.annualized_return
        ),
        "annualized_volatility": float(
            metrics.annualized_volatility
        ),
        "sharpe_ratio": float(
            metrics.sharpe_ratio
        ),
        "max_drawdown": float(
            metrics.max_drawdown
        ),
        "calmar_ratio": float(
            metrics.calmar_ratio
        ),
        "average_monthly_turnover": float(
            metrics.average_monthly_turnover
        ),
    }


def align_risk_free(
    dates: pd.DatetimeIndex,
) -> np.ndarray:
    rf = load_daily_risk_free()

    series = pd.Series(
        rf.returns,
        index=rf.dates,
    )

    aligned = series.reindex(
        dates
    )

    if aligned.isna().any():
        raise ValueError(
            "Risk-free series does not cover every test date"
        )

    return aligned.to_numpy(
        dtype=np.float64
    )


def construct_weights(
    scenarios: np.ndarray,
    method: str,
) -> np.ndarray:
    n_time, _, n_assets = (
        scenarios.shape
    )

    weights = np.empty(
        (n_time, n_assets),
        dtype=np.float64,
    )

    for t in tqdm(
        range(n_time),
        desc=method,
    ):
        scenarios_t = (
            scenarios[t]
            .astype(
                np.float64,
                copy=False,
            )
        )

        mean = scenarios_t.mean(
            axis=0
        )

        covariance = np.cov(
            scenarios_t,
            rowvar=False,
            ddof=1,
        )

        if (
            method
            == "diffusion_min_variance"
        ):
            weights[t] = (
                solve_long_only_minimum_variance(
                    covariance
                )
            )

        elif (
            method
            == "diffusion_tangency"
        ):
            weights[t] = (
                solve_long_only_tangency(
                    mean,
                    covariance,
                )
            )

        elif (
            method
            == "diffusion_mean_cvar"
        ):
            # Match our historical Mean-CVaR convention:
            # require at least the equal-weight expected
            # return under the same information set.
            equal_weight_expected_return = float(
                mean.mean()
            )

            weights[t] = (
                solve_long_only_mean_cvar(
                    scenarios_t,
                    confidence_level=0.95,
                    minimum_expected_return=(
                        equal_weight_expected_return
                    ),
                )
            )

        else:
            raise ValueError(
                f"Unknown method: {method}"
            )

    return weights


def main() -> None:
    args = parse_args()

    run_dir = Path(
        args.run_dir
    )

    config_path = (
        run_dir
        / "config.yaml"
    )

    scenario_path = (
        run_dir
        / "test_scenarios.npz"
    )

    cfg = load_config(
        config_path
    )

    archive = np.load(
        scenario_path,
        allow_pickle=False,
    )

    scenarios = archive[
        "scenarios"
    ].astype(
        np.float64
    )

    observed_excess = archive[
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
        str(column)
        for column
        in archive[
            "columns"
        ]
    )

    if scenarios.ndim != 3:
        raise ValueError(
            "scenarios must have shape "
            "[time, scenarios, assets]"
        )

    if (
        observed_excess.shape
        != (
            scenarios.shape[0],
            scenarios.shape[2],
        )
    ):
        raise ValueError(
            "Observed-return shape does not match scenarios"
        )

    print("=" * 72)
    print("DIFFUSION PORTFOLIO EVALUATION")
    print("=" * 72)

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

    # ---------------------------------------------------------
    # Total returns required only for realistic weight drift
    # in transaction-cost backtesting.
    # ---------------------------------------------------------
    rf = align_risk_free(
        dates
    )

    observed_total = (
        observed_excess
        + rf[:, None]
    )

    all_weights = {}

    results = {}

    for method in METHODS:
        print()
        print("=" * 72)
        print(method)
        print("=" * 72)

        weights = construct_weights(
            scenarios,
            method,
        )

        all_weights[
            method
        ] = weights.astype(
            np.float32
        )

        # -----------------------------------------------------
        # Daily frictionless benchmark
        # -----------------------------------------------------
        frictionless = (
            evaluate_portfolio(
                observed_excess,
                weights,
                dates,
                annualization_factor=(
                    cfg.portfolio
                    .annualization_factor
                ),
                allow_short=False,
            )
        )

        # -----------------------------------------------------
        # Same robustness convention as Stage 4:
        # rebalance every 5 trading days, 10 bps per
        # one-way turnover.
        # -----------------------------------------------------
        friction = (
            evaluate_rebalanced_portfolio(
                asset_excess_returns=(
                    observed_excess
                ),
                asset_total_returns=(
                    observed_total
                ),
                target_weights=weights,
                dates=dates,
                rebalance_interval=5,
                transaction_cost_bps=(
                    cfg.portfolio
                    .transaction_cost_bps
                ),
                annualization_factor=(
                    cfg.portfolio
                    .annualization_factor
                ),
                allow_short=False,
            )
        )

        result = {
            "frictionless": (
                metric_dict(
                    frictionless.metrics
                )
            ),
            "five_day_gross": (
                metric_dict(
                    friction.gross_metrics
                )
            ),
            "five_day_net": (
                metric_dict(
                    friction.net_metrics
                )
            ),
            "five_day_total_cost": float(
                friction.transaction_costs.sum()
            ),
            "five_day_total_turnover": float(
                friction.one_way_turnover.sum()
            ),
        }

        results[
            method
        ] = result

        print(
            "Frictionless Sharpe:",
            result[
                "frictionless"
            ][
                "sharpe_ratio"
            ],
        )

        print(
            "5-day gross Sharpe:",
            result[
                "five_day_gross"
            ][
                "sharpe_ratio"
            ],
        )

        print(
            "5-day net Sharpe:",
            result[
                "five_day_net"
            ][
                "sharpe_ratio"
            ],
        )

        print(
            "5-day monthly turnover:",
            result[
                "five_day_net"
            ][
                "average_monthly_turnover"
            ],
        )

        print(
            "5-day total transaction cost:",
            result[
                "five_day_total_cost"
            ],
        )

    # ---------------------------------------------------------
    # Persist exact target weights.
    # ---------------------------------------------------------
    np.savez(
        run_dir
        / "test_diffusion_weights.npz",
        dates=dates.values.astype(
            "datetime64[D]"
        ),
        columns=np.asarray(
            columns
        ),
        **all_weights,
    )

    output = {
        "test_start": str(
            dates[0].date()
        ),
        "test_end": str(
            dates[-1].date()
        ),
        "n_test": int(
            len(dates)
        ),
        "n_scenarios": int(
            scenarios.shape[1]
        ),
        "transaction_cost_bps": float(
            cfg.portfolio
            .transaction_cost_bps
        ),
        "rebalance_interval": 5,
        "methods": results,
    }

    output_path = (
        run_dir
        / "test_portfolio_metrics.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print()
    print("=" * 72)
    print("PORTFOLIO EVALUATION COMPLETE")
    print("=" * 72)

    print(
        "Metrics:",
        output_path,
    )

    print(
        "Weights:",
        run_dir
        / "test_diffusion_weights.npz",
    )


if __name__ == "__main__":
    main()