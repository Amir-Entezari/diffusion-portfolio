"""Paired multi-seed confirmation for the retained Neural CDE."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from diffusion_portfolio.config import load_config
from diffusion_portfolio.data.preparation import load_standardizer
from diffusion_portfolio.utils.research_checks import check_checkpoint_pair, require_completed_run


METRICS = (
    "validation_loss",
    "crps",
    "energy",
    "mean_abs_ace",
)


def run_command(
    command: list[str],
    *,
    cwd: Path,
) -> None:
    print()
    print("$", " ".join(command))

    subprocess.run(
        command,
        cwd=cwd,
        check=True,
    )


def make_seed_config(
    base_config: Path,
    destination: Path,
    *,
    seed: int,
) -> None:
    with base_config.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(handle)

    config["seed"] = seed

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with destination.open(
        "w",
        encoding="utf-8",
    ) as handle:
        yaml.safe_dump(
            config,
            handle,
            sort_keys=False,
        )


def mean_abs_ace(
    probabilistic_metrics: dict,
) -> float:
    return float(
        np.mean(
            [
                abs(item["ace"])
                for item
                in probabilistic_metrics[
                    "calibration"
                ]
            ]
        )
    )


def read_result(
    run_dir: Path,
    *,
    architecture: str,
    seed: int,
) -> dict:
    summary_path = (
        run_dir
        / "summary.json"
    )

    metrics_path = (
        run_dir
        / "validation_probabilistic_metrics.json"
    )

    if not summary_path.exists():
        raise FileNotFoundError(
            summary_path
        )

    if not metrics_path.exists():
        raise FileNotFoundError(
            metrics_path
        )

    with summary_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        summary = json.load(handle)

    with metrics_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        metrics = json.load(handle)

    return {
        "architecture": architecture,
        "seed": seed,
        "best_epoch": int(
            summary["best_epoch"]
        ),
        "validation_loss": float(
            summary["best_val_loss"]
        ),
        "crps": float(
            metrics["crps_mean"]
        ),
        "energy": float(
            metrics["energy_score"]
        ),
        "mean_abs_ace": (
            mean_abs_ace(metrics)
        ),
    }


def ensure_run(
    *,
    repo_root: Path,
    base_config: Path,
    run_dir: Path,
    seed: int,
    device: str,
    config_dir: Path,
    architecture: str,
) -> None:
    summary_path = (
        run_dir
        / "summary.json"
    )

    metrics_path = (
        run_dir
        / "validation_probabilistic_metrics.json"
    )

    seed_config = (
        config_dir
        / f"{architecture}_seed_{seed}.yaml"
    )

    make_seed_config(
        base_config,
        seed_config,
        seed=seed,
    )

    if (run_dir / "config.yaml").exists():
        if load_config(run_dir / "config.yaml") != load_config(seed_config):
            raise RuntimeError(f"Existing run belongs to another configuration: {run_dir}")
        if summary_path.exists():
            require_completed_run(run_dir, seed_config, validation=metrics_path.exists())
    if not summary_path.exists():
        check_checkpoint_pair(run_dir)

    if not summary_path.exists():
        print()
        print(
            "=" * 72
        )
        print(
            f"TRAINING {architecture.upper()} "
            f"SEED {seed}"
        )
        print(
            "=" * 72
        )

        run_command(
            [
                sys.executable,
                "-u",
                "scripts/train_diffusion.py",
                "--config",
                str(seed_config),
                "--output-dir",
                str(run_dir),
                "--device",
                device,
                "--resume",
            ],
            cwd=repo_root,
        )

    else:
        print(
            f"Training already complete: "
            f"{architecture} seed {seed}"
        )

    if not metrics_path.exists():
        print()
        print(
            f"Evaluating validation: "
            f"{architecture} seed {seed}"
        )

        run_command(
            [
                sys.executable,
                "-u",
                "scripts/evaluate_diffusion.py",
                "--run-dir",
                str(run_dir),
                "--device",
                device,
                "--split", "validation",
            ],
            cwd=repo_root,
        )

    else:
        print(
            f"Validation evaluation already complete: "
            f"{architecture} seed {seed}"
        )
    if (run_dir / "config.yaml").exists():
        require_completed_run(run_dir, seed_config, validation=True)


def aggregate(
    rows: list[dict],
) -> tuple[
    pd.DataFrame,
    dict,
]:
    frame = pd.DataFrame(rows)

    frame = frame.sort_values(
        [
            "seed",
            "architecture",
        ]
    ).reset_index(
        drop=True
    )

    architecture_summary = {}

    for architecture in (
        "vanilla",
        "cde",
    ):
        subset = frame[
            frame[
                "architecture"
            ]
            == architecture
        ]

        architecture_summary[
            architecture
        ] = {}

        for metric in METRICS:
            values = (
                subset[metric]
                .to_numpy(
                    dtype=float
                )
            )

            architecture_summary[
                architecture
            ][
                metric
            ] = {
                "mean": float(
                    values.mean()
                ),
                "std": float(
                    values.std(
                        ddof=1
                    )
                )
                if len(values) > 1
                else 0.0,
            }

    paired = []

    seeds = sorted(
        set(
            frame["seed"].tolist()
        )
    )

    for seed in seeds:
        vanilla = frame[
            (
                frame["seed"]
                == seed
            )
            & (
                frame[
                    "architecture"
                ]
                == "vanilla"
            )
        ]

        cde = frame[
            (
                frame["seed"]
                == seed
            )
            & (
                frame[
                    "architecture"
                ]
                == "cde"
            )
        ]

        if (
            len(vanilla) != 1
            or len(cde) != 1
        ):
            raise RuntimeError(
                f"Missing paired result "
                f"for seed {seed}"
            )

        vanilla_row = (
            vanilla.iloc[0]
        )

        cde_row = (
            cde.iloc[0]
        )

        result = {
            "seed": int(seed),
        }

        for metric in METRICS:
            result[
                f"{metric}_delta"
            ] = float(
                cde_row[metric]
                - vanilla_row[metric]
            )

            result[
                f"{metric}_cde_wins"
            ] = bool(
                cde_row[metric]
                < vanilla_row[metric]
            )

        paired.append(
            result
        )

    paired_frame = pd.DataFrame(
        paired
    )

    paired_summary = {}

    for metric in METRICS:
        deltas = paired_frame[
            f"{metric}_delta"
        ].to_numpy(
            dtype=float
        )

        wins = paired_frame[
            f"{metric}_cde_wins"
        ].to_numpy(
            dtype=bool
        )

        paired_summary[
            metric
        ] = {
            "mean_delta_cde_minus_vanilla":
                float(
                    deltas.mean()
                ),
            "std_delta":
                float(
                    deltas.std(
                        ddof=1
                    )
                )
                if len(deltas) > 1
                else 0.0,
            "cde_wins": int(
                wins.sum()
            ),
            "n_seeds": int(
                len(wins)
            ),
        }

    result = {
        "architectures":
            architecture_summary,
        "paired":
            paired_summary,
    }

    return (
        frame,
        {
            "summary": result,
            "paired_rows":
                paired_frame.to_dict(
                    orient="records"
                ),
        },
    )


def run_multiseed(args, *, repo_root) -> None:
    # The CLI supplies the checkout root. Resolve caller paths before changing
    # subprocess cwd so relative outputs refer to the same files in both places.
    repo_root = Path(repo_root).resolve()

    output_root = Path(
        args.output_root
    ).resolve()

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_dir = (
        output_root
        / "configs"
    )

    vanilla42 = Path(
        args.seed42_vanilla_dir
    ).resolve()

    cde42 = Path(
        args.seed42_cde_dir
    ).resolve()

    vanilla_base = repo_root / (getattr(args, "vanilla_config", None) or "configs/mvp.yaml")
    cde_base = repo_root / (getattr(args, "cde_config", None) or "configs/cde.yaml")
    explicit_configs = getattr(args, "vanilla_config", None) or getattr(args, "cde_config", None)
    if not explicit_configs:
        for reference in (vanilla42, cde42):
            config = reference / "config.yaml"
            if config.is_file() and load_config(config).data.dataset != "ken_french_12":
                raise ValueError("Non-KF12 references require --vanilla-config and --cde-config; KF12 defaults are unsafe")
    if explicit_configs:
        vanilla_cfg = require_completed_run(vanilla42, vanilla_base, validation=True)
        cde_cfg = require_completed_run(cde42, cde_base, validation=True)
        vanilla_scaler = load_standardizer(vanilla42 / "standardizer.npz")
        cde_scaler = load_standardizer(cde42 / "standardizer.npz")
        if (vanilla_scaler.columns != cde_scaler.columns
                or not np.array_equal(vanilla_scaler.mean, cde_scaler.mean)
                or not np.array_equal(vanilla_scaler.std, cde_scaler.std)):
            raise ValueError("Seed-42 references do not have matched training scalers")
        if (vanilla_cfg.data != cde_cfg.data or vanilla_cfg.seed != 42 or cde_cfg.seed != 42
                or vanilla_cfg.training != cde_cfg.training or vanilla_cfg.evaluation != cde_cfg.evaluation
                or vanilla_cfg.model.history_encoder != "mlp" or cde_cfg.model.history_encoder != "cde"):
            raise ValueError("Multi-seed references must be matched Vanilla/CDE seed-42 experiments")
    if 42 in args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Additional seeds must be unique and exclude the existing seed 42")

    print(
        "=" * 72
    )
    print(
        "CDE — PAIRED MULTI-SEED CONFIRMATION"
    )
    print(
        "=" * 72
    )

    print(
        "Existing seed 42 Vanilla:",
        vanilla42,
    )

    print(
        "Existing seed 42 CDE:",
        cde42,
    )

    print(
        "New paired seeds:",
        args.seeds,
    )

    rows = [
        read_result(
            vanilla42,
            architecture="vanilla",
            seed=42,
        ),
        read_result(
            cde42,
            architecture="cde",
            seed=42,
        ),
    ]

    for seed in args.seeds:
        vanilla_dir = (
            output_root
            / "vanilla"
            / f"seed_{seed}"
        )

        cde_dir = (
            output_root
            / "cde"
            / f"seed_{seed}"
        )

        ensure_run(
            repo_root=repo_root,
            base_config=vanilla_base,
            run_dir=vanilla_dir,
            seed=seed,
            device=args.device,
            config_dir=config_dir,
            architecture="vanilla",
        )

        rows.append(
            read_result(
                vanilla_dir,
                architecture="vanilla",
                seed=seed,
            )
        )

        ensure_run(
            repo_root=repo_root,
            base_config=cde_base,
            run_dir=cde_dir,
            seed=seed,
            device=args.device,
            config_dir=config_dir,
            architecture="cde",
        )

        rows.append(
            read_result(
                cde_dir,
                architecture="cde",
                seed=seed,
            )
        )

        if explicit_configs:
            for run_dir in (vanilla_dir, cde_dir):
                require_completed_run(run_dir, validation=True)
                reference_scaler = load_standardizer(vanilla42 / "standardizer.npz")
                scaler = load_standardizer(run_dir / "standardizer.npz")
                if (scaler.columns != reference_scaler.columns
                        or not np.array_equal(scaler.mean, reference_scaler.mean)
                        or not np.array_equal(scaler.std, reference_scaler.std)):
                    raise RuntimeError("Paired seeds do not use the same training normalization/source data")

    (
        per_seed,
        aggregate_result,
    ) = aggregate(
        rows
    )

    per_seed.to_csv(
        output_root
        / "per_seed_results.csv",
        index=False,
    )

    with (
        output_root
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            aggregate_result,
            handle,
            indent=2,
        )

    print()
    print(
        "=" * 72
    )
    print(
        "PER-SEED RESULTS"
    )
    print(
        "=" * 72
    )

    print(
        per_seed.to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 72
    )
    print(
        "PAIRED CDE - VANILLA SUMMARY"
    )
    print(
        "=" * 72
    )

    for metric, values in (
        aggregate_result[
            "summary"
        ][
            "paired"
        ].items()
    ):
        print(
            f"{metric}: "
            f"mean delta="
            f"{values['mean_delta_cde_minus_vanilla']:.9f}, "
            f"CDE wins="
            f"{values['cde_wins']}/"
            f"{values['n_seeds']}"
        )

    print()
    print(
        "Results saved to:",
        output_root,
    )

    print()
    print(
        "TEST SPLIT WAS NOT EVALUATED."
    )
