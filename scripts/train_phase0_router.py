"""Train Phase 0B2 residual routers on frozen Phase 0A."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from diffusion_portfolio.config import load_config
from diffusion_portfolio.data import (
    TrainStandardizer,
    build_window_datasets,
    collate_return_batch,
    load_daily_risk_free,
    load_kf12_daily,
    slice_return_table,
    to_excess_returns,
)
from diffusion_portfolio.evaluation import (
    evaluate_probabilistic_forecast,
)
from diffusion_portfolio.models.diffusion import (
    ConditionalDiffusionModel,
)
from diffusion_portfolio.models.evidential import (
    EvidentialRegimeHead,
)
from diffusion_portfolio.models.routing import (
    EvidentialResidualRouter,
    SoftmaxResidualRouter,
)
from diffusion_portfolio.utils.seed import set_global_seed


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cde-run-dir",
        required=True,
    )
    parser.add_argument(
        "--b1-run-dir",
        required=True,
    )
    parser.add_argument(
        "--config",
        default="configs/phase0_router.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default="/kaggle/working/phase0_router",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    return parser.parse_args()


def resolve_device(requested):
    if requested == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable"
        )

    return torch.device(requested)


def load_yaml(path):
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise TypeError(
            "router config must be a mapping"
        )

    return raw


def build_model(cfg, n_assets):
    return ConditionalDiffusionModel(
        lookback=cfg.data.lookback,
        n_assets=n_assets,
        condition_dim=cfg.model.condition_dim,
        history_hidden_dim=cfg.model.history_hidden_dim,
        diffusion_steps=cfg.model.diffusion_steps,
        schedule_type=cfg.model.schedule,
        prediction_type=cfg.model.prediction_type,
        channels=list(cfg.model.channels),
        time_embed_dim=cfg.model.time_embed_dim,
        n_res_blocks=cfg.model.n_res_blocks,
        history_encoder_type=cfg.model.history_encoder,
        cde_hidden_dim=cfg.model.cde_hidden_dim,
        cde_drift_hidden_dim=cfg.model.cde_drift_hidden_dim,
        cde_sensitivity_hidden_dim=(
            cfg.model.cde_sensitivity_hidden_dim
        ),
        cde_solver=cfg.model.cde_solver,
        cde_rtol=cfg.model.cde_rtol,
        cde_atol=cfg.model.cde_atol,
        cde_use_adjoint=cfg.model.cde_use_adjoint,
        cde_fixed_steps_per_interval=(
            cfg.model.cde_fixed_steps_per_interval
        ),
    )


def load_scaler(path):
    saved = np.load(
        path,
        allow_pickle=False,
    )

    return TrainStandardizer(
        mean=saved["mean"],
        std=saved["std"],
        columns=tuple(
            str(column)
            for column in saved["columns"]
        ),
    )


def prepare_datasets(cfg, scaler):
    assets = slice_return_table(
        load_kf12_daily(),
        start=cfg.data.sample_start,
        end=cfg.data.sample_end,
    )

    excess = to_excess_returns(
        assets,
        load_daily_risk_free(),
    )

    scaled = scaler.transform(excess)

    return build_window_datasets(
        scaled,
        excess,
        lookback=cfg.data.lookback,
        horizon=cfg.data.horizon,
        train_end=cfg.data.train_end,
        val_end=cfg.data.val_end,
        test_end=cfg.data.sample_end,
    )


@torch.no_grad()
def extract_conditions(
    encoder,
    dataset,
    *,
    batch_size,
    device,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_return_batch,
    )

    parts = []

    encoder.eval()

    for batch in tqdm(
        loader,
        desc="CDE embeddings",
        leave=False,
    ):
        history = batch["history"].to(device)

        parts.append(
            encoder(history).cpu()
        )

    return torch.cat(
        parts,
        dim=0,
    )


def model_targets(dataset):
    targets = dataset.model_windows.target

    if targets.ndim != 3 or targets.shape[1] != 1:
        raise RuntimeError(
            "Phase 0B2 expects horizon=1"
        )

    return torch.from_numpy(
        targets[:, 0, :]
    ).float()


def freeze_module(module):
    module.eval()

    for parameter in module.parameters():
        parameter.requires_grad_(False)


def routed_diffusion_loss(
    model,
    router,
    condition,
    target,
    timesteps,
    noise,
):
    x_t = model.noise_schedule.q_sample(
        target,
        timesteps,
        noise,
    )

    routed_condition = router(condition)

    prediction = model.score_network(
        x_t,
        timesteps,
        routed_condition,
    )

    training_target = model._make_training_target(
        target,
        noise,
        timesteps,
    )

    return F.mse_loss(
        prediction,
        training_target,
    )


@torch.no_grad()
def evaluate_router_loss(
    model,
    router,
    conditions,
    targets,
    *,
    batch_size,
    validation_seed,
    device,
):
    router.eval()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(validation_seed)

    total_loss = 0.0
    total_samples = 0

    for start in range(
        0,
        len(targets),
        batch_size,
    ):
        end = min(
            start + batch_size,
            len(targets),
        )

        condition = conditions[start:end].to(device)
        target = targets[start:end].to(device)

        n = len(target)

        timesteps = torch.randint(
            0,
            model.diffusion_steps,
            (n,),
            generator=generator,
            device="cpu",
        ).to(device)

        noise = torch.randn(
            (n, model.n_assets),
            generator=generator,
            device="cpu",
            dtype=target.dtype,
        ).to(device)

        loss = routed_diffusion_loss(
            model,
            router,
            condition,
            target,
            timesteps,
            noise,
        )

        total_loss += float(loss.cpu()) * n
        total_samples += n

    return total_loss / total_samples


def make_training_loader(
    conditions,
    targets,
    *,
    batch_size,
    seed,
):
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        TensorDataset(
            conditions,
            targets,
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )


def fit_router(
    model,
    router,
    train_conditions,
    train_targets,
    val_conditions,
    val_targets,
    *,
    epochs,
    batch_size,
    learning_rate,
    weight_decay,
    gradient_clip_norm,
    validation_batch_size,
    validation_seed,
    training_seed,
    device,
    label,
):
    set_global_seed(
        training_seed,
        deterministic=False,
    )

    router.to(device)

    trainable = [
        parameter
        for parameter in router.parameters()
        if parameter.requires_grad
    ]

    if not trainable:
        raise RuntimeError(
            "Router has no trainable parameters"
        )

    optimizer = torch.optim.AdamW(
        trainable,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    loader = make_training_loader(
        train_conditions,
        train_targets,
        batch_size=batch_size,
        seed=training_seed,
    )

    corruption_generator = torch.Generator(
        device="cpu"
    )
    corruption_generator.manual_seed(
        training_seed + 10_000
    )

    best_state = None
    best_epoch = -1
    best_val_loss = float("inf")
    history = []

    initial_val_loss = evaluate_router_loss(
        model,
        router,
        val_conditions,
        val_targets,
        batch_size=validation_batch_size,
        validation_seed=validation_seed,
        device=device,
    )

    print(
        f"{label} initial val loss: "
        f"{initial_val_loss:.9f}"
    )

    for epoch in range(1, epochs + 1):
        router.train()

        total_loss = 0.0
        total_samples = 0

        for condition, target in loader:
            condition = condition.to(device)
            target = target.to(device)

            n = len(target)

            timesteps = torch.randint(
                0,
                model.diffusion_steps,
                (n,),
                generator=corruption_generator,
                device="cpu",
            ).to(device)

            noise = torch.randn(
                (n, model.n_assets),
                generator=corruption_generator,
                device="cpu",
                dtype=target.dtype,
            ).to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            loss = routed_diffusion_loss(
                model,
                router,
                condition,
                target,
                timesteps,
                noise,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite router loss"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                trainable,
                gradient_clip_norm,
            )

            optimizer.step()

            total_loss += (
                float(loss.detach().cpu())
                * n
            )
            total_samples += n

        train_loss = (
            total_loss
            / total_samples
        )

        val_loss = evaluate_router_loss(
            model,
            router,
            val_conditions,
            val_targets,
            batch_size=validation_batch_size,
            validation_seed=validation_seed,
            device=device,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        print(
            f"{label} "
            f"{epoch:03d}/{epochs:03d} | "
            f"train={train_loss:.6f} | "
            f"val={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(
                router.state_dict()
            )

    if best_state is None:
        raise RuntimeError(
            "No best router state"
        )

    router.load_state_dict(best_state)

    return (
        router,
        history,
        best_epoch,
        best_val_loss,
    )


@torch.no_grad()
def sample_from_conditions(
    model,
    router,
    conditions,
    *,
    n_scenarios,
    batch_size,
    sampling_seed,
    device,
):
    router.eval()

    torch.manual_seed(sampling_seed)
    np.random.seed(sampling_seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(
            sampling_seed
        )

    forecast = np.empty(
        (
            len(conditions),
            n_scenarios,
            model.n_assets,
        ),
        dtype=np.float32,
    )

    offset = 0

    for start in tqdm(
        range(
            0,
            len(conditions),
            batch_size,
        ),
        desc="Validation sampling",
    ):
        end = min(
            start + batch_size,
            len(conditions),
        )

        condition = conditions[
            start:end
        ].to(device)

        routed_condition = router(
            condition
        )

        n = len(routed_condition)

        repeated_condition = (
            routed_condition.repeat_interleave(
                n_scenarios,
                dim=0,
            )
        )

        x = torch.randn(
            (
                n,
                n_scenarios,
                model.n_assets,
            ),
            device=device,
            dtype=routed_condition.dtype,
        ).reshape(
            n * n_scenarios,
            model.n_assets,
        )

        for step in reversed(
            range(model.diffusion_steps)
        ):
            timesteps = torch.full(
                (
                    n * n_scenarios,
                ),
                step,
                device=device,
                dtype=torch.long,
            )

            if step > 0:
                step_noise = torch.randn_like(x)
            else:
                step_noise = torch.zeros_like(x)

            x = model.reverse_step(
                x,
                timesteps,
                repeated_condition,
                noise=step_noise,
            )

        forecast[
            offset:
            offset + n
        ] = (
            x.reshape(
                n,
                n_scenarios,
                model.n_assets,
            )
            .cpu()
            .numpy()
        )

        offset += n

    return forecast


@torch.no_grad()
def routing_diagnostics(
    router,
    conditions,
    *,
    batch_size,
    device,
):
    router.eval()

    weights = []
    vacuities = []

    for start in range(
        0,
        len(conditions),
        batch_size,
    ):
        output = router.route(
            conditions[
                start:
                start + batch_size
            ].to(device)
        )

        weights.append(
            output.weights.cpu()
        )

        if output.vacuity is not None:
            vacuities.append(
                output.vacuity.cpu()
            )

    weights = torch.cat(
        weights,
        dim=0,
    )

    entropy = (
        -weights
        * torch.log(
            weights.clamp_min(1e-8)
        )
    ).sum(dim=-1)

    result = {
        "mean_weights": [
            float(value)
            for value
            in weights.mean(dim=0)
        ],
        "mean_entropy": float(
            entropy.mean()
        ),
    }

    if vacuities:
        vacuity = torch.cat(
            vacuities,
            dim=0,
        )

        result[
            "mean_vacuity"
        ] = float(
            vacuity.mean()
        )

        if isinstance(
            router,
            EvidentialResidualRouter,
        ):
            blend = torch.sigmoid(
                router.transition_steepness
                * (
                    vacuity
                    - router.uncertainty_threshold
                )
            )

            result[
                "mean_uniform_blend"
            ] = float(
                blend.mean()
            )

    return result


def probability_metrics(
    forecast_raw,
    observed_raw,
    *,
    sampling_seed,
    n_scenarios,
):
    metrics = evaluate_probabilistic_forecast(
        forecast_raw,
        observed_raw,
    )

    return {
        "split": "validation",
        "sampling_seed": sampling_seed,
        "n_validation": len(observed_raw),
        "n_scenarios": n_scenarios,
        "scenario_mean": float(
            forecast_raw.mean()
        ),
        "scenario_std": float(
            forecast_raw.std()
        ),
        "scenario_max_abs": float(
            np.abs(forecast_raw).max()
        ),
        "crps_mean": float(
            metrics.crps_mean
        ),
        "crps_std": float(
            metrics.crps_std
        ),
        "energy_score": float(
            metrics.energy_score
        ),
        "calibration": [
            {
                "level": float(level),
                "picp": float(picp),
                "ace": float(ace),
            }
            for level, picp, ace
            in zip(
                metrics.calibration.levels,
                metrics.calibration.picp,
                metrics.calibration.ace,
            )
        ],
    }


def mean_abs_ace(metrics):
    return float(
        np.mean([
            abs(item["ace"])
            for item
            in metrics["calibration"]
        ])
    )


def main():
    args = parse_args()

    cde_run_dir = Path(args.cde_run_dir)
    b1_run_dir = Path(args.b1_run_dir)
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = resolve_device(
        args.device
    )

    router_cfg = load_yaml(
        config_path
    )

    seed = int(
        router_cfg["seed"]
    )

    set_global_seed(
        seed,
        deterministic=False,
    )

    # ---------------------------------------------------------
    # Phase 0A base
    # ---------------------------------------------------------
    cde_cfg = load_config(
        cde_run_dir / "config.yaml"
    )

    with (
        cde_run_dir
        / "summary.json"
    ).open(
        "r",
        encoding="utf-8",
    ) as handle:
        cde_summary = json.load(handle)

    checkpoint = torch.load(
        cde_run_dir / "best.pt",
        map_location="cpu",
        weights_only=False,
    )

    if (
        checkpoint["epoch"]
        != cde_summary["best_epoch"]
    ):
        raise RuntimeError(
            "Phase 0A checkpoint mismatch"
        )

    scaler = load_scaler(
        cde_run_dir
        / "standardizer.npz"
    )

    datasets = prepare_datasets(
        cde_cfg,
        scaler,
    )

    model = build_model(
        cde_cfg,
        len(scaler.columns),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    freeze_module(model)

    print("=" * 72)
    print(
        "PHASE 0B2 — UNCERTAINTY-AWARE ROUTING"
    )
    print("=" * 72)

    print("Device:", device)
    print(
        "Phase 0A checkpoint:",
        checkpoint["epoch"],
        checkpoint["val_loss"],
    )

    # ---------------------------------------------------------
    # Frozen Phase 0A conditions
    # ---------------------------------------------------------
    extraction_batch_size = (
        cde_cfg.training.batch_size
    )

    print()
    print(
        "Extracting frozen CDE conditions..."
    )

    train_conditions = extract_conditions(
        model.history_encoder,
        datasets.train,
        batch_size=extraction_batch_size,
        device=device,
    )

    val_conditions = extract_conditions(
        model.history_encoder,
        datasets.val,
        batch_size=extraction_batch_size,
        device=device,
    )

    train_targets = model_targets(
        datasets.train
    )

    val_targets = model_targets(
        datasets.val
    )

    print(
        "Train conditions:",
        tuple(train_conditions.shape),
    )
    print(
        "Validation conditions:",
        tuple(val_conditions.shape),
    )

    # ---------------------------------------------------------
    # Phase 0B1 heads
    # ---------------------------------------------------------
    section = router_cfg["router"]

    n_experts = int(
        section["n_experts"]
    )

    expert_hidden_dim = int(
        section["expert_hidden_dim"]
    )

    gamma = float(
        section[
            "transition_steepness"
        ]
    )

    evidential_checkpoint = torch.load(
        b1_run_dir
        / "evidential_best.pt",
        map_location="cpu",
        weights_only=False,
    )

    softmax_checkpoint = torch.load(
        b1_run_dir
        / "softmax_best.pt",
        map_location="cpu",
        weights_only=False,
    )

    temporary_head = EvidentialRegimeHead(
        input_dim=cde_cfg.model.condition_dim,
        n_classes=n_experts,
    ).to(device)

    temporary_head.load_state_dict(
        evidential_checkpoint[
            "model_state_dict"
        ]
    )

    freeze_module(
        temporary_head
    )

    with torch.no_grad():
        train_vacuity = temporary_head(
            train_conditions.to(device)
        ).vacuity.cpu()

    tau = float(
        torch.median(train_vacuity)
    )

    print()
    print(
        "Train-only tau:",
        tau,
    )

    # ---------------------------------------------------------
    # Identically initialized routers
    # ---------------------------------------------------------
    set_global_seed(
        seed,
        deterministic=False,
    )

    softmax_router = SoftmaxResidualRouter(
        condition_dim=(
            cde_cfg.model.condition_dim
        ),
        n_experts=n_experts,
        expert_hidden_dim=(
            expert_hidden_dim
        ),
    )

    softmax_router.routing_head.load_state_dict(
        softmax_checkpoint[
            "model_state_dict"
        ]
    )

    freeze_module(
        softmax_router.routing_head
    )

    initial_experts = copy.deepcopy(
        softmax_router.experts.state_dict()
    )

    set_global_seed(
        seed,
        deterministic=False,
    )

    evidential_router = (
        EvidentialResidualRouter(
            condition_dim=(
                cde_cfg.model.condition_dim
            ),
            n_experts=n_experts,
            expert_hidden_dim=(
                expert_hidden_dim
            ),
            uncertainty_threshold=tau,
            transition_steepness=gamma,
        )
    )

    evidential_router.evidential_head.load_state_dict(
        evidential_checkpoint[
            "model_state_dict"
        ]
    )

    evidential_router.experts.load_state_dict(
        initial_experts
    )

    freeze_module(
        evidential_router.evidential_head
    )

    # ---------------------------------------------------------
    # Confirm both start exactly at Phase 0A
    # ---------------------------------------------------------
    validation_seed = (
        cde_cfg.training.validation_seed
    )

    validation_batch_size = (
        cde_cfg.training.batch_size
    )

    baseline_val = float(
        cde_summary["best_val_loss"]
    )

    softmax_initial = evaluate_router_loss(
        model,
        softmax_router.to(device),
        val_conditions,
        val_targets,
        batch_size=validation_batch_size,
        validation_seed=validation_seed,
        device=device,
    )

    evidential_initial = evaluate_router_loss(
        model,
        evidential_router.to(device),
        val_conditions,
        val_targets,
        batch_size=validation_batch_size,
        validation_seed=validation_seed,
        device=device,
    )

    print()
    print(
        "Phase 0A val loss:",
        baseline_val,
    )
    print(
        "Softmax initial:",
        softmax_initial,
    )
    print(
        "Evidential initial:",
        evidential_initial,
    )

    for name, value in [
        ("Softmax", softmax_initial),
        ("Evidential", evidential_initial),
    ]:
        if not np.isclose(
            value,
            baseline_val,
            rtol=1e-5,
            atol=1e-6,
        ):
            raise RuntimeError(
                f"{name} router does not "
                "start at Phase 0A"
            )

    # ---------------------------------------------------------
    # Train ONLY experts
    # ---------------------------------------------------------
    training = router_cfg["training"]

    common = dict(
        epochs=int(
            training["epochs"]
        ),
        batch_size=int(
            training["batch_size"]
        ),
        learning_rate=float(
            training["learning_rate"]
        ),
        weight_decay=float(
            training["weight_decay"]
        ),
        gradient_clip_norm=float(
            training[
                "gradient_clip_norm"
            ]
        ),
        validation_batch_size=(
            validation_batch_size
        ),
        validation_seed=(
            validation_seed
        ),
        training_seed=seed,
        device=device,
    )

    print()
    print(
        "Training Softmax MoE..."
    )

    (
        softmax_router,
        softmax_history,
        softmax_best_epoch,
        softmax_best_val,
    ) = fit_router(
        model,
        softmax_router,
        train_conditions,
        train_targets,
        val_conditions,
        val_targets,
        label="Softmax",
        **common,
    )

    print()
    print(
        "Training Evidential MoE..."
    )

    (
        evidential_router,
        evidential_history,
        evidential_best_epoch,
        evidential_best_val,
    ) = fit_router(
        model,
        evidential_router,
        train_conditions,
        train_targets,
        val_conditions,
        val_targets,
        label="Evidential",
        **common,
    )

    # ---------------------------------------------------------
    # Save training artifacts
    # ---------------------------------------------------------
    softmax_dir = (
        output_dir / "softmax_moe"
    )

    evidential_dir = (
        output_dir / "evidential_moe"
    )

    softmax_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    evidential_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "router_state_dict":
                softmax_router.state_dict(),
            "best_epoch":
                softmax_best_epoch,
            "best_val_loss":
                softmax_best_val,
        },
        softmax_dir / "best.pt",
    )

    torch.save(
        {
            "router_state_dict":
                evidential_router.state_dict(),
            "best_epoch":
                evidential_best_epoch,
            "best_val_loss":
                evidential_best_val,
            "uncertainty_threshold": tau,
            "transition_steepness": gamma,
        },
        evidential_dir / "best.pt",
    )

    pd.DataFrame(
        softmax_history
    ).to_csv(
        softmax_dir / "history.csv",
        index=False,
    )

    pd.DataFrame(
        evidential_history
    ).to_csv(
        evidential_dir / "history.csv",
        index=False,
    )

    shutil.copy2(
        config_path,
        output_dir / "config.yaml",
    )

    # ---------------------------------------------------------
    # Validation probabilistic forecasts
    # ---------------------------------------------------------
    n_scenarios = (
        cde_cfg.evaluation.n_scenarios
    )

    sampling_batch_size = int(
        router_cfg[
            "evaluation"
        ][
            "sampling_batch_size"
        ]
    )

    observed_raw = (
        datasets.val
        .raw_windows
        .target[
            :,
            0,
            :,
        ]
        .astype(
            np.float32,
            copy=True,
        )
    )

    dates = (
        datasets.val
        .raw_windows
        .target_dates
    )

    variant_results = {}

    variants = [
        (
            "softmax_moe",
            softmax_router,
            softmax_dir,
            softmax_best_epoch,
            softmax_best_val,
        ),
        (
            "evidential_moe",
            evidential_router,
            evidential_dir,
            evidential_best_epoch,
            evidential_best_val,
        ),
    ]

    for (
        name,
        router,
        variant_dir,
        best_epoch,
        best_val,
    ) in variants:
        print()
        print(
            f"Sampling {name}..."
        )

        standardized = sample_from_conditions(
            model,
            router,
            val_conditions,
            n_scenarios=n_scenarios,
            batch_size=sampling_batch_size,
            sampling_seed=validation_seed,
            device=device,
        )

        raw = scaler.inverse_transform(
            standardized
        )

        np.savez(
            variant_dir
            / "validation_scenarios.npz",
            scenarios=raw,
            observed=observed_raw,
            dates=dates.values.astype(
                "datetime64[D]"
            ),
            columns=np.asarray(
                scaler.columns
            ),
        )

        metrics = probability_metrics(
            raw,
            observed_raw,
            sampling_seed=validation_seed,
            n_scenarios=n_scenarios,
        )

        metrics[
            "best_router_epoch"
        ] = best_epoch

        metrics[
            "best_validation_loss"
        ] = best_val

        metrics["routing"] = (
            routing_diagnostics(
                router,
                val_conditions,
                batch_size=(
                    validation_batch_size
                ),
                device=device,
            )
        )

        with (
            variant_dir
            / "validation_probabilistic_metrics.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                metrics,
                handle,
                indent=2,
            )

        variant_results[name] = metrics

    # ---------------------------------------------------------
    # Compare with Phase 0A
    # ---------------------------------------------------------
    with (
        cde_run_dir
        / "validation_probabilistic_metrics.json"
    ).open(
        "r",
        encoding="utf-8",
    ) as handle:
        baseline_metrics = json.load(
            handle
        )

    comparison = {
        "CDE": {
            "validation_loss":
                baseline_val,
            "crps":
                baseline_metrics[
                    "crps_mean"
                ],
            "energy":
                baseline_metrics[
                    "energy_score"
                ],
            "mean_abs_ace":
                mean_abs_ace(
                    baseline_metrics
                ),
        },
        "Softmax MoE": {
            "validation_loss":
                softmax_best_val,
            "crps":
                variant_results[
                    "softmax_moe"
                ][
                    "crps_mean"
                ],
            "energy":
                variant_results[
                    "softmax_moe"
                ][
                    "energy_score"
                ],
            "mean_abs_ace":
                mean_abs_ace(
                    variant_results[
                        "softmax_moe"
                    ]
                ),
        },
        "Evidential MoE": {
            "validation_loss":
                evidential_best_val,
            "crps":
                variant_results[
                    "evidential_moe"
                ][
                    "crps_mean"
                ],
            "energy":
                variant_results[
                    "evidential_moe"
                ][
                    "energy_score"
                ],
            "mean_abs_ace":
                mean_abs_ace(
                    variant_results[
                        "evidential_moe"
                    ]
                ),
        },
        "tau": tau,
        "gamma": gamma,
    }

    with (
        output_dir
        / "comparison.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            comparison,
            handle,
            indent=2,
        )

    table = pd.DataFrame(
        {
            name: [
                values["validation_loss"],
                values["crps"],
                values["energy"],
                values["mean_abs_ace"],
            ]
            for name, values
            in comparison.items()
            if isinstance(values, dict)
        },
        index=[
            "Validation loss ↓",
            "CRPS ↓",
            "Energy Score ↓",
            "Mean |ACE| ↓",
        ],
    )

    print()
    print("=" * 72)
    print(
        "PHASE 0B2 VALIDATION COMPARISON"
    )
    print("=" * 72)

    print(
        table.to_string()
    )

    print()
    print(
        "Softmax routing:"
    )
    print(
        json.dumps(
            variant_results[
                "softmax_moe"
            ][
                "routing"
            ],
            indent=2,
        )
    )

    print()
    print(
        "Evidential routing:"
    )
    print(
        json.dumps(
            variant_results[
                "evidential_moe"
            ][
                "routing"
            ],
            indent=2,
        )
    )

    print()
    print(
        "TEST SPLIT WAS NOT EVALUATED."
    )


if __name__ == "__main__":
    main()