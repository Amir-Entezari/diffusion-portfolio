"""Tiny real-data overfit test for the vanilla conditional DDPM.

This is a DEBUGGING experiment, not a scientific validation experiment.

The same small subset is deliberately used for training and validation.
The goal is to establish that:
1. the real KF12 pipeline reaches the model correctly;
2. the DDPM objective can be optimized;
3. validation loss on seen examples falls materially;
4. reverse sampling remains finite after fitting.
"""

from __future__ import annotations

import numpy as np
import torch

from torch.utils.data import (
    DataLoader,
    Subset,
)

from diffusion_portfolio.config import (
    load_config,
)

from diffusion_portfolio.data import (
    TrainStandardizer,
    build_window_datasets,
    collate_return_batch,
    load_daily_risk_free,
    load_kf12_daily,
    slice_return_table,
    to_excess_returns,
)

from diffusion_portfolio.models.diffusion import (
    ConditionalDiffusionModel,
)

from diffusion_portfolio.training import (
    evaluate_diffusion_loss,
    fit_diffusion,
)

from diffusion_portfolio.utils.seed import (
    set_global_seed,
)


N_OVERFIT_SAMPLES = 32
OVERFIT_EPOCHS = 200
N_SAMPLE_SCENARIOS = 32


def main() -> None:
    # ---------------------------------------------------------
    # Configuration / reproducibility
    # ---------------------------------------------------------
    cfg = load_config(
        "configs/mvp.yaml"
    )

    set_global_seed(
        cfg.seed,
        deterministic=False,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    # ---------------------------------------------------------
    # Real KF12 excess-return pipeline
    # ---------------------------------------------------------
    assets = slice_return_table(
        load_kf12_daily(),
        start=cfg.data.sample_start,
        end=cfg.data.sample_end,
    )

    risk_free = (
        load_daily_risk_free()
    )

    excess = to_excess_returns(
        assets,
        risk_free,
    )

    scaler = TrainStandardizer.fit(
        excess,
        train_end=cfg.data.train_end,
    )

    scaled = scaler.transform(
        excess
    )

    datasets = build_window_datasets(
        scaled,
        excess,
        lookback=cfg.data.lookback,
        horizon=cfg.data.horizon,
        train_end=cfg.data.train_end,
        val_end=cfg.data.val_end,
        test_end=cfg.data.sample_end,
    )

    # ---------------------------------------------------------
    # Pick a fixed, spread-out random subset of TRAIN data.
    #
    # We deliberately reuse these SAME samples for validation.
    # That is appropriate only for this overfit diagnostic.
    # ---------------------------------------------------------
    rng = np.random.default_rng(
        cfg.seed
    )

    indices = rng.choice(
        len(datasets.train),
        size=N_OVERFIT_SAMPLES,
        replace=False,
    )

    indices = np.sort(
        indices
    ).tolist()

    tiny_dataset = Subset(
        datasets.train,
        indices,
    )

    train_generator = (
        torch.Generator()
    )

    train_generator.manual_seed(
        cfg.seed
    )

    train_loader = DataLoader(
        tiny_dataset,
        batch_size=N_OVERFIT_SAMPLES,
        shuffle=True,
        generator=train_generator,
        num_workers=0,
        collate_fn=collate_return_batch,
        drop_last=False,
    )

    val_loader = DataLoader(
        tiny_dataset,
        batch_size=N_OVERFIT_SAMPLES,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_return_batch,
        drop_last=False,
    )

    # ---------------------------------------------------------
    # Exact MVP architecture
    # ---------------------------------------------------------
    model = ConditionalDiffusionModel(
        lookback=cfg.data.lookback,
        n_assets=12,
        condition_dim=(
            cfg.model.condition_dim
        ),
        history_hidden_dim=(
            cfg.model.history_hidden_dim
        ),
        diffusion_steps=(
            cfg.model.diffusion_steps
        ),
        schedule_type=(
            cfg.model.schedule
        ),
        channels=list(
            cfg.model.channels
        ),
        time_embed_dim=(
            cfg.model.time_embed_dim
        ),
        n_res_blocks=(
            cfg.model.n_res_blocks
        ),
    )

    model.to(
        device
    )

    # ---------------------------------------------------------
    # Initial fixed-corruption validation loss
    # ---------------------------------------------------------
    initial_val_loss = (
        evaluate_diffusion_loss(
            model,
            val_loader,
            device=device,
            validation_seed=(
                cfg.training.validation_seed
            ),
        )
    )

    print()
    print(
        "Initial validation loss:",
        initial_val_loss,
    )

    # ---------------------------------------------------------
    # Deliberately overfit
    # ---------------------------------------------------------
    result = fit_diffusion(
        model,
        train_loader,
        val_loader,
        epochs=OVERFIT_EPOCHS,
        learning_rate=(
            cfg.training.learning_rate
        ),
        weight_decay=0.0,
        gradient_clip_norm=(
            cfg.training.gradient_clip_norm
        ),
        validation_seed=(
            cfg.training.validation_seed
        ),
        device=device,
        checkpoint_path=(
            "checkpoints/"
            "overfit_debug_best.pt"
        ),
        verbose=False,
    )

    final_record = (
        result.history[-1]
    )

    reduction_ratio = (
        result.best_val_loss
        / initial_val_loss
    )

    print()
    print("=" * 72)
    print(
        "TINY REAL-DATA OVERFIT TEST"
    )
    print("=" * 72)

    print(
        "Samples:",
        N_OVERFIT_SAMPLES,
    )

    print(
        "Epochs:",
        OVERFIT_EPOCHS,
    )

    print(
        "Initial val loss:",
        initial_val_loss,
    )

    print(
        "Best val loss:",
        result.best_val_loss,
    )

    print(
        "Best epoch:",
        result.best_epoch,
    )

    print(
        "Final train loss:",
        final_record.train_loss,
    )

    print(
        "Final val loss:",
        final_record.val_loss,
    )

    print(
        "Best / initial ratio:",
        reduction_ratio,
    )

    # Show a few points from the trajectory without printing
    # all 200 epochs.
    selected_epochs = {
        1,
        10,
        25,
        50,
        100,
        150,
        200,
    }

    print()
    print(
        "Selected loss trajectory:"
    )

    for record in result.history:
        if (
            record.epoch
            in selected_epochs
        ):
            print(
                f"epoch={record.epoch:3d} "
                f"train={record.train_loss:.6f} "
                f"val={record.val_loss:.6f}"
            )

    # ---------------------------------------------------------
    # Reverse-sampling sanity check AFTER fitting
    # ---------------------------------------------------------
    sample = datasets.train[
        indices[0]
    ]

    history = (
        sample["history"]
        .unsqueeze(0)
        .to(device)
    )

    model.eval()

    with torch.no_grad():
        generated = model.sample(
            history,
            n_scenarios=(
                N_SAMPLE_SCENARIOS
            ),
        )

    generated_model_space = (
        generated
        .detach()
        .cpu()
        .numpy()
    )

    generated_raw = (
        scaler.inverse_transform(
            generated_model_space
        )
    )

    actual_target_model = (
        sample["target"]
        .numpy()
    )

    actual_target_raw = (
        sample["target_raw"]
        .numpy()
    )

    print()
    print("=" * 72)
    print(
        "POST-TRAINING SAMPLING CHECK"
    )
    print("=" * 72)

    print(
        "Scenario shape:",
        generated_model_space.shape,
    )

    print(
        "Finite:",
        bool(
            np.isfinite(
                generated_model_space
            ).all()
        ),
    )

    print()
    print(
        "Generated standardized mean:",
        generated_model_space.mean(),
    )

    print(
        "Generated standardized std:",
        generated_model_space.std(),
    )

    print(
        "Generated standardized max abs:",
        np.abs(
            generated_model_space
        ).max(),
    )

    print()
    print(
        "Generated raw excess-return mean:",
        generated_raw.mean(),
    )

    print(
        "Generated raw excess-return std:",
        generated_raw.std(),
    )

    print(
        "Generated raw max abs:",
        np.abs(
            generated_raw
        ).max(),
    )

    print()
    print(
        "Seen target standardized:",
        actual_target_model,
    )

    print(
        "Seen target raw:",
        actual_target_raw,
    )


if __name__ == "__main__":
    main()