from pathlib import Path

import torch
from torch.utils.data import DataLoader

from diffusion_portfolio.baselines.diffolio import (
    DiffolioObjective,
)
from diffusion_portfolio.baselines.diffolio.training import diffolio_learning_rate, fit_diffolio_steps


def make_model():
    torch.manual_seed(
        123
    )

    training_returns = torch.randn(
        100,
        3,
    )

    centered = (
        training_returns
        - training_returns.mean(
            dim=0,
            keepdim=True,
        )
    )

    training_covariance = (
        centered.T
        @ centered
    ) / training_returns.shape[0]

    return DiffolioObjective(
        training_covariance=(
            training_covariance
        ),
        n_assets=3,
        n_asset_characteristics=2,
        n_systematic=2,
        lookback=8,
        hidden_dim=16,
        num_heads=4,
        mlp_dim=32,
        time_embedding_dim=8,
        diffusion_steps=10,
        lambda_corr=0.05,
    )


def make_loader(
    *,
    n_samples: int = 8,
    batch_size: int = 4,
):
    torch.manual_seed(
        456
    )

    dataset = []

    for _ in range(
        n_samples
    ):
        dataset.append(
            {
                "return_history": torch.randn(
                    8,
                    3,
                ),
                "return_history_raw": (
                    torch.randn(
                        8,
                        3,
                    )
                    * 0.01
                ),
                "asset_covariates": torch.randn(
                    8,
                    3,
                    2,
                ),
                "systematic_covariates": torch.randn(
                    8,
                    2,
                ),
                "target": torch.randn(
                    3
                ),
                "target_raw": (
                    torch.randn(
                        3
                    )
                    * 0.01
                ),
            }
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )


def test_learning_rate_schedule_boundaries():
    max_lr = 1e-3

    lr_1 = diffolio_learning_rate(
        1,
        total_steps=6,
        warmup_steps=2,
        max_learning_rate=max_lr,
    )

    lr_2 = diffolio_learning_rate(
        2,
        total_steps=6,
        warmup_steps=2,
        max_learning_rate=max_lr,
    )

    lr_4 = diffolio_learning_rate(
        4,
        total_steps=6,
        warmup_steps=2,
        max_learning_rate=max_lr,
    )

    lr_6 = diffolio_learning_rate(
        6,
        total_steps=6,
        warmup_steps=2,
        max_learning_rate=max_lr,
    )

    assert lr_1 == 0.5 * max_lr
    assert lr_2 == max_lr

    torch.testing.assert_close(
        torch.tensor(
            lr_4
        ),
        torch.tensor(
            0.5 * max_lr
        ),
    )

    torch.testing.assert_close(
        torch.tensor(
            lr_6
        ),
        torch.tensor(
            0.0
        ),
        atol=1e-12,
        rtol=0.0,
    )


def test_fit_updates_parameters_and_runs_exact_steps():
    model = make_model()

    loader = make_loader()

    before = {
        name: parameter
        .detach()
        .clone()
        for name, parameter
        in model.named_parameters()
    }

    result = fit_diffolio_steps(
        model,
        loader,
        total_steps=5,
        warmup_steps=2,
        max_learning_rate=1e-3,
        weight_decay=0.0,
        device="cpu",
        record_every=1,
        verbose=False,
    )

    assert result.final_step == 5

    assert len(
        result.history
    ) == 5

    assert [
        record.step
        for record
        in result.history
    ] == [
        1,
        2,
        3,
        4,
        5,
    ]

    changed = any(
        not torch.equal(
            before[name],
            parameter.detach(),
        )
        for name, parameter
        in model.named_parameters()
    )

    assert changed

    for record in result.history:
        assert torch.isfinite(
            torch.tensor(
                record.loss
            )
        )

        assert torch.isfinite(
            torch.tensor(
                record.ddpm_loss
            )
        )

        assert torch.isfinite(
            torch.tensor(
                record.correlation_loss
            )
        )


def test_fit_cycles_dataloader():
    model = make_model()

    # Exactly one minibatch per dataloader pass.
    loader = make_loader(
        n_samples=4,
        batch_size=4,
    )

    result = fit_diffolio_steps(
        model,
        loader,
        total_steps=3,
        warmup_steps=1,
        max_learning_rate=1e-3,
        weight_decay=0.0,
        device="cpu",
        record_every=1,
        verbose=False,
    )

    assert result.final_step == 3

    assert len(
        result.history
    ) == 3


def test_fit_writes_final_checkpoint(
    tmp_path: Path,
):
    model = make_model()

    loader = make_loader()

    checkpoint = (
        tmp_path
        / "diffolio.pt"
    )

    result = fit_diffolio_steps(
        model,
        loader,
        total_steps=2,
        warmup_steps=1,
        max_learning_rate=1e-3,
        weight_decay=0.0,
        device="cpu",
        checkpoint_path=checkpoint,
        record_every=1,
        verbose=False,
    )

    assert checkpoint.exists()

    saved = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    assert saved[
        "step"
    ] == 2

    assert saved[
        "training"
    ][
        "total_steps"
    ] == 2

    assert (
        "model_state_dict"
        in saved
    )

    assert (
        "optimizer_state_dict"
        in saved
    )

    assert result.checkpoint_path == str(
        checkpoint
    )
    
def test_training_can_resume_from_checkpoint(
    tmp_path: Path,
):
    checkpoint = (
        tmp_path
        / "resume.pt"
    )

    first_model = make_model()

    loader = make_loader()

    first_result = fit_diffolio_steps(
        first_model,
        loader,
        total_steps=2,
        warmup_steps=1,
        max_learning_rate=1e-3,
        weight_decay=0.0,
        device="cpu",
        checkpoint_path=checkpoint,
        record_every=1,
        verbose=False,
    )

    assert first_result.final_step == 2

    resumed_model = make_model()

    resumed_result = fit_diffolio_steps(
        resumed_model,
        loader,
        total_steps=4,
        warmup_steps=1,
        max_learning_rate=1e-3,
        weight_decay=0.0,
        device="cpu",
        checkpoint_path=checkpoint,
        resume_checkpoint=checkpoint,
        record_every=1,
        verbose=False,
    )

    assert resumed_result.final_step == 4

    assert [
        record.step
        for record
        in resumed_result.history
    ] == [
        3,
        4,
    ]

    saved = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    assert saved[
        "step"
    ] == 4
    

def test_checkpoint_callback_receives_periodic_and_final_checkpoints(
    tmp_path: Path,
):
    model = make_model()

    loader = make_loader()

    checkpoint = (
        tmp_path
        / "callback.pt"
    )

    calls = []

    def callback(
        step: int,
        path: Path,
    ):
        calls.append(
            (
                step,
                path.exists(),
            )
        )

    fit_diffolio_steps(
        model,
        loader,
        total_steps=4,
        warmup_steps=1,
        max_learning_rate=1e-3,
        weight_decay=0.0,
        device="cpu",
        checkpoint_path=checkpoint,
        checkpoint_every=2,
        checkpoint_callback=callback,
        record_every=1,
        verbose=False,
    )

    assert calls == [
        (
            2,
            True,
        ),
        (
            4,
            True,
        ),
    ]