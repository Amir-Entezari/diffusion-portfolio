from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from diffusion_portfolio.models.diffusion import (
    ConditionalDiffusionModel,
)
from diffusion_portfolio.training import (
    evaluate_diffusion_loss,
    fit_diffusion,
    train_one_epoch,
)


def make_model():
    return ConditionalDiffusionModel(
        lookback=5,
        n_assets=4,
        condition_dim=8,
        history_hidden_dim=16,
        diffusion_steps=10,
        schedule_type="cosine",
        channels=[
            8,
            16,
        ],
        time_embed_dim=8,
        prediction_type="v_prediction",
        n_res_blocks=1,
    )


def make_loader(
    *,
    n_samples: int = 12,
    batch_size: int = 4,
) -> DataLoader:
    torch.manual_seed(
        42
    )

    histories = torch.randn(
        n_samples,
        5,
        4,
    )

    targets = torch.randn(
        n_samples,
        1,
        4,
    )

    dataset = [
        {
            "history": histories[i],
            "target": targets[i],
        }
        for i in range(
            n_samples
        )
    ]

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )


def test_train_one_epoch_is_finite_and_updates_parameters():
    model = make_model()

    loader = make_loader()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    before = {
        name: parameter.detach().clone()
        for name, parameter
        in model.named_parameters()
    }

    loss = train_one_epoch(
        model,
        loader,
        optimizer,
        device=torch.device("cpu"),
        gradient_clip_norm=1.0,
    )

    assert loss > 0.0
    assert torch.isfinite(
        torch.tensor(loss)
    )

    changed = any(
        not torch.equal(
            before[name],
            parameter.detach(),
        )
        for name, parameter
        in model.named_parameters()
    )

    assert changed


def test_validation_loss_is_reproducible():
    model = make_model()

    loader = make_loader()

    loss_a = evaluate_diffusion_loss(
        model,
        loader,
        device=torch.device("cpu"),
        validation_seed=123,
    )

    loss_b = evaluate_diffusion_loss(
        model,
        loader,
        device=torch.device("cpu"),
        validation_seed=123,
    )

    assert loss_a == pytest.approx(
        loss_b,
        abs=1e-12,
    )


def test_validation_does_not_update_parameters():
    model = make_model()

    loader = make_loader()

    before = {
        name: parameter.detach().clone()
        for name, parameter
        in model.named_parameters()
    }

    evaluate_diffusion_loss(
        model,
        loader,
        device=torch.device("cpu"),
        validation_seed=123,
    )

    for name, parameter in (
        model.named_parameters()
    ):
        torch.testing.assert_close(
            parameter,
            before[name],
        )


def test_fit_returns_complete_history():
    model = make_model()

    train_loader = make_loader(
        n_samples=12,
    )

    val_loader = make_loader(
        n_samples=8,
    )

    result = fit_diffusion(
        model,
        train_loader,
        val_loader,
        epochs=3,
        learning_rate=1e-3,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
        validation_seed=123,
        device="cpu",
        verbose=False,
    )

    assert len(
        result.history
    ) == 3

    assert result.best_epoch in {
        1,
        2,
        3,
    }

    assert result.best_val_loss == pytest.approx(
        min(
            record.val_loss
            for record
            in result.history
        )
    )


def test_fit_writes_best_checkpoint(
    tmp_path: Path,
):
    model = make_model()

    train_loader = make_loader()
    val_loader = make_loader()

    checkpoint = (
        tmp_path
        / "best.pt"
    )

    result = fit_diffusion(
        model,
        train_loader,
        val_loader,
        epochs=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
        validation_seed=123,
        device="cpu",
        checkpoint_path=checkpoint,
        verbose=False,
    )

    assert checkpoint.exists()

    saved = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    assert saved["epoch"] == (
        result.best_epoch
    )

    assert saved["val_loss"] == pytest.approx(
        result.best_val_loss
    )

    assert "model_state_dict" in saved
    assert "optimizer_state_dict" in saved


def test_invalid_epoch_count_is_rejected():
    model = make_model()

    loader = make_loader()

    with pytest.raises(
        ValueError,
        match="epochs",
    ):
        fit_diffusion(
            model,
            loader,
            loader,
            epochs=0,
            learning_rate=1e-3,
            weight_decay=0.0,
            gradient_clip_norm=1.0,
            validation_seed=123,
            device="cpu",
            verbose=False,
        )