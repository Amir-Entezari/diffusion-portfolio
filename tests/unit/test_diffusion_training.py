from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

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
from diffusion_portfolio.training import diffusion as diffusion_training
from diffusion_portfolio.models.diffusion.levy import (
    ConditionalLevyDiffusionModel,
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


def test_fit_resumes_from_last_checkpoint(
    tmp_path: Path,
):
    best_path = (
        tmp_path
        / "best.pt"
    )

    last_path = (
        tmp_path
        / "last.pt"
    )

    model = make_model()

    first = fit_diffusion(
        model,
        make_loader(),
        make_loader(),
        epochs=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
        validation_seed=123,
        device="cpu",
        checkpoint_path=best_path,
        resume_path=last_path,
        stop_after_epoch=2,
        verbose=False,
    )

    assert len(
        first.history
    ) == 2

    assert last_path.exists()

    resumed_model = make_model()

    second = fit_diffusion(
        resumed_model,
        make_loader(),
        make_loader(),
        epochs=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
        validation_seed=123,
        device="cpu",
        checkpoint_path=best_path,
        resume_path=last_path,
        stop_after_epoch=4,
        verbose=False,
    )

    assert [
        record.epoch
        for record
        in second.history
    ] == [
        1,
        2,
        3,
        4,
    ]
    

def test_fit_reports_flushed_epoch_and_batch_progress(monkeypatch, capsys):
    printer = Mock(wraps=print)
    monkeypatch.setattr("builtins.print", printer)
    monkeypatch.setattr(diffusion_training, "time", SimpleNamespace(
        perf_counter=Mock(side_effect=[100.0, 198.2]),
    ))
    model = make_model()
    loss_calls = Mock(wraps=model.training_loss)
    monkeypatch.setattr(model, "training_loss", loss_calls)
    result = fit_diffusion(
        model, make_loader(n_samples=45, batch_size=2),
        make_loader(n_samples=25, batch_size=2),
        epochs=50, stop_after_epoch=1, learning_rate=1e-3, weight_decay=0.0,
        gradient_clip_norm=1.0, validation_seed=123, device="cpu", verbose=True,
    )
    record = result.history[0]
    assert capsys.readouterr().out.splitlines() == [
        "Epoch 001/050 | Starting...",
        "Epoch 001/050 | Train 10/23 batches",
        "Epoch 001/050 | Train 20/23 batches",
        "Epoch 001/050 | Train 23/23 batches",
        "Epoch 001/050 | Validation 10/13 batches",
        "Epoch 001/050 | Validation 13/13 batches",
        f"Epoch 001/050 | train={record.train_loss:.6f} | val={record.val_loss:.6f} | elapsed=98.2s",
    ]
    assert all(call.kwargs.get("flush") is True for call in printer.call_args_list)
    assert loss_calls.call_count == 23 + 13


def test_progress_preserves_losses_checkpoints_and_resume(tmp_path, capsys):
    results, checkpoints = [], []
    for name, verbose, resume in (("quiet", False, False), ("live", True, False),
                                  ("resumed", True, True)):
        torch.manual_seed(123)
        model = make_model()
        train_data = make_loader(n_samples=11).dataset
        train_loader = DataLoader(
            train_data, batch_size=4, shuffle=True,
            generator=torch.Generator().manual_seed(91),
        )
        val_loader = make_loader(n_samples=7)
        torch.manual_seed(1001)
        folder = tmp_path / name
        folder.mkdir()
        options = dict(
            epochs=5, learning_rate=1e-3, weight_decay=0.0, gradient_clip_norm=1.0,
            validation_seed=123, device="cpu", checkpoint_path=folder / "best.pt",
            resume_path=folder / "last.pt", verbose=verbose,
        )
        if resume:
            pilot = fit_diffusion(model, train_loader, val_loader, stop_after_epoch=2, **options)
            assert len(pilot.history) == 2
            model = make_model()  # Resume must restore model, optimizer, and RNG state.
            capsys.readouterr()
        result = fit_diffusion(model, train_loader, val_loader, stop_after_epoch=3, **options)
        output = capsys.readouterr().out
        if not verbose:
            assert output == ""
        if resume:
            assert "Resuming from epoch 2" in output
            assert "Epoch 003/005 | Starting..." in output
            assert "Epoch 001/005" not in output and "Epoch 004/005" not in output
        assert [record.epoch for record in result.history] == [1, 2, 3]
        results.append((result.history, result.best_epoch, result.best_val_loss))
        checkpoints.append({name: torch.load(folder / name, weights_only=False, map_location="cpu")
                            for name in ("best.pt", "last.pt")})

    assert results[0] == results[1] == results[2]
    for saved in checkpoints[1:]:
        torch.testing.assert_close(saved, checkpoints[0], rtol=0, atol=0)
    assert set(checkpoints[0]["best.pt"]) == {
        "epoch", "model_state_dict", "optimizer_state_dict", "train_loss", "val_loss",
    }
    assert set(checkpoints[0]["last.pt"]) == {
        "epoch", "model_state_dict", "optimizer_state_dict", "history", "best_epoch",
        "best_val_loss", "torch_rng_state", "cuda_rng_state", "loader_rng_state",
    }


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
        
        
        
def test_levy_validation_loss_is_reproducible():
    torch.manual_seed(
        42
    )

    model = ConditionalLevyDiffusionModel(
        alpha=1.7,
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
        n_res_blocks=1,
    )

    loader = make_loader()

    loss_a = evaluate_diffusion_loss(
        model,
        loader,
        device=torch.device(
            "cpu"
        ),
        validation_seed=123,
    )

    loss_b = evaluate_diffusion_loss(
        model,
        loader,
        device=torch.device(
            "cpu"
        ),
        validation_seed=123,
    )

    assert loss_a == pytest.approx(
        loss_b,
        abs=1e-12,
    )
