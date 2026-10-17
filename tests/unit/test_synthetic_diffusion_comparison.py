"""Limited inference, provenance, paired scores and block-bootstrap regression checks."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from diffusion_portfolio.config import load_config
from diffusion_portfolio.data.identity import file_sha256
from diffusion_portfolio.data.normalization import TrainStandardizer
from diffusion_portfolio.evaluation import forecasts
from diffusion_portfolio.evaluation.forecast_comparison import compare_synthetic_forecasts, moving_block_bootstrap
from diffusion_portfolio.evaluation.probabilistic import (
    crps_by_date, energy_score_by_date, evaluate_probabilistic_forecast, marginal_crps_by_date,
)
from diffusion_portfolio.evaluation.synthetic_benchmarks import benchmark_synthetic_forecasts

ROOT = Path(__file__).parents[2]


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    """Real two-epoch CPU training of a tiny fixture, never a scientific run."""
    from diffusion_portfolio.training import runs

    root = tmp_path_factory.mktemp("synthetic-comparison")
    cfg = load_config(ROOT / "configs/synthetic_gaussian.yaml")
    dates = pd.bdate_range("2000-01-03", periods=180)
    cfg = replace(cfg, data=replace(cfg.data, sample_start=str(dates[0].date()),
                  sample_end=str(dates[-1].date()), train_end=str(dates[99].date()), val_end=str(dates[139].date()),
                  lookback=4, source_options={"n_assets": 3, "n_observations": 180, "n_factors": 2,
                                              "seed": 19, "start_date": str(dates[0].date())}),
                  model=replace(cfg.model, condition_dim=8, history_hidden_dim=16, channels=(8,),
                                time_embed_dim=8, diffusion_steps=4),
                  training=replace(cfg.training, batch_size=32), evaluation=replace(cfg.evaluation, n_scenarios=8))
    config_path = root / "tiny.yaml"
    config_path.write_text(yaml.safe_dump(asdict(cfg)))
    run = root / "run"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runs, "git_commit", lambda: pytest.fail("A pilot must not query Git"))
        runs.train_diffusion(config_path, run, device="cpu", resume=True, stop_after_epoch=2)
    assert (run / "data_identity.json").exists() and not (run / "summary.json").exists()
    return SimpleNamespace(root=root, cfg=cfg, config=config_path, run=run)


@pytest.fixture(scope="module")
def matched(tiny_run):
    baseline, diffusion = tiny_run.root / "baseline", tiny_run.root / "diffusion"
    benchmark_synthetic_forecasts(tiny_run.config, output_dir=baseline, n_scenarios=8)
    forecasts.evaluate_diffusion(tiny_run.run, limit=8, output_dir=diffusion, batch_size=4, device="cpu")
    return SimpleNamespace(**vars(tiny_run), baseline=baseline, diffusion=diffusion)


def test_limited_inference_slices_before_loader_reuses_scaler_and_never_accesses_test(tiny_run, tmp_path, monkeypatch):
    original_prepare, original_generate = forecasts.prepare_return_data, forecasts.generate_scenarios
    full, _ = original_prepare(tiny_run.cfg, standardizer_path=tiny_run.run / "standardizer.npz")

    def validation_only(cfg, *, standardizer_path):
        assert Path(standardizer_path) == tiny_run.run / "standardizer.npz"
        datasets, scaler = original_prepare(cfg, standardizer_path=standardizer_path)
        return SimpleNamespace(val=datasets.val), scaler  # No test attribute.

    def generate(model, loader, scaler, **kwargs):
        assert len(loader.dataset) == 3 and kwargs["n_samples"] == 3
        assert not model.training and kwargs["n_scenarios"] == 8
        before = {key: value.clone() for key, value in model.state_dict().items()}
        checkpoint = torch.load(tiny_run.run / "best.pt", weights_only=False)
        assert all(torch.equal(value, checkpoint["model_state_dict"][key]) for key, value in before.items())
        result = original_generate(model, loader, scaler, **kwargs)
        assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
        return result

    monkeypatch.setattr(forecasts, "prepare_return_data", validation_only)
    monkeypatch.setattr(forecasts, "generate_scenarios", generate)
    monkeypatch.setattr(TrainStandardizer, "fit", lambda *a, **kw: pytest.fail("Saved scaler was refitted"))
    first, second = tmp_path / "first", tmp_path / "second"
    for output in (first, second):
        result = forecasts.evaluate_diffusion(tiny_run.run, limit=3, output_dir=output, batch_size=2, device="cpu")
        assert result["n_validation"] == 3 and result["available_samples"] == 40
    with np.load(first / "validation_scenarios.npz") as a, np.load(second / "validation_scenarios.npz") as b:
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_array_equal(a["dates"], full.val.raw_windows.target_dates[:3].values.astype("datetime64[D]"))
        np.testing.assert_array_equal(a["observed"], full.val.raw_windows.target[:3, 0])
        assert a["columns"].tolist() == ["asset_001", "asset_002", "asset_003"]
    metadata = json.loads((first / "forecast_metadata.json").read_text())
    assert metadata["training_run"]["status"] == "preliminary"
    assert metadata["training_run"]["completed_epochs"] == 2 and metadata["training_run"]["configured_epochs"] == 50
    assert metadata["batch_size"] == 2 and metadata["device"] == "cpu"
    assert not (tiny_run.run / "validation_scenarios.npz").exists()


def test_default_full_validation_paths_and_sample_count_unchanged(tiny_run, tmp_path):
    run = tmp_path / "copy"
    shutil.copytree(tiny_run.run, run)
    # Legacy default evaluation does not demand the new provenance sidecar.
    (run / "data_identity.json").unlink()
    result = forecasts.evaluate_diffusion(run, batch_size=8, device="cpu")
    assert result["n_validation"] == 40 and "evaluation_limit" not in result
    assert (run / "validation_scenarios.npz").is_file()
    assert not (run / "forecast_metadata.json").exists()


def test_training_provenance_recording_preserves_exact_history_weights_and_rng(tiny_run, tmp_path, monkeypatch):
    from diffusion_portfolio.training import runs
    monkeypatch.setattr(runs, "COMPARISON_DATASETS", set())
    monkeypatch.setattr(runs, "git_commit", lambda: pytest.fail("Unexpected Git query"))
    control = tmp_path / "without-sidecar"
    runs.train_diffusion(tiny_run.config, control, device="cpu", resume=True, stop_after_epoch=2)
    assert not (control / "data_identity.json").exists()
    assert (control / "history.csv").read_bytes() == (tiny_run.run / "history.csv").read_bytes()
    first = torch.load(control / "last.pt", weights_only=False)
    second = torch.load(tiny_run.run / "last.pt", weights_only=False)
    for key, value in first["model_state_dict"].items():
        assert torch.equal(value, second["model_state_dict"][key])
    assert torch.equal(first["torch_rng_state"], second["torch_rng_state"])
    assert torch.equal(first["loader_rng_state"], second["loader_rng_state"])


def test_training_contract_exists_before_the_first_training_step(tiny_run, tmp_path, monkeypatch):
    from diffusion_portfolio.training import runs

    class BeforeTraining(Exception):
        pass

    def inspect(model, train_loader, val_loader, cfg, output_dir, **kwargs):
        recorded = json.loads((output_dir / "data_identity.json").read_text())["training_contract"]
        assert recorded["seed"] == cfg.seed and recorded["optimizer"] == "torch.optim.AdamW"
        assert recorded["training"] == asdict(cfg.training)
        assert recorded["evaluation"] == asdict(cfg.evaluation)
        assert recorded["model"] == json.loads(json.dumps(asdict(cfg.model)))
        assert recorded["training"]["epochs"] == 50
        assert not (output_dir / "best.pt").exists() and not (output_dir / "last.pt").exists()
        raise BeforeTraining

    monkeypatch.setattr(runs, "fit_and_record", inspect)
    with pytest.raises(BeforeTraining):
        runs.train_diffusion(tiny_run.config, tmp_path / "new", device="cpu", resume=True, stop_after_epoch=2)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_inference_limit(limit, tmp_path):
    with pytest.raises(ValueError, match="limit"):
        forecasts.evaluate_diffusion(tmp_path, limit=limit, output_dir=tmp_path / "out")


def test_no_limited_test_or_implicit_overwrite(tiny_run, tmp_path):
    with pytest.raises(ValueError, match="validation"):
        forecasts.evaluate_diffusion(tiny_run.run, limit=3, split="test", output_dir=tmp_path / "out")
    with pytest.raises(ValueError, match="output-dir"):
        forecasts.evaluate_diffusion(tiny_run.run, limit=3)
    with pytest.raises(FileExistsError, match="empty"):
        forecasts.evaluate_diffusion(tiny_run.run, limit=3, output_dir=tiny_run.run)


def test_inference_limit_exactly_available_is_accepted(tiny_run, tmp_path, monkeypatch):
    generate = forecasts.generate_scenarios

    def verify_count(model, loader, scaler, **kwargs):
        assert len(loader.dataset) == kwargs["n_samples"] == 40
        return generate(model, loader, scaler, **kwargs)

    monkeypatch.setattr(forecasts, "generate_scenarios", verify_count)
    result = forecasts.evaluate_diffusion(tiny_run.run, limit=40, output_dir=tmp_path / "exact", batch_size=8, device="cpu")
    assert result["n_validation"] == result["available_samples"] == result["evaluation_limit"] == 40


def test_excessive_inference_limit_fails_before_model_or_sampling(tiny_run, tmp_path, monkeypatch):
    monkeypatch.setattr(forecasts, "build_diffusion", lambda *a, **kw: pytest.fail("Model built for an excessive limit"))
    monkeypatch.setattr(forecasts, "generate_scenarios", lambda *a, **kw: pytest.fail("Sampling ran for an excessive limit"))
    with pytest.raises(ValueError, match="limit=41 exceeds the 40 available validation dates"):
        forecasts.evaluate_diffusion(tiny_run.run, limit=41, output_dir=tmp_path / "excessive", device="cpu")
    assert not (tmp_path / "excessive").exists()


@pytest.mark.parametrize("corruption,match", [("identity", "Missing training-time"), ("scaler", "train-only"),
                                            ("pair", "not bound"), ("weights", "not bound"),
                                            ("summary", "summary conflicts")])
def test_inference_rejects_unverifiable_or_inconsistent_run(tiny_run, tmp_path, corruption, match):
    run = tmp_path / "run"
    shutil.copytree(tiny_run.run, run)
    if corruption == "identity":
        (run / "data_identity.json").unlink()
    elif corruption == "scaler":
        with np.load(run / "standardizer.npz") as saved:
            mean, std, columns = saved["mean"], saved["std"], saved["columns"]
        np.savez(run / "standardizer.npz", mean=mean + .01, std=std, columns=columns)
    elif corruption == "pair":
        saved = torch.load(run / "last.pt", weights_only=False)
        saved["best_epoch"] += 1
        torch.save(saved, run / "last.pt")
    elif corruption == "weights":
        saved = torch.load(run / "best.pt", weights_only=False)
        first = next(iter(saved["model_state_dict"]))
        saved["model_state_dict"][first] = saved["model_state_dict"][first] + .001
        torch.save(saved, run / "best.pt")  # Epoch and loss still match; weights differ.
    else:
        (run / "summary.json").write_text(json.dumps({"best_epoch": 1, "best_val_loss": -10}))
    with pytest.raises((ValueError, RuntimeError), match=match):
        forecasts.evaluate_diffusion(run, limit=3, output_dir=tmp_path / "out", device="cpu")
    assert not (tmp_path / "out").exists()


def test_rewriting_original_budget_cannot_manufacture_completion(tiny_run, tmp_path):
    from diffusion_portfolio.data.preparation import load_standardizer
    from diffusion_portfolio.evaluation.forecast_identity import verify_synthetic_checkpoint
    run = tmp_path / "complete"
    shutil.copytree(tiny_run.run, run)
    cfg = replace(tiny_run.cfg, training=replace(tiny_run.cfg.training, epochs=2))
    (run / "config.yaml").write_text(yaml.safe_dump(asdict(cfg)))
    best = torch.load(run / "best.pt", weights_only=False)
    (run / "summary.json").write_text(json.dumps({"best_epoch": best["epoch"], "best_val_loss": best["val_loss"]}))
    original_provenance = (run / "data_identity.json").read_bytes()
    with pytest.raises(ValueError, match="Immutable training contract differs"):
        verify_synthetic_checkpoint(run, cfg, load_standardizer(run / "standardizer.npz"), best)
    assert (run / "data_identity.json").read_bytes() == original_provenance


def test_two_epoch_pilot_resumes_to_its_original_fifty_epoch_budget(tiny_run, tmp_path, monkeypatch):
    from diffusion_portfolio.training import runs
    from diffusion_portfolio.data.preparation import load_standardizer
    from diffusion_portfolio.evaluation.forecast_identity import verify_synthetic_checkpoint

    run = tmp_path / "continued"
    shutil.copytree(tiny_run.run, run)
    original_contract = json.loads((run / "data_identity.json").read_text())["training_contract"]
    assert original_contract["training"]["epochs"] == 50
    assert torch.load(run / "last.pt", weights_only=False)["epoch"] == 2
    monkeypatch.setattr(runs, "git_commit", lambda: None)  # Never execute Git.
    # Tiny three-asset fixture only; no research-sized 50-epoch experiment.
    result = runs.train_diffusion(tiny_run.config, run, device="cpu", resume=True)
    assert [row.epoch for row in result.history] == list(range(1, 51))
    assert json.loads((run / "data_identity.json").read_text())["training_contract"] == original_contract
    assert load_config(run / "config.yaml") == tiny_run.cfg
    best = torch.load(run / "best.pt", weights_only=False)
    metadata = verify_synthetic_checkpoint(run, tiny_run.cfg, load_standardizer(run / "standardizer.npz"), best)
    assert metadata["training_run"]["status"] == "completed_declared_budget"
    assert metadata["training_run"]["configured_epochs"] == metadata["training_run"]["completed_epochs"] == 50


@pytest.mark.parametrize("section,key,value", [
    (None, "seed", 43), ("model", "history_hidden_dim", 32), ("model", "history_encoder", "cde"),
    ("training", "epochs", 2), ("training", "epochs", 51), ("training", "learning_rate", .001),
    ("training", "weight_decay", .1), ("training", "batch_size", 16),
    ("training", "gradient_clip_norm", .5), ("training", "validation_seed", 99),
    ("evaluation", "n_scenarios", 16), ("data", "lookback", 5),
])
def test_incompatible_resume_preserves_all_original_artifacts(tiny_run, tmp_path, monkeypatch, section, key, value):
    from diffusion_portfolio.training import runs
    run = tmp_path / "run"
    shutil.copytree(tiny_run.run, run)
    cfg = (replace(tiny_run.cfg, **{key: value}) if section is None else
           replace(tiny_run.cfg, **{section: replace(getattr(tiny_run.cfg, section), **{key: value})}))
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(asdict(cfg)))
    before = {p: file_sha256(p) for p in run.iterdir() if p.is_file()}
    monkeypatch.setattr(runs, "prepare_return_data", lambda *a, **kw: pytest.fail("Incompatible resume reached data loading"))
    with pytest.raises(ValueError, match="Immutable training contract differs"):
        runs.train_diffusion(path, run, device="cpu", resume=True, stop_after_epoch=3)
    assert all(file_sha256(p) == digest for p, digest in before.items())


def test_completed_shortcut_cannot_bypass_contract_validation(tiny_run, tmp_path):
    from diffusion_portfolio.training import runs
    run = tmp_path / "run"
    shutil.copytree(tiny_run.run, run)
    (run / "summary.json").write_text("{}")
    cfg = replace(tiny_run.cfg, training=replace(tiny_run.cfg.training, epochs=2))
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(asdict(cfg)))
    with pytest.raises(ValueError, match="Immutable training contract differs"):
        runs.train_diffusion(path, run, device="cpu", resume=True)


def test_legacy_resume_cannot_retroactively_certify_a_training_contract(tiny_run, tmp_path):
    from diffusion_portfolio.training import runs
    run = tmp_path / "legacy"
    shutil.copytree(tiny_run.run, run)
    identity = json.loads((run / "data_identity.json").read_text())
    del identity["training_contract"]
    (run / "data_identity.json").write_text(json.dumps(identity))
    result = runs.train_diffusion(tiny_run.config, run, device="cpu", resume=True, stop_after_epoch=3)
    assert result.history[-1].epoch == 3
    continued = json.loads((run / "data_identity.json").read_text())
    assert "training_contract" not in continued
    assert continued["source_identity"] == identity["source_identity"]
    before = {p: file_sha256(p) for p in run.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="Missing immutable training contract"):
        forecasts.evaluate_diffusion(run, limit=3, output_dir=tmp_path / "strict", device="cpu")
    assert all(file_sha256(p) == digest for p, digest in before.items())
    # Historical default evaluation remains available without certification.
    result = forecasts.evaluate_diffusion(run, batch_size=8, device="cpu")
    assert result["n_validation"] == 40
    assert all(file_sha256(p) == digest for p, digest in before.items())


def test_changed_contract_rejected_by_default_and_strict_evaluation(tiny_run, tmp_path):
    run = tmp_path / "changed"
    shutil.copytree(tiny_run.run, run)
    cfg = replace(tiny_run.cfg, training=replace(tiny_run.cfg.training, learning_rate=.001))
    (run / "config.yaml").write_text(yaml.safe_dump(asdict(cfg)))
    before = {p: file_sha256(p) for p in run.iterdir() if p.is_file()}
    for options in ({}, {"limit": 3, "output_dir": tmp_path / "strict"}):
        with pytest.raises(ValueError, match="Immutable training contract differs"):
            forecasts.evaluate_diffusion(run, device="cpu", **options)
    assert all(file_sha256(p) == digest for p, digest in before.items())


def test_per_date_scores_match_brute_force_and_historical_aggregates():
    rng = np.random.default_rng(22)
    samples, observed = rng.normal(0, .01, (7, 6, 3)), rng.normal(0, .01, (7, 3))
    marginal, energy = [], []
    for x, y in zip(samples, observed):
        marginal.append(abs(x - y).mean(0) - .5 * abs(x[:, None] - x[None, :]).mean((0, 1)))
        energy.append(np.linalg.norm(x - y, axis=1).mean() - .5 * np.linalg.norm(x[:, None] - x[None, :], axis=2).mean())
    np.testing.assert_allclose(marginal_crps_by_date(samples, observed), marginal, rtol=1e-14)
    np.testing.assert_allclose(crps_by_date(samples, observed), np.mean(marginal, axis=1), rtol=1e-14)
    np.testing.assert_allclose(energy_score_by_date(samples, observed), energy, rtol=1e-14)
    aggregate = evaluate_probabilistic_forecast(samples, observed)
    np.testing.assert_allclose(crps_by_date(samples, observed).mean(), aggregate.crps_mean, rtol=1e-14)
    np.testing.assert_allclose(np.mean(energy), aggregate.energy_score, rtol=1e-14)


@pytest.mark.parametrize("block_length", [1, 3, 8])
def test_bootstrap_reproducible_paired_contiguous_blocks(block_length):
    values = np.column_stack((np.arange(8.), -np.arange(8.)))
    first = moving_block_bootstrap(values, block_length=block_length, n_bootstrap=50, seed=18)
    np.testing.assert_array_equal(first, moving_block_bootstrap(values, block_length=block_length, n_bootstrap=50, seed=18))
    rng, means = np.random.default_rng(18), []
    for _ in range(50):
        starts = rng.integers(0, 8 - block_length + 1, size=(8 + block_length - 1) // block_length)
        sequence = np.concatenate([values[start:start + block_length] for start in starts])[:8]
        means.append(sequence.mean(axis=0))
    np.testing.assert_array_equal(first, np.quantile(means, [.025, .975], axis=0))
    np.testing.assert_allclose(first[:, 0], -first[::-1, 1])
    if block_length == 8:
        np.testing.assert_array_equal(first[:, 0], [3.5, 3.5])


@pytest.mark.parametrize("values,kwargs", [([1], {}), ([np.nan, 1], {}), ([], {}),
    ([1, 2], {"block_length": 3}), ([1, 2], {"block_length": 0}), ([1, 2], {"block_length": True}),
    ([1, 2], {"block_length": 1, "n_bootstrap": 1}), ([1, 2], {"block_length": 1, "seed": -1})])
def test_invalid_bootstrap_inputs(values, kwargs):
    with pytest.raises(ValueError):
        moving_block_bootstrap(values, **kwargs)


def test_full_baseline_subset_comparison_scores_signs_and_preservation(matched, tmp_path):
    paths = [path for path in matched.baseline.rglob('*') if path.is_file()]
    original_hashes = {path: file_sha256(path) for path in paths}
    output = tmp_path / "comparison"
    report = compare_synthetic_forecasts(matched.baseline, matched.diffusion, output_dir=output,
                                         limit=6, block_length=2, n_bootstrap=50, bootstrap_seed=17)
    assert report["n_dates"] == 6 and report["training_run"]["status"] == "preliminary"
    assert report["training_contract"]["training"]["epochs"] == 50
    table = pd.read_csv(output / "per_date_scores.csv")
    for row in report["paired_comparisons"]:
        baseline, metric = row["baseline"], row["metric"]
        paired = table[f"diffusion_{metric}"] - table[f"{baseline}_{metric}"]
        np.testing.assert_allclose(paired, table[f"diffusion_minus_{baseline}_{metric}"], atol=1e-15)
        np.testing.assert_allclose(row["mean_difference"], paired.mean(), atol=1e-15)
        np.testing.assert_allclose(row["paired_sample_std"], paired.std(ddof=1), atol=1e-15)
        np.testing.assert_allclose(row["relative_improvement"], -row["mean_difference"] / row["baseline_mean"], atol=1e-14)
    assert len(pd.read_csv(output / "calibration.csv")) == 20
    assert all(file_sha256(path) == value for path, value in original_hashes.items())
    assert json.loads((output / "comparison_report.json").read_text()) == report


@pytest.mark.parametrize("corruption,match", [("config", "Immutable training contract differs"),
                                            ("metadata", "metadata differs from the immutable"),
                                            ("missing", "Missing immutable training contract")])
def test_comparison_verifies_immutable_training_contract(matched, tmp_path, corruption, match):
    diffusion = tmp_path / "diffusion"
    shutil.copytree(matched.diffusion, diffusion)
    metadata_path = diffusion / "forecast_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if corruption == "config":
        path = diffusion / "config.yaml"
        cfg = yaml.safe_load(path.read_text())
        cfg["training"]["epochs"] = 2
        path.write_text(yaml.safe_dump(cfg))
        metadata["checkpoint_files_sha256"]["config.yaml"] = file_sha256(path)
        metadata["training_run"].update(configured_epochs=2, completed_epochs=2, status="completed_declared_budget")
    elif corruption == "metadata":
        metadata["training_contract"]["training"]["epochs"] = 2
    else:
        path = diffusion / "data_identity.json"
        identity = json.loads(path.read_text())
        del identity["training_contract"]
        path.write_text(json.dumps(identity))
    metadata_path.write_text(json.dumps(metadata))
    before = {p: file_sha256(p) for p in diffusion.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match=match):
        compare_synthetic_forecasts(matched.baseline, diffusion, output_dir=tmp_path / "out", block_length=2)
    assert not (tmp_path / "out").exists()
    assert all(file_sha256(p) == digest for p, digest in before.items())


@pytest.mark.parametrize("corruption,match", [
    ("seed", "configuration mismatch"), ("dataset", "configuration mismatch"),
    ("lookback", "configuration mismatch"), ("risk_free", "configuration mismatch"),
    ("dates", "dates"), ("columns", "Asset order"), ("observed", "observations"),
    ("archive_count", "Scenario count"), ("report_count", "scenario counts"),
    ("missing_identity", "Missing baseline"), ("fingerprint", "fingerprint mismatch"),
    ("units", "Return units"), ("split", "validation"), ("missing_metadata", "Missing required provenance"),
])
def test_incompatible_archives_are_rejected(matched, tmp_path, corruption, match):
    baseline, diffusion = tmp_path / "baseline", tmp_path / "diffusion"
    shutil.copytree(matched.baseline, baseline)
    shutil.copytree(matched.diffusion, diffusion)
    report = json.loads((baseline / "report.json").read_text())
    if corruption in ("seed", "dataset", "lookback", "risk_free"):
        cfg = yaml.safe_load((baseline / "config.yaml").read_text())
        if corruption == "seed":
            cfg["data"]["source_options"]["seed"] += 1
        elif corruption == "dataset":
            cfg["data"]["dataset"] = "synthetic_mean_reverting"
        elif corruption == "lookback":
            cfg["data"]["lookback"] += 1
        else:
            cfg["data"]["subtract_risk_free"] = True
        (baseline / "config.yaml").write_text(yaml.safe_dump(cfg))
    elif corruption in ("dates", "columns", "observed", "archive_count"):
        path = baseline / "gaussian/validation_scenarios.npz"
        with np.load(path) as saved:
            data = {key: saved[key].copy() for key in saved.files}
        if corruption == "dates":
            data["dates"][0] -= np.timedelta64(1, 'D')
        elif corruption == "columns":
            data["columns"] = data["columns"][::-1]
        elif corruption == "observed":
            data["observed"][0, 0] += .001
        else:
            data["scenarios"] = data["scenarios"][:, :-1]
        np.savez(path, **data)
    elif corruption == "report_count":
        report["n_scenarios"] += 1
    elif corruption == "missing_identity":
        del report["source_identity"]
    elif corruption == "fingerprint":
        report["source_identity"]["train_validation_panel_sha256"] = "0" * 64
    elif corruption == "units":
        report["return_units"] = "standardized returns"
    elif corruption == "split":
        report["split"] = "test"
    else:
        (diffusion / "forecast_metadata.json").unlink()
    (baseline / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match=match):
        compare_synthetic_forecasts(baseline, diffusion, output_dir=tmp_path / "out", block_length=2, n_bootstrap=20)
    assert not (tmp_path / "out").exists()


def test_comparison_rejects_missing_provenance_and_tampered_diffusion_archive(matched, tmp_path):
    diffusion = tmp_path / "diffusion"
    shutil.copytree(matched.diffusion, diffusion)
    with (diffusion / "validation_scenarios.npz").open('ab') as stream:
        stream.write(b'tampered')
    with pytest.raises(ValueError, match="archive differs"):
        compare_synthetic_forecasts(matched.baseline, diffusion, output_dir=tmp_path / "out", block_length=2)


def test_comparison_limit_cannot_silently_exceed_pilot(matched, tmp_path):
    with pytest.raises(ValueError, match="limit exceeds"):
        compare_synthetic_forecasts(matched.baseline, matched.diffusion, output_dir=tmp_path / "out", limit=20)


def test_perfect_forecast_fixture_has_negative_paired_differences(matched, tmp_path):
    # Deliberately artificial score fixture, not a trained model or oracle feature.
    diffusion = tmp_path / "fixture"
    shutil.copytree(matched.diffusion, diffusion)
    path = diffusion / "validation_scenarios.npz"
    with np.load(path) as saved:
        data = {key: saved[key].copy() for key in saved.files}
    data["scenarios"] = np.repeat(data["observed"][:, None, :], 8, axis=1)
    np.savez(path, **data)
    metadata = json.loads((diffusion / "forecast_metadata.json").read_text())
    metadata["archive_sha256"] = file_sha256(path)
    (diffusion / "forecast_metadata.json").write_text(json.dumps(metadata))
    report = compare_synthetic_forecasts(matched.baseline, diffusion, output_dir=tmp_path / "out",
                                         block_length=2, n_bootstrap=20)
    for row in report["paired_comparisons"]:
        assert row["diffusion_mean"] == 0 and row["mean_difference"] < 0
        assert row["relative_improvement"] == 1 and row["ci95_upper"] < 0


def test_notebook_schema_sync_syntax_and_cli_arguments(monkeypatch):
    import ast
    import runpy
    import shlex
    import sys
    nbformat = pytest.importorskip("nbformat")
    transform = pytest.importorskip("IPython.core.inputtransformer2").TransformerManager()
    book = nbformat.read(ROOT / "notebooks/synthetic/synthetic_diffusion_comparison.ipynb", as_version=4)
    reference = nbformat.read(ROOT / "notebooks/synthetic/synthetic_forecast_benchmarks.ipynb", as_version=4)
    nbformat.validate(book)
    sync = next(cell.source for cell in book.cells if cell.id == "github-sync")
    assert sync == next(cell.source for cell in reference.cells if cell.id == "github-sync")
    assert 'BRANCH = "feature/synthetic-market"' in sync
    commands = {}
    for cell in book.cells:
        if cell.cell_type != "code":
            continue
        assert cell.outputs == [] and cell.execution_count is None
        assert not any(term in cell.source for term in ("MODE", "%%bash", "subprocess", "%%capture"))
        source = transform.transform_cell(cell.source)
        compile(source, cell.id, "exec")
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call) or getattr(node.func, "attr", None) != "system":
                continue
            command = ast.literal_eval(node.args[0])
            tokens = shlex.split(command)
            scripts = [token for token in tokens if token.startswith("scripts/")]
            if not scripts:
                continue  # Never execute Git, install or test commands.
            script = scripts[0]
            monkeypatch.setattr(sys, "argv", tokens[tokens.index(script):])
            args = runpy.run_path(str(ROOT / script))["parse_args"]()
            commands[script] = args
            lines = cell.source.splitlines()
            system_line = next(i for i, line in enumerate(lines) if line.lstrip().startswith("!PYTHONUNBUFFERED"))
            while lines[system_line].rstrip().endswith("\\"):
                system_line += 1
            assert lines[system_line + 1].lstrip().startswith("assert _exit_code == 0")
    assert commands["scripts/train_diffusion.py"].stop_after_epoch == 2
    assert commands["scripts/train_diffusion.py"].resume
    inference = commands["scripts/evaluate_diffusion.py"]
    assert inference.split == "validation" and inference.limit == 64 and inference.batch_size == 16
    assert inference.run_dir != inference.output_dir
    assert commands["scripts/benchmark_synthetic_forecasts.py"].n_scenarios == 256
    assert commands["scripts/compare_synthetic_forecasts.py"].block_length == 20
