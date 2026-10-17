"""Controlled history availability, immutable policy and archive-only diagnostics."""
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
from diffusion_portfolio.data.identity import file_sha256, synthetic_training_contract
from diffusion_portfolio.data.preparation import prepare_return_data
from diffusion_portfolio.evaluation.forecast_comparison import compare_synthetic_forecasts
from diffusion_portfolio.evaluation.forecasts import evaluate_diffusion
from diffusion_portfolio.evaluation.history_ablation import conditional_mean_diagnostics, diagnose_history_ablation
from diffusion_portfolio.evaluation.probabilistic import evaluate_probabilistic_forecast
from diffusion_portfolio.evaluation.synthetic_benchmarks import benchmark_synthetic_forecasts
from diffusion_portfolio.models.diffusion.build import build_diffusion
from diffusion_portfolio.models.encoders.history_mask import history_mask_policy, mask_history
from diffusion_portfolio.training import runs

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs/ablations/synthetic_ar1_context.yaml"


def test_config_changes_only_the_predeclared_epoch_budget():
    original = load_config(ROOT / "configs/synthetic_mean_reverting.yaml")
    cfg = load_config(CONFIG)
    assert cfg == replace(original, training=replace(original.training, epochs=15))
    assert cfg.data.lookback == 60 and cfg.data.source_options["n_assets"] == 12
    assert "history_mask" not in synthetic_training_contract(original)


@pytest.mark.parametrize("keep", [0, 1, 5, 60])
def test_mask_preserves_input_and_keeps_only_the_requested_standardized_history(keep):
    history = torch.arange(2 * 60 * 3).reshape(2, 60, 3).float().requires_grad_()
    original = history.detach().clone()
    result = mask_history(history, keep)
    assert torch.equal(history, original)
    assert torch.count_nonzero(result[:, :60 - keep]) == 0
    if keep:
        assert torch.equal(result[:, -keep:], original[:, -keep:])
    if keep == 60:
        assert result is history
    # Changing unavailable observations cannot affect the masked history.
    changed = original.clone()
    changed[:, :60 - keep] += 1000
    assert torch.equal(result, mask_history(changed, keep))


@pytest.mark.parametrize("keep", [-1, 61, 1.5, True])
def test_invalid_history_length(keep):
    with pytest.raises(ValueError, match="history-keep-last"):
        history_mask_policy(load_config(CONFIG), keep)


@pytest.mark.parametrize("change", ["dataset", "encoder", "lookback", "horizon", "risk_free"])
def test_unsupported_ablation_configuration(change):
    cfg = load_config(CONFIG)
    if change == "encoder":
        cfg = replace(cfg, model=replace(cfg.model, history_encoder="cde"))
    else:
        changes = {"dataset": {"dataset": "synthetic_gaussian"}, "lookback": {"lookback": 5},
                   "horizon": {"horizon": 2}, "risk_free": {"subtract_risk_free": True}}[change]
        cfg = replace(cfg, data=replace(cfg.data, **changes))
    with pytest.raises(ValueError, match="History masking requires"):
        history_mask_policy(cfg, 1)


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    """Four tiny two-epoch CPU fixtures, not full-size research experiments."""
    root = tmp_path_factory.mktemp("history-ablation")
    cfg = load_config(CONFIG)
    dates = pd.bdate_range("2000-01-03", periods=180)
    cfg = replace(cfg, data=replace(cfg.data, sample_start=str(dates[0].date()),
                  sample_end=str(dates[-1].date()), train_end=str(dates[99].date()), val_end=str(dates[139].date()),
                  source_options={"n_assets": 3, "n_observations": 180, "n_factors": 2,
                                  "seed": 19, "start_date": str(dates[0].date()), "ar_coefficients": .4}),
                  model=replace(cfg.model, condition_dim=8, history_hidden_dim=16, channels=(8,),
                                time_embed_dim=8, diffusion_steps=4),
                  training=replace(cfg.training, batch_size=20), evaluation=replace(cfg.evaluation, n_scenarios=8))
    config = root / "config.yaml"
    config.write_text(yaml.safe_dump(asdict(cfg)))
    run_dirs, archives = {}, {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runs, "git_commit", lambda: pytest.fail("No Git operations"))
        for keep in (0, 1, 5, 60):
            label = f"K{keep}"
            run_dirs[label], archives[label] = root / label, root / f"{label}_inference"
            runs.train_diffusion(config, run_dirs[label], device="cpu", resume=True,
                                 stop_after_epoch=2, history_keep_last=keep)
            evaluate_diffusion(run_dirs[label], limit=8, batch_size=4, device="cpu", output_dir=archives[label])
    baseline = root / "baseline"
    benchmark_synthetic_forecasts(config, output_dir=baseline, n_scenarios=8)
    return SimpleNamespace(root=root, cfg=cfg, config=config, runs=run_dirs, archives=archives, baseline=baseline)


@pytest.mark.parametrize("keep", [0, 1, 5, 60])
def test_hook_covers_loss_and_sampling_without_changing_raw_data_targets_or_parameters(experiment, keep):
    cfg = experiment.cfg
    data, _ = prepare_return_data(cfg)
    sample = data.val[0]
    before = {key: value.clone() for key, value in sample.items() if isinstance(value, torch.Tensor)}
    history, target = sample["history"][None], sample["target"][None]
    torch.manual_seed(3)
    model = build_diffusion(cfg, n_assets=3, history_mask=history_mask_policy(cfg, keep))
    control = build_diffusion(cfg, n_assets=3)
    control.load_state_dict(model.state_dict())
    assert model.state_dict().keys() == control.state_dict().keys()
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in control.parameters())
    seen = []
    model.history_encoder.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
    for training in (True, False):
        model.train(training)
        torch.manual_seed(77)
        actual = model.training_loss(history, target)
        torch.manual_seed(77)
        expected = control.training_loss(mask_history(history, keep), target)
        assert torch.equal(actual.loss, expected.loss)
        assert torch.equal(actual.training_target, expected.training_target)
    torch.manual_seed(9)
    actual = model.sample(history, n_scenarios=8)
    torch.manual_seed(9)
    expected = control.sample(mask_history(history, keep), n_scenarios=8)
    assert torch.equal(actual, expected)
    assert len(seen) == 3
    assert all(torch.equal(value, mask_history(history, keep)) for value in seen)
    for key, value in before.items():
        assert torch.equal(value, sample[key])
    # The hook also leaves arrays owned by paired datasets unchanged.
    np.testing.assert_array_equal(data.val.raw_windows.history[0], before["history_raw"].numpy())
    np.testing.assert_array_equal(data.val.raw_windows.target[0], before["target_raw"].numpy())


def test_all_arms_have_identical_initial_weights_dates_and_parameter_count(experiment):
    cfg = experiment.cfg
    originals = None
    for keep in (0, 1, 5, 60):
        torch.manual_seed(cfg.seed)
        model = build_diffusion(cfg, n_assets=3, history_mask=history_mask_policy(cfg, keep))
        state = model.state_dict()
        if originals is None:
            originals = state
        assert all(torch.equal(value, originals[key]) for key, value in state.items())
    evidence = [json.loads((path / "forecast_metadata.json").read_text()) for path in experiment.archives.values()]
    assert all(record["target_dates"] == evidence[0]["target_dates"] for record in evidence)
    assert len({record["training_run"]["parameter_count"] for record in evidence}) == 1
    assert all(record["training_contract"]["data"] == evidence[0]["training_contract"]["data"] for record in evidence)


def test_k60_training_is_bitwise_identical_to_the_legacy_unmasked_path(experiment, tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "git_commit", lambda: pytest.fail("No Git operations"))
    runs.train_diffusion(experiment.config, tmp_path, device="cpu", resume=True, stop_after_epoch=2)
    control = torch.load(tmp_path / "last.pt", weights_only=False)
    full = torch.load(experiment.runs["K60"] / "last.pt", weights_only=False)
    assert control["history"] == full["history"]
    assert control.keys() == full.keys()
    for key, value in control["model_state_dict"].items():
        assert torch.equal(value, full["model_state_dict"][key])
    for key in ("torch_rng_state", "loader_rng_state"):
        assert torch.equal(control[key], full[key])


@pytest.mark.parametrize("requested", [None, 5, 60])
def test_k1_cannot_resume_with_another_policy_or_without_its_policy(experiment, tmp_path, requested):
    run = tmp_path / "run"
    shutil.copytree(experiment.runs["K1"], run)
    before = {p: file_sha256(p) for p in run.iterdir()}
    with pytest.raises(ValueError, match="History-mask policy differs"):
        runs.train_diffusion(experiment.config, run, resume=True, stop_after_epoch=3, history_keep_last=requested)
    assert all(file_sha256(path) == digest for path, digest in before.items())


def test_legitimate_masked_resume_matches_uninterrupted_training(experiment, tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "git_commit", lambda: pytest.fail("No Git operations"))
    resumed, uninterrupted = tmp_path / "resumed", tmp_path / "uninterrupted"
    shutil.copytree(experiment.runs["K1"], resumed)
    runs.train_diffusion(experiment.config, resumed, resume=True, device="cpu", stop_after_epoch=3, history_keep_last=1)
    runs.train_diffusion(experiment.config, uninterrupted, resume=True, device="cpu", stop_after_epoch=3, history_keep_last=1)
    a, b = (torch.load(path / "last.pt", weights_only=False) for path in (resumed, uninterrupted))
    assert a["history"] == b["history"] and a["epoch"] == 3
    assert all(torch.equal(value, b["model_state_dict"][key]) for key, value in a["model_state_dict"].items())
    assert json.loads((resumed / "data_identity.json").read_text())["training_contract"]["training"]["epochs"] == 15


@pytest.mark.parametrize("removed", ["marker", "contract_policy", "identity"])
def test_default_evaluation_rejects_missing_original_mask_evidence(experiment, tmp_path, removed):
    run = tmp_path / "run"
    shutil.copytree(experiment.runs["K1"], run)
    if removed == "marker":
        (run / "experiment.json").unlink()
    elif removed == "identity":
        (run / "data_identity.json").unlink()
    else:
        path = run / "data_identity.json"
        identity = json.loads(path.read_text())
        del identity["training_contract"]["history_mask"]
        path.write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="mask|contract"):
        evaluate_diffusion(run, device="cpu")


def test_automatic_evaluation_restores_original_mask_and_is_reproducible(experiment, tmp_path, monkeypatch):
    from diffusion_portfolio.evaluation import forecasts
    build = forecasts.build_diffusion
    seen = []

    def inspect(cfg, **kwargs):
        seen.append(kwargs["history_mask"]["keep_last"])
        return build(cfg, **kwargs)

    monkeypatch.setattr(forecasts, "build_diffusion", inspect)
    evaluate_diffusion(experiment.runs["K1"], limit=8, batch_size=4, device="cpu", output_dir=tmp_path / "repeat")
    assert seen == [1]
    with np.load(tmp_path / "repeat/validation_scenarios.npz") as a, np.load(experiment.archives["K1"] / "validation_scenarios.npz") as b:
        assert all(np.array_equal(a[key], b[key]) for key in a.files)


def test_comparison_accepts_matched_baselines_but_rejects_altered_policy(experiment, tmp_path):
    report = compare_synthetic_forecasts(experiment.baseline, experiment.archives["K1"], output_dir=tmp_path / "ok",
                                         block_length=2, n_bootstrap=20)
    assert report["training_contract"]["history_mask"]["keep_last"] == 1
    copied = tmp_path / "changed"
    shutil.copytree(experiment.archives["K1"], copied)
    path = copied / "forecast_metadata.json"
    metadata = json.loads(path.read_text())
    metadata["history_mask"]["keep_last"] = 5
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="history-mask policy differs"):
        compare_synthetic_forecasts(experiment.baseline, copied, output_dir=tmp_path / "rejected", block_length=2)


def test_historical_fifty_epoch_config_statistical_archives_remain_reusable(experiment, tmp_path):
    original = replace(experiment.cfg, training=replace(experiment.cfg.training, epochs=50))
    path = tmp_path / "original_config.yaml"
    path.write_text(yaml.safe_dump(asdict(original)))
    baseline = tmp_path / "original_baseline"
    benchmark_synthetic_forecasts(path, output_dir=baseline, n_scenarios=8, limit=8)
    before = {p: file_sha256(p) for p in baseline.rglob('*') if p.is_file()}
    report = compare_synthetic_forecasts(baseline, experiment.archives["K1"], output_dir=tmp_path / "comparison",
                                         block_length=2, n_bootstrap=20)
    assert report["training_contract"]["training"]["epochs"] == 15
    assert all(file_sha256(p) == digest for p, digest in before.items())


def test_diagnostic_formula_matches_independent_log_space_fixture():
    rng = np.random.default_rng(11)
    last_log = rng.normal(0, .01, (80, 3)) + np.array([.02, -.01, .03])
    means = .4 * last_log + np.array([.01, -.02, .005])
    errors = rng.normal(0, .005, last_log.shape)
    observed = np.expm1(means + errors)
    # Log ensemble mean is exactly means; mean arithmetic return differs.
    samples = np.expm1(means[:, None] + np.array([-.02, .02])[None, :, None])
    values = conditional_mean_diagnostics(samples, observed, np.expm1(last_log))
    assert values["pooled_lag_slope"] == pytest.approx(.4, abs=1e-14)
    np.testing.assert_allclose(values["mse_by_date"], (errors ** 2).mean(axis=1), rtol=1e-13)
    assert values["log_mean_mse"] == pytest.approx(np.mean(errors ** 2))
    assert not np.allclose(np.log1p(samples.mean(axis=1)), means)


def test_matched_arm_diagnostics_reuse_scores_and_pairing(experiment, tmp_path):
    report = diagnose_history_ablation(experiment.baseline, experiment.archives, output_dir=tmp_path / "all",
                                       block_length=2, n_bootstrap=30)
    assert report["n_dates"] == 8 and len(report["models"]) == 7
    assert len(report["paired_comparisons"]) == 18  # six arm pairs, three metrics
    scores = pd.read_csv(tmp_path / "all/per_date_scores.csv")
    with np.load(experiment.archives["K1"] / "validation_scenarios.npz") as archive:
        expected = evaluate_probabilistic_forecast(archive["scenarios"], archive["observed"])
    assert scores.K1_crps.mean() == pytest.approx(expected.crps_mean)
    assert scores.K1_energy_score.mean() == pytest.approx(expected.energy_score)
    pair = next(row for row in report["paired_comparisons"]
                if row["left"] == "K1" and row["right"] == "K60" and row["metric"] == "crps")
    assert pair["mean_difference"] == pytest.approx((scores.K1_crps - scores.K60_crps).mean())
    again = diagnose_history_ablation(experiment.baseline, experiment.archives, output_dir=tmp_path / "again",
                                      block_length=2, n_bootstrap=30)
    assert report["paired_comparisons"] == again["paired_comparisons"]


def test_diagnostics_reject_different_date_counts_instead_of_taking_intersection(experiment, tmp_path):
    short = tmp_path / "short"
    evaluate_diffusion(experiment.runs["K1"], limit=4, batch_size=4, device="cpu", output_dir=short)
    with pytest.raises(ValueError, match="different contracts, dates"):
        diagnose_history_ablation(experiment.baseline, {"K1": short, "K60": experiment.archives["K60"]},
                                  output_dir=tmp_path / "rejected", block_length=2)


def test_diagnostics_reject_mixing_two_and_three_epoch_pilots(experiment, tmp_path):
    run = tmp_path / "continued"
    shutil.copytree(experiment.runs["K1"], run)
    runs.train_diffusion(experiment.config, run, device="cpu", resume=True, stop_after_epoch=3, history_keep_last=1)
    inference = tmp_path / "inference"
    evaluate_diffusion(run, limit=8, batch_size=4, device="cpu", output_dir=inference)
    with pytest.raises(ValueError, match="completed budgets"):
        diagnose_history_ablation(experiment.baseline, {"K1": inference, "K60": experiment.archives["K60"]},
                                  output_dir=tmp_path / "rejected", block_length=2)


def test_earlier_forecast_cannot_see_modified_current_or_future_targets(experiment):
    from diffusion_portfolio.data.core import ReturnTable
    from diffusion_portfolio.data.dataset import build_window_datasets
    from diffusion_portfolio.data.registry import load_return_table

    cfg = experiment.cfg
    data, scaler = prepare_return_data(cfg)
    original = load_return_table(cfg.data.dataset, **cfg.data.source_options)
    changed = original.returns.copy()
    changed[100:] += .1  # All validation/test targets; no training/history changes.
    table = ReturnTable(original.dates, changed, original.columns)
    altered = build_window_datasets(scaler.transform(table), table, lookback=60, horizon=1,
                                    train_end=cfg.data.train_end, val_end=cfg.data.val_end,
                                    test_end=cfg.data.sample_end)
    assert not np.array_equal(data.val.raw_windows.target[0], altered.val.raw_windows.target[0])
    np.testing.assert_array_equal(data.train.model_windows.target_dates, altered.train.model_windows.target_dates)
    for keep in (0, 1, 5, 60):
        model = build_diffusion(cfg, n_assets=3, history_mask=history_mask_policy(cfg, keep)).eval()
        torch.manual_seed(51)
        before = model.sample(data.val[0]["history"][None], n_scenarios=4)
        torch.manual_seed(51)
        after = model.sample(altered.val[0]["history"][None], n_scenarios=4)
        assert torch.equal(before, after)


def test_legacy_checkpoint_cannot_be_retrofitted_with_a_mask(experiment, tmp_path):
    run = tmp_path / "legacy"
    shutil.copytree(experiment.runs["K60"], run)
    (run / "experiment.json").unlink()
    (run / "data_identity.json").unlink()
    before = {p: file_sha256(p) for p in run.iterdir()}
    with pytest.raises(ValueError, match="legacy checkpoints"):
        runs.train_diffusion(experiment.config, run, resume=True, history_keep_last=1, stop_after_epoch=3)
    assert all(file_sha256(path) == digest for path, digest in before.items())


def test_ablation_notebook_schema_ipython_and_real_cli_arguments(monkeypatch):
    import ast
    import runpy
    import shlex
    import sys
    import nbformat
    from IPython.core.inputtransformer2 import TransformerManager

    book = nbformat.read(ROOT / "notebooks/synthetic/synthetic_history_ablation.ipynb", as_version=4)
    reference = nbformat.read(ROOT / "notebooks/synthetic/synthetic_diffusion_comparison.ipynb", as_version=4)
    nbformat.validate(book)
    sync = next(c.source for c in book.cells if c.id == "github-sync")
    assert sync == next(c.source for c in reference.cells if c.id == "github-sync")
    assert 'BRANCH = "feature/synthetic-market"' in sync
    commands = []
    for cell in book.cells:
        if cell.cell_type != "code":
            continue
        assert cell.outputs == [] and cell.execution_count is None
        assert not any(term in cell.source for term in ("MODE", "%%bash", "subprocess", "%%capture"))
        transformed = TransformerManager().transform_cell(cell.source)
        compile(transformed, cell.id, "exec")
        for node in ast.walk(ast.parse(transformed)):
            if not isinstance(node, ast.Call) or getattr(node.func, "attr", None) != "system":
                continue
            tokens = shlex.split(ast.literal_eval(node.args[0]))
            scripts = [t for t in tokens if t.startswith("scripts/")]
            if not scripts:
                continue  # Never execute synchronization, install or test commands.
            script = scripts[0]
            monkeypatch.setattr(sys, "argv", tokens[tokens.index(script):])
            args = runpy.run_path(str(ROOT / script))["parse_args"]()
            commands.append((script, args))
            lines = cell.source.splitlines()
            index = next(i for i, line in enumerate(lines) if line.lstrip().startswith("!PYTHONUNBUFFERED"))
            while lines[index].rstrip().endswith("\\"):
                index += 1
            assert lines[index + 1].lstrip().startswith("assert _exit_code == 0")
    training = [args for script, args in commands if script == "scripts/train_diffusion.py"]
    assert len(training) == 2 and {args.history_keep_last for args in training} == {1, 60}
    assert all(args.stop_after_epoch == 2 and args.resume and args.config == str(CONFIG.relative_to(ROOT)) for args in training)
    inference = [args for script, args in commands if script == "scripts/evaluate_diffusion.py"]
    assert len(inference) == 2 and all(args.split == "validation" and args.limit == 64 for args in inference)
    assert len({args.run_dir for args in inference}) == len({args.output_dir for args in inference}) == 2
    assert {args.output_dir for args in training} == {args.run_dir for args in inference}
    manual = "\n".join(c.source for c in book.cells if c.cell_type == "markdown")
    assert all(f"--history-keep-last {keep}" in manual for keep in (0, 1, 5, 60))


def test_notebook_loss_plots_execute_on_partial_checkpoint_histories(experiment, monkeypatch):
    import nbformat
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")
    monkeypatch.setattr(plt, "show", lambda: None)  # Figures are inspected below on this headless backend.
    book = nbformat.read(ROOT / "notebooks/synthetic/synthetic_history_ablation.ipynb", as_version=4)
    namespace = {"display": lambda value: None}
    for cell in book.cells:
        if cell.cell_type == "code" and 'history.plot(' in cell.source:
            source = cell.source
            for label in ("K1", "K60"):
                source = source.replace(f"/kaggle/working/synthetic_context/{label.lower()}", str(experiment.runs[label]))
            exec(compile(source, cell.id, "exec"), namespace)
            history = namespace["history"]
            assert history.epoch.tolist() == [1, 2]
            lines = plt.gca().get_lines()
            assert len(lines) == 3  # train, validation, best-epoch marker
            np.testing.assert_array_equal(lines[0].get_ydata(), history.train_loss)
            np.testing.assert_array_equal(lines[1].get_ydata(), history.val_loss)
            plt.close("all")


def test_default_masked_evaluation_still_checks_checkpoint_byte_binding(experiment, tmp_path):
    run = tmp_path / "run"
    shutil.copytree(experiment.runs["K1"], run)
    # Same architecture/config, wrong arm's weights: epoch/loss alone are insufficient.
    shutil.copy2(experiment.runs["K5"] / "best.pt", run / "best.pt")
    with pytest.raises(ValueError, match="not bound"):
        evaluate_diffusion(run, device="cpu")
    assert not (run / "validation_scenarios.npz").exists()


def test_default_masked_evaluation_restores_policy_without_changing_legacy_interface(experiment, tmp_path):
    run = tmp_path / "run"
    shutil.copytree(experiment.runs["K1"], run)
    result = evaluate_diffusion(run, batch_size=4, device="cpu")
    assert result["n_validation"] == 40
    record = json.loads((run / "forecast_metadata.json").read_text())
    assert record["history_mask"]["keep_last"] == 1
