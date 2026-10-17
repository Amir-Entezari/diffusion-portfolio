"""Denoiser-only capacity changes, strict factorial alignment and paired effects."""
from copy import deepcopy
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
from diffusion_portfolio.evaluation import capacity_ablation as capacity
from diffusion_portfolio.evaluation.forecasts import evaluate_diffusion
from diffusion_portfolio.evaluation.history_ablation import diagnose_history_ablation
from diffusion_portfolio.evaluation.probabilistic import evaluate_probabilistic_forecast
from diffusion_portfolio.evaluation.synthetic_benchmarks import benchmark_synthetic_forecasts
from diffusion_portfolio.models.diffusion.build import build_diffusion
from diffusion_portfolio.models.encoders.history_mask import history_mask_policy, mask_history
from diffusion_portfolio.training import runs

ROOT = Path(__file__).parents[2]
LARGE = ROOT / "configs/ablations/synthetic_ar1_context.yaml"
SMALL = ROOT / "configs/ablations/synthetic_ar1_capacity_small.yaml"


def test_small_config_changes_only_denoiser_channels():
    large, small = load_config(LARGE), load_config(SMALL)
    assert small == replace(large, model=replace(large.model, channels=(16, 32, 64)))
    assert large.model.channels == (32, 64, 128)
    assert small.training.epochs == 15 and small.evaluation.n_scenarios == 256


def test_actual_architecture_counts_history_initialization_and_checkpoint_names():
    history_state, key_names = None, None
    for path, expected in ((LARGE, (1044417, 826433)), (SMALL, (565281, 347297))):
        cfg = load_config(path)
        for keep in (1, 60):
            torch.manual_seed(cfg.seed)
            model = build_diffusion(cfg, n_assets=12, history_mask=history_mask_policy(cfg, keep))
            counts = capacity.capacity_parameter_counts(cfg, 12)
            assert (counts["parameter_count"], counts["denoiser_parameter_count"]) == expected
            assert counts["history_encoder_parameter_count"] == 217984
            if history_state is None:
                history_state, key_names = model.history_encoder.state_dict(), set(model.state_dict())
            assert set(model.state_dict()) == key_names
            assert all(torch.equal(value, history_state[key]) for key, value in model.history_encoder.state_dict().items())
            for module in model.modules():
                if isinstance(module, torch.nn.GroupNorm):
                    assert module.num_channels % module.num_groups == 0


@pytest.mark.parametrize("config", [LARGE, SMALL])
@pytest.mark.parametrize("keep", [1, 60])
def test_actual_four_arm_cpu_forward_loss_backward_sampling_and_mask(config, keep):
    cfg = load_config(config)
    torch.manual_seed(cfg.seed)
    model = build_diffusion(cfg, n_assets=12, history_mask=history_mask_policy(cfg, keep))
    history, target = torch.randn(2, 60, 12), torch.randn(2, 1, 12)
    original = history.clone()
    control = build_diffusion(cfg, n_assets=12)
    control.load_state_dict(model.state_dict())
    seen = []
    model.history_encoder.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
    prediction = model.predict_model_output(torch.zeros(2, 12), torch.tensor([0, 99]), history)
    assert prediction.shape == (2, 12) and torch.isfinite(prediction).all()
    for training in (True, False):
        model.train(training)
        torch.manual_seed(93)
        actual = model.training_loss(history, target)
        torch.manual_seed(93)
        expected = control.training_loss(mask_history(history, keep), target)
        assert actual.prediction.shape == (2, 12) and torch.equal(actual.loss, expected.loss)
        assert torch.equal(actual.training_target, expected.training_target)
    actual.loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    torch.manual_seed(9)
    samples = model.sample(history, n_scenarios=2)
    torch.manual_seed(9)
    expected = control.sample(mask_history(history, keep), n_scenarios=2)
    assert samples.shape == (2, 2, 12) and torch.isfinite(samples).all()
    assert torch.equal(samples, expected)
    assert torch.equal(history, original)
    assert all(torch.equal(value, mask_history(history, keep)) for value in seen)


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    """Tiny CPU fixtures use true training histories, not fabricated completion.

    Data size, conditioning widths and diffusion steps are reduced equally in
    all arms; the actual named channel widths, lookback and 15-epoch budget stay.
    No full-size scientific model is trained or evaluated.
    """
    root = tmp_path_factory.mktemp("capacity-ablation")
    cfg = load_config(LARGE)
    dates = pd.bdate_range("2000-01-03", periods=180)
    cfg = replace(cfg, data=replace(cfg.data, sample_start=str(dates[0].date()),
                  sample_end=str(dates[-1].date()), train_end=str(dates[99].date()), val_end=str(dates[139].date()),
                  source_options={"n_assets": 3, "n_observations": 180, "n_factors": 2, "seed": 42,
                                  "start_date": str(dates[0].date()), "ar_coefficients": .4}),
                  model=replace(cfg.model, condition_dim=8, history_hidden_dim=16, time_embed_dim=8, diffusion_steps=4),
                  evaluation=replace(cfg.evaluation, n_scenarios=8))
    configs, run_dirs, archives, pilots, pilot_runs = {}, {}, {}, {}, {}
    for name, channels in capacity.CAPACITY_CHANNELS.items():
        path = root / f"{name}.yaml"
        path.write_text(yaml.safe_dump(asdict(replace(cfg, model=replace(cfg.model, channels=channels)))))
        configs[name] = path
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runs, "git_commit", lambda: None)  # Never run Git.
        for label, (keep, size) in capacity.CAPACITY_ARMS.items():
            run_dirs[label], archives[label] = root / label, root / f"{label}_inference"
            if size == "Small":
                runs.train_diffusion(configs[size], run_dirs[label], device="cpu", resume=True,
                                     stop_after_epoch=2, history_keep_last=keep)
                pilot_runs[label] = root / f"{label}_pilot_run"
                shutil.copytree(run_dirs[label], pilot_runs[label])
                pilots[label] = root / f"{label}_pilot_inference"
                evaluate_diffusion(pilot_runs[label], limit=8, batch_size=16, device="cpu", output_dir=pilots[label])
            runs.train_diffusion(configs[size], run_dirs[label], device="cpu", resume=True, history_keep_last=keep)
            evaluate_diffusion(run_dirs[label], batch_size=16, device="cpu", output_dir=archives[label])
    baseline = root / "baseline"
    benchmark_synthetic_forecasts(configs["Large"], output_dir=baseline, n_scenarios=8)
    return SimpleNamespace(root=root, cfg=cfg, configs=configs, runs=run_dirs, archives=archives,
                           pilot_runs=pilot_runs, pilots=pilots, baseline=baseline)


@pytest.mark.parametrize("old,new", [("Large", "Small"), ("Small", "Large")])
def test_capacity_config_cannot_be_exchanged_on_resume(experiment, old, new):
    run = experiment.runs[f"K1-{old}"]
    before = {p: file_sha256(p) for p in run.iterdir()}
    with pytest.raises(ValueError, match="Immutable training contract differs"):
        runs.train_diffusion(experiment.configs[new], run, resume=True, history_keep_last=1)
    assert all(file_sha256(p) == digest for p, digest in before.items())


def test_small_history_policy_and_evaluation_capacity_cannot_be_switched(experiment, tmp_path):
    run = experiment.runs["K1-Small"]
    with pytest.raises(ValueError, match="History-mask policy differs"):
        runs.train_diffusion(experiment.configs["Small"], run, resume=True, history_keep_last=60)
    copied = tmp_path / "run"
    shutil.copytree(run, copied)
    shutil.copy2(experiment.configs["Large"], copied / "config.yaml")
    with pytest.raises(ValueError, match="Immutable training contract differs"):
        evaluate_diffusion(copied, device="cpu")
    for label in ("K1-Small", "K60-Small"):
        metadata = json.loads((experiment.archives[label] / "forecast_metadata.json").read_text())
        assert metadata["history_mask"]["keep_last"] == capacity.CAPACITY_ARMS[label][0]
        assert metadata["training_contract"]["model"]["channels"] == [16, 32, 64]


@pytest.fixture(scope="module")
def full_result(experiment):
    output = experiment.root / "full_comparison"
    report = capacity.diagnose_capacity_ablation(experiment.baseline, experiment.archives,
              run_dirs=experiment.runs, output_dir=output, block_length=2, n_bootstrap=30)
    return SimpleNamespace(report=report, output=output)


def test_full_scores_curves_counts_and_checkpoint_gaps(experiment, full_result):
    report, output = full_result.report, full_result.output
    assert report["n_dates"] == 40 and not report["pilot"]
    assert len(report["models"]) == 7 and len(report["contrasts"]) == 15
    assert (output / "loss_curves.png").stat().st_size > 1000
    scores = pd.read_csv(output / "per_date_scores.csv")
    curves = pd.read_csv(output / "training_history.csv")
    for row in report["models"][:4]:
        label = row["model"]
        with np.load(experiment.archives[label] / "validation_scenarios.npz") as archive:
            metric = evaluate_probabilistic_forecast(archive["scenarios"], archive["observed"])
        assert scores[f"{label}_crps"].mean() == pytest.approx(metric.crps_mean)
        assert scores[f"{label}_energy_score"].mean() == pytest.approx(metric.energy_score)
        history = curves.loc[curves.model == label]
        best = history.loc[history.epoch == row["checkpoint_epoch"]].iloc[0]
        assert row["selected_train_val_gap"] == pytest.approx(best.val_loss - best.train_loss)
        assert row["final_train_val_gap"] == pytest.approx(history.val_loss.iloc[-1] - history.train_loss.iloc[-1])
        assert history.epoch.tolist() == list(range(1, 16))
    calibration = pd.read_csv(output / "calibration.csv")
    assert len(calibration) == 35 and set(calibration.level) == {.5, .8, .9, .95, .99}


def test_two_epoch_pilot_is_separate_from_completed_large_runs(experiment, tmp_path):
    report = capacity.diagnose_capacity_ablation(experiment.baseline, experiment.pilots,
              run_dirs=experiment.pilot_runs, pilot=True, output_dir=tmp_path / "pilot", block_length=2, n_bootstrap=20)
    assert report["pilot"] and len(report["contrasts"]) == 3
    assert all(row["contrast"].startswith("context_Small_") for row in report["contrasts"])
    mixed = dict(experiment.archives, **experiment.pilots)
    with pytest.raises(ValueError, match="completed 15-epoch"):
        capacity.diagnose_capacity_ablation(experiment.baseline, mixed, run_dirs=experiment.runs, output_dir=tmp_path / "bad")
    with pytest.raises(ValueError, match="pilot.*labels"):
        capacity.diagnose_capacity_ablation(experiment.baseline, mixed, pilot=True, output_dir=tmp_path / "bad")


def test_full_comparison_rejects_a_limited_completed_archive(experiment, tmp_path):
    limited = tmp_path / "limited"
    evaluate_diffusion(experiment.runs["K1-Small"], limit=8, batch_size=16, device="cpu", output_dir=limited)
    archives = {**experiment.archives, "K1-Small": limited}
    with pytest.raises(ValueError, match="all validation targets"):
        capacity.diagnose_capacity_ablation(experiment.baseline, archives, run_dirs=experiment.runs, output_dir=tmp_path / "bad")


@pytest.mark.parametrize("change,match", [("fingerprint", "fingerprint mismatch"), ("columns", "Asset order"),
                                        ("dates", "dates"), ("observed", "observations")])
def test_archive_identity_alignment_stays_strict(experiment, tmp_path, change, match):
    copied = tmp_path / "changed"
    shutil.copytree(experiment.archives["K1-Small"], copied)
    meta_path = copied / "forecast_metadata.json"
    meta = json.loads(meta_path.read_text())
    if change == "fingerprint":
        meta["source_identity"]["train_validation_panel_sha256"] = "0" * 64
    else:
        path = copied / "validation_scenarios.npz"
        with np.load(path) as saved:
            data = {key: saved[key].copy() for key in saved.files}
        if change == "columns":
            data["columns"] = data["columns"][::-1]
        elif change == "dates":
            data["dates"][0] -= np.timedelta64(1, "D")
        else:
            data["observed"][0, 0] += .001
        np.savez(path, **data)
        meta["archive_sha256"] = file_sha256(path)
    meta_path.write_text(json.dumps(meta))
    archives = {**experiment.archives, "K1-Small": copied}
    with pytest.raises(ValueError, match=match):
        capacity.diagnose_capacity_ablation(experiment.baseline, archives, run_dirs=experiment.runs, output_dir=tmp_path / "bad")


def test_label_mapping_is_verified_not_inferred_from_paths(experiment, tmp_path):
    archives = {**experiment.archives, "K1-Small": experiment.archives["K60-Small"]}
    with pytest.raises(ValueError, match="label K1-Small"):
        capacity.diagnose_capacity_ablation(experiment.baseline, archives, run_dirs=experiment.runs, output_dir=tmp_path / "bad")


@pytest.mark.parametrize("field,key,value", [(None, "seed", 43), ("training", "learning_rate", .001), ("training", "weight_decay", .1),
                                          ("model", "history_hidden_dim", 32), ("model", "condition_dim", 16),
                                          ("model", "time_embed_dim", 16), ("evaluation", "report_distribution_metrics", False)])
def test_only_declared_factor_differences_are_allowed(experiment, tmp_path, monkeypatch, field, key, value):
    loader = capacity.load_matched_synthetic_forecasts

    def changed(baseline, directory):
        matched = loader(baseline, directory)
        if Path(directory) == experiment.archives["K1-Small"]:
            matched.contract = deepcopy(matched.contract)
            if field is None:
                matched.contract[key] = value
            else:
                matched.contract[field][key] = value
        return matched

    monkeypatch.setattr(capacity, "load_matched_synthetic_forecasts", changed)
    with pytest.raises(ValueError, match="differ beyond"):
        capacity.diagnose_capacity_ablation(experiment.baseline, experiment.archives,
                                            run_dirs=experiment.runs, output_dir=tmp_path / "bad")


def test_unchanged_history_diagnostic_still_rejects_cross_capacity_arms(experiment, tmp_path):
    with pytest.raises(ValueError, match="different contracts"):
        diagnose_history_ablation(experiment.baseline,
          {"K1": experiment.archives["K1-Small"], "K60": experiment.archives["K60-Large"]}, output_dir=tmp_path / "bad")


def test_missing_or_wrong_training_history_is_not_invented(experiment, tmp_path):
    with pytest.raises(ValueError, match="verified training history"):
        capacity.diagnose_capacity_ablation(experiment.baseline, experiment.archives, output_dir=tmp_path / "missing")
    runs_wrong = {**experiment.runs, "K1-Large": experiment.runs["K60-Large"]}
    with pytest.raises(ValueError, match="verified training history"):
        capacity.diagnose_capacity_ablation(experiment.baseline, experiment.archives, run_dirs=runs_wrong, output_dir=tmp_path / "wrong")


def test_planned_contrast_and_interaction_signs():
    # Deterministic score fixture, not simulated or trained financial results.
    series = np.arange(10, dtype=float)
    levels = {"K1-Large": 10, "K60-Large": 14, "K1-Small": 9, "K60-Small": 11}
    scores = {label: {m: series + value for m in capacity.SCORE_NAMES} for label, value in levels.items()}
    effects = capacity.capacity_contrasts(scores)
    for m in capacity.SCORE_NAMES:
        np.testing.assert_array_equal(effects[f"capacity_K1_{m}"], -np.ones(10))
        np.testing.assert_array_equal(effects[f"capacity_K60_{m}"], -3 * np.ones(10))
        np.testing.assert_array_equal(effects[f"context_Large_{m}"], 4 * np.ones(10))
        np.testing.assert_array_equal(effects[f"context_Small_{m}"], 2 * np.ones(10))
        np.testing.assert_array_equal(effects[f"interaction_{m}"], -2 * np.ones(10))


def test_one_joint_bootstrap_receives_per_date_interactions(experiment, tmp_path, monkeypatch):
    original, calls = capacity.moving_block_bootstrap, []

    def inspect(values, **kwargs):
        calls.append(values.copy())
        return original(values, **kwargs)

    monkeypatch.setattr(capacity, "moving_block_bootstrap", inspect)
    report = capacity.diagnose_capacity_ablation(experiment.baseline, experiment.archives,
              run_dirs=experiment.runs, output_dir=tmp_path / "result", block_length=3, n_bootstrap=25, bootstrap_seed=9)
    assert len(calls) == 1 and calls[0].shape == (40, 15)
    rows = pd.read_csv(tmp_path / "result/per_date_scores.csv")
    expected = np.column_stack([rows[row["contrast"]] for row in report["contrasts"]])
    np.testing.assert_allclose(calls[0], expected, rtol=1e-11, atol=1e-16)
    # Independently resample the same chronological rows for all 15 series.
    rng = np.random.default_rng(9)
    means = []
    for _ in range(25):
        starts = rng.integers(0, 38, size=14)
        indices = (starts[:, None] + np.arange(3)).ravel()[:40]
        means.append(calls[0][indices].mean(axis=0))
    bounds = np.quantile(means, [.025, .975], axis=0)
    np.testing.assert_allclose(bounds[0], [row["ci95_lower"] for row in report["contrasts"]])
    np.testing.assert_allclose(bounds[1], [row["ci95_upper"] for row in report["contrasts"]])


@pytest.mark.parametrize("key,value", [("batch_size", 8), ("sampling_seed", 42)])
def test_inference_protocol_cannot_change(experiment, tmp_path, key, value):
    copied = tmp_path / "changed"
    shutil.copytree(experiment.archives["K1-Small"], copied)
    path = copied / "forecast_metadata.json"
    metadata = json.loads(path.read_text())
    metadata[key] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="protocol requires"):
        capacity.diagnose_capacity_ablation(experiment.baseline, {**experiment.archives, "K1-Small": copied},
                                            run_dirs=experiment.runs, output_dir=tmp_path / "bad")


def test_new_notebook_schema_ipython_and_real_cli_arguments(monkeypatch):
    import ast
    import re
    import runpy
    import shlex
    import sys
    import nbformat
    from IPython.core.inputtransformer2 import TransformerManager

    book = nbformat.read(ROOT / "notebooks/synthetic/synthetic_capacity_ablation.ipynb", as_version=4)
    reference = nbformat.read(ROOT / "notebooks/synthetic/synthetic_history_ablation.ipynb", as_version=4)
    nbformat.validate(book)
    sync = next(cell.source for cell in book.cells if cell.id == "github-sync")
    assert sync == next(cell.source for cell in reference.cells if cell.id == "github-sync")
    assert 'BRANCH = "feature/synthetic-market"' in sync
    commands, manual = [], []

    def parse(command):
        tokens = shlex.split(command)
        scripts = [token for token in tokens if token.startswith("scripts/")]
        if not scripts:
            return None
        script = scripts[0]
        monkeypatch.setattr(sys, "argv", tokens[tokens.index(script):])
        return script, runpy.run_path(str(ROOT / script))["parse_args"]()

    for cell in book.cells:
        if cell.cell_type == "markdown":
            for block in re.findall(r"```bash\n(.*?)```", cell.source, flags=re.S):
                for line in block.replace("\\\n", " ").splitlines():
                    if line.startswith("python "):
                        manual.append(parse(line))
            continue
        assert cell.outputs == [] and cell.execution_count is None
        assert not any(term in cell.source for term in ("MODE", "REVISION", "%%bash", "subprocess", "%%capture"))
        transformed = TransformerManager().transform_cell(cell.source)
        compile(transformed, cell.id, "exec")
        for node in ast.walk(ast.parse(transformed)):
            if not isinstance(node, ast.Call) or getattr(node.func, "attr", None) != "system":
                continue
            command = ast.literal_eval(node.args[0])
            parsed = parse(command)
            if parsed is not None:
                commands.append(parsed)
                lines = cell.source.splitlines()
                index = next(i for i, line in enumerate(lines) if line.lstrip().startswith("!PYTHONUNBUFFERED"))
                while lines[index].rstrip().endswith("\\"):
                    index += 1
                assert lines[index + 1].lstrip().startswith("assert _exit_code == 0")
    training = [args for script, args in commands if script == "scripts/train_diffusion.py"]
    assert len(training) == 2 and {args.history_keep_last for args in training} == {1, 60}
    assert all(args.config.endswith("synthetic_ar1_capacity_small.yaml") and args.resume and args.stop_after_epoch == 2 for args in training)
    inference = [args for script, args in commands if script == "scripts/evaluate_diffusion.py"]
    assert len(inference) == 2 and all(args.split == "validation" and args.limit == 64 and args.batch_size == 16 for args in inference)
    assert {args.output_dir for args in training} == {args.run_dir for args in inference}
    assert len({args.output_dir for args in inference}) == 2
    pilot = next(args for script, args in commands if script == "scripts/diagnose_capacity_ablation.py")
    assert pilot.pilot and set(pilot.diffusion_dirs) == {"K1-Small", "K60-Small"}
    assert set(pilot.run_dirs) == set(pilot.diffusion_dirs)
    full_training = [args for script, args in manual if script == "scripts/train_diffusion.py"]
    assert len(full_training) == 2 and all(args.resume and args.stop_after_epoch is None for args in full_training)
    full = next(args for script, args in manual if script == "scripts/diagnose_capacity_ablation.py")
    assert not full.pilot and set(full.diffusion_dirs) == set(capacity.CAPACITY_ARMS)


def test_new_notebook_plots_read_latest_partial_checkpoints(experiment, monkeypatch):
    import matplotlib.pyplot as plt
    import nbformat

    plt.switch_backend("Agg")
    monkeypatch.setattr(plt, "show", lambda: None)
    book = nbformat.read(ROOT / "notebooks/synthetic/synthetic_capacity_ablation.ipynb", as_version=4)
    namespace = {"Path": Path, "torch": torch, "pd": pd, "display": lambda value: None}
    for cell in book.cells:
        if cell.cell_type == "code" and "history.plot(" in cell.source:
            source = cell.source
            for keep in (1, 60):
                source = source.replace(f'SMALL_ROOT / "k{keep}_small"', f'Path({str(experiment.pilot_runs[f"K{keep}-Small"])!r})')
            exec(compile(source, cell.id, "exec"), namespace)
            assert namespace["history"].epoch.tolist() == [1, 2]
            lines = plt.gca().get_lines()
            np.testing.assert_array_equal(lines[0].get_ydata(), namespace["history"].train_loss)
            np.testing.assert_array_equal(lines[1].get_ydata(), namespace["history"].val_loss)
            assert len(lines) == 3
            plt.close("all")
