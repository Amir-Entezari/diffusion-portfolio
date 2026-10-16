"""Stage 4C specification tests; all statistical expectations concern log returns."""
from argparse import Namespace
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from diffusion_portfolio.config import load_config
from diffusion_portfolio.data.sources.synthetic_market import simulate_market

ROOT = Path(__file__).parents[2]
SCENARIOS = ("synthetic_student_t", "synthetic_mean_reverting", "synthetic_whipsaw")
REFERENCES = json.loads((ROOT / "tests/fixtures/synthetic_market_v1_hashes.json").read_text())


@pytest.mark.parametrize("case", REFERENCES["cases"], ids=lambda c: c["scenario"] + ("-custom" if c["options"] else "-default"))
def test_stage4ab_arrays_match_captured_pre_extension_hashes(case):
    result = simulate_market(case["scenario"], **case["options"])
    arrays = {"returns": result.returns, "log_returns": result.log_returns, "prices": result.prices,
              "dates": result.dates.values.astype("datetime64[ns]"), "columns": np.asarray(result.columns),
              **result.ground_truth}
    hashes = {name: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() for name, value in arrays.items()}
    assert hashes == case["hashes"], f"Original arrays changed; fixture NumPy={REFERENCES['numpy_version']}, current={np.__version__}"


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_reproducibility_prefix_prices_and_global_rng_isolation(scenario):
    state = np.random.get_state()
    kwargs = dict(n_assets=5, n_factors=3, n_observations=200, seed=37)
    first = simulate_market(scenario, **kwargs)
    again = simulate_market(scenario, **kwargs)
    longer = simulate_market(scenario, **{**kwargs, "n_observations": 400})
    assert first.returns.shape == (200, 5)
    assert np.isfinite(first.returns).all() and (first.returns > -1).all()
    for name in ("returns", "log_returns", "prices"):
        np.testing.assert_array_equal(getattr(first, name), getattr(again, name))
        np.testing.assert_array_equal(getattr(first, name), getattr(longer, name)[:200])
    for name, value in first.ground_truth.items():
        np.testing.assert_array_equal(value, longer.ground_truth[name][:200])
    assert not np.array_equal(first.returns, simulate_market(scenario, **{**kwargs, "seed": 38}).returns)
    np.testing.assert_allclose(first.prices / np.vstack((np.full(5, 100.), first.prices[:-1])) - 1,
                               first.returns, atol=2e-15)
    for before, after in zip(state, np.random.get_state()):
        np.testing.assert_array_equal(before, after)
    json.dumps(first.parameters, allow_nan=False)


@pytest.fixture(scope="module")
def student_market():
    # nu=12 has finite eighth moments, enabling a finite-variance kurtosis SE.
    return simulate_market("synthetic_student_t", n_assets=4, degrees_of_freedom=12,
                           n_observations=60000, seed=14, include_prices=False)


def test_student_common_scale_is_applied_to_whole_gaussian_vector():
    kwargs = dict(n_assets=5, n_factors=3, n_observations=300, seed=11)
    gaussian = simulate_market("synthetic_gaussian", **kwargs)
    student = simulate_market("synthetic_student_t", **kwargs)
    mu = np.asarray(student.parameters["drift"])[0]
    scales = student.ground_truth["student_t_scales"]
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(11, spawn_key=(4,))))
    np.testing.assert_array_equal(scales, np.sqrt(6 / rng.chisquare(8, size=300)))
    np.testing.assert_allclose(student.log_returns - mu, (gaussian.log_returns - mu) * scales[:, None], atol=2e-17)
    assert not student.ground_truth["jump_events"].any()


def test_student_covariance_and_kurtosis_with_monte_carlo_uncertainty(student_market):
    result = student_market
    x, n, nu = result.log_returns, len(result.log_returns), result.parameters["degrees_of_freedom"]
    mean = np.asarray(result.parameters["drift"])[0]
    cov = np.asarray(result.parameters["conditional_log_covariance"])[0]
    c = (nu - 2) / (nu - 4)
    se = np.sqrt((c * np.outer(cov.diagonal(), cov.diagonal()) + (2 * c - 1) * cov ** 2) / n)
    assert np.all(abs(x.mean(0) - mean) < 6 * np.sqrt(cov.diagonal() / n))
    assert np.all(abs(np.cov(x, rowvar=False) - cov) < 6 * se)
    z = (x - mean) / np.sqrt(cov.diagonal())
    m4, m6 = 3 * c, 15 * (nu - 2) ** 2 / ((nu - 4) * (nu - 6))
    m8 = 105 * (nu - 2) ** 3 / ((nu - 4) * (nu - 6) * (nu - 8))
    kurtosis_se = np.sqrt((m8 - m4 ** 2 + 4 * m4 ** 3 - 4 * m4 * m6) / n)
    observed = (z ** 4).mean(0) / (z ** 2).mean(0) ** 2 - 3
    assert np.all(abs(observed - 6 / (nu - 4)) < 6 * kurtosis_se)


def test_default_student_tail_and_joint_exceedances_against_matched_gaussian():
    kwargs = dict(n_assets=2, n_factors=1, factor_loadings=0, idiosyncratic_vol=.01,
                  n_observations=60000, seed=25, include_prices=False, drift=0)
    student = simulate_market("synthetic_student_t", **kwargs)
    gaussian = simulate_market("synthetic_gaussian", **kwargs)
    z, g = student.log_returns / .01, gaussian.log_returns / .01
    probability = 2 * stats.t.sf(3 / np.sqrt(6 / 8), df=8)
    observed = (abs(z) > 3).mean(0)
    assert np.all(abs(observed - probability) < 6 * np.sqrt(probability * (1 - probability) / len(z)))
    assert observed.mean() > (abs(g) > 3).mean() + .002
    # Independent pre-scale assets still share Student-t tail events.
    expected_joint = integrate.quad(lambda w: stats.norm.sf(3 * np.sqrt(w / 6)) ** 2 * stats.chi2.pdf(w, 8),
                                    0, np.inf, epsabs=1e-12)[0]
    joint = ((z[:, 0] < -3) & (z[:, 1] < -3)).mean()
    assert abs(joint - expected_joint) < 6 * np.sqrt(expected_joint * (1 - expected_joint) / len(z))
    assert expected_joint > 5 * stats.norm.sf(3) ** 2
    assert np.isfinite(z).all()


@pytest.mark.parametrize("scenario,control,coefficients", [
    ("synthetic_mean_reverting", "synthetic_gaussian", [.4]),
    ("synthetic_whipsaw", "synthetic_regime", [.1, -.65]),
])
def test_causal_ar_recursion_and_stationary_fixed_regime_initialization(scenario, control, coefficients):
    kwargs = dict(n_assets=3, n_observations=400, seed=18)
    result, reference = simulate_market(scenario, **kwargs), simulate_market(control, **kwargs)
    states = result.ground_truth["regimes"]
    np.testing.assert_array_equal(states, reference.ground_truth["regimes"])
    means = np.asarray(result.parameters["drift"])[states]
    drivers = reference.log_returns - means
    deviations = result.log_returns - means
    np.testing.assert_array_equal(result.log_returns[0], reference.log_returns[0])
    phi = np.asarray(coefficients)[states[1:]]
    expected = phi[:, None] * deviations[:-1] + np.sqrt(1 - phi ** 2)[:, None] * drivers[1:]
    np.testing.assert_allclose(deviations[1:], expected, atol=2e-17)
    if scenario == "synthetic_whipsaw":
        assert result.parameters["conditional_log_covariance"] is None
        assert result.parameters["innovation_covariances"] != result.parameters["driver_covariances"]
    for seed in range(12):
        # Exact initial Gaussian law; no zero initialization or length-dependent burn-in.
        first = simulate_market("synthetic_mean_reverting", seed=seed, n_observations=1)
        baseline = simulate_market("synthetic_gaussian", seed=seed, n_observations=1)
        np.testing.assert_array_equal(first.returns, baseline.returns)


def test_stationary_ar_mean_covariance_and_multilag_autocovariance():
    result = simulate_market("synthetic_mean_reverting", n_assets=4, n_observations=60000, include_prices=False, seed=23)
    phi = .4
    x = result.log_returns
    cov = np.asarray(result.parameters["conditional_log_covariance"])[0]
    mean = np.asarray(result.parameters["drift"])[0]
    n = len(x)
    assert np.all(abs(x.mean(0) - mean) < 6 * np.sqrt(cov.diagonal() / n * (1 + phi) / (1 - phi)))
    se = np.sqrt((cov ** 2 + np.outer(cov.diagonal(), cov.diagonal())) / (n - 1) * (1 + phi ** 2) / (1 - phi ** 2))
    assert np.all(abs(np.cov(x, rowvar=False) - cov) < 6 * se)
    y = x - mean
    a = (1 + phi ** 2) / (1 - phi ** 2)
    for lag in (1, 3, 5):
        observed = y[lag:].T @ y[:-lag] / (n - lag)
        b = phi ** (2 * lag) * (2 * lag + 1 + 2 * phi ** 2 / (1 - phi ** 2))
        se = np.sqrt((np.outer(cov.diagonal(), cov.diagonal()) * a + cov ** 2 * b) / (n - lag))
        assert np.all(abs(observed - phi ** lag * cov) < 6 * se)


@pytest.mark.parametrize("coefficient", [-.95, 0., .95])
def test_valid_ar_coefficients_remain_stable_and_zero_matches_iid(coefficient):
    result = simulate_market("synthetic_mean_reverting", ar_coefficients=coefficient,
                             n_assets=3, n_observations=500, seed=9)
    assert np.isfinite(result.returns).all() and np.isfinite(result.prices).all()
    if coefficient == 0:
        baseline = simulate_market("synthetic_gaussian", n_assets=3, n_observations=500, seed=9)
        np.testing.assert_array_equal(result.returns, baseline.returns)


def test_whipsaw_negative_within_spell_dependence_reversals_and_transitions():
    result = simulate_market("synthetic_whipsaw", n_assets=4, n_observations=60000, include_prices=False, seed=31)
    states = result.ground_truth["regimes"]
    y = result.log_returns - np.asarray(result.parameters["drift"])[states]
    reversals, raw_reversals, correlations = [], [], []
    for state in range(2):
        pairs = (states[1:] == state) & (states[:-1] == state)
        previous, current = y[:-1][pairs], y[1:][pairs]
        correlations.append(np.corrcoef(previous[:, 0], current[:, 0])[0, 1])
        reversals.append(np.mean(previous * current < 0))
        raw_reversals.append(np.mean(result.log_returns[:-1][pairs] * result.log_returns[1:][pairs] < 0))
        slope = np.sum(previous * current, axis=0) / np.sum(previous ** 2, axis=0)
        assert np.all(abs(slope - result.parameters["ar_coefficients"][state]) < .05)
        following = states[1:][states[:-1] == state]
        p = np.asarray(result.parameters["transition_matrix"])[state]
        error = abs(np.bincount(following, minlength=2) / len(following) - p)
        assert np.all(error < 6 * np.sqrt(p * (1 - p) / len(following)))
    assert correlations[1] < -.5 < 0 < correlations[0]
    assert reversals[1] > reversals[0] + .15
    assert raw_reversals[1] > raw_reversals[0] + .15
    edges = np.r_[0, np.flatnonzero(np.diff(states)) + 1, len(states)]
    durations, labels = np.diff(edges)[1:-1], states[edges[1:-2]]
    for state in range(2):
        spells = durations[labels == state]
        stay = result.parameters["transition_matrix"][state][state]
        se = np.sqrt(stay / (1 - stay) ** 2 / len(spells))
        assert abs(spells.mean() - 1 / (1 - stay)) < 6 * se


@pytest.mark.parametrize("scenario,options", [
    ("synthetic_student_t", {"degrees_of_freedom": v}) for v in (4, 2, -1, np.nan, np.inf, True)
] + [
    ("synthetic_mean_reverting", {"ar_coefficients": v}) for v in (-1, 1, 1.1, np.nan, [0, .4])
] + [
    ("synthetic_student_t", {"ar_coefficients": .2}),
    ("synthetic_mean_reverting", {"innovation_distribution": "student_t"}),
    ("synthetic_gaussian", {"ar_coefficients": .4}),
    ("synthetic_gaussian", {"degrees_of_freedom": 8}),
    ("synthetic_whipsaw", {"ar_coefficients": [.1, .5]}),
    ("synthetic_whipsaw", {"jump_probabilities": [.01, .02]}),
])
def test_invalid_or_unvalidated_mechanism_combinations(scenario, options):
    with pytest.raises(ValueError):
        simulate_market(scenario, **options)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_new_sources_pipeline_saved_scaler_and_cpu_model_smoke(scenario, tmp_path, monkeypatch):
    import yaml
    from diffusion_portfolio.data import preparation, TrainStandardizer
    from diffusion_portfolio.data.diagnostics import run_data_check
    from diffusion_portfolio.data.preparation import save_standardizer
    from diffusion_portfolio.data.registry import load_return_table

    cfg = load_config(ROOT / f"configs/{scenario}.yaml")
    original = load_config(ROOT / "configs/synthetic_gaussian.yaml")
    assert cfg.model == original.model and cfg.training == original.training
    assert not cfg.data.subtract_risk_free and cfg.data.lookback == 60 and cfg.data.horizon == 1
    monkeypatch.setattr(preparation, "load_daily_risk_free", lambda: pytest.fail("Synthetic source loaded RF"))
    table = load_return_table(scenario, **cfg.data.source_options)
    result = simulate_market(scenario, **cfg.data.source_options)
    assert set(vars(table)) == {"dates", "returns", "columns"}
    np.testing.assert_array_equal(table.returns, result.returns)
    assert table.returns.shape == (8000, 12)
    datasets, scaler = preparation.prepare_return_data(cfg)
    assert [len(datasets.train), len(datasets.val), len(datasets.test)] == [4940, 1500, 1500]
    np.testing.assert_array_equal(scaler.mean, table.returns[:5000].mean(0))
    np.testing.assert_array_equal(scaler.std, table.returns[:5000].std(0))
    assert table.dates[4999] == pd.Timestamp(cfg.data.train_end)
    assert table.dates[6499] == pd.Timestamp(cfg.data.val_end)
    save_standardizer(tmp_path / "scaler.npz", scaler)
    with monkeypatch.context() as saved_scaler:
        saved_scaler.setattr(TrainStandardizer, "fit", lambda *a, **k: pytest.fail("Saved scaler refitted"))
        restored, _ = preparation.prepare_return_data(replace(cfg, seed=43), standardizer_path=tmp_path / "scaler.npz")
    np.testing.assert_array_equal(datasets.val.model_windows.history, restored.val.model_windows.history)
    # Change only fixture model size; the checked-in config retains the full baseline.
    small = replace(cfg, model=replace(cfg.model, condition_dim=8, history_hidden_dim=16,
                                       channels=(8,), time_embed_dim=8, diffusion_steps=4))
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml.safe_dump(asdict(small)))
    report = run_data_check(Namespace(config=path, model_smoke=True, smoke_batch_size=2, device="cpu"))
    assert report["model_smoke"]["gradients_finite"] and report["model_smoke"]["prediction_shape"] == [2, 12]
    # Post-training observations cannot change the fitted scale or training windows.
    table.returns[5000:] += 100
    monkeypatch.setattr(preparation, "load_return_table", lambda *a, **k: table)
    altered, changed_scaler = preparation.prepare_return_data(cfg)
    np.testing.assert_array_equal(changed_scaler.mean, scaler.mean)
    np.testing.assert_array_equal(altered.train.model_windows.history, datasets.train.model_windows.history)


def test_dynamics_pairs_never_cross_regime_boundaries_even_if_endpoints_match():
    from diffusion_portfolio.evaluation.synthetic_market import within_regime_dynamics

    states = np.array([0, 0, 1, 0, 0, 0, 1, 1, 1, 1])
    x = np.array([1, -2, 5, -3, 2, 4, -1, 3, -2, 1.], dtype=float)[:, None]
    means = np.array([[-.1], [1.5]])
    rows = within_regime_dynamics(x, states, means, np.array([.1, -.65]),
                                  fixed_regime=False, lags=(1, 2))
    for row in rows:
        state, lag = row["state"], row["lag"]
        indices = [i for i in range(lag, len(x)) if np.all(states[i - lag:i + 1] == state)]
        assert row["pairs"] == len(indices)
        assert row["theoretical_correlation"] is None
        raw_a, raw_b = x[np.array(indices) - lag], x[indices]
        a, b = raw_a - means[state], raw_b - means[state]
        assert row["raw_sign_reversal_rate"] == np.mean(raw_a * raw_b < 0)
        assert row["centered_sign_reversal_rate"] == np.mean(a * b < 0)
        if len(indices) >= 2:
            np.testing.assert_allclose(row["observed_correlation_by_asset"], [np.corrcoef(a[:, 0], b[:, 0])[0, 1]])


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_extended_diagnostics_cutoff_tail_definitions_and_plots(scenario):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from diffusion_portfolio.evaluation.synthetic_market import market_diagnostics, plot_market_diagnostics, plot_stress_diagnostics

    result = simulate_market(scenario, n_assets=3, n_observations=400, seed=13)
    end = result.dates[299]
    report = market_diagnostics(result, end_date=end)
    assert report["observations"] == 300
    if scenario == "synthetic_whipsaw":
        assert report["moments"]["whipsaw"]["configured_covariance"] is None
        assert all(row["theoretical_correlation"] is None for row in report["dynamics"])
        assert all(row["configured_mean_log_volatility"] is None for row in report["summary"])
    if scenario == "synthetic_student_t":
        assert report["tails"]["configured_excess_kurtosis"] == 1.5
        z = (result.log_returns[:300] - np.asarray(result.parameters["drift"])[0]) / np.sqrt(np.diag(result.parameters["driver_covariances"][0]))
        for row in report["tails"]["exceedances"]:
            threshold = row["threshold_log_standard_deviations"]
            lower = (z < -threshold).astype(float)
            np.testing.assert_array_equal(row["observed_two_sided_by_asset"], (abs(z) > threshold).mean(0))
            np.testing.assert_array_equal(row["observed_lower_joint_probabilities"], lower.T @ lower / 300)
        np.testing.assert_allclose(report["tails"]["observed_excess_kurtosis"], stats.kurtosis(z, axis=0, fisher=True, bias=False))
    json.dumps(report, allow_nan=False)
    result.log_returns[300:] = np.nan
    result.returns[300:] = np.nan
    result.ground_truth["regimes"][300:] = 99
    assert market_diagnostics(result, end_date=end) == report
    for plot in (plot_market_diagnostics, plot_stress_diagnostics):
        fig = plot(result, end_date=end)
        plt.close(fig)


def test_stage4c_notebook_schema_syntax_branch_and_actual_cli_arguments(monkeypatch):
    import ast
    import runpy
    import shlex
    import sys

    nbformat = pytest.importorskip("nbformat")
    transform = pytest.importorskip("IPython.core.inputtransformer2").TransformerManager()
    book = nbformat.read(ROOT / "notebooks/synthetic/synthetic_market_stage4c_validation.ipynb", as_version=4)
    reference = nbformat.read(ROOT / "notebooks/synthetic/synthetic_market_validation.ipynb", as_version=4)
    nbformat.validate(book)
    cells = {c.id: c for c in book.cells}
    assert cells["github-sync"].source == next(c.source for c in reference.cells if c.id == "github-sync")
    assert 'BRANCH = "feature/synthetic-market"' in cells["github-sync"].source
    assert "end_date=cfg.data.train_end" in cells["diagnostic-reports"].source
    seen = []
    for cell in book.cells:
        if cell.cell_type != "code":
            continue
        assert cell.outputs == [] and cell.execution_count is None
        assert not any(s in cell.source for s in ("subprocess", "%%bash", "MODE", "%%capture"))
        transformed = transform.transform_cell(cell.source)
        compile(transformed, cell.id, "exec")
        for node in ast.walk(ast.parse(transformed)):
            if not isinstance(node, ast.Call) or getattr(node.func, "attr", None) != "system":
                continue
            command = ast.literal_eval(node.args[0])
            if "scripts/data_smoke_test.py" not in command:
                continue  # Never execute Git, dependency-installation or test commands.
            argv = shlex.split(command)
            script = argv.index("scripts/data_smoke_test.py")
            monkeypatch.setattr(sys, "argv", argv[script:])
            namespace = runpy.run_path(str(ROOT / "scripts/data_smoke_test.py"), run_name="cli_validation_only")
            args = namespace["parse_args"]()
            cfg = load_config(ROOT / args.config)
            assert cfg.data.dataset in SCENARIOS and not cfg.data.subtract_risk_free
            assert args.model_smoke and args.device == "cpu" and args.smoke_batch_size == 2
            assert cell.source.splitlines()[-1].startswith("assert _exit_code == 0")
            seen.append(cfg.data.dataset)
    assert set(seen) == set(SCENARIOS)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_stage4c_optional_export_keeps_new_hidden_arrays_separate(scenario, tmp_path):
    nbformat = pytest.importorskip("nbformat")
    transform = pytest.importorskip("IPython.core.inputtransformer2").TransformerManager()
    book = nbformat.read(ROOT / "notebooks/synthetic/synthetic_market_stage4c_validation.ipynb", as_version=4)
    source = next(c.source for c in book.cells if c.id == "optional-export")
    result = simulate_market(scenario, n_assets=3, n_observations=80, include_prices=False)
    directory = tmp_path / scenario
    directory.mkdir()

    class Shell:
        def getoutput(self, command):
            assert command == "git rev-parse HEAD"
            return ["f" * 40]  # Fixture only, never execute Git.

    exec(transform.transform_cell(source), {"get_ipython": Shell, "markets": {scenario: result},
                                            "OUT": tmp_path, "np": np, "json": json})
    with np.load(directory / "observations.npz", allow_pickle=False) as saved:
        assert set(saved.files) == {"dates", "returns", "columns"}
        np.testing.assert_array_equal(saved["returns"], result.returns)
    with np.load(directory / "ground_truth.npz", allow_pickle=False) as saved:
        assert set(saved.files) == {"dates", *result.ground_truth}
        for name, values in result.ground_truth.items():
            np.testing.assert_array_equal(saved[name], values)
    assert json.loads((directory / "parameters.json").read_text())["source_revision"] == "f" * 40
