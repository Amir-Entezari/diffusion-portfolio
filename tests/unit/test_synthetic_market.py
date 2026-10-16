"""Controlled-market mathematics; simulation assumptions are not empirical realism."""
import numpy as np
import pandas as pd
import pytest

from diffusion_portfolio.data.sources.synthetic_market import simulate_market


def assert_gaussian_moments(values, mean, covariance):
    # Gaussian sample-mean and Wishart element standard errors, six-sigma bounds.
    n = len(values)
    np.testing.assert_array_less(np.abs(values.mean(0) - mean),
                                 6 * np.sqrt(np.diag(covariance) / n))
    se = np.sqrt((covariance ** 2 + np.outer(np.diag(covariance), np.diag(covariance))) / (n - 1))
    np.testing.assert_array_less(np.abs(np.cov(values, rowvar=False) - covariance), 6 * se)


def test_gaussian_shape_prices_reproducibility_and_isolated_rng():
    state = np.random.get_state()
    kwargs = dict(n_assets=7, n_observations=200, n_factors=3, seed=18,
                  initial_prices=np.linspace(50, 110, 7))
    first, repeat = simulate_market(**kwargs), simulate_market(**kwargs)
    assert first.returns.shape == first.prices.shape == (200, 7)
    assert first.dates.equals(pd.bdate_range("1990-01-01", periods=200))
    assert len(set(first.columns)) == 7
    np.testing.assert_array_equal(first.returns, repeat.returns)
    assert not np.array_equal(first.returns, simulate_market(**{**kwargs, "seed": 19}).returns)
    for a, b in zip(state, np.random.get_state()):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(first.returns, np.expm1(first.log_returns))
    preceding = np.vstack((kwargs["initial_prices"], first.prices[:-1]))
    np.testing.assert_allclose(first.prices / preceding - 1, first.returns, atol=2e-15)
    assert np.isfinite(first.prices).all() and np.isfinite(first.returns).all()
    assert (first.returns > -1).all() and not np.all(first.returns[0] == 0)
    longer = simulate_market(**{**kwargs, "n_observations": 250})
    np.testing.assert_array_equal(first.returns, longer.returns[:200])


def test_gaussian_theoretical_mean_covariance_and_psd():
    market = simulate_market(n_assets=6, n_factors=3, n_observations=40000, include_prices=False)
    p = market.parameters
    b, d, s = map(np.asarray, (p["factor_loadings"], p["idiosyncratic_vol"], p["volatility_multipliers"]))
    covariance = s[0] ** 2 * (b[0] @ b[0].T + np.diag(d[0] ** 2))
    np.testing.assert_allclose(p["diffusive_covariances"][0], covariance, atol=1e-19)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-15
    assert_gaussian_moments(market.log_returns, np.asarray(p["drift"])[0], covariance)
    # Singular covariance is valid: this is a PSD construction, not Cholesky repair.
    singular = simulate_market(n_assets=3, n_factors=1, idiosyncratic_vol=0)
    assert np.linalg.eigvalsh(singular.parameters["diffusive_covariances"])[0].min() >= -1e-15


@pytest.mark.parametrize("options", [
    {"n_assets": 0}, {"n_observations": True}, {"n_factors": -1}, {"seed": None},
    {"start_date": "2024-01-06"}, {"drift": float("nan")}, {"idiosyncratic_vol": -.1},
    {"volatility_multipliers": 0}, {"initial_prices": -1}, {"factor_loadings": [[1, 2, 3]]},
    {"drift": -100}, {"drift": 1000},
])
def test_invalid_gaussian_parameters_fail_explicitly(options):
    with pytest.raises(ValueError):
        simulate_market(**options)


@pytest.fixture(scope="module")
def regime_market():
    return simulate_market("synthetic_regime_jump", n_assets=4, n_factors=2,
                           n_observations=60000, seed=14, include_prices=False)


def test_markov_transitions_occupancy_and_complete_durations(regime_market):
    market = regime_market
    states = market.ground_truth["regimes"]
    p = np.asarray(market.parameters["transition_matrix"])
    for state in range(2):
        following = states[1:][states[:-1] == state]
        counts = np.bincount(following, minlength=2)
        error = abs(counts / len(following) - p[state])
        assert np.all(error < 6 * np.sqrt(p[state] * (1 - p[state]) / len(following)) + 1 / len(following))
    stationary = np.asarray(market.parameters["stationary_distribution"])
    eigenvalue = p[0, 0] + p[1, 1] - 1
    occupancy_se = np.sqrt(stationary * (1 - stationary) / len(states) * (1 + eigenvalue) / (1 - eigenvalue))
    assert np.all(abs(np.bincount(states) / len(states) - stationary) < 6 * occupancy_se)
    starts = np.r_[0, np.flatnonzero(np.diff(states)) + 1]
    lengths = np.diff(np.r_[starts, len(states)])
    # Exclude both boundary-censored spells from duration estimates.
    for state in range(2):
        durations = lengths[1:-1][states[starts[1:-1]] == state]
        expected = 1 / (1 - p[state, state])
        se = np.sqrt(p[state, state] / (1 - p[state, state]) ** 2 / len(durations))
        assert abs(durations.mean() - expected) < 6 * se


def test_conditional_gaussian_and_jump_moments_and_stress_dependence(regime_market):
    market, truth, params = regime_market, regime_market.ground_truth, regime_market.parameters
    states = truth["regimes"]
    b = np.asarray(params["jump_exposures"])
    diffusive = market.log_returns + truth["jump_magnitudes"][:, None] * b[states]
    covariances = np.asarray(params["diffusive_covariances"])
    correlations = []
    for state in range(2):
        mask = states == state
        assert_gaussian_moments(diffusive[mask], np.asarray(params["drift"])[state], covariances[state])
        assert np.linalg.eigvalsh(covariances[state]).min() >= -1e-15
        covariance = np.asarray(params["conditional_log_covariance"])[state]
        mean = np.asarray(params["conditional_log_mean"])[state]
        n = mask.sum()
        assert np.all(abs(market.log_returns[mask].mean(0) - mean) < 6 * np.sqrt(np.diag(covariance) / n))
        probability = params["jump_probabilities"][state]
        # Fourth cumulant of a fixed Bernoulli jump, added to Gaussian covariance SE.
        cumulant = (probability * (1 - probability) * (1 - 6 * probability * (1 - probability))
                    * params["jump_magnitude"] ** 4 * np.outer(b[state] ** 2, b[state] ** 2))
        se = np.sqrt((covariance ** 2 + np.outer(np.diag(covariance), np.diag(covariance)) + cumulant) / (n - 1))
        assert np.all(abs(np.cov(market.log_returns[mask], rowvar=False) - covariance) < 6 * se)
        correlations.append(np.corrcoef(diffusive[mask], rowvar=False)[np.triu_indices(4, 1)].mean())
    assert np.all(np.diag(covariances[1]) > np.diag(covariances[0]))
    assert correlations[1] > correlations[0] + .2


def test_jump_frequency_and_common_negative_counterfactual(regime_market):
    market = regime_market
    truth, params = market.ground_truth, market.parameters
    for state, probability in enumerate(params["jump_probabilities"]):
        events = truth["jump_events"][truth["regimes"] == state]
        assert abs(events.mean() - probability) < 6 * np.sqrt(probability * (1 - probability) / len(events))
    assert np.all(truth["jump_magnitudes"] >= 0)
    assert np.all(truth["jump_magnitudes"][truth["jump_events"]] > 0)
    no_jump = simulate_market("synthetic_regime", n_assets=4, n_factors=2,
                              n_observations=60000, seed=14, include_prices=False)
    np.testing.assert_array_equal(truth["regimes"], no_jump.ground_truth["regimes"])
    difference = market.log_returns - no_jump.log_returns
    expected = -truth["jump_magnitudes"][:, None] * np.asarray(params["jump_exposures"])[truth["regimes"]]
    np.testing.assert_allclose(difference, expected, atol=2e-17)
    assert np.all(difference[truth["jump_events"]] < 0)
    np.testing.assert_array_equal(difference[~truth["jump_events"]], 0)


def test_regime_continuity_prefix_explicit_initial_state_and_price_path():
    kwargs = dict(scenario="synthetic_regime_jump", n_assets=3,
                  transition_matrix=[[.8, .2], [.3, .7]], seed=5, initial_regime=1)
    short = simulate_market(**kwargs, n_observations=80)
    longer = simulate_market(**kwargs, n_observations=160)
    assert short.ground_truth["regimes"][0] == 1
    np.testing.assert_array_equal(short.returns, longer.returns[:80])
    np.testing.assert_array_equal(short.ground_truth["regimes"], longer.ground_truth["regimes"][:80])
    previous = np.vstack((np.full(3, 100.), short.prices[:-1]))
    np.testing.assert_allclose(short.prices / previous - 1, short.returns, atol=2e-15)
    assert np.any(np.diff(short.ground_truth["regimes"]) != 0)
    assert (short.dates[1:] - short.dates[:-1]).max() == pd.Timedelta(days=3)
    # Uniform volatility scaling alone leaves factor/idiosyncratic correlation unchanged.
    low = simulate_market("synthetic_regime", n_assets=3, volatility_multipliers=[1, 1])
    high = simulate_market("synthetic_regime", n_assets=3, volatility_multipliers=[2, 3])
    for c1, c2 in zip(low.parameters["diffusive_covariances"], high.parameters["diffusive_covariances"]):
        c1, c2 = np.asarray(c1), np.asarray(c2)
        np.testing.assert_allclose(c1 / np.sqrt(np.outer(c1.diagonal(), c1.diagonal())),
                                   c2 / np.sqrt(np.outer(c2.diagonal(), c2.diagonal())))


@pytest.mark.parametrize("options", [
    {"transition_matrix": [.9, .1]},
    {"transition_matrix": [[.9, .2], [.1, .9]]},
    {"transition_matrix": [[1.1, -.1], [.1, .9]]},
    {"transition_matrix": [[1, 0], [0, 1]]},
    {"initial_regime": 2}, {"initial_regime": True},
    {"jump_probabilities": [-.1, .2]}, {"jump_probabilities": [.2, .1]},
    {"jump_probabilities": [.1, 1.1]},
    {"jump_magnitude": 0}, {"jump_exposures": -1},
])
def test_invalid_regime_and_jump_parameters(options):
    with pytest.raises(ValueError):
        simulate_market("synthetic_regime_jump", **options)


def test_systemic_stress_uses_the_same_parameterized_factor_model():
    result = simulate_market("synthetic_regime", n_assets=5, n_factors=1, n_observations=100,
                             factor_loadings=[[[.01]], [[.02]]], idiosyncratic_vol=[[.006], [.004]],
                             volatility_multipliers=[1.5, 2.5])
    covariance = np.asarray(result.parameters["diffusive_covariances"])
    correlations = covariance / np.sqrt(covariance.diagonal(axis1=1, axis2=2)[:, :, None]
                                       * covariance.diagonal(axis1=1, axis2=2)[:, None, :])
    assert correlations[1, 0, 1] > correlations[0, 0, 1] > .7
    assert np.all(covariance[1].diagonal() > covariance[0].diagonal())


def test_nonuniform_jump_exposures_and_zero_sensitivity():
    options = dict(n_assets=3, n_observations=200, seed=22, jump_exposures=[0, .5, 1.5])
    market = simulate_market("synthetic_regime_jump", jump_probabilities=[.1, .4], **options)
    control = simulate_market("synthetic_regime", **options)
    impact = -market.ground_truth["jump_magnitudes"][:, None] * np.array([0, .5, 1.5])
    np.testing.assert_allclose(market.log_returns - control.log_returns, impact, atol=2e-17)
    np.testing.assert_array_equal(market.log_returns[:, 0], control.log_returns[:, 0])
