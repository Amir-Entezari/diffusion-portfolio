"""Mathematics, sampling and input validation for observation-only baselines."""
import numpy as np
import pandas as pd
import pytest

from diffusion_portfolio.baselines.statistical import (
    EWMAGaussian, fit_ar1, fit_gaussian, observed_log_returns, sample_gaussian_returns,
)
from diffusion_portfolio.data.sources.synthetic_market import simulate_market


def test_gaussian_mle_and_joint_sampling_with_monte_carlo_tolerances():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(5000, 3)) @ np.array([[.01, .004, 0], [0, .008, .002], [0, 0, .006]]) + .0002
    model = fit_gaussian(np.expm1(x))
    np.testing.assert_allclose(model.mean, x.mean(0), rtol=1e-13, atol=1e-17)
    np.testing.assert_allclose(model.covariance, np.cov(x, rowvar=False, ddof=0), rtol=1e-13)
    assert np.linalg.eigvalsh(model.covariance).min() >= 0
    assert model.phi == 0
    np.testing.assert_array_equal(model.conditional_mean(None), model.mean)
    state = np.random.get_state()
    samples = sample_gaussian_returns(model.mean, model.covariance, n_scenarios=60000, rng=np.random.default_rng(91))
    repeated = sample_gaussian_returns(model.mean, model.covariance, n_scenarios=60000, rng=np.random.default_rng(91))
    np.testing.assert_array_equal(samples, repeated)
    assert samples.shape == (60000, 3) and np.isfinite(samples).all() and (samples > -1).all()
    z, n, cov = np.log1p(samples), len(samples), model.covariance
    assert np.all(abs(z.mean(0) - model.mean) < 6 * np.sqrt(cov.diagonal() / n))
    se = np.sqrt((cov ** 2 + np.outer(cov.diagonal(), cov.diagonal())) / n)
    assert np.all(abs(np.cov(z, rowvar=False) - cov) < 6 * se)
    for before, after in zip(state, np.random.get_state()):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("scenario,phi", [("synthetic_gaussian", 0.), ("synthetic_mean_reverting", .4)])
def test_ar1_recovers_shared_coefficient_mean_and_innovation_covariance(scenario, phi):
    result = simulate_market(scenario, n_assets=3, n_observations=30000, seed=29, include_prices=False)
    model = fit_ar1(result.returns, dates=result.dates)
    x = np.log1p(result.returns)
    p, q = x[:-1], x[1:]
    # Independent pooled-OLS calculation, including the fitted asset intercepts.
    centered_p, centered_q = p - p.mean(0), q - q.mean(0)
    expected_phi = np.sum(centered_p * centered_q) / np.sum(centered_p ** 2)
    np.testing.assert_allclose(model.phi, expected_phi, rtol=1e-14)
    assert abs(model.phi - phi) < 6 / np.sqrt(len(x))
    intercept = q.mean(0) - expected_phi * p.mean(0)
    np.testing.assert_allclose(model.intercept, intercept, atol=1e-18)
    residual = q - (intercept + expected_phi * p)
    np.testing.assert_allclose(model.covariance, np.cov(residual, rowvar=False, ddof=0), rtol=1e-13)
    true_covariance = np.asarray(result.parameters["diffusive_covariances"])[0]
    omega = (1 - phi ** 2) * true_covariance
    se = np.sqrt((omega ** 2 + np.outer(omega.diagonal(), omega.diagonal())) / len(residual))
    assert np.all(abs(model.covariance - omega) < 6 * se)
    mu = np.asarray(result.parameters["drift"])[0]
    assert np.all(abs(model.mean - mu) < 6 * np.sqrt(true_covariance.diagonal() / len(x) * (1 + phi) / (1 - phi)))
    conditional_mean = model.mean + model.phi * (np.log1p(result.returns[-1]) - model.mean)
    np.testing.assert_array_equal(model.conditional_mean(result.returns[-1]), conditional_mean)
    if phi:
        assert np.all(model.covariance.diagonal() < .95 * true_covariance.diagonal())


def test_ar1_excludes_pairs_across_calendar_gaps():
    x = np.array([.01, -.01, .005, .0, -.005, .002])[:, None]
    dates = pd.DatetimeIndex(["2000-01-03", "2000-01-04", "2000-01-05", "2005-01-03", "2005-01-04", "2005-01-05"])
    result = fit_ar1(np.expm1(x), dates=dates)
    p, q = x[[0, 1, 3, 4]], x[[1, 2, 4, 5]]
    expected = np.sum((p - p.mean(0)) * (q - q.mean(0))) / np.sum((p - p.mean(0)) ** 2)
    np.testing.assert_allclose(result.phi, expected)


def test_ar1_rejects_nonstationarity_and_degenerate_predictors():
    with pytest.raises(ValueError, match="nonstationary"):
        fit_ar1(np.expm1(np.array([.001, .002, .004, .008, .016])[:, None]))
    with pytest.raises(ValueError, match="no variation"):
        fit_ar1(np.full((10, 2), .01))
    with pytest.raises(ValueError, match="consecutive"):
        fit_ar1(np.array([[.01], [.02]]))


def test_ewma_is_centered_exponential_mixture_and_remains_psd():
    x = np.array([[.01, -.01], [-.01, .02], [.005, -.01], [0, .01]])
    fitted = EWMAGaussian.fit(np.expm1(x), decay=.8)
    mean, covariance = x.mean(0), np.cov(x, rowvar=False, ddof=0)
    np.testing.assert_allclose(fitted.mean, mean, atol=1e-18)
    np.testing.assert_allclose(fitted.covariance, covariance, rtol=1e-14)
    for next_x in ([.08, -.04], [-.1, .02], [.03, .01]):
        next_x = np.array(next_x)
        expected_mean = .8 * mean + .2 * next_x
        # Central covariance of .8*old distribution + .2*new point mass.
        expected_covariance = .8 * (covariance + np.outer(mean - expected_mean, mean - expected_mean))
        expected_covariance += .2 * np.outer(next_x - expected_mean, next_x - expected_mean)
        fitted.update(np.expm1(next_x))
        np.testing.assert_allclose(fitted.mean, expected_mean, atol=2e-17)
        np.testing.assert_allclose(fitted.covariance, expected_covariance, rtol=1e-13)
        assert np.linalg.eigvalsh(fitted.covariance).min() >= 0
        mean, covariance = expected_mean, expected_covariance
    assert fitted.covariance.trace() > np.cov(x, rowvar=False, ddof=0).trace()


def test_sampling_singular_covariance_without_artificial_ridge():
    samples = sample_gaussian_returns(np.zeros(3), np.ones((3, 3)) * .0001,
                                      n_scenarios=20, rng=np.random.default_rng(1))
    np.testing.assert_allclose(samples[:, 0], samples[:, 1], atol=1e-10)
    np.testing.assert_allclose(samples[:, 0], samples[:, 2], atol=1e-10)
    zero = sample_gaussian_returns(np.array([.01, -.02]), np.zeros((2, 2)),
                                   n_scenarios=5, rng=np.random.default_rng(2))
    np.testing.assert_array_equal(zero, np.broadcast_to(np.expm1([.01, -.02]), (5, 2)))
    near_psd = np.diag([.001, -1e-18])
    sample_gaussian_returns(np.zeros(2), near_psd, n_scenarios=2, rng=np.random.default_rng(3))
    with pytest.raises(ValueError, match="positive semidefinite"):
        sample_gaussian_returns(np.zeros(2), np.diag([.001, -.0001]), n_scenarios=2, rng=np.random.default_rng(3))


@pytest.mark.parametrize("values", [[], [1., 2.], [[-1.]], [[-1.1]], [[np.nan]], [[np.inf]]])
def test_invalid_observable_returns_are_rejected(values):
    with pytest.raises(ValueError):
        observed_log_returns(values)


@pytest.mark.parametrize("decay", [0, 1, -.1, np.nan, np.inf, True])
def test_invalid_ewma_decay(decay):
    with pytest.raises(ValueError, match="decay"):
        EWMAGaussian.fit(np.array([[.01], [-.01]]), decay=decay)


@pytest.mark.parametrize("mean", [1000., -1000.])
def test_unrepresentable_arithmetic_samples_are_rejected_without_clipping(mean):
    with pytest.raises(ValueError, match="arithmetic returns"):
        sample_gaussian_returns(np.array([mean]), np.zeros((1, 1)), n_scenarios=3, rng=np.random.default_rng(3))
