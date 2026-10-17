"""Joint Gaussian forecasts fitted to observable log1p(arithmetic returns).

Covariances are maximum-likelihood estimates (divisor T, not T-1). No
simulator parameters or hidden state enter these models.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


def observed_log_returns(returns):
    """Transform a finite [time, assets] arithmetic panel without clipping."""
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) == 0:
        raise ValueError("Returns must have nonempty shape [time, assets]")
    if not np.isfinite(values).all() or np.any(values <= -1):
        raise ValueError("Arithmetic returns must be finite and strictly greater than -1")
    return np.log1p(values)


def _covariance(values):
    centered = values - values.mean(axis=0)
    return centered.T @ centered / len(values)


@dataclass(frozen=True)
class GaussianForecast:
    """Fixed fitted mean and one-step covariance; phi=0 is unconditional."""
    mean: np.ndarray
    covariance: np.ndarray
    phi: float = 0.0

    @property
    def intercept(self):
        return (1 - self.phi) * self.mean

    def conditional_mean(self, last_return):
        if self.phi == 0:
            return self.mean.copy()
        last = observed_log_returns(np.asarray(last_return)[None, :])[0]
        if last.shape != self.mean.shape:
            raise ValueError("History asset dimension differs from fitted model")
        return self.mean + self.phi * (last - self.mean)


def fit_gaussian(training_returns):
    """Fit one mean/covariance to unique training observations, then freeze."""
    x = observed_log_returns(training_returns)
    if len(x) < 2:
        raise ValueError("Gaussian fitting requires at least two observations")
    return GaussianForecast(x.mean(axis=0), _covariance(x))


def fit_ar1(training_returns, *, dates=None, max_gap_days=7):
    """Pooled OLS with asset-specific intercepts and one shared scalar phi.

    Demean predictors/responses separately across admissible training pairs,
    fit phi, then c = mean(response)-phi*mean(predictor), mu = c/(1-phi).
    Innovation covariance is the centered residual MLE. Reject nonstationary
    fits rather than silently clipping phi. Pairs across long gaps are excluded.
    """
    x = observed_log_returns(training_returns)
    previous, following = x[:-1], x[1:]
    if dates is not None:
        dates = pd.DatetimeIndex(dates)
        if len(dates) != len(x) or dates.hasnans or not dates.is_monotonic_increasing or dates.has_duplicates:
            raise ValueError("Training dates must be unique, chronological and match returns")
        if isinstance(max_gap_days, bool) or not isinstance(max_gap_days, int) or max_gap_days <= 0:
            raise ValueError("max_gap_days must be a positive integer")
        valid = (dates[1:] - dates[:-1]) <= pd.Timedelta(days=max_gap_days)
        previous, following = previous[valid], following[valid]
    if len(previous) < 2:
        raise ValueError("AR(1) fitting requires at least two consecutive training pairs")
    p_mean, q_mean = previous.mean(axis=0), following.mean(axis=0)
    p, q = previous - p_mean, following - q_mean
    denominator = np.sum(p * p)
    if denominator <= 0:
        raise ValueError("AR(1) predictors have no variation")
    phi = float(np.sum(p * q) / denominator)
    if not np.isfinite(phi) or abs(phi) >= 1:
        raise ValueError(f"AR(1) fit is nonstationary: phi={phi}; require abs(phi) < 1")
    intercept = q_mean - phi * p_mean
    residuals = following - (intercept + phi * previous)
    return GaussianForecast(intercept / (1 - phi), _covariance(residuals), phi)


@dataclass
class EWMAGaussian:
    """Causal exponential-mixture moments in log space, default decay 0.94.

    Initialized once with the full training MLE at train_end. Thereafter each
    newly observed return is incorporated once. With d=x-m_old:
      m_new = m_old + (1-decay)*d
      C_new = decay*C_old + decay*(1-decay)*d*d.T
    This centered population-moment recursion preserves PSD.
    """
    mean: np.ndarray
    covariance: np.ndarray
    decay: float = .94

    @classmethod
    def fit(cls, training_returns, *, decay=.94):
        if isinstance(decay, bool) or not np.isfinite(decay) or not 0 < decay < 1:
            raise ValueError("EWMA decay must lie strictly between 0 and 1")
        initial = fit_gaussian(training_returns)
        return cls(initial.mean.copy(), initial.covariance.copy(), float(decay))

    def update(self, observed_return):
        x = observed_log_returns(np.asarray(observed_return)[None, :])[0]
        if x.shape != self.mean.shape:
            raise ValueError("Observation asset dimension differs from EWMA state")
        delta = x - self.mean
        self.mean = self.mean + (1 - self.decay) * delta
        self.covariance = self.decay * self.covariance + self.decay * (1 - self.decay) * np.outer(delta, delta)


def sample_gaussian_returns(mean, covariance, *, n_scenarios, rng):
    """Sample joint log Gaussians, then expm1 into arithmetic decimal returns.

    A PSD eigensquare root supports singular covariance without a ridge.
    Only roundoff-negative eigenvalues (<=100*eps*spectral scale) are set to
    zero; a materially indefinite covariance is rejected. This is numerical
    PSD correction, not truncation of returns or addition of predictive noise.
    """
    mean, covariance = np.asarray(mean, dtype=np.float64), np.asarray(covariance, dtype=np.float64)
    if mean.ndim != 1 or len(mean) == 0 or covariance.shape != (len(mean), len(mean)):
        raise ValueError("Gaussian mean/covariance asset dimensions differ")
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError("Gaussian moments must be finite")
    if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=0):
        raise ValueError("Gaussian covariance must be symmetric")
    if isinstance(n_scenarios, bool) or not isinstance(n_scenarios, (int, np.integer)) or n_scenarios < 2:
        raise ValueError("n_scenarios must be an integer >= 2")
    eigenvalues, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
    tolerance = 100 * np.finfo(np.float64).eps * max(np.abs(eigenvalues).max(), np.finfo(np.float64).tiny)
    if eigenvalues.min() < -tolerance:
        raise ValueError("Gaussian covariance must be positive semidefinite")
    root = vectors * np.sqrt(np.maximum(eigenvalues, 0))
    try:
        with np.errstate(over="raise", invalid="raise"):
            log_samples = mean + rng.standard_normal((n_scenarios, len(mean))) @ root.T
            samples = np.expm1(log_samples)
    except FloatingPointError as exc:
        raise ValueError("Gaussian samples cannot be represented as finite arithmetic returns") from exc
    if not np.isfinite(samples).all() or np.any(samples <= -1):
        raise ValueError("Gaussian samples must be finite arithmetic returns greater than -1; no clipping is applied")
    return samples
