"""Controlled daily factor markets, independent of forecasting models.

All drift, loading and idiosyncratic volatility parameters are in daily *log*
return units. Factors and idiosyncratic innovations are independent N(0, I).
The public return panel contains arithmetic decimal returns, expm1(log_return).
Weekdays are a simulation calendar, not an exchange holiday calendar.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from diffusion_portfolio.data.core import ReturnTable


@dataclass(frozen=True)
class SyntheticMarketResult:
    dates: pd.DatetimeIndex
    columns: tuple[str, ...]
    returns: np.ndarray
    log_returns: np.ndarray
    # prices[t] is AFTER return[t]; initial_prices precedes the first date.
    prices: np.ndarray | None
    ground_truth: dict
    parameters: dict


def _array(value, shape, name):
    try:
        array = np.broadcast_to(np.asarray(value, dtype=np.float64), shape).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must broadcast to {shape}") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array


def simulate_market(
    scenario="synthetic_gaussian", *, n_assets=12, n_observations=8000,
    n_factors=2, seed=42, start_date="1990-01-01", drift=None,
    factor_loadings=None, idiosyncratic_vol=None, volatility_multipliers=None,
    transition_matrix=None, initial_regime="stationary", jump_probabilities=None,
    jump_magnitude=.04, jump_exposures=1.0,
    initial_prices=100.0, include_prices=True,
) -> SyntheticMarketResult:
    """Simulate one continuous trajectory using explicit, illustrative parameters.

    Drift/idiosyncratic volatility accept scalars, [assets], or [regimes, assets].
    Loadings accept [assets, factors] or [regimes, assets, factors]. Multipliers
    accept a scalar or [regimes]. Parameters are assumptions, not fitted estimates.
    Separate seeded PCG64 streams isolate state, factor, idiosyncratic and jump
    draws. Extending the observation count preserves the already generated prefix.
    Reproduce exact arrays with the same code and NumPy version (recorded below).

    Regimes are daily Markov states; P[i,j] = Pr(z[t+1]=j | z[t]=i).
    J[t] is a daily Bernoulli event, NOT a continuous-time Poisson rate.
    Event magnitude is fixed and strictly positive in log-return units; its
    negative impact is magnitude * nonnegative asset exposure. All innovations,
    transitions and event uniforms use independent streams. No price resets.
    """
    if scenario not in {"synthetic_gaussian", "synthetic_regime", "synthetic_regime_jump"}:
        raise ValueError(f"Unknown synthetic market scenario: {scenario}")
    for name, value in (("n_assets", n_assets), ("n_observations", n_observations),
                        ("n_factors", n_factors), ("seed", seed)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < (0 if name == "seed" else 1):
            raise ValueError(f"{name} must be {'nonnegative' if name == 'seed' else 'positive'} integer")
    start = pd.Timestamp(start_date)
    if pd.isna(start) or start.tz is not None or start != start.normalize() or start.dayofweek >= 5:
        raise ValueError("start_date must be a timezone-naive weekday at midnight")
    dates = pd.bdate_range(start, periods=n_observations)
    n_regimes = 1 if scenario == "synthetic_gaussian" else 2
    if factor_loadings is None:
        base = np.empty((n_assets, n_factors))
        base[:, 0] = np.linspace(.004, .008, n_assets)
        for k in range(1, n_factors):
            base[:, k] = .002 * np.cos(2 * np.pi * k * np.arange(n_assets) / n_assets) / np.sqrt(n_factors - 1)
        factor_loadings = base[None] if n_regimes == 1 else np.stack((base, 2 * base))
    loadings = _array(factor_loadings, (n_regimes, n_assets, n_factors), "factor_loadings")
    mu = _array(np.array([.0002, -.0004][:n_regimes])[:, None] if drift is None else drift,
                (n_regimes, n_assets), "drift")
    idio = _array(np.array([.01, .0075][:n_regimes])[:, None] if idiosyncratic_vol is None else idiosyncratic_vol,
                  (n_regimes, n_assets), "idiosyncratic_vol")
    sigma = _array([1.0, 1.5][:n_regimes] if volatility_multipliers is None else volatility_multipliers,
                   (n_regimes,), "volatility_multipliers")
    initial = _array(initial_prices, (n_assets,), "initial_prices")
    if np.any(idio < 0) or np.any(sigma <= 0) or np.any(initial <= 0):
        raise ValueError("Idiosyncratic volatility must be nonnegative; multipliers and prices must be positive")
    if transition_matrix is None:
        transition_matrix = [[1.0]] if n_regimes == 1 else [[.99, .01], [.04, .96]]
    if np.shape(transition_matrix) != (n_regimes, n_regimes):
        raise ValueError(f"transition_matrix must have shape {(n_regimes, n_regimes)}")
    transition = _array(transition_matrix, (n_regimes, n_regimes), "transition_matrix")
    if (np.any(transition < 0) or np.any(transition > 1)
            or not np.allclose(transition.sum(1), 1, rtol=0, atol=1e-12)):
        raise ValueError("transition_matrix must be row-stochastic with probabilities in [0, 1]")
    stationary = np.ones(1) if n_regimes == 1 else None
    if n_regimes == 2 and transition[0, 1] + transition[1, 0] > 0:
        stationary = np.array([transition[1, 0], transition[0, 1]]) / (transition[0, 1] + transition[1, 0])
    if initial_regime == "stationary":
        if stationary is None:
            raise ValueError("Stationary distribution is not unique; specify initial_regime explicitly")
    elif (isinstance(initial_regime, bool) or not isinstance(initial_regime, (int, np.integer))
          or not 0 <= initial_regime < n_regimes):
        raise ValueError("initial_regime must be 'stationary' or a valid integer state")
    probabilities = _array(([.002, .03] if scenario == "synthetic_regime_jump" else 0)
                           if jump_probabilities is None else jump_probabilities,
                           (n_regimes,), "jump_probabilities")
    exposures = _array(jump_exposures, (n_regimes, n_assets), "jump_exposures")
    magnitude = _array(jump_magnitude, (), "jump_magnitude").item()
    if np.any(probabilities < 0) or np.any(probabilities > 1) or np.any(exposures < 0) or magnitude <= 0:
        raise ValueError("Jump probabilities must be in [0, 1], exposures nonnegative and magnitude positive")
    if scenario != "synthetic_regime_jump" and np.any(probabilities != 0):
        raise ValueError("Only synthetic_regime_jump permits nonzero jump probabilities")
    if scenario == "synthetic_regime_jump" and probabilities[1] <= probabilities[0]:
        raise ValueError("The jump scenario requires stress probability > normal probability")
    covariance = sigma[:, None, None] ** 2 * (
        loadings @ loadings.transpose(0, 2, 1)
        + np.eye(n_assets)[None] * idio[:, :, None] ** 2
    )
    state_rng, factor_rng, idio_rng, jump_rng = [
        np.random.Generator(np.random.PCG64(s)) for s in np.random.SeedSequence(seed).spawn(4)
    ]
    states = np.zeros(n_observations, dtype=np.int64)
    if initial_regime == "stationary":
        states[0] = state_rng.choice(n_regimes, p=stationary)
    else:
        states[0] = initial_regime
    uniforms = state_rng.random(n_observations - 1)
    cumulative = np.cumsum(transition, axis=1)
    cumulative[:, -1] = 1.0  # Avoid endpoint roundoff; validated rows already sum to one.
    for t, uniform in enumerate(uniforms, start=1):
        states[t] = np.searchsorted(cumulative[states[t - 1]], uniform, side="right")
    factors = factor_rng.standard_normal((n_observations, n_factors))
    epsilon = idio_rng.standard_normal((n_observations, n_assets))
    log_returns = mu[states] + sigma[states, None] * (
        np.einsum("tnk,tk->tn", loadings[states], factors) + idio[states] * epsilon
    )
    jumps = jump_rng.random(n_observations) < probabilities[states]
    magnitudes = magnitude * jumps
    log_returns -= magnitudes[:, None] * exposures[states]
    conditional_mean = mu - probabilities[:, None] * magnitude * exposures
    conditional_covariance = covariance + (
        probabilities * (1 - probabilities) * magnitude ** 2
    )[:, None, None] * exposures[:, :, None] * exposures[:, None, :]
    with np.errstate(over="raise", invalid="raise"):
        try:
            returns = np.expm1(log_returns)
            prices = np.exp(np.log(initial) + np.cumsum(log_returns, axis=0)) if include_prices else None
        except FloatingPointError as exc:
            raise ValueError("Simulation exceeds floating-point return/price range; revise parameters") from exc
    if (not np.isfinite(returns).all() or np.any(returns <= -1)
            or (prices is not None and (not np.isfinite(prices).all() or np.any(prices <= 0)))):
        raise ValueError("Simulation cannot represent finite returns > -1 and positive finite prices")
    parameters = {
        "simulator_version": 1, "scenario": scenario, "seed": int(seed), "numpy_version": np.__version__,
        "rng": "PCG64; SeedSequence children: states, factors, idiosyncratic, jumps",
        "n_assets": int(n_assets), "n_observations": int(n_observations), "n_factors": int(n_factors),
        "start_date": str(start.date()), "calendar": "business weekdays, no exchange holidays",
        "regime_names": ["normal", "stress"][:n_regimes],
        "transition_matrix": transition.tolist(),
        "initial_regime": initial_regime if isinstance(initial_regime, str) else int(initial_regime),
        "stationary_distribution": None if stationary is None else stationary.tolist(),
        "jump_probabilities": probabilities.tolist(), "jump_magnitude": magnitude,
        "jump_exposures": exposures.tolist(), "jump_distribution": "fixed positive log-return magnitude",
        "drift": mu.tolist(), "factor_loadings": loadings.tolist(),
        "idiosyncratic_vol": idio.tolist(), "volatility_multipliers": sigma.tolist(),
        "diffusive_covariances": covariance.tolist(), "conditional_log_mean": conditional_mean.tolist(),
        "conditional_log_covariance": conditional_covariance.tolist(), "initial_prices": initial.tolist(),
        "units": {"drift": "daily log return", "loadings_and_idiosyncratic_vol": "daily log-return standard deviation",
                  "volatility_multipliers_and_jump_exposures": "dimensionless",
                  "jump_probabilities": "per business-day Bernoulli probability", "jump_magnitude": "log return",
                  "returns": "arithmetic decimal", "prices": "same arbitrary units as initial_prices"},
    }
    return SyntheticMarketResult(
        dates, tuple(f"asset_{i + 1:03d}" for i in range(n_assets)), returns, log_returns, prices,
        {"regimes": states, "jump_events": jumps, "jump_magnitudes": magnitudes,
         "volatility_multipliers": sigma[states]},
        parameters,
    )


def load_synthetic_market(scenario="synthetic_gaussian", **options) -> ReturnTable:
    """Expose only arithmetic observations; hidden states never enter ReturnTable."""
    result = simulate_market(scenario, include_prices=False, **options)
    return ReturnTable(result.dates, result.returns, result.columns)
