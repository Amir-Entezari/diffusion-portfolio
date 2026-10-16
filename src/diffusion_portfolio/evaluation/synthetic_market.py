"""Small simulator diagnostics in daily log-return units, not a realism test."""
from __future__ import annotations

import numpy as np
import pandas as pd


def correlation_matrix(covariance):
    """Undefined correlations (zero marginal variance) remain NaN."""
    covariance = np.asarray(covariance, dtype=np.float64)
    scale = np.sqrt(np.outer(covariance.diagonal(), covariance.diagonal()))
    return np.divide(covariance, scale, out=np.full_like(covariance, np.nan), where=scale > 0)


def _pair_mean(correlation):
    values = correlation[np.triu_indices(len(correlation), 1)]
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if len(finite) else None


def _correlation_list(matrix):
    return [[float(value) if np.isfinite(value) else None for value in row] for row in matrix]


def _prefix(result, end_date):
    count = int((result.dates <= pd.Timestamp(end_date)).sum())
    if count < 2:
        raise ValueError("Diagnostics need at least two observations before end_date")
    return count


def _finite_list(values):
    values = np.asarray(values, dtype=float)
    return np.where(np.isfinite(values), values, None).tolist()


def within_regime_dynamics(log_returns, states, means, coefficients, *, fixed_regime, lags=(1, 2, 5)):
    """Pair only observations in the SAME uninterrupted spell, including all intermediate dates.

    Switching AR coefficients specify conditional slopes, not stationary lag
    correlations. Sign reversals refer to deviations from the configured log mean.
    """
    segments = np.r_[0, np.cumsum(states[1:] != states[:-1])]
    deviations = log_returns - means[states]
    rows = []
    for state, coefficient in enumerate(coefficients):
        for lag in lags:
            if lag >= len(states):
                continue
            pairs = (states[lag:] == state) & (segments[lag:] == segments[:-lag])
            a, b = deviations[:-lag][pairs], deviations[lag:][pairs]
            correlation = np.full(log_returns.shape[1], np.nan)
            slope = correlation.copy()
            if len(a) >= 2:
                ac, bc = a - a.mean(0), b - b.mean(0)
                denominator = np.sqrt(np.sum(ac ** 2, axis=0) * np.sum(bc ** 2, axis=0))
                np.divide(np.sum(ac * bc, axis=0), denominator, out=correlation, where=denominator > 0)
                square = np.sum(a ** 2, axis=0)
                np.divide(np.sum(a * b, axis=0), square, out=slope, where=square > 0)
            finite = correlation[np.isfinite(correlation)]
            rows.append({"state": state, "lag": lag, "pairs": int(pairs.sum()),
                         "configured_lag_slope": float(coefficient ** lag),
                         "theoretical_correlation": float(coefficient ** lag) if fixed_regime else None,
                         "observed_correlation_by_asset": _finite_list(correlation),
                         "observed_mean_correlation": float(finite.mean()) if len(finite) else None,
                         "observed_slope_by_asset": _finite_list(slope),
                         "raw_sign_reversal_rate": float(np.mean(log_returns[:-lag][pairs] * log_returns[lag:][pairs] < 0)) if len(a) else None,
                         "centered_sign_reversal_rate": float(np.mean(a * b < 0)) if len(a) else None,
                         "theoretical_centered_sign_reversal_rate": float(np.arccos(coefficient ** lag) / np.pi) if fixed_regime else None})
    return rows


def log_tail_diagnostics(centered_log_returns, covariance, *, degrees_of_freedom, thresholds=(2., 3., 4.)):
    """Finite-threshold co-exceedances, not an asymptotic tail-dependence estimate.

    Standardization uses known marginal LOG variance, not a fitted tail threshold.
    Student-t arithmetic returns exp(x)-1 have no finite positive population
    moments; finite log moments must not be presented as arithmetic moments.
    """
    from scipy import stats

    vol = np.sqrt(np.diag(covariance))
    z = np.divide(centered_log_returns, vol, out=np.full_like(centered_log_returns, np.nan), where=vol > 0)
    levels = np.array([.01, .05, .5, .95, .99])
    nu = degrees_of_freedom
    distribution = stats.t(df=nu, scale=np.sqrt((nu - 2) / nu))
    rows = []
    for threshold in thresholds:
        lower = (z < -threshold).astype(float)
        joint = lower.T @ lower / len(z)
        joint[(vol <= 0), :] = np.nan
        joint[:, (vol <= 0)] = np.nan
        observed = np.where(vol > 0, (abs(z) > threshold).mean(0), np.nan)
        rows.append({"threshold_log_standard_deviations": threshold,
                     "configured_two_sided_probability": float(2 * distribution.sf(threshold)),
                     "gaussian_two_sided_probability": float(2 * stats.norm.sf(threshold)),
                     "observed_two_sided_by_asset": _finite_list(observed),
                     "observed_lower_joint_probabilities": _finite_list(joint),
                     "observed_mean_lower_pair_probability": _pair_mean(joint)})
    return {"units": "configured standard-deviation units of centered daily LOG returns",
            "degrees_of_freedom": nu, "configured_excess_kurtosis": float(6 / (nu - 4)),
            "observed_excess_kurtosis": _finite_list(stats.kurtosis(z, axis=0, fisher=True, bias=False)) if len(z) > 3 else None,
            "quantile_levels": levels.tolist(), "configured_quantiles": distribution.ppf(levels).tolist(),
            "observed_quantiles_by_asset": _finite_list(np.quantile(z, levels, axis=0)), "exceedances": rows}


def market_diagnostics(result, *, end_date):
    """Compare configured conditional log moments with one Monte Carlo path.

    A cutoff is required so an experiment can explicitly restrict reports to its
    training period. These diagnostics do not select parameters or test a model.
    Complete interior regime spells exclude both boundary-censored spells.
    Missing/undefined estimates and infinite expected durations are None.
    """
    count = _prefix(result, end_date)
    x = result.log_returns[:count]
    states = result.ground_truth["regimes"][:count]
    events = result.ground_truth["jump_events"][:count]
    magnitudes = result.ground_truth["jump_magnitudes"][:count]
    p = result.parameters
    transition = np.asarray(p["transition_matrix"])
    transition_counts = np.zeros_like(transition, dtype=int)
    np.add.at(transition_counts, (states[:-1], states[1:]), 1)
    observed_transition = [row / row.sum() if row.sum() else [None] * len(row)
                           for row in transition_counts]
    edges = np.r_[0, np.flatnonzero(np.diff(states)) + 1, count]
    spell_states, lengths = states[edges[:-1]], np.diff(edges)
    rows, moments = [], {}
    for state, name in enumerate(p["regime_names"]):
        mask = states == state
        n = int(mask.sum())
        mean = np.asarray(p["conditional_log_mean"])[state]
        cov = None if p["conditional_log_covariance"] is None else np.asarray(p["conditional_log_covariance"])[state]
        corr = None if cov is None else correlation_matrix(cov)
        observed_cov = np.atleast_2d(np.cov(x[mask], rowvar=False)) if n > 1 else None
        observed_corr = correlation_matrix(observed_cov) if observed_cov is not None else None
        durations = lengths[1:-1][spell_states[1:-1] == state]
        event_sizes = magnitudes[mask & events]
        exposures = np.asarray(p["jump_exposures"])[state]
        rows.append({
            "regime": name, "observations": n, "occupancy": n / count,
            "stationary_occupancy": None if p["stationary_distribution"] is None else p["stationary_distribution"][state],
            "expected_duration_days": float(1 / (1 - transition[state, state])) if transition[state, state] < 1 else None,
            "complete_spells": len(durations), "observed_duration_days": float(durations.mean()) if len(durations) else None,
            "configured_mean_log_return": float(mean.mean()),
            "observed_mean_log_return": float(x[mask].mean()) if n else None,
            "configured_mean_log_volatility": float(np.sqrt(cov.diagonal()).mean()) if cov is not None else None,
            "observed_mean_log_volatility": float(np.sqrt(observed_cov.diagonal()).mean()) if observed_cov is not None else None,
            "configured_mean_correlation": _pair_mean(corr) if corr is not None else None,
            "observed_mean_correlation": _pair_mean(observed_corr) if observed_corr is not None else None,
            "configured_diffusive_correlation": _pair_mean(correlation_matrix(np.asarray(p["diffusive_covariances"])[state])),
            "configured_jump_frequency": p["jump_probabilities"][state],
            "observed_jump_frequency": float(events[mask].mean()) if n else None,
            "event_count": int(len(event_sizes)), "configured_event_magnitude": p["jump_magnitude"],
            "observed_event_magnitude": float(event_sizes.mean()) if len(event_sizes) else None,
            # A jump is a negative contribution; the net observed return can still be positive.
            "mean_common_log_jump_term": float(-magnitudes[mask].mean() * exposures.mean()) if n else None,
        })
        moments[name] = {
            "configured_mean": mean.tolist(), "observed_mean": x[mask].mean(0).tolist() if n else None,
            "configured_covariance": cov.tolist() if cov is not None else None,
            "observed_covariance": observed_cov.tolist() if observed_cov is not None else None,
            "configured_correlation": _correlation_list(corr) if corr is not None else None,
            "observed_correlation": _correlation_list(observed_corr) if observed_corr is not None else None,
        }
    report = {"scenario": p["scenario"], "seed": p["seed"], "observations": count,
            "start": str(result.dates[0].date()), "end": str(result.dates[count - 1].date()),
            "units": "daily log returns; jumps are daily Bernoulli events; durations in weekdays",
            "summary": rows, "moments": moments, "configured_transition": transition.tolist(),
            "observed_transition": [list(row) for row in observed_transition],
            "transition_counts": transition_counts.tolist()}
    if "innovation_distribution" in p:
        report["covariance_interpretation"] = p["covariance_interpretation"]
        report["driver_covariances"] = p["driver_covariances"]
        report["innovation_covariances"] = p["innovation_covariances"]
        report["dynamics"] = within_regime_dynamics(
            x, states, np.asarray(p["drift"]), np.asarray(p["ar_coefficients"]),
            fixed_regime=len(transition) == 1,
        )
        if p["innovation_distribution"] == "student_t":
            report["tails"] = log_tail_diagnostics(x - np.asarray(p["drift"])[0],
                                                   np.asarray(p["driver_covariances"])[0],
                                                   degrees_of_freedom=p["degrees_of_freedom"])
    return report


def plot_market_diagnostics(result, *, end_date):
    """Five small diagnostic views; uses observations only through end_date."""
    import matplotlib.pyplot as plt

    report = market_diagnostics(result, end_date=end_date)
    count = report["observations"]
    dates, returns = result.dates[:count], result.returns[:count]
    states = result.ground_truth["regimes"][:count]
    events = result.ground_truth["jump_events"][:count]
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    axes[0, 0].plot(dates, returns[:, :min(3, len(result.columns))], lw=.5)
    axes[0, 0].set(title="Arithmetic decimal returns (first assets)", ylabel="Daily arithmetic return")
    axes[0, 1].step(dates, states, lw=.7, label="Hidden regime (diagnostic only)")
    axes[0, 1].set(title="Regime and trailing market log-return volatility", ylabel="Regime")
    vol = pd.Series(result.log_returns[:count].mean(1), index=dates).rolling(20).std()
    right = axes[0, 1].twinx()
    right.plot(dates, vol, color="tab:orange", lw=.8)
    right.set_ylabel("Trailing 20-day log volatility", color="tab:orange")
    for ax, state in zip(axes[1], (0, 1)):
        if state >= len(result.parameters["regime_names"]):
            ax.text(.5, .5, "Single regime: no stress state", ha="center")
            ax.set_axis_off()
            continue
        name = result.parameters["regime_names"][state]
        correlation = report["moments"][name]["observed_correlation"]
        if correlation is None:
            ax.text(.5, .5, f"No estimable {name} correlation", ha="center")
            ax.set_axis_off()
            continue
        image = ax.imshow(np.asarray(correlation, dtype=float), vmin=-1, vmax=1, cmap="coolwarm")
        ax.set(title=f"Observed {name} log-return correlation", xlabel="Asset", ylabel="Asset")
        fig.colorbar(image, ax=ax, fraction=.045)
    market = returns.mean(1)
    axes[2, 0].plot(dates, market, lw=.5, label="Mean asset arithmetic return")
    axes[2, 0].scatter(dates[events], market[events], color="red", s=12, label="Common negative jump event")
    axes[2, 0].set(title="Jump events and observed market returns", ylabel="Arithmetic return")
    axes[2, 0].legend(fontsize=8)
    table = pd.DataFrame(report["summary"]).set_index("regime")
    columns = ["configured_mean_log_volatility", "observed_mean_log_volatility"]
    if result.parameters["conditional_log_covariance"] is None:
        columns = columns[1:]
    table[columns].plot.bar(ax=axes[2, 1], rot=0)
    axes[2, 1].set(title="Average asset volatility (return covariance)" if len(columns) == 1 else "Configured versus observed average asset volatility", ylabel="Daily log volatility")
    axes[2, 1].legend(["Observed"] if len(columns) == 1 else ["Configured", "Observed"], fontsize=8)
    fig.suptitle(f"{report['scenario']} — {report['start']} to {report['end']} (no later observations)")
    fig.tight_layout()
    return fig


def plot_stress_diagnostics(result, *, end_date):
    """Return dynamics and tails, with no invalid switching-covariance reference."""
    import matplotlib.pyplot as plt
    from scipy import stats

    report = market_diagnostics(result, end_date=end_date)
    count = report["observations"]
    p, dates = result.parameters, result.dates[:count]
    states = result.ground_truth["regimes"][:count]
    x = result.log_returns[:count]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    entries = np.flatnonzero(states == 1)
    start = max(0, int(entries[0]) - 25) if len(entries) else 0
    stop = min(count, start + 250)
    ax = axes[0, 0]
    ax.plot(dates[start:stop], x[start:stop, 0], lw=.8)
    if len(p["regime_names"]) == 2:
        timeline = ax.twinx()
        timeline.step(dates[start:stop], states[start:stop], color="tab:orange", alpha=.5)
        timeline.set_ylabel("Hidden regime (diagnostic)")
    ax.set(title="Representative first-asset log-return trajectory", ylabel="Daily log return")
    centered = x[:, 0] - np.asarray(p["drift"])[states, 0]
    sd = (np.sqrt(np.asarray(p["conditional_log_covariance"])[0, 0, 0])
          if p["conditional_log_covariance"] is not None else centered.std())
    if sd > 0:
        z = centered / sd
        axes[0, 1].hist(z, bins=80, density=True, alpha=.6, label="Observed")
        grid = np.linspace(np.quantile(z, .001), np.quantile(z, .999), 250)
        if p["innovation_distribution"] == "student_t":
            nu = p["degrees_of_freedom"]
            axes[0, 1].plot(grid, stats.t.pdf(grid, nu, scale=np.sqrt((nu - 2) / nu)), label="Configured Student-t")
        elif len(p["regime_names"]) == 1:
            axes[0, 1].plot(grid, stats.norm.pdf(grid), label="Stationary Gaussian")
        axes[0, 1].legend()
    axes[0, 1].set(title="First-asset centered log-return histogram", xlabel="Log standard deviations")
    dynamics = pd.DataFrame(report["dynamics"])
    for state, name in enumerate(p["regime_names"]):
        frame = dynamics[dynamics.state == state]
        axes[1, 0].plot(frame.lag, frame.observed_mean_correlation, marker="o", label=name)
    if len(p["regime_names"]) == 1:
        axes[1, 0].plot(dynamics.lag, dynamics.theoretical_correlation, "k--", label="Configured stationary ACF")
    axes[1, 0].axhline(0, color="gray", lw=.5)
    axes[1, 0].set(title="Within-spell mean asset log-return correlation", xlabel="Lag (observations)")
    axes[1, 0].legend()
    if "tails" in report:
        tails = report["tails"]["exceedances"]
        threshold = [row["threshold_log_standard_deviations"] for row in tails]
        axes[1, 1].plot(threshold, [np.mean(row["observed_two_sided_by_asset"]) for row in tails], "o-", label="Observed")
        axes[1, 1].plot(threshold, [row["configured_two_sided_probability"] for row in tails], "--", label="Configured Student-t")
        axes[1, 1].plot(threshold, [row["gaussian_two_sided_probability"] for row in tails], ":", label="Covariance-matched Gaussian")
        axes[1, 1].set(title="Two-sided log-return tail exceedances", xlabel="Log standard deviations", yscale="log")
        axes[1, 1].legend()
    else:
        reversal = dynamics[dynamics.lag == 1].copy()
        reversal.index = p["regime_names"]
        columns = ["raw_sign_reversal_rate", "centered_sign_reversal_rate"]
        if len(p["regime_names"]) == 1:
            columns += ["theoretical_centered_sign_reversal_rate"]
        reversal[columns].rename(columns={"raw_sign_reversal_rate": "Raw log returns",
                                         "centered_sign_reversal_rate": "Centered log returns",
                                         "theoretical_centered_sign_reversal_rate": "Configured centered"}).plot.bar(ax=axes[1, 1], rot=0)
        axes[1, 1].set(title="Within-spell direction reversals", ylabel="Frequency")
    fig.suptitle(f"{p['scenario']} — training cutoff {report['end']}")
    fig.tight_layout()
    return fig
