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
        cov = np.asarray(p["conditional_log_covariance"])[state]
        corr = correlation_matrix(cov)
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
            "configured_mean_log_volatility": float(np.sqrt(cov.diagonal()).mean()),
            "observed_mean_log_volatility": float(np.sqrt(observed_cov.diagonal()).mean()) if observed_cov is not None else None,
            "configured_mean_correlation": _pair_mean(corr),
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
            "configured_covariance": cov.tolist(),
            "observed_covariance": observed_cov.tolist() if observed_cov is not None else None,
            "configured_correlation": _correlation_list(corr),
            "observed_correlation": _correlation_list(observed_corr) if observed_corr is not None else None,
        }
    return {"scenario": p["scenario"], "seed": p["seed"], "observations": count,
            "start": str(result.dates[0].date()), "end": str(result.dates[count - 1].date()),
            "units": "daily log returns; jumps are daily Bernoulli events; durations in weekdays",
            "summary": rows, "moments": moments, "configured_transition": transition.tolist(),
            "observed_transition": [list(row) for row in observed_transition],
            "transition_counts": transition_counts.tolist()}


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
    table[["configured_mean_log_volatility", "observed_mean_log_volatility"]].plot.bar(ax=axes[2, 1], rot=0)
    axes[2, 1].set(title="Configured versus observed average asset volatility", ylabel="Daily log volatility")
    axes[2, 1].legend(["Configured", "Observed"], fontsize=8)
    fig.suptitle(f"{report['scenario']} — {report['start']} to {report['end']} (no later observations)")
    fig.tight_layout()
    return fig
