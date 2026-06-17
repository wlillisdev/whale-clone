"""Performance metrics and bootstrap confidence intervals (pure).

All functions take plain pandas Series of *simple* returns or value levels and
return floats. No IO, no globals — unit-testable with tiny fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cagr(value: pd.Series, *, periods_per_year: int = TRADING_DAYS) -> float:
    """Compound annual growth rate from a value-level series."""
    value = value.dropna()
    if len(value) < 2:
        return float("nan")
    total_return = value.iloc[-1] / value.iloc[0]
    years = (len(value) - 1) / periods_per_year
    if years <= 0 or total_return <= 0:
        return float("nan")
    return float(total_return ** (1.0 / years) - 1.0)


def annualised_vol(returns: pd.Series, *, periods_per_year: int = TRADING_DAYS) -> float:
    returns = returns.dropna()
    if len(returns) < 2:
        return float("nan")
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe(
    returns: pd.Series,
    *,
    risk_free_annual: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Annualised Sharpe ratio from a per-period simple-return series."""
    returns = returns.dropna()
    if len(returns) < 2:
        return float("nan")
    rf_per_period = risk_free_annual / periods_per_year
    excess = returns - rf_per_period
    sd = excess.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def max_drawdown(value: pd.Series) -> float:
    """Most negative peak-to-trough drawdown of a value-level series (<= 0)."""
    value = value.dropna()
    if value.empty:
        return float("nan")
    running_max = value.cummax()
    drawdown = value / running_max - 1.0
    return float(drawdown.min())


@dataclass(frozen=True)
class BootstrapCI:
    mean: float
    lower: float
    upper: float
    confidence: float


def bootstrap_mean_ci(
    sample: pd.Series,
    *,
    iterations: int = 5000,
    confidence: float = 0.95,
    seed: int = 1234,
) -> BootstrapCI:
    """Percentile bootstrap CI for the mean of ``sample``.

    Used on per-period excess returns vs the benchmark. A lower bound above zero
    is the cost-adjusted expectancy gate (brief, section 5, gate 1).
    """
    data = sample.dropna().to_numpy(dtype=float)
    if data.size < 2:
        return BootstrapCI(float("nan"), float("nan"), float("nan"), confidence)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, data.size, size=(iterations, data.size))
    means = data[idx].mean(axis=1)
    alpha = 1.0 - confidence
    lower = float(np.quantile(means, alpha / 2.0))
    upper = float(np.quantile(means, 1.0 - alpha / 2.0))
    return BootstrapCI(float(data.mean()), lower, upper, confidence)
