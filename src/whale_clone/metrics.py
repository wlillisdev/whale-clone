"""Performance metrics and bootstrap confidence intervals (pure).

All functions take plain pandas Series of *simple* returns or value levels and
return floats. No IO, no globals — unit-testable with tiny fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

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


def downside_deviation(
    returns: pd.Series, *, target: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    """Annualised downside deviation: RMS of returns below ``target`` only.

    Unlike standard deviation, this ignores upside volatility — the right
    denominator for strategies (like short-vol) whose risk is one-sided.
    """
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    downside = np.minimum(r.to_numpy(dtype=float) - target, 0.0)
    dd = float(np.sqrt((downside**2).mean()))
    return float(dd * np.sqrt(periods_per_year))


def sortino(
    returns: pd.Series,
    *,
    risk_free_annual: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Annualised Sortino ratio: excess return per unit of *downside* deviation.

    Sharpe divides by total volatility and so is blind to negative skew; a
    high-Sharpe strategy can still have a catastrophic left tail. Sortino only
    penalises downside, exposing exactly that.
    """
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    rf = risk_free_annual / periods_per_year
    excess = r.to_numpy(dtype=float) - rf
    dd = float(np.sqrt((np.minimum(excess, 0.0) ** 2).mean()))
    if dd == 0:
        return float("nan")
    return float(excess.mean() / dd * np.sqrt(periods_per_year))


def cvar(returns: pd.Series, *, alpha: float = 0.95) -> float:
    """Conditional Value-at-Risk (expected shortfall) of the worst ``1-alpha`` tail.

    Returns the mean of the worst (1-alpha) fraction of period returns — a
    negative number measuring the *typical* loss in a bad tail event, which a
    Sharpe/standard-deviation view hides.
    """
    data = returns.dropna().to_numpy(dtype=float)
    if data.size < 2:
        return float("nan")
    q = float(np.quantile(data, 1.0 - alpha))
    tail = data[data <= q]
    if tail.size == 0:
        return q
    return float(tail.mean())


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


def block_bootstrap_mean_ci(
    sample: pd.Series,
    *,
    block_len: int = 21,
    iterations: int = 5000,
    confidence: float = 0.95,
    seed: int = 1234,
) -> BootstrapCI:
    """Moving-block bootstrap CI for the mean — for autocorrelated series.

    A long/flat timer holds the same position for weeks, so its daily excess
    returns are strongly autocorrelated and clustered. The IID bootstrap then
    badly *understates* the CI width and waves through noise. Resampling
    contiguous blocks of length ``block_len`` (roughly the holding period)
    preserves that dependence, giving an honest interval.
    """
    data = sample.dropna().to_numpy(dtype=float)
    n = data.size
    if n < 2:
        return BootstrapCI(float("nan"), float("nan"), float("nan"), confidence)
    block_len = max(1, min(block_len, n))
    n_blocks = int(np.ceil(n / block_len))
    max_start = n - block_len
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=float)
    for i in range(iterations):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        sampled = np.concatenate([data[s : s + block_len] for s in starts])[:n]
        means[i] = sampled.mean()
    alpha = 1.0 - confidence
    lower = float(np.quantile(means, alpha / 2.0))
    upper = float(np.quantile(means, 1.0 - alpha / 2.0))
    return BootstrapCI(float(data.mean()), lower, upper, confidence)


# --------------------------------------------------------------------------- #
# Deflated Sharpe Ratio — guards against selecting the best of many trials
# (Bailey & López de Prado). The point: a Sharpe that looks good is only real
# if it beats what the BEST of N noise strategies would produce by luck.
# --------------------------------------------------------------------------- #
def probabilistic_sharpe(
    returns: pd.Series,
    *,
    sr_benchmark: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """P(true Sharpe > ``sr_benchmark``) given the observed SR, n, skew, kurtosis.

    Both Sharpes are annualised. Returns a probability in [0, 1].
    """
    r = returns.dropna()
    n = len(r)
    if n < 3 or r.std(ddof=1) == 0:
        return float("nan")
    sr = sharpe(r, periods_per_year=periods_per_year) / np.sqrt(periods_per_year)
    sr_b = sr_benchmark / np.sqrt(periods_per_year)
    g3 = float(((r - r.mean()) ** 3).mean() / r.std(ddof=1) ** 3)  # skew
    g4 = float(((r - r.mean()) ** 4).mean() / r.std(ddof=1) ** 4)  # kurtosis
    denom = np.sqrt(max(1e-12, 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2))
    return float(norm.cdf((sr - sr_b) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(
    n_trials: int, *, sr_std: float, periods_per_year: int = TRADING_DAYS
) -> float:
    """Expected maximum (annualised) Sharpe from ``n_trials`` independent noise
    strategies whose per-trial Sharpes scatter with std ``sr_std`` (annualised).
    """
    if n_trials < 2 or sr_std <= 0:
        return 0.0
    e = 0.5772156649015329  # Euler-Mascheroni
    z = norm.ppf(1.0 - 1.0 / n_trials) * (1.0 - e) + norm.ppf(1.0 - 1.0 / (n_trials * np.e)) * e
    return float(sr_std * z)


def deflated_sharpe(
    returns: pd.Series,
    *,
    n_trials: int,
    trials_sr_std: float,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Deflated Sharpe Ratio: PSR with the benchmark set to the expected max
    Sharpe of ``n_trials`` noise strategies. A value > 0.95 means the strategy's
    true Sharpe beats the data-mining baseline at 95% confidence.
    """
    sr0 = expected_max_sharpe(n_trials, sr_std=trials_sr_std, periods_per_year=periods_per_year)
    return probabilistic_sharpe(returns, sr_benchmark=sr0, periods_per_year=periods_per_year)
