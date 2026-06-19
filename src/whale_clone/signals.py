"""Trading signals for single-asset timing strategies (pure, no IO).

A signal maps a daily price series to a daily *target weight* in {0, 1}
(long/flat) or {-1, 0, 1} (long/flat/short). The no-look-ahead boundary lives
here and only here: :func:`monthly_targets` applies each month-end decision
starting the *next* trading day, so a position is never taken on information
from the bar it is executed on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def momentum_signal(
    prices: pd.Series, *, lookback: int = 252, allow_short: bool = False
) -> pd.Series:
    """Time-series momentum: long when trailing ``lookback``-day return > 0.

    Returns a daily raw signal (NaN until enough history exists). Causality is
    enforced later by :func:`monthly_targets`.
    """
    mom = prices / prices.shift(lookback) - 1.0
    sig = np.sign(mom) if allow_short else (mom > 0).astype(float)
    out = pd.Series(sig, index=prices.index, dtype=float)
    out[mom.isna()] = np.nan
    return out


def sma_signal(prices: pd.Series, *, window: int = 200, allow_short: bool = False) -> pd.Series:
    """Moving-average trend: long when price is above its ``window``-day SMA."""
    sma = prices.rolling(window).mean()
    sig = np.where(prices > sma, 1.0, -1.0) if allow_short else (prices > sma).astype(float)
    out = pd.Series(sig, index=prices.index, dtype=float)
    out[sma.isna()] = np.nan
    return out


def monthly_targets(daily_signal: pd.Series) -> pd.Series:
    """Convert a daily raw signal into monthly-rebalanced, causal target weights.

    The decision made on the last trading day of each month is applied from the
    *following* trading day onward (a single ``shift(1)``), so we never act on
    the bar we observe. Days before the first valid decision are flat (0).
    """
    s = daily_signal.dropna()
    if s.empty:
        return pd.Series(0.0, index=daily_signal.index)
    idx = pd.DatetimeIndex(s.index)
    period = idx.to_period("M")
    # Mark the last trading day of each month.
    is_month_end = pd.Series(period, index=idx) != pd.Series(period, index=idx).shift(-1)
    decisions = s[is_month_end.to_numpy()]
    target = decisions.reindex(daily_signal.index, method="ffill").shift(1)
    return target.fillna(0.0)
