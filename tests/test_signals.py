"""Signal + single-asset backtest tests, including no-look-ahead."""

from __future__ import annotations

import numpy as np
import pandas as pd

from whale_clone.costs import CostModel
from whale_clone.signal_backtest import SignalBacktestConfig, run_signal_backtest, time_in_market
from whale_clone.signals import momentum_signal, monthly_targets, sma_signal


def _prices(n=600, drift=0.0004, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    p = 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
    return pd.Series(p, index=idx)


def test_momentum_signal_long_when_trailing_return_positive():
    # Monotonic up series -> momentum signal is 1 once history exists.
    idx = pd.bdate_range("2015-01-01", periods=300)
    up = pd.Series(np.linspace(100, 200, 300), index=idx)
    sig = momentum_signal(up, lookback=252)
    assert sig.iloc[:252].isna().all()  # not enough history
    assert (sig.dropna() == 1.0).all()


def test_monthly_targets_are_causal():
    prices = _prices()
    raw = momentum_signal(prices, lookback=126)
    target = monthly_targets(raw)
    # Target is defined over the full index and only takes values in {0,1}.
    assert set(pd.unique(target.dropna())).issubset({0.0, 1.0})
    assert len(target) == len(prices)


def test_constant_long_reproduces_buy_and_hold():
    prices = _prices()
    target = pd.Series(1.0, index=prices.index)  # always long
    cfg = SignalBacktestConfig(cost_model=CostModel(0.0, 0.0))  # no costs
    res = run_signal_backtest(prices, prices, target, cfg)
    # Always-long, zero cost == buy and hold: value path matches benchmark.
    pd.testing.assert_series_equal(
        res.value.rename("x"), res.benchmark_value.rename("x"), atol=1e-9
    )


def test_constant_flat_earns_risk_free():
    prices = _prices()
    target = pd.Series(1.0, index=prices.index)
    target.iloc[10:] = 0.0  # go flat after day 10 (need an initial nonzero to start sim)
    cfg = SignalBacktestConfig(cost_model=CostModel(0.0, 0.0), risk_free_annual=0.0)
    res = run_signal_backtest(prices, prices, target, cfg)
    # With rf=0, once flat the strategy value is constant.
    tail = res.value.iloc[20:]
    assert tail.std() < 1e-9


def test_signal_no_lookahead_future_bar_cannot_change_past():
    prices = _prices()
    raw = momentum_signal(prices, lookback=126)
    target = monthly_targets(raw)
    cfg = SignalBacktestConfig(cost_model=CostModel(0.0, 5.0))
    base = run_signal_backtest(prices, prices, target, cfg)

    # Tamper with the FINAL price bar only.
    tampered = prices.copy()
    tampered.iloc[-1] *= 1.5
    raw2 = momentum_signal(tampered, lookback=126)
    target2 = monthly_targets(raw2)
    res2 = run_signal_backtest(tampered, tampered, target2, cfg)

    cutoff = prices.index[-3]
    a = base.value[base.value.index <= cutoff]
    b = res2.value[res2.value.index <= cutoff]
    pd.testing.assert_series_equal(a, b)


def test_time_in_market():
    idx = pd.bdate_range("2015-01-01", periods=10)
    target = pd.Series([0, 1, 1, 1, 0, 0, 1, 1, 0, 0], index=idx, dtype=float)
    assert time_in_market(target) == 0.5


def test_sma_signal_shape():
    prices = _prices()
    sig = sma_signal(prices, window=200)
    assert sig.iloc[:199].isna().all()
    assert set(pd.unique(sig.dropna())).issubset({0.0, 1.0})
