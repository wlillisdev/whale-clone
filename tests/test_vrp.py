"""VRP put-writing tests: Black-Scholes, tail metrics, no-look-ahead, tail gate."""

from __future__ import annotations

import numpy as np
import pandas as pd

from whale_clone.backtest import BacktestConfig, BacktestResult
from whale_clone.config import load_settings
from whale_clone.gates import GateConfig, gate_tail_risk
from whale_clone.metrics import cvar, downside_deviation, sortino
from whale_clone.vrp import bs_put_price, evaluate_vrp, load_vrp_data, simulate_put_write


def test_bs_put_price_sane_and_monotone():
    # ATM put with positive vol has positive time value.
    p = bs_put_price(100.0, 100.0, rate=0.0, sigma=0.2, t_years=1.0)
    assert p > 0.0
    # Higher vol -> more expensive put.
    p_hi = bs_put_price(100.0, 100.0, rate=0.0, sigma=0.4, t_years=1.0)
    assert p_hi > p
    # Degenerate (no time) falls back to intrinsic value.
    assert bs_put_price(90.0, 100.0, rate=0.0, sigma=0.2, t_years=0.0) == 10.0
    assert bs_put_price(110.0, 100.0, rate=0.0, sigma=0.2, t_years=0.0) == 0.0


def test_tail_metrics_capture_downside():
    # A series with a fat negative tail: downside dev > 0, CVaR negative, Sortino finite.
    r = pd.Series([0.01, 0.01, 0.01, 0.01, -0.20])
    assert downside_deviation(r, periods_per_year=12) > 0.0
    assert cvar(r, alpha=0.8) < 0.0
    # All-positive returns have zero downside deviation -> Sortino is nan (no risk).
    up = pd.Series([0.01, 0.02, 0.015, 0.03])
    assert np.isnan(sortino(up, periods_per_year=12))


def _series(n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    px = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    iv = pd.Series(0.2, index=idx)
    return pd.Series(px, index=idx), iv


def test_simulate_put_write_no_lookahead():
    index, iv = _series()
    base = simulate_put_write(index, iv, dte_days=21, moneyness=1.0, cost_bps=0.0)

    tampered = index.copy()
    tampered.iloc[-1] *= 1.5  # change only the very last day
    after = simulate_put_write(tampered, iv, dte_days=21, moneyness=1.0, cost_bps=0.0)

    # Rolls that end before the tampered day must be byte-for-byte identical.
    cutoff = base.returns.index[-2]
    pd.testing.assert_series_equal(
        base.returns[base.returns.index < cutoff],
        after.returns[after.returns.index < cutoff],
    )


def test_put_write_eats_the_crash():
    # A sharp drop below the strike must produce a large negative roll return.
    idx = pd.bdate_range("2020-01-01", periods=44)
    px = np.full(44, 100.0)
    px[21:] = 70.0  # -30% gap held into expiry
    index = pd.Series(px, index=idx)
    iv = pd.Series(0.2, index=idx)
    res = simulate_put_write(index, iv, dte_days=21, moneyness=1.0, cost_bps=0.0)
    assert res.returns.min() < -0.15  # the seller wears the fall, minus premium


def test_gate_tail_risk_fails_negative_skew():
    # Build a strategy that beats on mean but has a worse drawdown than benchmark.
    idx = pd.bdate_range("2020-01-01", periods=10)
    strat = pd.Series([0.02, 0.02, 0.02, 0.02, -0.30, 0.02, 0.02, 0.02, 0.02, 0.02], index=idx)
    bench = pd.Series(0.005, index=idx)
    res = BacktestResult(
        value=(1 + strat).cumprod(),
        benchmark_value=(1 + bench).cumprod(),
        returns=strat,
        benchmark_returns=bench,
        rebalances=[],
        weights={},
    )
    cfg = BacktestConfig(trading_days_per_year=12)
    gc = GateConfig(trading_days_per_year=12)
    gate = gate_tail_risk(res, cfg, gc)
    assert gate.passed is False  # the deep drawdown must trip the tail gate


def test_evaluate_vrp_demo_smoke():
    s = load_settings(vrp_source="demo", bootstrap_iterations=200)
    index, iv = load_vrp_data(s)
    verdict, diag = evaluate_vrp(index, iv, s)
    assert len(verdict.gates) == 6  # five standard + the added tail-risk gate
    assert any("Tail-risk" in g.name for g in verdict.gates)
    assert diag["n_periods"] > 0
