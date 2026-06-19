"""Tests added in the pre-merge hardening pass (from the audit findings)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from whale_clone.allocation import AllocationConfig, _simulate_fixed_weight, run_allocation
from whale_clone.backtest import BacktestConfig, BacktestResult, run_backtest
from whale_clone.config import load_settings
from whale_clone.costs import CostModel
from whale_clone.gates import GateConfig, _gate_walk_forward
from whale_clone.portfolio import holdings_known_on, validate_holdings
from whale_clone.signals import monthly_targets


def _panel(seed=0, n=900):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame(
        {a: 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))) for a in ("SPY", "IEF", "GLD")},
        index=idx,
    )


# --- C2 / H1: holdings dedup --------------------------------------------------
def test_holdings_known_on_keeps_only_latest_period_on_shared_filing_date():
    rows = [
        ("M", "2020-Q1", "2020-08-14", "AAA", 100.0),
        ("M", "2020-Q2", "2020-08-14", "BBB", 100.0),  # same filing date, later period
    ]
    h = validate_holdings(
        pd.DataFrame(rows, columns=["manager", "period", "filing_date", "ticker", "value"])
    )
    visible = holdings_known_on(h, pd.Timestamp("2020-08-14"))
    assert set(visible["period"]) == {"2020-Q2"}
    assert set(visible["ticker"]) == {"BBB"}


# --- The missing look-ahead guarantee for the allocation engine ---------------
def test_allocation_no_lookahead():
    panel = _panel()
    cfg = AllocationConfig({"SPY": 0.5, "GLD": 0.5}, {"SPY": 1.0}, cost_model=CostModel(0.0, 5.0))
    base = run_allocation(panel, cfg)

    tampered = panel.copy()
    tampered.iloc[-1] *= 1.5  # blow up the final bar only
    after = run_allocation(tampered, cfg)

    cutoff = panel.index[-3]
    a = base.value[base.value.index <= cutoff]
    b = after.value[after.value.index <= cutoff]
    pd.testing.assert_series_equal(a, b)


def test_simulate_fixed_weight_compounds_correctly():
    # Constant equal returns -> weights stay 50/50, value compounds cleanly.
    idx = pd.bdate_range("2020-01-01", periods=2)
    daily = pd.DataFrame({"A": [0.01, 0.01], "B": [0.01, 0.01]}, index=idx)
    vals, _rets, _rb = _simulate_fixed_weight(
        daily, {"A": 0.5, "B": 0.5}, set(), CostModel(0.0, 0.0)
    )
    assert vals.iloc[-1] == pytest.approx(1.01**2, rel=1e-12)


# --- H2: walk-forward share uses the positive total ---------------------------
def _result_from_returns(strat: list[float], bench: list[float]) -> BacktestResult:
    idx = pd.bdate_range("2020-01-01", periods=len(strat))
    s = pd.Series(strat, index=idx)
    b = pd.Series(bench, index=idx)
    return BacktestResult(
        value=(1 + s).cumprod(),
        benchmark_value=(1 + b).cumprod(),
        returns=s,
        benchmark_returns=b,
        rebalances=[],
        weights={},
    )


def test_walk_forward_rejects_single_dominant_window():
    # Window 1 carries ~90% of the positive total -> share check must fail.
    strat = [0.05, 0.05, 0.05, 0.005, 0.005, 0.005, -0.02, -0.02, -0.02]
    res = _result_from_returns(strat, [0.0] * 9)
    gc = GateConfig(walk_forward_windows=3, max_single_window_share=0.70)
    gate = _gate_walk_forward(res, BacktestConfig(), gc)
    assert not gate.passed
    assert gate.metrics["max_window_share"] > 0.70


# --- backtest guard branches --------------------------------------------------
def test_run_backtest_raises_without_benchmark(simple_holdings, simple_prices):
    h = validate_holdings(simple_holdings)
    no_bench = simple_prices.drop(columns=["SPY"])
    with pytest.raises(ValueError, match="benchmark"):
        run_backtest(h, no_bench, BacktestConfig(benchmark="SPY"))


def test_run_backtest_raises_when_no_actionable_rebalances(simple_holdings, simple_prices):
    h = validate_holdings(simple_holdings).copy()
    h["filing_date"] = pd.Timestamp("2099-01-01")  # all filings past the price calendar
    with pytest.raises(ValueError, match="no actionable rebalances"):
        run_backtest(h, simple_prices, BacktestConfig(benchmark="SPY"))


# --- signals: the literal shift(1) causality ----------------------------------
def test_monthly_targets_applies_decision_next_day_not_same_day():
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    sig = pd.Series(1.0, index=idx)  # always "risk-on"
    target = monthly_targets(sig)
    jan = idx[idx.to_period("M") == pd.Period("2020-01")]
    jan_end = jan[-1]
    nxt = idx[idx.get_loc(jan_end) + 1]
    assert target.loc[jan_end] == 0.0  # decision NOT applied on its own day
    assert target.loc[nxt] == 1.0  # applied the next trading day


# --- glue / CLI smoke (demo, fast bootstrap) ----------------------------------
def test_pipeline_run_demo_end_to_end():
    from whale_clone.pipeline import run

    s = load_settings(holdings_source="demo", price_source="demo", bootstrap_iterations=200)
    verdict = run(s)
    assert len(verdict.gates) == 4
    assert "excess_cagr" in verdict.headline


def test_evaluate_gold_demo_smoke():
    from whale_clone.gold import evaluate_gold, load_gold_prices

    s = load_settings(price_source="demo", bootstrap_iterations=200)
    prices = load_gold_prices(s)
    verdict, diag = evaluate_gold(prices, s)
    assert len(verdict.gates) == 4
    assert 0.0 <= diag["time_in_market"] <= 1.0
    assert diag["n_trades"] >= 0


def test_evaluate_allocation_demo_smoke():
    from whale_clone.allocation import evaluate_allocation, load_alloc_prices

    s = load_settings(price_source="demo", bootstrap_iterations=200)
    prices = load_alloc_prices(s)
    verdict, _ = evaluate_allocation(prices, s)
    assert len(verdict.gates) == 4
    assert "strategy_sharpe" in verdict.headline and "strategy_maxdd" in verdict.headline


def test_tracker_cli_demo_runs():
    from whale_clone.tracker import main

    assert main(["--demo", "--top", "3"]) == 0
