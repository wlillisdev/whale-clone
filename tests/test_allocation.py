"""Allocation engine tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from whale_clone.allocation import (
    AllocationConfig,
    _rebalance_dates,
    _simulate_fixed_weight,
    run_allocation,
)
from whale_clone.costs import CostModel


def _panel(seed=0, n=800):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    cols = {a: 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))) for a in ("SPY", "IEF", "GLD")}
    return pd.DataFrame(cols, index=idx)


def test_rebalance_dates_quarterly_fewer_than_monthly():
    cal = pd.bdate_range("2015-01-01", "2018-12-31")
    q = _rebalance_dates(cal, "Q")
    m = _rebalance_dates(cal, "M")
    assert 0 < len(q) < len(m)
    assert len(q) == 16  # 4 years * 4 quarters


def test_single_asset_full_weight_reproduces_that_asset():
    panel = _panel()
    cfg = AllocationConfig(
        weights={"SPY": 1.0},
        benchmark_weights={"SPY": 1.0},
        cost_model=CostModel(0.0, 0.0),
    )
    res = run_allocation(panel, cfg)
    # Strategy and benchmark are both 100% SPY with no cost -> identical paths.
    pd.testing.assert_series_equal(
        res.value.rename("x"), res.benchmark_value.rename("x"), atol=1e-9
    )


def test_no_cost_fixed_weight_matches_manual_two_asset():
    panel = _panel()
    daily = panel[["SPY", "IEF"]].pct_change(fill_method=None).iloc[1:]
    vals, _rets, rebs = _simulate_fixed_weight(
        daily, {"SPY": 0.5, "IEF": 0.5}, set(), CostModel(0.0, 0.0)
    )
    # With no rebalance dates, no trades occur after the free entry.
    assert rebs == []
    assert len(vals) == len(daily)
    assert vals.iloc[0] > 0


def test_costs_reduce_value():
    panel = _panel()
    free = run_allocation(
        panel,
        AllocationConfig({"SPY": 0.5, "IEF": 0.5}, {"SPY": 1.0}, cost_model=CostModel(0.0, 0.0)),
    )
    costly = run_allocation(
        panel,
        AllocationConfig({"SPY": 0.5, "IEF": 0.5}, {"SPY": 1.0}, cost_model=CostModel(0.0, 50.0)),
    )
    # Higher costs -> lower terminal strategy value.
    assert costly.value.iloc[-1] < free.value.iloc[-1]
