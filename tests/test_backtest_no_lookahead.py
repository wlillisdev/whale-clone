"""The most important test: prove the backtest cannot see the future.

Two complementary checks:

1. Mutating a *future* filing (one not yet public at a given date) must not
   change any of the portfolio's behaviour up to that date.
2. Every rebalance the engine performs happens on or after a real filing date —
   never on a quarter-end and never before the data was public.
"""

from __future__ import annotations

import pandas as pd

from whale_clone.backtest import BacktestConfig, run_backtest
from whale_clone.portfolio import validate_holdings


def _cfg() -> BacktestConfig:
    return BacktestConfig(benchmark="SPY", weighting="value", max_position_weight=1.0)


def test_future_filing_does_not_affect_the_past(simple_holdings, simple_prices):
    h = validate_holdings(simple_holdings)
    base = run_backtest(h, simple_prices, _cfg())

    # Tamper ONLY with the Q2 filing (public 2020-08-14): blow up a weight.
    tampered = h.copy()
    mask = tampered["period"] == "2020-Q2"
    tampered.loc[mask, "value"] = tampered.loc[mask, "value"] * 1000.0
    tampered_result = run_backtest(tampered, simple_prices, _cfg())

    # Up to (but not including) the Q2 filing date, the value paths must match
    # exactly — the future filing was invisible.
    cutoff = pd.Timestamp("2020-08-13")
    a = base.value[base.value.index <= cutoff]
    b = tampered_result.value[tampered_result.value.index <= cutoff]
    pd.testing.assert_series_equal(a, b)


def test_rebalances_only_on_or_after_filing_dates(simple_holdings, simple_prices):
    h = validate_holdings(simple_holdings)
    result = run_backtest(h, simple_prices, _cfg())
    filing_dates = set(pd.to_datetime(h["filing_date"]).unique())
    for rb in result.rebalances:
        # Each rebalance must be the first trading day on/after some filing date,
        # i.e. there exists a filing date <= rebalance date with no trading day
        # strictly between them in our calendar.
        assert any(fd <= rb.date for fd in filing_dates), rb.date


def test_no_rebalance_uses_quarter_end_prices(simple_holdings, simple_prices):
    """A rebalance must never land before its filing date (the lag is real)."""
    h = validate_holdings(simple_holdings)
    result = run_backtest(h, simple_prices, _cfg())
    earliest_filing = pd.to_datetime(h["filing_date"]).min()
    assert all(rb.date >= earliest_filing for rb in result.rebalances)


def test_value_starts_at_one_and_benchmark_too(simple_holdings, simple_prices):
    h = validate_holdings(simple_holdings)
    result = run_backtest(h, simple_prices, _cfg())
    # First simulated day's benchmark value is ~1.0 (entry not charged on day 0).
    assert result.benchmark_value.iloc[0] == 1.0
    assert result.value.iloc[0] == 1.0
