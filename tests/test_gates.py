"""Gate behaviour tests.

We construct holdings + prices where the answer is known, and assert the gates
return the honest verdict. Critically: a strategy that does NOT beat the
benchmark must FAIL, and pure noise must not be reported as an edge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from whale_clone.backtest import BacktestConfig
from whale_clone.costs import CostModel
from whale_clone.gates import GateConfig, evaluate_gates
from whale_clone.portfolio import validate_holdings


def _quarterly_holdings(tickers, start="2015-01-01", end="2022-12-31"):
    quarter_ends = pd.date_range(start=start, end=end, freq="QE")
    rows = []
    for qe in quarter_ends:
        for t in tickers:
            rows.append(
                {
                    "manager": "M1",
                    "period": pd.Timestamp(qe).to_period("Q").strftime("%Y-Q%q"),
                    "filing_date": qe + pd.Timedelta(days=45),
                    "ticker": t,
                    "value": 100.0,
                }
            )
    return validate_holdings(pd.DataFrame(rows))


def _gc() -> GateConfig:
    return GateConfig(bootstrap_iterations=1500, walk_forward_windows=3, random_seed=1)


def _cfg() -> BacktestConfig:
    return BacktestConfig(
        benchmark="SPY", weighting="value", max_position_weight=1.0, cost_model=CostModel()
    )


def test_strategy_identical_to_benchmark_does_not_pass():
    """If the only holding IS the benchmark, excess is ~0 -> cannot clear gates."""
    dates = pd.bdate_range("2015-01-01", "2022-12-31")
    n = len(dates)
    spy = pd.Series([100 * 1.0004**i for i in range(n)], index=dates)
    prices = pd.DataFrame({"SPY": spy})
    holdings = _quarterly_holdings(["SPY"])

    verdict = evaluate_gates(holdings, prices, _cfg(), _gc())
    # Holding SPY itself cannot beat SPY after any cost; benchmark-beating fails.
    assert not verdict.passed
    bb = next(g for g in verdict.gates if g.name.startswith("Benchmark-beating"))
    assert not bb.passed


def test_noise_does_not_produce_a_positive_expectancy():
    """Random-walk holdings vs random-walk SPY: expectancy CI lower bound <= 0."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2015-01-01", "2022-12-31")
    n = len(dates)

    def walk(mu):
        return pd.Series(100 * np.exp(np.cumsum(rng.normal(mu, 0.01, n))), index=dates)

    prices = pd.DataFrame({"AAA": walk(0.0003), "BBB": walk(0.0003), "SPY": walk(0.0003)})
    holdings = _quarterly_holdings(["AAA", "BBB"])

    verdict = evaluate_gates(holdings, prices, _cfg(), _gc())
    exp_gate = next(g for g in verdict.gates if g.name.startswith("Cost-adjusted"))
    assert not exp_gate.passed  # no real edge in noise


def test_clear_outperformer_passes_expectancy_and_benchmark_gates():
    """A holding that strictly dominates SPY every day clears those two gates."""
    dates = pd.bdate_range("2015-01-01", "2022-12-31")
    n = len(dates)
    spy = pd.Series([100 * 1.0002**i for i in range(n)], index=dates)
    winner = pd.Series([100 * 1.0006**i for i in range(n)], index=dates)
    prices = pd.DataFrame({"WIN": winner, "SPY": spy})
    holdings = _quarterly_holdings(["WIN"])

    verdict = evaluate_gates(holdings, prices, _cfg(), _gc())
    exp_gate = next(g for g in verdict.gates if g.name.startswith("Cost-adjusted"))
    bb_gate = next(g for g in verdict.gates if g.name.startswith("Benchmark-beating"))
    assert exp_gate.passed
    assert bb_gate.passed
    assert verdict.headline["excess_cagr"] > 0


def test_verdict_renders():
    dates = pd.bdate_range("2015-01-01", "2018-12-31")
    n = len(dates)
    prices = pd.DataFrame(
        {
            "WIN": [100 * 1.0006**i for i in range(n)],
            "SPY": [100 * 1.0002**i for i in range(n)],
        },
        index=dates,
    )
    holdings = _quarterly_holdings(["WIN"], start="2015-01-01", end="2018-12-31")
    verdict = evaluate_gates(holdings, prices, _cfg(), _gc())
    text = verdict.render()
    assert "WHALE-CLONE VERDICT" in text
    assert "FINAL VERDICT" in text
