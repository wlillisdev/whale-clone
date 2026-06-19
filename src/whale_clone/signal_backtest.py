"""Single-asset timing backtest (pure).

Reuses the cost model and the :class:`BacktestResult` shape from the 13F engine,
so every metric and gate consumes the output unchanged. The only genuinely new
mechanic versus the 13F loop: when the strategy is flat (or partially invested),
the idle capital earns the risk-free rate rather than nothing — otherwise a
long/flat timer would be unfairly penalised for sitting in cash.

Fairness: both the strategy and the buy-and-hold benchmark are entered once at
the first simulated day with no charge (it cancels in the comparison); every
subsequent position change pays the cost model.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .backtest import BacktestResult, Rebalance
from .costs import CostModel, one_way_turnover, traded_notional


@dataclass(frozen=True)
class SignalBacktestConfig:
    cost_model: CostModel
    risk_free_annual: float = 0.0
    trading_days_per_year: int = 252


def run_signal_backtest(
    prices: pd.Series,
    benchmark_prices: pd.Series,
    target: pd.Series,
    config: SignalBacktestConfig,
) -> BacktestResult:
    """Backtest a daily target-weight series against buy-and-hold of the asset.

    ``target`` holds the weight (in {-1,0,1}) *held during* each day; it must
    already be causal (see :func:`signals.monthly_targets`).
    """
    prices = prices.sort_index().astype(float)
    benchmark_prices = benchmark_prices.reindex(prices.index).astype(float)
    target = target.reindex(prices.index).fillna(0.0)

    ret = prices.pct_change(fill_method=None)
    bench_ret = benchmark_prices.pct_change(fill_method=None)
    rf_daily = config.risk_free_annual / config.trading_days_per_year

    # Start once the strategy first takes a non-flat position.
    nonzero = target[target != 0.0]
    if nonzero.empty:
        raise ValueError("signal is flat for the entire sample — nothing to test")
    start = nonzero.index[0]
    sim = prices.index[prices.index >= start]

    value = 1.0
    bench_value = 1.0
    w_prev = float(target.loc[start])  # established free on the first sim day
    rebalances: list[Rebalance] = []
    weights_log: dict[pd.Timestamp, dict[str, float]] = {start: {"ASSET": w_prev}}

    value_series: dict[pd.Timestamp, float] = {}
    ret_series: dict[pd.Timestamp, float] = {}
    bench_value_series: dict[pd.Timestamp, float] = {}
    bench_ret_series: dict[pd.Timestamp, float] = {}

    first = True
    for d in sim:
        w = float(target.loc[d])

        # Trade cost when the target changes (free on the very first day).
        rebalance_cost = 0.0
        if not first and w != w_prev:
            traded = traded_notional({"ASSET": w_prev}, {"ASSET": w})
            rebalance_cost = config.cost_model.cost_for_traded(traded)
            value *= 1.0 - rebalance_cost
            rebalances.append(
                Rebalance(d, one_way_turnover({"ASSET": w_prev}, {"ASSET": w}), rebalance_cost, 1)
            )
            weights_log[d] = {"ASSET": w}

        # Earn the day's return: invested sleeve in the asset, idle sleeve at rf.
        r = ret.loc[d]
        r = 0.0 if pd.isna(r) else float(r)
        gross = w * r + (1.0 - abs(w)) * rf_daily
        value *= 1.0 + gross

        b = bench_ret.loc[d]
        b = 0.0 if pd.isna(b) else float(b)
        bench_day = 0.0 if first else b  # benchmark entered free on day 0
        bench_value *= 1.0 + bench_day

        net_day = (1.0 - rebalance_cost) * (1.0 + gross) - 1.0
        value_series[d] = value
        ret_series[d] = net_day
        bench_value_series[d] = bench_value
        bench_ret_series[d] = bench_day
        w_prev = w
        first = False

    return BacktestResult(
        value=pd.Series(value_series, name="strategy"),
        benchmark_value=pd.Series(bench_value_series, name="benchmark"),
        returns=pd.Series(ret_series, name="strategy_ret"),
        benchmark_returns=pd.Series(bench_ret_series, name="benchmark_ret"),
        rebalances=rebalances,
        weights=weights_log,
    )


def time_in_market(target: pd.Series) -> float:
    """Fraction of days the strategy holds a non-flat position."""
    t = target.dropna()
    if t.empty:
        return 0.0
    return float((t != 0.0).mean())
