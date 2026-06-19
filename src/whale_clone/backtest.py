"""Quarterly-rebalance backtest loop (pure).

Given holdings (with filing dates), a price panel (adjusted close, including the
benchmark column) and a :class:`BacktestConfig`, simulate the clone portfolio
day by day, net of costs, against a buy-and-hold benchmark.

No-look-ahead invariants enforced here:

* We only ever rebalance on a real 13F *filing date* (mapped forward to the
  next available trading day), never the quarter-end.
* On each rebalance we use :func:`portfolio.holdings_known_on`, which can only
  see filings already public on that date.
* The day's market return is earned on the weights held *coming into* the day;
  the rebalance happens at that day's close. We never trade on information from
  a future bar.

Fairness: both the strategy and the benchmark are bought once at the start and
that identical entry trade is not charged (it would cancel in the comparison).
Every *subsequent* rebalance of the strategy pays the cost model. The benchmark
never trades again. So the only cost difference measured is the strategy's
ongoing turnover — exactly the thing we want to test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .costs import CostModel, one_way_turnover, traded_notional
from .portfolio import filing_dates, holdings_known_on, target_weights


@dataclass(frozen=True)
class BacktestConfig:
    benchmark: str = "SPY"
    weighting: str = "value"
    max_position_weight: float = 0.25
    top_n: int | None = None  # keep only each manager's top-N positions (concentration)
    cost_model: CostModel = field(default_factory=CostModel)
    risk_free_annual: float = 0.0
    trading_days_per_year: int = 252


@dataclass
class Rebalance:
    date: pd.Timestamp
    turnover: float
    cost: float
    n_positions: int


@dataclass
class BacktestResult:
    value: pd.Series  # strategy value level, net of costs, starts at 1.0
    benchmark_value: pd.Series  # benchmark value level, starts at 1.0
    returns: pd.Series  # daily strategy simple returns (net)
    benchmark_returns: pd.Series  # daily benchmark simple returns
    rebalances: list[Rebalance]
    weights: dict[pd.Timestamp, dict[str, float]]

    @property
    def excess_returns(self) -> pd.Series:
        return (self.returns - self.benchmark_returns).rename("excess")

    @property
    def total_cost(self) -> float:
        return sum(r.cost for r in self.rebalances)

    @property
    def avg_turnover(self) -> float:
        if not self.rebalances:
            return 0.0
        return sum(r.turnover for r in self.rebalances) / len(self.rebalances)


def _row_returns(daily_ret: pd.DataFrame, day: pd.Timestamp) -> dict[str, float]:
    """One day's simple returns as a plain ``{ticker: float}`` dict, NaNs -> 0.0.

    NaN returns (a name not yet listed / no quote) are treated as 0.0, i.e. that
    sleeve holds value flat for the day — a simple, conservative stand-in.
    """
    row = daily_ret.loc[day]
    out: dict[str, float] = {}
    for ticker, val in row.items():
        out[str(ticker)] = 0.0 if pd.isna(val) else float(val)
    return out


def _next_trading_day(calendar: pd.DatetimeIndex, day: pd.Timestamp) -> pd.Timestamp | None:
    """First trading day on or after ``day`` (None if past the calendar)."""
    pos = int(calendar.searchsorted(day, side="left"))
    if pos >= len(calendar):
        return None
    return pd.Timestamp(calendar[pos])


def run_backtest(
    holdings: pd.DataFrame,
    prices: pd.DataFrame,
    config: BacktestConfig,
) -> BacktestResult:
    """Run the clone backtest. ``prices`` must include the benchmark column."""
    if config.benchmark not in prices.columns:
        raise ValueError(f"benchmark {config.benchmark!r} not present in prices")

    prices = prices.sort_index()
    calendar = pd.DatetimeIndex(prices.index)
    daily_ret = prices.pct_change(fill_method=None)

    # Map each filing date forward to the next real trading day.
    rebalance_targets: dict[pd.Timestamp, dict[str, float]] = {}
    for f_date in filing_dates(holdings):
        td = _next_trading_day(calendar, pd.Timestamp(f_date))
        if td is None:
            continue
        visible = holdings_known_on(holdings, pd.Timestamp(f_date))
        weights = target_weights(
            visible,
            weighting=config.weighting,
            max_position_weight=config.max_position_weight,
            top_n=config.top_n,
        )
        # Keep only names we actually have prices for; renormalise.
        weights = {t: w for t, w in weights.items() if t in prices.columns}
        wsum = sum(weights.values())
        if wsum > 0:
            weights = {t: w / wsum for t, w in weights.items()}
            rebalance_targets[td] = weights

    if not rebalance_targets:
        raise ValueError("no actionable rebalances within the price calendar")

    first_rebalance = min(rebalance_targets)
    sim_dates = calendar[calendar >= first_rebalance]

    value = 1.0
    bench_value = 1.0
    weights_now: dict[str, float] = {}
    rebalances: list[Rebalance] = []
    weights_log: dict[pd.Timestamp, dict[str, float]] = {}

    value_series: dict[pd.Timestamp, float] = {}
    ret_series: dict[pd.Timestamp, float] = {}
    bench_value_series: dict[pd.Timestamp, float] = {}
    bench_ret_series: dict[pd.Timestamp, float] = {}

    is_first_rebalance = True
    bench_invested = False  # benchmark is bought at the first rebalance close, like the strategy
    for d in sim_dates:
        day_returns = _row_returns(daily_ret, d)

        # 1) Earn the day's return on weights held coming into the day.
        day_ret = sum(w * day_returns.get(t, 0.0) for t, w in weights_now.items())
        value *= 1.0 + day_ret

        # Drift the held weights with the realised returns.
        if weights_now and day_ret > -1.0:
            weights_now = {
                t: w * (1.0 + day_returns.get(t, 0.0)) / (1.0 + day_ret)
                for t, w in weights_now.items()
            }

        # 2) Benchmark earns its own daily return (buy-and-hold), but only once
        #    it has been bought (at the first rebalance close) — same entry
        #    timing as the strategy, so the comparison is fair.
        bench_day_ret = day_returns.get(config.benchmark, 0.0) if bench_invested else 0.0
        bench_value *= 1.0 + bench_day_ret

        # 3) Rebalance at the close if a filing became actionable today.
        rebalance_cost = 0.0
        if d in rebalance_targets:
            target = rebalance_targets[d]
            traded = traded_notional(weights_now, target)
            turnover = one_way_turnover(weights_now, target)
            if is_first_rebalance:
                # Entry trade is not charged (cancels vs the benchmark entry).
                is_first_rebalance = False
            else:
                rebalance_cost = config.cost_model.cost_for_traded(traded)
                value *= 1.0 - rebalance_cost
            weights_now = dict(target)
            weights_log[d] = weights_now
            rebalances.append(Rebalance(d, turnover, rebalance_cost, len(target)))
            bench_invested = True  # buy the benchmark at this same close

        # The strategy's net daily return includes any cost charged today.
        net_day_ret = (1.0 + day_ret) * (1.0 - rebalance_cost) - 1.0
        value_series[d] = value
        ret_series[d] = net_day_ret
        bench_value_series[d] = bench_value
        bench_ret_series[d] = bench_day_ret

    return BacktestResult(
        value=pd.Series(value_series, name="strategy"),
        benchmark_value=pd.Series(bench_value_series, name="benchmark"),
        returns=pd.Series(ret_series, name="strategy_ret"),
        benchmark_returns=pd.Series(bench_ret_series, name="benchmark_ret"),
        rebalances=rebalances,
        weights=weights_log,
    )
