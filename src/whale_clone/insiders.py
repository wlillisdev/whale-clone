"""Insider cluster-buying strategy: signal + basket backtest + verdict.

Thesis (the most promising new lead from the research fan-out): when multiple
insiders — especially officers — buy their own stock on the open market in a
short window ("cluster buy"), it predicts outperformance. Unlike 13F (45-day
lag), Form 4 is filed within ~2 days, so the signal is fresh.

We hold an equal-weight basket of names with a recent qualifying cluster buy,
entered the day *after* the filing (no look-ahead), for a fixed horizon, vs a
buy-and-hold benchmark — judged by the same five gates incl. the deflated-Sharpe
overfitting guard. Honest prior: the edge has decayed ~60-70% out of sample and
lives in small caps; expect a near-miss at best.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, BacktestResult, Rebalance
from .config import Settings, load_settings
from .costs import CostModel, one_way_turnover, traded_notional
from .data.insiders import load_insider_buys
from .gates import (
    GateConfig,
    GateResult,
    Verdict,
    _full_sample_metrics,
    _gate_benchmark_beating,
    _gate_walk_forward,
)
from .metrics import block_bootstrap_mean_ci, cagr
from .rigor import deflated_sharpe_gate
from .store import Store

_INSIDER_TRIALS = 20  # implicit trials (window/hold/min-buyers choices) for deflated Sharpe


def cluster_events(
    buys: pd.DataFrame,
    *,
    min_buyers: int,
    require_officer: bool,
    min_value: float,
    window_days: int,
) -> pd.DataFrame:
    """Qualifying cluster-buy events: (ticker, event_date) (pure, causal).

    An event fires on a purchase filing date if, within the trailing
    ``window_days``, ``>= min_buyers`` distinct insiders bought, total value
    ``>= min_value``, and (optionally) at least one was an officer.
    """
    if buys.empty:
        return pd.DataFrame(columns=["ticker", "event_date"])
    out: list[dict[str, object]] = []
    win = pd.Timedelta(days=window_days)
    for ticker, g in buys.groupby("ticker"):
        g = g.sort_values("filing_date")
        for d in g["filing_date"].unique():
            w = g[(g["filing_date"] > d - win) & (g["filing_date"] <= d)]
            if (
                w["owner"].nunique() >= min_buyers
                and float(w["value"].sum()) >= min_value
                and (not require_officer or bool(w["is_officer"].any()))
            ):
                out.append({"ticker": ticker, "event_date": pd.Timestamp(d)})
    return pd.DataFrame(out, columns=["ticker", "event_date"])


def build_target_schedule(
    events: pd.DataFrame, calendar: pd.DatetimeIndex, *, hold_days: int
) -> pd.DataFrame:
    """Daily equal-weight target weights for held names (pure, no look-ahead).

    A name is held from the first trading day *after* its event date through
    ``hold_days`` later; weights are equal across all names held that day.
    """
    cal = pd.DatetimeIndex(calendar).sort_values()
    tickers = sorted(events["ticker"].unique()) if not events.empty else []
    member = pd.DataFrame(0.0, index=cal, columns=tickers)
    for _, ev in events.iterrows():
        pos = int(cal.searchsorted(pd.Timestamp(ev["event_date"]), side="right"))  # strictly after
        if pos >= len(cal):
            continue
        start = cal[pos]
        end = start + pd.Timedelta(days=hold_days)
        member.loc[start:end, ev["ticker"]] = 1.0
    counts = member.sum(axis=1)
    weights = member.div(counts.where(counts > 0, 1.0), axis=0)
    return weights


def run_basket_backtest(
    prices: pd.DataFrame,
    targets: pd.DataFrame,
    benchmark: str,
    *,
    cost_model: CostModel,
    risk_free_annual: float = 0.0,
    trading_days_per_year: int = 252,
) -> BacktestResult:
    """Backtest a time-varying equal-weight basket vs buy-and-hold benchmark."""
    if benchmark not in prices.columns:
        raise ValueError(f"benchmark {benchmark!r} not in prices")
    prices = prices.sort_index()
    daily_ret = prices.pct_change(fill_method=None)
    rf_daily = risk_free_annual / trading_days_per_year

    held = targets.reindex(columns=[c for c in targets.columns if c in prices.columns])
    active = held.index[held.sum(axis=1) > 0]
    if len(active) == 0:
        raise ValueError("no days with any holdings — signal never fires")
    sim = prices.index[(prices.index >= active[0]) & (prices.index <= held.index[-1])]

    value = 1.0
    bench_value = 1.0
    w_prev: dict[str, float] = {}
    bench_invested = False
    rebalances: list[Rebalance] = []
    vseries: dict[pd.Timestamp, float] = {}
    rseries: dict[pd.Timestamp, float] = {}
    bvseries: dict[pd.Timestamp, float] = {}
    brseries: dict[pd.Timestamp, float] = {}

    first = True
    for d in sim:
        row = held.loc[d] if d in held.index else None
        w = {t: float(v) for t, v in row.items() if v > 0} if row is not None else {}

        cost = 0.0
        if not first and (w or w_prev):
            traded = traded_notional(w_prev, w)
            if traded > 0:
                cost = cost_model.cost_for_traded(traded)
                value *= 1.0 - cost
                rebalances.append(Rebalance(d, one_way_turnover(w_prev, w), cost, len(w)))

        ret = daily_ret.loc[d]
        gross = 0.0
        for t, wt in w.items():
            r = ret.get(t, 0.0)
            gross += wt * (0.0 if pd.isna(r) else float(r))
        gross += (1.0 - sum(w.values())) * rf_daily
        value *= 1.0 + gross

        b = ret.get(benchmark, 0.0)
        b = 0.0 if pd.isna(b) else float(b)
        bench_day = b if bench_invested else 0.0
        bench_value *= 1.0 + bench_day
        bench_invested = True

        vseries[d] = value
        rseries[d] = (1.0 - cost) * (1.0 + gross) - 1.0
        bvseries[d] = bench_value
        brseries[d] = bench_day
        w_prev = w
        first = False

    return BacktestResult(
        value=pd.Series(vseries, name="strategy"),
        benchmark_value=pd.Series(bvseries, name="benchmark"),
        returns=pd.Series(rseries, name="strategy_ret"),
        benchmark_returns=pd.Series(brseries, name="benchmark_ret"),
        rebalances=rebalances,
        weights={},
    )


def _events_for(settings: Settings, buys: pd.DataFrame, *, min_buyers: int) -> pd.DataFrame:
    return cluster_events(
        buys,
        min_buyers=min_buyers,
        require_officer=settings.insider_require_officer,
        min_value=settings.insider_min_value,
        window_days=settings.insider_window_days,
    )


def evaluate_insiders(
    buys: pd.DataFrame, prices: pd.DataFrame, settings: Settings
) -> tuple[Verdict, dict[str, float]]:
    ppy = settings.trading_days_per_year
    cost = CostModel(0.0, settings.gold_slippage_bps)
    cal = pd.DatetimeIndex(prices.index)

    events = _events_for(settings, buys, min_buyers=settings.insider_min_buyers)
    targets = build_target_schedule(events, cal, hold_days=settings.insider_hold_days)
    result = run_basket_backtest(
        prices,
        targets,
        settings.benchmark,
        cost_model=cost,
        risk_free_annual=settings.risk_free_annual,
        trading_days_per_year=ppy,
    )
    bc = BacktestConfig(
        benchmark=settings.benchmark,
        risk_free_annual=settings.risk_free_annual,
        trading_days_per_year=ppy,
    )
    gc = GateConfig(
        bootstrap_iterations=settings.bootstrap_iterations,
        bootstrap_confidence=settings.bootstrap_confidence,
        walk_forward_windows=settings.walk_forward_windows,
        max_single_window_share=settings.max_single_window_share,
        random_seed=settings.random_seed,
        trading_days_per_year=ppy,
    )
    headline = _full_sample_metrics(result, bc)
    headline["excess_cagr"] = headline["strategy_cagr"] - headline["benchmark_cagr"]

    ci = block_bootstrap_mean_ci(
        result.excess_returns,
        block_len=max(21, len(result.returns) // 40),
        iterations=gc.bootstrap_iterations,
        confidence=gc.bootstrap_confidence,
        seed=gc.random_seed,
    )
    g1 = GateResult(
        "Cost-adjusted expectancy (block bootstrap)",
        bool(ci.lower > 0 and not np.isnan(ci.lower)),
        f"Mean daily excess {ci.mean:+.4%}; 95% CI [{ci.lower:+.4%}, {ci.upper:+.4%}].",
        {},
    )
    g2 = _gate_walk_forward(result, bc, gc)
    # Robustness: vary the min-buyers threshold (the key signal knob).
    rob: list[tuple[str, float]] = []
    for mb in (2, 3, 4):
        try:
            ev = _events_for(settings, buys, min_buyers=mb)
            tg = build_target_schedule(ev, cal, hold_days=settings.insider_hold_days)
            r = run_basket_backtest(
                prices,
                tg,
                settings.benchmark,
                cost_model=cost,
                risk_free_annual=settings.risk_free_annual,
                trading_days_per_year=ppy,
            )
            rob.append(
                (
                    f"min_buyers={mb}",
                    cagr(r.value, periods_per_year=ppy)
                    - cagr(r.benchmark_value, periods_per_year=ppy),
                )
            )
        except Exception:
            rob.append((f"min_buyers={mb} (err)", float("nan")))
    beats = sum(1 for _, e in rob if not np.isnan(e) and e > 0)
    g3 = GateResult(
        "Robustness (min-buyers plateau)",
        beats > len(rob) / 2,
        f"{beats}/{len(rob)} thresholds beat benchmark. ["
        + "; ".join(f"{n} {e:+.2%}" if not np.isnan(e) else n for n, e in rob)
        + "]",
        {},
    )
    g4 = _gate_benchmark_beating(headline)
    g5 = deflated_sharpe_gate(
        result.excess_returns,
        n_strategies_tried=_INSIDER_TRIALS,
        trials_sr_std=settings.trial_sharpe_dispersion,
        threshold=settings.deflated_sharpe_threshold,
        periods_per_year=ppy,
    )

    gates = [g1, g2, g3, g4, g5]
    diagnostics = {
        "n_events": float(len(events)),
        "n_names": float(events["ticker"].nunique()) if not events.empty else 0.0,
        "avg_turnover": result.avg_turnover,
    }
    return Verdict(all(g.passed for g in gates), gates, headline), diagnostics


def load_insider_data(
    settings: Settings, *, refresh: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from .data.prices import load_prices

    store = Store(settings.cache_dir)
    buys = load_insider_buys(
        settings.insider_universe,
        source=settings.insider_source,
        start=settings.start_date,
        end=settings.end_date,
        store=store,
        refresh=refresh,
        seed=settings.random_seed,
    )
    prices = load_prices(
        settings.insider_universe,
        start=settings.start_date,
        end=settings.end_date,
        source=settings.price_source,
        benchmark=settings.benchmark,
        store=store,
        refresh=refresh,
        seed=settings.random_seed,
    )
    return buys, prices


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="whale-insiders",
        description="Backtest an insider cluster-buying strategy vs the index, after costs.",
    )
    parser.add_argument("--demo", action="store_true", help="Offline synthetic data.")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch data.")
    args = parser.parse_args(argv)

    overrides: dict[str, object] = {}
    if args.demo:
        overrides["insider_source"] = "demo"
        overrides["price_source"] = "demo"
    settings = load_settings(**overrides)
    if settings.insider_source == "demo":
        print("[note] DEMO synthetic insider data — NOT a market claim.\n", file=sys.stderr)

    try:
        buys, prices = load_insider_data(settings, refresh=args.refresh)
        verdict, diag = evaluate_insiders(buys, prices, settings)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if settings.insider_source != "demo":
            print(
                "\nIf data hosts are unreachable here, try: whale-insiders --demo", file=sys.stderr
            )
        return 2

    print(verdict.render())
    print(
        f"Diagnostics: {int(diag['n_events'])} cluster events across "
        f"{int(diag['n_names'])} names | avg turnover/rebalance {diag['avg_turnover']:.1%}"
    )
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
