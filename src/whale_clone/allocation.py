"""Multi-asset allocation backtest + risk-adjusted verdict (pure engine + glue).

This tests the one idea our research graded as genuinely evidence-backed:
*diversification*. The question is deliberately NOT "beat the index on return"
(a lower-volatility portfolio will often trail on raw CAGR) but "deliver a
better RISK-ADJUSTED outcome than a 60/40 benchmark after costs" — higher Sharpe
and smaller max drawdown, robustly.

Engine: fixed target weights, periodic rebalance, daily drift, costs on
turnover — for both the strategy portfolio and the benchmark portfolio (both are
real portfolios that must be rebalanced, so both pay costs; fair comparison).
Reuses :class:`BacktestResult`, the cost model, and the metrics/gate dataclasses.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backtest import BacktestResult, Rebalance
from .config import Settings, load_settings
from .costs import CostModel, one_way_turnover, traded_notional
from .gates import GateResult, Verdict
from .metrics import annualised_vol, cagr, max_drawdown, sharpe
from .store import Store


@dataclass(frozen=True)
class AllocationConfig:
    weights: dict[str, float]
    benchmark_weights: dict[str, float]
    rebalance: str = "Q"  # "M" monthly | "Q" quarterly
    cost_model: CostModel = field(default_factory=lambda: CostModel(0.0, 5.0))
    risk_free_annual: float = 0.0
    trading_days_per_year: int = 252


def _rebalance_dates(calendar: pd.DatetimeIndex, freq: str) -> set[pd.Timestamp]:
    """Last trading day of each month/quarter within the calendar."""
    period = calendar.to_period("M" if freq == "M" else "Q")
    is_last = pd.Series(period, index=calendar) != pd.Series(period, index=calendar).shift(-1)
    return set(calendar[is_last.to_numpy()])


def _simulate_fixed_weight(
    daily_ret: pd.DataFrame,
    target: dict[str, float],
    rebal_dates: set[pd.Timestamp],
    cost_model: CostModel,
) -> tuple[pd.Series, pd.Series, list[Rebalance]]:
    """Simulate a fixed-weight, periodically-rebalanced portfolio (net of costs)."""
    assets = list(target)
    value = 1.0
    w = dict(target)  # entered free on day 0
    values: dict[pd.Timestamp, float] = {}
    rets: dict[pd.Timestamp, float] = {}
    rebalances: list[Rebalance] = []
    first = True
    for d in daily_ret.index:
        row = daily_ret.loc[d]
        port_ret = 0.0
        for a in assets:
            r = row.get(a, 0.0)
            port_ret += w[a] * (0.0 if pd.isna(r) else float(r))
        value *= 1.0 + port_ret

        # Drift weights with realised returns.
        if port_ret > -1.0:
            drifted = {}
            for a in assets:
                r = row.get(a, 0.0)
                r = 0.0 if pd.isna(r) else float(r)
                drifted[a] = w[a] * (1.0 + r) / (1.0 + port_ret)
            w = drifted

        cost = 0.0
        if d in rebal_dates:
            if not first:
                traded = traded_notional(w, target)
                cost = cost_model.cost_for_traded(traded)
                value *= 1.0 - cost
                rebalances.append(Rebalance(d, one_way_turnover(w, target), cost, len(target)))
            w = dict(target)
            first = False

        values[d] = value
        rets[d] = (1.0 + port_ret) * (1.0 - cost) - 1.0
    return pd.Series(values), pd.Series(rets), rebalances


def run_allocation(prices: pd.DataFrame, config: AllocationConfig) -> BacktestResult:
    """Backtest the strategy portfolio vs the benchmark portfolio."""
    cols = sorted(set(config.weights) | set(config.benchmark_weights))
    missing = [c for c in cols if c not in prices.columns]
    if missing:
        raise ValueError(f"missing price columns for: {missing}")
    px = prices[cols].dropna().sort_index()
    if len(px) < config.trading_days_per_year:
        raise ValueError("not enough overlapping history across all assets")
    daily_ret = px.pct_change(fill_method=None).iloc[1:]
    rebal = _rebalance_dates(pd.DatetimeIndex(daily_ret.index), config.rebalance)

    s_val, s_ret, s_rb = _simulate_fixed_weight(daily_ret, config.weights, rebal, config.cost_model)
    b_val, b_ret, _ = _simulate_fixed_weight(
        daily_ret, config.benchmark_weights, rebal, config.cost_model
    )
    return BacktestResult(
        value=s_val.rename("strategy"),
        benchmark_value=b_val.rename("benchmark"),
        returns=s_ret.rename("strategy_ret"),
        benchmark_returns=b_ret.rename("benchmark_ret"),
        rebalances=s_rb,
        weights={},
    )


# --------------------------------------------------------------------------- #
# Risk-adjusted gates (Sharpe edge, not raw return)
# --------------------------------------------------------------------------- #
def _sharpe(returns: pd.Series, rf: float, ppy: int) -> float:
    return sharpe(returns, risk_free_annual=rf, periods_per_year=ppy)


def _bootstrap_sharpe_edge(
    strat: pd.Series,
    bench: pd.Series,
    *,
    rf: float,
    ppy: int,
    block_len: int,
    iterations: int,
    confidence: float,
    seed: int,
) -> tuple[float, float, float]:
    """Block-bootstrap CI for the Sharpe DIFFERENCE (strategy - benchmark).

    Paired block resampling preserves autocorrelation and the cross-asset
    relationship, giving an honest interval for the risk-adjusted edge.
    """
    df = pd.concat([strat.rename("s"), bench.rename("b")], axis=1).dropna()
    s = df["s"].to_numpy()
    b = df["b"].to_numpy()
    n = len(s)
    if n < block_len * 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_len))
    max_start = n - block_len
    diffs = np.empty(iterations)
    for i in range(iterations):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx = np.concatenate([np.arange(st, st + block_len) for st in starts])[:n]
        ss = pd.Series(s[idx])
        bb = pd.Series(b[idx])
        diffs[i] = _sharpe(ss, rf, ppy) - _sharpe(bb, rf, ppy)
    point = _sharpe(strat, rf, ppy) - _sharpe(bench, rf, ppy)
    alpha = 1.0 - confidence
    return point, float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))


def _variations(weights: dict[str, float]) -> list[tuple[str, dict[str, float], str]]:
    """Weight/rebalance perturbations for the robustness plateau (name, weights, freq)."""

    def norm(w: dict[str, float]) -> dict[str, float]:
        tot = sum(w.values())
        return {k: v / tot for k, v in w.items()}

    out: list[tuple[str, dict[str, float], str]] = [("base", weights, "Q")]
    if "GLD" in weights:
        for g in (0.10, 0.20):
            w = dict(weights)
            w["GLD"] = g
            out.append((f"gold {g:.0%}", norm(w), "Q"))
    if "DBC" in weights:
        w = dict(weights)
        w["DBC"] = 0.05
        out.append(("commod 5%", norm(w), "Q"))
    out.append(("monthly", weights, "M"))
    out.append(("equal weight", {k: 1.0 / len(weights) for k in weights}, "Q"))
    return out


def evaluate_allocation(
    prices: pd.DataFrame, settings: Settings
) -> tuple[Verdict, dict[str, float]]:
    cfg = _alloc_cfg(settings)
    ppy = settings.trading_days_per_year
    rf = settings.risk_free_annual
    result = run_allocation(prices, cfg)

    s_sharpe = _sharpe(result.returns, rf, ppy)
    b_sharpe = _sharpe(result.benchmark_returns, rf, ppy)
    s_dd = max_drawdown(result.value)
    b_dd = max_drawdown(result.benchmark_value)
    headline = {
        "strategy_cagr": cagr(result.value, periods_per_year=ppy),
        "benchmark_cagr": cagr(result.benchmark_value, periods_per_year=ppy),
        "strategy_sharpe": s_sharpe,
        "benchmark_sharpe": b_sharpe,
        "strategy_vol": annualised_vol(result.returns, periods_per_year=ppy),
        "benchmark_vol": annualised_vol(result.benchmark_returns, periods_per_year=ppy),
        "strategy_maxdd": s_dd,
        "benchmark_maxdd": b_dd,
        "avg_turnover": result.avg_turnover,
        "total_cost": result.total_cost,
        "excess_cagr": cagr(result.value, periods_per_year=ppy)
        - cagr(result.benchmark_value, periods_per_year=ppy),
    }

    # Gate 1: Sharpe edge significant (block bootstrap CI lower bound > 0).
    block = max(21, len(result.returns) // 40)
    pt, lo, hi = _bootstrap_sharpe_edge(
        result.returns,
        result.benchmark_returns,
        rf=rf,
        ppy=ppy,
        block_len=block,
        iterations=settings.bootstrap_iterations,
        confidence=settings.bootstrap_confidence,
        seed=settings.random_seed,
    )
    conf_pct = int(settings.bootstrap_confidence * 100)
    g1 = GateResult(
        "Risk-adjusted edge (bootstrap Sharpe diff)",
        bool(lo > 0 and not np.isnan(lo)),
        f"Sharpe edge {pt:+.2f}; {conf_pct}% CI [{lo:+.2f}, {hi:+.2f}]; "
        f"lower bound {'>' if (lo > 0) else '<='} 0.",
        {},
    )

    # Gate 2: walk-forward — Sharpe edge positive in the majority of windows.
    n = settings.walk_forward_windows
    chunks = np.array_split(np.arange(len(result.returns)), n)
    edges = [
        _sharpe(result.returns.iloc[c], rf, ppy)
        - _sharpe(result.benchmark_returns.iloc[c], rf, ppy)
        for c in chunks
    ]
    pos = sum(1 for e in edges if e > 0)
    g2 = GateResult(
        "Walk-forward (Sharpe edge across windows)",
        pos > n / 2,
        f"{pos}/{n} windows positive Sharpe edge; per-window {[round(e, 2) for e in edges]}.",
        {},
    )

    # Gate 3: robustness — Sharpe edge positive across weight/timing variations.
    rob: list[tuple[str, float]] = []
    for name, w, freq in _variations(cfg.weights):
        try:
            r = run_allocation(prices, _alloc_cfg(settings, weights=w, rebalance=freq))
            rob.append((name, _sharpe(r.returns, rf, ppy) - _sharpe(r.benchmark_returns, rf, ppy)))
        except Exception as exc:  # record and continue
            rob.append((f"{name} (err: {exc})", float("nan")))
    valid = [e for _, e in rob if not np.isnan(e)]
    beats = sum(1 for e in valid if e > 0)
    g3 = GateResult(
        "Robustness (Sharpe-edge plateau)",
        bool(valid) and beats > len(valid) / 2,
        f"{beats}/{len(valid)} variants keep a positive Sharpe edge. ["
        + "; ".join(f"{n} {e:+.2f}" if not np.isnan(e) else n for n, e in rob)
        + "]",
        {},
    )

    # Gate 4: risk-adjusted benchmark-beating — higher Sharpe AND smaller drawdown.
    sharpe_ok = s_sharpe > b_sharpe
    dd_ok = s_dd > b_dd  # less negative = smaller drawdown
    g4 = GateResult(
        "Risk-adjusted beating (Sharpe & drawdown)",
        bool(sharpe_ok and dd_ok),
        f"Sharpe {s_sharpe:.2f} vs {b_sharpe:.2f} ({'beats' if sharpe_ok else 'loses'}); "
        f"max drawdown {s_dd:.1%} vs {b_dd:.1%} ({'smaller' if dd_ok else 'larger'}).",
        {},
    )

    gates = [g1, g2, g3, g4]
    return Verdict(all(g.passed for g in gates), gates, headline), {}


def _alloc_cfg(
    settings: Settings, *, weights: dict[str, float] | None = None, rebalance: str | None = None
) -> AllocationConfig:
    return AllocationConfig(
        weights=weights if weights is not None else dict(settings.alloc_weights),
        benchmark_weights=dict(settings.alloc_benchmark_weights),
        rebalance=rebalance if rebalance is not None else settings.alloc_rebalance,
        cost_model=CostModel(0.0, settings.gold_slippage_bps),
        risk_free_annual=settings.risk_free_annual,
        trading_days_per_year=settings.trading_days_per_year,
    )


def _render(verdict: Verdict, h: dict[str, float], settings: Settings) -> str:
    lines = ["=" * 64, "WHALE-CLONE ALLOCATION VERDICT (diversified vs 60/40)", "=" * 64]
    lines.append(
        f"Strategy: CAGR {h['strategy_cagr']:+.2%} | vol {h['strategy_vol']:.1%} | "
        f"Sharpe {h['strategy_sharpe']:.2f} | maxDD {h['strategy_maxdd']:.1%}"
    )
    lines.append(
        f"60/40:    CAGR {h['benchmark_cagr']:+.2%} | vol {h['benchmark_vol']:.1%} | "
        f"Sharpe {h['benchmark_sharpe']:.2f} | maxDD {h['benchmark_maxdd']:.1%}"
    )
    lines.append(f"Costs {h['total_cost']:.2%} | avg turnover/rebalance {h['avg_turnover']:.1%}")
    lines.append("-" * 64)
    for g in verdict.gates:
        lines.append(f"[{'PASS' if g.passed else 'FAIL'}] {g.name}")
        lines.append(f"        {g.detail}")
    lines.append("-" * 64)
    final = (
        "PASS — better risk-adjusted outcome than 60/40 after costs"
        if verdict.passed
        else "FAIL — does not clear the risk-adjusted gates"
    )
    lines.append(f"FINAL VERDICT: {final}")
    lines.append("=" * 64)
    return "\n".join(lines)


def load_alloc_prices(settings: Settings, *, refresh: bool = False) -> pd.DataFrame:
    from .data.prices import load_prices

    store = Store(settings.cache_dir)
    assets = sorted(set(settings.alloc_weights) | set(settings.alloc_benchmark_weights))
    return load_prices(
        assets,
        start=settings.start_date,
        end=settings.end_date,
        source=settings.price_source,
        benchmark=assets[0],
        store=store,
        refresh=refresh,
        seed=settings.random_seed,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="whale-alloc",
        description="Test a diversified portfolio vs 60/40 on risk-adjusted terms, after costs.",
    )
    parser.add_argument("--demo", action="store_true", help="Offline synthetic data (no network).")
    parser.add_argument("--refresh", action="store_true", help="Ignore cache and re-fetch.")
    parser.add_argument("--price-source", choices=["yahoo", "stooq", "demo"], help="Price source.")
    args = parser.parse_args(argv)

    overrides: dict[str, object] = {}
    if args.demo:
        overrides["price_source"] = "demo"
    if args.price_source:
        overrides["price_source"] = args.price_source
    settings = load_settings(**overrides)

    if settings.price_source == "demo":
        print(
            "[note] DEMO synthetic prices — pipeline smoke test, NOT a market claim.\n",
            file=sys.stderr,
        )

    try:
        prices = load_alloc_prices(settings, refresh=args.refresh)
        verdict, _ = evaluate_allocation(prices, settings)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if settings.price_source != "demo":
            print("\nIf data hosts are unreachable here, try: whale-alloc --demo", file=sys.stderr)
        return 2

    print(_render(verdict, verdict.headline, settings))
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
