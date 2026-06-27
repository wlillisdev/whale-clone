"""The combined system: blend the best bits into one risk-managed book.

Every strategy in this repo failed *on its own* to beat buy-and-hold on raw
return. But that is the wrong test for a portfolio. The pro move is to combine
the pieces that actually carried signal and judge the *book*:

* **Index core** — beta; the thing that actually compounds.
* **VRP overlay** — cash-secured put-writing, the one sleeve that beat the index
  on a risk-adjusted basis (higher Sharpe, shallower drawdown).
* **Insider tilt** — an equal-weight basket of recent cluster-buy names, the
  satellite with real (if decayed) academic support.

The lever that turns "higher Sharpe" into "more money": **volatility targeting.**
If the blend's Sharpe beats the index, then scaling the blend's exposure up to
the index's own volatility (with a hard leverage cap and an honest financing
cost) makes it beat the index on *return* too. That is the legitimate way the
best bits become a system that wins — and it is also exactly where levering a
short-vol book can blow up, which is why the tail-risk gate is non-negotiable
here. No tuning to pass; the gates get the final word.

All sleeves are reduced to a common monthly return series; pure combination math
is fixture-tested. Network only via the loaders; ``--demo`` runs offline.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, BacktestResult, Rebalance
from .config import Settings, load_settings
from .costs import CostModel
from .gates import (
    GateConfig,
    GateResult,
    Verdict,
    _full_sample_metrics,
    _gate_benchmark_beating,
    _gate_walk_forward,
    gate_tail_risk,
)
from .insiders import build_target_schedule, cluster_events, run_basket_backtest
from .metrics import block_bootstrap_mean_ci, cagr, sharpe
from .rigor import deflated_sharpe_gate
from .store import Store
from .vrp import _realized_iv, simulate_put_write

_SYSTEM_TRIALS = 30  # be conservative: many strategies/variants were tried across the repo


def _to_monthly_returns(value: pd.Series) -> pd.Series:
    """Month-end simple returns from a daily value-level series."""
    m = value.resample("ME").last().dropna()
    return m.pct_change().dropna()


def combine_sleeves(
    sleeves: pd.DataFrame,
    weights: dict[str, float],
    *,
    vol_target: float,
    max_leverage: float,
    borrow_spread: float,
    rf_annual: float,
    periods_per_year: int = 12,
) -> pd.Series:
    """Blend monthly sleeve returns into one net monthly return series (pure, causal).

    ``sleeves`` columns are monthly returns per sleeve; missing months are treated
    as that sleeve sitting in cash (0). With ``vol_target > 0`` the whole blend is
    scaled toward that annual volatility using a leverage factor computed *only*
    from trailing data (no look-ahead), capped at ``max_leverage``; the levered
    portion pays ``rf_annual + borrow_spread``.
    """
    w = pd.Series(weights, dtype=float)
    aligned = sleeves.reindex(columns=w.index).fillna(0.0)
    blend = aligned.mul(w, axis=1).sum(axis=1)

    if vol_target <= 0:
        return blend

    # Causal leverage: trailing 12m vol known through the prior month only.
    trailing_vol = blend.rolling(periods_per_year).std(ddof=1) * np.sqrt(periods_per_year)
    lev = (vol_target / trailing_vol.shift(1)).clip(upper=max_leverage)
    lev = lev.fillna(1.0).clip(lower=0.0)
    borrow = (lev - 1.0).clip(lower=0.0) * (rf_annual + borrow_spread) / periods_per_year
    return pd.Series(lev * blend - borrow)


def _value_from_returns(returns: pd.Series, name: str) -> pd.Series:
    return (1.0 + returns).cumprod().rename(name)


def build_sleeves(settings: Settings, *, refresh: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(monthly_sleeve_returns, index_monthly_returns)`` on a common calendar."""
    from .data.prices import load_prices

    store = Store(settings.cache_dir)
    index_sym = settings.vrp_index
    universe = sorted({*settings.insider_universe, index_sym})
    prices = load_prices(
        universe,
        start=settings.start_date,
        end=settings.end_date,
        source=settings.price_source,
        benchmark=index_sym,
        store=store,
        refresh=refresh,
        seed=settings.random_seed,
    )
    index_px = prices[index_sym].dropna()
    index_ret = _to_monthly_returns(index_px)

    # VRP sleeve: cash-secured monthly put-writes on the index.
    iv = _realized_iv(index_px, markup=settings.vrp_iv_markup, floor=settings.vrp_iv_floor)
    vrp_res = simulate_put_write(
        index_px,
        iv,
        rf_annual=settings.risk_free_annual,
        dte_days=settings.vrp_dte_days,
        moneyness=settings.vrp_moneyness,
        cost_bps=settings.vrp_cost_bps,
        trading_days_per_year=settings.trading_days_per_year,
    )
    vrp_ret = _to_monthly_returns(vrp_res.value)

    # Insider sleeve: equal-weight basket of recent cluster-buy names.
    from .data.insiders import load_insider_buys

    buys = load_insider_buys(
        settings.insider_universe,
        source=settings.insider_source,
        start=settings.start_date,
        end=settings.end_date,
        store=store,
        refresh=refresh,
        seed=settings.random_seed,
    )
    events = cluster_events(
        buys,
        min_buyers=settings.insider_min_buyers,
        require_officer=settings.insider_require_officer,
        min_value=settings.insider_min_value,
        window_days=settings.insider_window_days,
    )
    cal = pd.DatetimeIndex(prices.index)
    targets = build_target_schedule(events, cal, hold_days=settings.insider_hold_days)
    try:
        ins_res = run_basket_backtest(
            prices,
            targets,
            index_sym,
            cost_model=CostModel(0.0, settings.gold_slippage_bps),
            risk_free_annual=settings.risk_free_annual,
            trading_days_per_year=settings.trading_days_per_year,
        )
        insider_ret = _to_monthly_returns(ins_res.value)
    except ValueError:
        insider_ret = pd.Series(dtype=float)  # signal never fired -> sleeve sits in cash

    sleeves = pd.DataFrame({"index": index_ret, "vrp": vrp_ret, "insider": insider_ret}).dropna(
        how="all"
    )
    return sleeves, index_ret


def _result_from_monthly(strategy: pd.Series, benchmark: pd.Series) -> BacktestResult:
    common = strategy.index.intersection(benchmark.index)
    s = strategy.reindex(common).fillna(0.0)
    b = benchmark.reindex(common).fillna(0.0)
    rebs = [Rebalance(pd.Timestamp(d), 0.0, 0.0, 3) for d in common]  # monthly rebalance markers
    return BacktestResult(
        value=_value_from_returns(s, "strategy"),
        benchmark_value=_value_from_returns(b, "benchmark"),
        returns=s.rename("strategy_ret"),
        benchmark_returns=b.rename("benchmark_ret"),
        rebalances=rebs,
        weights={},
    )


def evaluate_system(
    sleeves: pd.DataFrame, index_ret: pd.Series, settings: Settings, *, lever: bool = True
) -> tuple[Verdict, dict[str, float]]:
    """Blend the sleeves, volatility-target to the index, and judge by all gates.

    With ``lever=False`` the plain (unscaled) blend is evaluated — useful to see
    the book's natural risk-adjusted profile before the leverage decision.
    """
    ppy = 12
    weights = {
        "index": settings.system_w_index,
        "vrp": settings.system_w_vrp,
        "insider": settings.system_w_insider,
    }
    # Default vol target = the index's own realised vol, so we compare at equal risk.
    index_vol = float(index_ret.std(ddof=1) * np.sqrt(ppy))
    if not lever:
        vol_target = 0.0
    elif settings.system_vol_target > 0:
        vol_target = settings.system_vol_target
    else:
        vol_target = index_vol

    def blend(w: dict[str, float]) -> pd.Series:
        return combine_sleeves(
            sleeves,
            w,
            vol_target=vol_target,
            max_leverage=settings.system_max_leverage,
            borrow_spread=settings.system_borrow_spread,
            rf_annual=settings.risk_free_annual,
            periods_per_year=ppy,
        )

    strat = blend(weights)
    result = _result_from_monthly(strat, index_ret)
    bc = BacktestConfig(
        benchmark="index", risk_free_annual=settings.risk_free_annual, trading_days_per_year=ppy
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
        block_len=max(3, len(result.returns) // 20),
        iterations=gc.bootstrap_iterations,
        confidence=gc.bootstrap_confidence,
        seed=gc.random_seed,
    )
    g1 = GateResult(
        "Cost-adjusted expectancy (block bootstrap)",
        bool(ci.lower > 0 and not np.isnan(ci.lower)),
        f"Mean monthly excess {ci.mean:+.4%}; 95% CI [{ci.lower:+.4%}, {ci.upper:+.4%}].",
        {"ci_lower": ci.lower},
    )
    g2 = _gate_walk_forward(result, bc, gc)

    # Robustness: shift weight between the VRP overlay and the index core.
    rob: list[tuple[str, float]] = []
    for shift in (-0.15, 0.0, 0.15):
        w = dict(weights)
        w["vrp"] = max(0.0, w["vrp"] + shift)
        w["index"] = max(0.0, w["index"] - shift)
        r = _result_from_monthly(blend(w), index_ret)
        rob.append(
            (
                f"vrp{w['vrp']:.2f}",
                cagr(r.value, periods_per_year=ppy) - cagr(r.benchmark_value, periods_per_year=ppy),
            )
        )
    beats = sum(1 for _, e in rob if not np.isnan(e) and e > 0)
    g3 = GateResult(
        "Robustness (mix plateau)",
        beats > len(rob) / 2,
        f"{beats}/{len(rob)} mixes beat index. ["
        + "; ".join(f"{n} {e:+.2%}" if not np.isnan(e) else n for n, e in rob)
        + "]",
        {},
    )
    g4 = gate_tail_risk(result, bc, gc)
    g5 = _gate_benchmark_beating(headline)
    g6 = deflated_sharpe_gate(
        result.returns,
        n_strategies_tried=_SYSTEM_TRIALS,
        trials_sr_std=settings.trial_sharpe_dispersion,
        threshold=settings.deflated_sharpe_threshold,
        periods_per_year=ppy,
    )

    gates = [g1, g2, g3, g4, g5, g6]
    if vol_target > 0:
        unlevered = combine_sleeves(
            sleeves,
            weights,
            vol_target=0.0,
            max_leverage=settings.system_max_leverage,
            borrow_spread=settings.system_borrow_spread,
            rf_annual=settings.risk_free_annual,
            periods_per_year=ppy,
        )
        ratio = (strat / unlevered).replace([np.inf, -np.inf], np.nan).dropna()
        avg_leverage = float(ratio.mean()) if not ratio.empty else 1.0
    else:
        avg_leverage = 1.0
    diagnostics = {
        "n_months": float(len(result.returns)),
        "vol_target": vol_target,
        "avg_leverage": avg_leverage,
        "strategy_vol": float(result.returns.std(ddof=1) * np.sqrt(ppy)),
        "index_vol": index_vol,
        "strategy_sharpe": sharpe(result.returns, periods_per_year=ppy),
    }
    return Verdict(all(g.passed for g in gates), gates, headline), diagnostics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="whale-system",
        description="Backtest the combined best-bits portfolio (index + VRP + insider), "
        "volatility-targeted to the index, judged by all gates incl. tail risk.",
    )
    parser.add_argument("--demo", action="store_true", help="Offline synthetic data.")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch data.")
    parser.add_argument(
        "--no-leverage", action="store_true", help="Disable volatility targeting / leverage."
    )
    args = parser.parse_args(argv)

    overrides: dict[str, object] = {}
    if args.demo:
        overrides["price_source"] = "demo"
        overrides["insider_source"] = "demo"
    settings = load_settings(**overrides)
    if settings.price_source == "demo":
        print("[note] DEMO synthetic data — NOT a market claim.\n", file=sys.stderr)

    try:
        sleeves, index_ret = build_sleeves(settings, refresh=args.refresh)
        verdict, diag = evaluate_system(sleeves, index_ret, settings, lever=not args.no_leverage)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if settings.price_source != "demo":
            print("\nIf data hosts are unreachable here, try: whale-system --demo", file=sys.stderr)
        return 2

    print(verdict.render())
    print(
        f"Diagnostics: {int(diag['n_months'])} months | avg leverage {diag['avg_leverage']:.2f}x | "
        f"strategy vol {diag['strategy_vol']:.1%} vs index {diag['index_vol']:.1%} | "
        f"Sharpe {diag['strategy_sharpe']:.2f}"
    )
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
