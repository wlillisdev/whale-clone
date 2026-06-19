"""Gold timing strategy: pipeline + the four gates, adapted for a single asset.

Pre-committed v1 (see config): 12-month time-series momentum on GLD, long/flat,
monthly rebalance, benchmarked against buy-and-hold GLD. Costs applied; the
expectancy gate uses a *block* bootstrap (timing returns are autocorrelated);
the robustness gate varies the signal lookback to demand a plateau, not a spike.

This module is the IO<->engine glue for the gold strategy, mirroring
``pipeline.py`` for the 13F clone. Everything it calls downstream is pure.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, BacktestResult
from .config import Settings, load_settings
from .costs import CostModel
from .gates import (
    GateConfig,
    GateResult,
    Verdict,
    _full_sample_metrics,
    _gate_benchmark_beating,
    _gate_walk_forward,
)
from .metrics import block_bootstrap_mean_ci, cagr
from .signal_backtest import SignalBacktestConfig, run_signal_backtest, time_in_market
from .signals import momentum_signal, monthly_targets, sma_signal
from .store import Store


def build_target(prices: pd.Series, *, signal: str, lookback: int, allow_short: bool) -> pd.Series:
    """Daily, causal target-weight series for the chosen signal."""
    if signal == "momentum":
        raw = momentum_signal(prices, lookback=lookback, allow_short=allow_short)
    elif signal == "sma":
        raw = sma_signal(prices, window=lookback, allow_short=allow_short)
    else:
        raise ValueError(f"unknown gold signal: {signal!r}")
    return monthly_targets(raw)


def _signal_cfg(settings: Settings) -> SignalBacktestConfig:
    return SignalBacktestConfig(
        cost_model=CostModel(commission_bps=0.0, slippage_bps=settings.gold_slippage_bps),
        risk_free_annual=settings.risk_free_annual,
        trading_days_per_year=settings.trading_days_per_year,
    )


def run_strategy(
    prices: pd.Series, settings: Settings, *, lookback: int | None = None
) -> tuple[BacktestResult, pd.Series]:
    lb = lookback if lookback is not None else settings.gold_lookback
    target = build_target(
        prices, signal=settings.gold_signal, lookback=lb, allow_short=settings.gold_allow_short
    )
    result = run_signal_backtest(prices, prices, target, _signal_cfg(settings))
    return result, target


def _bc_cfg(settings: Settings) -> BacktestConfig:
    return BacktestConfig(
        risk_free_annual=settings.risk_free_annual,
        trading_days_per_year=settings.trading_days_per_year,
    )


def _lookback_grid(settings: Settings) -> list[int]:
    base = settings.gold_lookback
    if settings.gold_signal == "momentum":
        return [126, 189, 252, 315, 378]  # 6, 9, 12, 15, 18 months
    return [100, 150, 200, 250, base]


def _gate_expectancy_block(
    result: BacktestResult, target: pd.Series, gc: GateConfig, block_len: int
) -> GateResult:
    n_trades = max(1, int((target.diff().fillna(0) != 0).sum()))
    if block_len <= 0:
        block_len = max(5, len(result.returns) // n_trades)  # ≈ average holding period
    ci = block_bootstrap_mean_ci(
        result.excess_returns,
        block_len=block_len,
        iterations=gc.bootstrap_iterations,
        confidence=gc.bootstrap_confidence,
        seed=gc.random_seed,
    )
    passed = bool(ci.lower > 0 and not np.isnan(ci.lower))
    detail = (
        f"Mean daily excess {ci.mean:+.4%}; block bootstrap (block={block_len}d) "
        f"{int(gc.bootstrap_confidence * 100)}% CI [{ci.lower:+.4%}, {ci.upper:+.4%}]; "
        f"lower bound {'>' if passed else '<='} 0."
    )
    return GateResult("Cost-adjusted expectancy (block bootstrap)", passed, detail, {})


def _gate_robustness_lookback(prices: pd.Series, settings: Settings, ppy: int) -> GateResult:
    results: list[tuple[str, float]] = []
    for lb in _lookback_grid(settings):
        try:
            r, _ = run_strategy(prices, settings, lookback=lb)
            excess = cagr(r.value, periods_per_year=ppy) - cagr(
                r.benchmark_value, periods_per_year=ppy
            )
            results.append((f"lookback {lb}", excess))
        except Exception as exc:
            results.append((f"lookback {lb} (err: {exc})", float("nan")))
    valid = [e for _, e in results if not np.isnan(e)]
    beats = sum(1 for e in valid if e > 0)
    passed = bool(valid) and beats > len(valid) / 2
    summary = "; ".join(f"{n} {e:+.2%}" if not np.isnan(e) else n for n, e in results)
    return GateResult(
        "Robustness (lookback plateau)",
        passed,
        f"{beats}/{len(valid)} lookbacks beat buy-and-hold. [{summary}]",
        {},
    )


def evaluate_gold(prices: pd.Series, settings: Settings) -> tuple[Verdict, dict[str, float]]:
    gc = GateConfig(
        bootstrap_iterations=settings.bootstrap_iterations,
        bootstrap_confidence=settings.bootstrap_confidence,
        walk_forward_windows=settings.walk_forward_windows,
        max_single_window_share=settings.max_single_window_share,
        random_seed=settings.random_seed,
        trading_days_per_year=settings.trading_days_per_year,
    )
    bc = _bc_cfg(settings)
    result, target = run_strategy(prices, settings)
    headline = _full_sample_metrics(result, bc)
    headline["excess_cagr"] = headline["strategy_cagr"] - headline["benchmark_cagr"]

    from .rigor import deflated_sharpe_gate

    gates = [
        _gate_expectancy_block(result, target, gc, settings.gold_block_bootstrap_len),
        _gate_walk_forward(result, bc, gc),
        _gate_robustness_lookback(prices, settings, settings.trading_days_per_year),
        _gate_benchmark_beating(headline),
        deflated_sharpe_gate(
            result.excess_returns,
            n_strategies_tried=settings.n_strategies_tried,
            trials_sr_std=settings.trial_sharpe_dispersion,
            threshold=settings.deflated_sharpe_threshold,
            periods_per_year=settings.trading_days_per_year,
        ),
    ]
    diagnostics = {
        "time_in_market": time_in_market(target),
        "n_trades": float(len(result.rebalances)),
    }
    verdict = Verdict(passed=all(g.passed for g in gates), gates=gates, headline=headline)
    return verdict, diagnostics


def load_gold_prices(settings: Settings, *, refresh: bool = False) -> pd.Series:
    store = Store(settings.cache_dir)
    from .data.prices import load_prices

    panel = load_prices(
        [settings.gold_instrument],
        start=settings.start_date,
        end=settings.end_date,
        source=settings.price_source,
        benchmark=settings.gold_instrument,
        store=store,
        refresh=refresh,
        seed=settings.random_seed,
    )
    return panel[settings.gold_instrument].dropna()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="whale-gold",
        description="Backtest a gold timing strategy vs buy-and-hold gold, after costs.",
    )
    parser.add_argument("--demo", action="store_true", help="Offline synthetic data (no network).")
    parser.add_argument("--refresh", action="store_true", help="Ignore cache and re-fetch.")
    parser.add_argument("--signal", choices=["momentum", "sma"], help="Override signal type.")
    parser.add_argument("--lookback", type=int, help="Override signal lookback (days).")
    parser.add_argument("--price-source", choices=["yahoo", "stooq", "demo"], help="Price source.")
    args = parser.parse_args(argv)

    overrides: dict[str, object] = {}
    if args.demo:
        overrides["price_source"] = "demo"
    if args.price_source:
        overrides["price_source"] = args.price_source
    if args.signal:
        overrides["gold_signal"] = args.signal
    if args.lookback:
        overrides["gold_lookback"] = args.lookback
    settings = load_settings(**overrides)

    if settings.price_source == "demo":
        print(
            "[note] DEMO synthetic prices — pipeline smoke test, NOT a market claim.\n",
            file=sys.stderr,
        )

    try:
        prices = load_gold_prices(settings, refresh=args.refresh)
        verdict, diag = evaluate_gold(prices, settings)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if settings.price_source != "demo":
            print("\nIf data hosts are unreachable here, try: whale-gold --demo", file=sys.stderr)
        return 2

    print(verdict.render())
    print(
        f"Diagnostics: time-in-market {diag['time_in_market']:.0%} | "
        f"trades {int(diag['n_trades'])} | "
        f"signal {settings.gold_signal} lookback {settings.gold_lookback}d on "
        f"{settings.gold_instrument}"
    )
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
