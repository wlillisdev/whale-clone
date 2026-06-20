"""Volatility risk premium: cash-secured index put-writing + tail-gated verdict.

Thesis (the most economically-grounded edge from the research fan-out): index
implied volatility runs persistently above subsequently-realised volatility, so a
seller of options is paid a risk premium. We harvest it the robust way — rolling
*fully cash-secured* at-the-money puts on an index (the CBOE PUT-index style),
never levered VIX-futures short-vol (which is what blew up XIV in 2018).

The catch, and the reason this module exists: short-vol returns are negatively
skewed with a fat left tail. A Sharpe / bootstrap pipeline will happily bless
them — high Sharpe, tight positive CI — while ignoring the steamroller. So the
verdict adds a **tail-risk gate** (max drawdown / Sortino / CVaR vs the
benchmark) on top of the usual five. That gate is the real upgrade here.

Pure functions (Black-Scholes pricing, the simulation) are fixture-tested; the
only IO is the price/IV loader. Network needs Yahoo/Stooq; use ``source="demo"``
offline. Demo data is synthetic and is a pipeline smoke test, never a claim.
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import norm

from .backtest import BacktestConfig, BacktestResult, Rebalance
from .config import Settings, load_settings
from .gates import (
    GateConfig,
    GateResult,
    Verdict,
    _full_sample_metrics,
    _gate_benchmark_beating,
    _gate_walk_forward,
    gate_tail_risk,
)
from .metrics import block_bootstrap_mean_ci, cagr
from .rigor import deflated_sharpe_gate
from .store import Store

_VRP_TRIALS = 12  # implicit trials (moneyness/tenor/IV choices) for the deflated Sharpe


def bs_put_price(spot: float, strike: float, *, rate: float, sigma: float, t_years: float) -> float:
    """Black-Scholes price of a European put (pure).

    Degenerate inputs (no time or no vol) fall back to intrinsic value.
    """
    if t_years <= 0.0 or sigma <= 0.0 or spot <= 0.0 or strike <= 0.0:
        return float(max(strike - spot, 0.0))
    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma**2) * t_years) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return float(strike * math.exp(-rate * t_years) * norm.cdf(-d2) - spot * norm.cdf(-d1))


def simulate_put_write(
    index: pd.Series,
    iv: pd.Series,
    *,
    rf_annual: float = 0.0,
    dte_days: int = 21,
    moneyness: float = 1.0,
    cost_bps: float = 10.0,
    trading_days_per_year: int = 252,
) -> BacktestResult:
    """Simulate rolling cash-secured put-writing vs buy-and-hold the index (pure).

    Each roll: sell one fully cash-secured put struck at ``moneyness * spot``,
    collect the Black-Scholes premium, earn the risk-free rate on the collateral,
    and at expiry pay any in-the-money loss. Returns are computed on the
    collateral base (= strike), so they are directly comparable to holding the
    index. Per-period returns are indexed by roll-end date.
    """
    index = index.dropna().sort_index()
    iv = iv.reindex(index.index).ffill().bfill()
    n = len(index)
    if n < 2 * dte_days:
        raise ValueError("not enough price history for even two option rolls")

    ppy = trading_days_per_year
    t_years = dte_days / ppy
    cost = cost_bps / 1e4

    starts = list(range(0, n - dte_days, dte_days))
    s_value = 1.0
    b_value = 1.0
    s_val: dict[pd.Timestamp, float] = {}
    s_ret: dict[pd.Timestamp, float] = {}
    b_val: dict[pd.Timestamp, float] = {}
    b_ret: dict[pd.Timestamp, float] = {}
    rebalances: list[Rebalance] = []

    for i in starts:
        j = i + dte_days
        t1 = index.index[j]
        spot = float(index.iloc[i])
        end = float(index.iloc[j])
        strike = moneyness * spot
        sigma = float(iv.iloc[i])

        premium = bs_put_price(spot, strike, rate=rf_annual, sigma=sigma, t_years=t_years)
        assignment_loss = max(strike - end, 0.0)
        # Return on the collateral (= strike): premium yield + interest - loss - cost.
        period_ret = (premium + strike * rf_annual * t_years - assignment_loss) / strike - cost
        bench_ret = end / spot - 1.0

        s_value *= 1.0 + period_ret
        b_value *= 1.0 + bench_ret
        s_val[t1] = s_value
        s_ret[t1] = period_ret
        b_val[t1] = b_value
        b_ret[t1] = bench_ret
        # Each roll is a full round-trip: turnover ~ 100% of notional.
        rebalances.append(Rebalance(t1, 1.0, cost, 1))

    result = BacktestResult(
        value=pd.Series(s_val, name="strategy"),
        benchmark_value=pd.Series(b_val, name="benchmark"),
        returns=pd.Series(s_ret, name="strategy_ret"),
        benchmark_returns=pd.Series(b_ret, name="benchmark_ret"),
        rebalances=rebalances,
        weights={},
    )
    return result


def _periods_per_year(settings: Settings) -> int:
    return max(1, round(settings.trading_days_per_year / settings.vrp_dte_days))


def evaluate_vrp(
    index: pd.Series, iv: pd.Series, settings: Settings
) -> tuple[Verdict, dict[str, float]]:
    """Run the put-write backtest through six gates (the five + a tail-risk gate)."""
    ppy = _periods_per_year(settings)

    def sim(*, moneyness: float, dte_days: int) -> BacktestResult:
        return simulate_put_write(
            index,
            iv,
            rf_annual=settings.risk_free_annual,
            dte_days=dte_days,
            moneyness=moneyness,
            cost_bps=settings.vrp_cost_bps,
            trading_days_per_year=settings.trading_days_per_year,
        )

    result = sim(moneyness=settings.vrp_moneyness, dte_days=settings.vrp_dte_days)
    bc = BacktestConfig(
        benchmark=settings.vrp_index,
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
        block_len=max(3, len(result.returns) // 20),
        iterations=gc.bootstrap_iterations,
        confidence=gc.bootstrap_confidence,
        seed=gc.random_seed,
    )
    g1 = GateResult(
        "Cost-adjusted expectancy (block bootstrap)",
        bool(ci.lower > 0 and not np.isnan(ci.lower)),
        f"Mean period excess {ci.mean:+.4%}; 95% CI [{ci.lower:+.4%}, {ci.upper:+.4%}].",
        {"ci_lower": ci.lower},
    )
    g2 = _gate_walk_forward(result, bc, gc)

    # Robustness: vary strike moneyness (the key short-vol knob) and tenor.
    rob: list[tuple[str, float]] = []
    for mny in (0.95, 1.0, 1.05):
        try:
            r = sim(moneyness=mny, dte_days=settings.vrp_dte_days)
            excess = cagr(r.value, periods_per_year=ppy) - cagr(
                r.benchmark_value, periods_per_year=ppy
            )
            rob.append((f"K/S={mny:.2f}", excess))
        except Exception:
            rob.append((f"K/S={mny:.2f} (err)", float("nan")))
    beats = sum(1 for _, e in rob if not np.isnan(e) and e > 0)
    g3 = GateResult(
        "Robustness (moneyness plateau)",
        beats > len(rob) / 2,
        f"{beats}/{len(rob)} strikes beat benchmark. ["
        + "; ".join(f"{n} {e:+.2%}" if not np.isnan(e) else n for n, e in rob)
        + "]",
        {},
    )
    g4 = gate_tail_risk(result, bc, gc)  # the upgrade: the gate Sharpe is blind to
    g5 = _gate_benchmark_beating(headline)
    g6 = deflated_sharpe_gate(
        result.returns,
        n_strategies_tried=_VRP_TRIALS,
        trials_sr_std=settings.trial_sharpe_dispersion,
        threshold=settings.deflated_sharpe_threshold,
        periods_per_year=ppy,
    )

    gates = [g1, g2, g3, g4, g5, g6]
    diagnostics = {
        "n_periods": float(len(result.returns)),
        "worst_period_ret": float(result.returns.min()) if len(result.returns) else float("nan"),
        "total_cost": result.total_cost,
    }
    return Verdict(all(g.passed for g in gates), gates, headline), diagnostics


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_vrp_data(settings: Settings, *, refresh: bool = False) -> tuple[pd.Series, pd.Series]:
    """Return ``(index_close, implied_vol)`` daily series for the put-write sim."""
    if settings.vrp_source == "demo":
        return _demo_vrp_data(
            settings.start_date,
            settings.end_date,
            seed=settings.random_seed,
            markup=settings.vrp_iv_markup,
            floor=settings.vrp_iv_floor,
        )

    from .data.prices import load_prices

    store = Store(settings.cache_dir)
    panel = load_prices(
        [settings.vrp_index],
        start=settings.start_date,
        end=settings.end_date,
        source=settings.vrp_source,
        benchmark=settings.vrp_index,
        store=store,
        refresh=refresh,
        seed=settings.random_seed,
    )
    index = panel[settings.vrp_index].dropna()

    iv = _try_load_vix(settings, store=store, refresh=refresh, calendar=index.index)
    if iv is None:
        iv = _realized_iv(index, markup=settings.vrp_iv_markup, floor=settings.vrp_iv_floor)
    return index, iv


def _try_load_vix(
    settings: Settings, *, store: Store, refresh: bool, calendar: pd.Index
) -> pd.Series | None:
    """Best-effort fetch of the VIX (annualised implied vol, decimal). None on failure."""
    from .data.prices import load_prices

    try:
        vix = load_prices(
            ["^VIX"],
            start=settings.start_date,
            end=settings.end_date,
            source=settings.vrp_source,
            benchmark="^VIX",
            store=store,
            refresh=refresh,
            seed=settings.random_seed,
        )["^VIX"]
    except Exception:
        return None
    return (vix / 100.0).reindex(calendar).ffill().bfill()


def _realized_iv(index: pd.Series, *, markup: float, floor: float) -> pd.Series:
    """Implied-vol proxy: trailing 21-day realised vol, annualised, marked up."""
    rets = index.pct_change(fill_method=None)
    realized = rets.rolling(21).std(ddof=1) * math.sqrt(252)
    iv = (realized * markup).clip(lower=floor)
    return iv.reindex(index.index).ffill().bfill()


def _demo_vrp_data(
    start: date, end: date, *, seed: int, markup: float, floor: float
) -> tuple[pd.Series, pd.Series]:
    """Synthetic index with occasional crashes + an IV series above realised vol.

    NOT real data. The markup makes implied > realised (so a VRP exists), while
    the crash clusters create the fat left tail the tail-gate must catch.
    """
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start=start, end=end)
    n = len(cal)
    rets = rng.normal(0.0004, 0.008, n)
    # Inject a few multi-day crash clusters (the steamroller).
    for _ in range(3):
        c = int(rng.integers(30, max(31, n - 30)))
        rets[c : c + 5] += rng.normal(-0.03, 0.02, 5)
    price = 100.0 * np.exp(np.cumsum(rets))
    index = pd.Series(price, index=cal, name="INDEX")
    iv = _realized_iv(index, markup=markup, floor=floor)
    return index, iv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="whale-vrp",
        description="Backtest cash-secured index put-writing (the volatility risk "
        "premium) vs buy-and-hold, with an added tail-risk gate.",
    )
    parser.add_argument("--demo", action="store_true", help="Offline synthetic data.")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch data.")
    args = parser.parse_args(argv)

    overrides: dict[str, object] = {}
    if args.demo:
        overrides["vrp_source"] = "demo"
    settings = load_settings(**overrides)
    if settings.vrp_source == "demo":
        print("[note] DEMO synthetic data — NOT a market claim.\n", file=sys.stderr)

    try:
        index, iv = load_vrp_data(settings, refresh=args.refresh)
        verdict, diag = evaluate_vrp(index, iv, settings)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if settings.vrp_source != "demo":
            print("\nIf data hosts are unreachable here, try: whale-vrp --demo", file=sys.stderr)
        return 2

    print(verdict.render())
    print(
        f"Diagnostics: {int(diag['n_periods'])} option rolls | "
        f"worst single-roll return {diag['worst_period_ret']:+.2%}"
    )
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
