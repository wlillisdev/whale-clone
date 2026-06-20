"""The four validation gates and the final verdict (brief, section 5).

A strategy is not "real" until it survives all four:

1. **Cost-adjusted expectancy** — bootstrap 95% CI lower bound on the per-period
   excess return vs the benchmark is > 0.
2. **Walk-forward** — the edge appears in the majority of >=3 sequential windows,
   with no single window carrying the whole result.
3. **Robustness** — the result survives sensible parameter variation as a
   plateau (majority of variants still beat the benchmark), not a single spike.
4. **Benchmark-beating** — full-sample net CAGR *and* Sharpe both beat
   buy-and-hold.

Report every number after costs. Never tune parameters to sneak past a gate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, BacktestResult, run_backtest
from .costs import CostModel
from .metrics import bootstrap_mean_ci, cagr, cvar, max_drawdown, sharpe, sortino


@dataclass(frozen=True)
class GateConfig:
    bootstrap_iterations: int = 5000
    bootstrap_confidence: float = 0.95
    walk_forward_windows: int = 3
    max_single_window_share: float = 0.70
    random_seed: int = 1234
    trading_days_per_year: int = 252
    n_strategies_tried: int = 8
    deflated_sharpe_threshold: float = 0.95
    trial_sharpe_dispersion: float = 0.5
    cvar_alpha: float = 0.95  # tail fraction for the CVaR / expected-shortfall check


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    detail: str
    metrics: dict[str, float]


@dataclass(frozen=True)
class Verdict:
    passed: bool
    gates: list[GateResult]
    headline: dict[str, float]

    def render(self) -> str:
        lines = ["=" * 64, "WHALE-CLONE VERDICT", "=" * 64]
        h = self.headline
        lines.append(
            f"Strategy CAGR {h['strategy_cagr']:+.2%} | "
            f"Benchmark CAGR {h['benchmark_cagr']:+.2%} | "
            f"Excess {h['excess_cagr']:+.2%}"
        )
        lines.append(
            f"Strategy Sharpe {h['strategy_sharpe']:.2f} | "
            f"Benchmark Sharpe {h['benchmark_sharpe']:.2f} | "
            f"Avg turnover/rebalance {h['avg_turnover']:.1%} | "
            f"Total costs {h['total_cost']:.2%}"
        )
        lines.append("-" * 64)
        for g in self.gates:
            mark = "PASS" if g.passed else "FAIL"
            lines.append(f"[{mark}] {g.name}")
            lines.append(f"        {g.detail}")
        lines.append("-" * 64)
        final = (
            "PASS — beats the index after costs"
            if self.passed
            else "FAIL — does not clear the gates"
        )
        lines.append(f"FINAL VERDICT: {final}")
        lines.append("=" * 64)
        return "\n".join(lines)


def _full_sample_metrics(result: BacktestResult, cfg: BacktestConfig) -> dict[str, float]:
    ppy = cfg.trading_days_per_year
    return {
        "strategy_cagr": cagr(result.value, periods_per_year=ppy),
        "benchmark_cagr": cagr(result.benchmark_value, periods_per_year=ppy),
        "strategy_sharpe": sharpe(
            result.returns, risk_free_annual=cfg.risk_free_annual, periods_per_year=ppy
        ),
        "benchmark_sharpe": sharpe(
            result.benchmark_returns, risk_free_annual=cfg.risk_free_annual, periods_per_year=ppy
        ),
        "avg_turnover": result.avg_turnover,
        "total_cost": result.total_cost,
    }


def _gate_expectancy(result: BacktestResult, gc: GateConfig) -> GateResult:
    ci = bootstrap_mean_ci(
        result.excess_returns,
        iterations=gc.bootstrap_iterations,
        confidence=gc.bootstrap_confidence,
        seed=gc.random_seed,
    )
    passed = ci.lower > 0 and not np.isnan(ci.lower)
    detail = (
        f"Mean daily excess {ci.mean:+.4%}; "
        f"{int(gc.bootstrap_confidence * 100)}% CI "
        f"[{ci.lower:+.4%}, {ci.upper:+.4%}]; lower bound {'>' if passed else '<='} 0."
    )
    return GateResult(
        "Cost-adjusted expectancy",
        passed,
        detail,
        {"mean_excess": ci.mean, "ci_lower": ci.lower, "ci_upper": ci.upper},
    )


def _gate_walk_forward(result: BacktestResult, cfg: BacktestConfig, gc: GateConfig) -> GateResult:
    n = gc.walk_forward_windows
    idx = result.returns.index
    if len(idx) < n * 2:
        return GateResult(
            "Walk-forward / out-of-sample", False, "not enough data to split into windows.", {}
        )
    chunks = np.array_split(np.arange(len(idx)), n)
    excess_logs: list[float] = []
    positive = 0
    for c in chunks:
        s = result.returns.iloc[c]
        b = result.benchmark_returns.iloc[c]
        # Cumulative log excess over the window.
        strat_log = float(np.log1p(s).sum())
        bench_log = float(np.log1p(b).sum())
        win_excess = strat_log - bench_log
        excess_logs.append(win_excess)
        if win_excess > 0:
            positive += 1

    majority = positive > n / 2
    # Share check: no single positive window may carry more than
    # max_single_window_share of the *positive* total (negative windows must not
    # distort the denominator).
    positive_total = sum(e for e in excess_logs if e > 0)
    if positive_total > 0:
        max_share = max(excess_logs) / positive_total
        share_ok = max_share <= gc.max_single_window_share
    else:
        max_share = float("nan")
        share_ok = False
    passed = bool(majority and share_ok)
    detail = (
        f"{positive}/{n} windows beat benchmark; "
        f"window log-excess {[round(e, 4) for e in excess_logs]}; "
        f"max single-window share {max_share:.0%} "
        f"(limit {gc.max_single_window_share:.0%})."
    )
    return GateResult(
        "Walk-forward / out-of-sample",
        passed,
        detail,
        {"windows_positive": float(positive), "max_window_share": max_share},
    )


def _variations(
    holdings: pd.DataFrame, cfg: BacktestConfig
) -> list[tuple[str, pd.DataFrame, BacktestConfig]]:
    """Sensible parameter variations for the robustness plateau test."""
    variants: list[tuple[str, pd.DataFrame, BacktestConfig]] = [("base", holdings, cfg)]

    # 2x slippage stress.
    stressed = replace(
        cfg,
        cost_model=CostModel(
            commission_bps=cfg.cost_model.commission_bps,
            slippage_bps=cfg.cost_model.slippage_bps * 2.0,
        ),
    )
    variants.append(("2x slippage", holdings, stressed))

    # Cap variations.
    variants.append(("cap 20%", holdings, replace(cfg, max_position_weight=0.20)))
    variants.append(("cap 30%", holdings, replace(cfg, max_position_weight=0.30)))

    # Weighting flip.
    other = "equal" if cfg.weighting == "value" else "value"
    variants.append((f"{other} weighting", holdings, replace(cfg, weighting=other)))

    # Concentration: vary top-N so the edge must hold as a plateau, not one N.
    for n in (3, 5, 8, 10):
        if n != cfg.top_n:
            variants.append((f"top {n}", holdings, replace(cfg, top_n=n)))

    # Drop one manager (the alphabetically first), if more than one.
    managers = sorted(holdings["manager"].unique())
    if len(managers) > 1:
        dropped = holdings[holdings["manager"] != managers[0]]
        variants.append((f"drop {managers[0]}", dropped, cfg))

    return variants


def _gate_robustness(
    holdings: pd.DataFrame, prices: pd.DataFrame, cfg: BacktestConfig, gc: GateConfig
) -> GateResult:
    results: list[tuple[str, float]] = []
    for name, h, c in _variations(holdings, cfg):
        try:
            r = run_backtest(h, prices, c)
        except Exception as exc:
            results.append((f"{name} (error: {exc})", float("nan")))
            continue
        excess = cagr(r.value, periods_per_year=c.trading_days_per_year) - cagr(
            r.benchmark_value, periods_per_year=c.trading_days_per_year
        )
        results.append((name, excess))

    # A variant that errored or blew up (NaN CAGR) counts as a FAIL, not as an
    # excluded sample — otherwise a losing config silently shrinks the plateau.
    total = len(results)
    beats = sum(1 for _, e in results if not np.isnan(e) and e > 0)
    passed = total > 0 and beats > total / 2
    summary = "; ".join(f"{n} {e:+.2%}" if not np.isnan(e) else f"{n} FAIL" for n, e in results)
    detail = f"{beats}/{total} variants beat benchmark (plateau). [{summary}]"
    return GateResult(
        "Robustness (parameter plateau)",
        passed,
        detail,
        {"variants_beating": float(beats), "variants_total": float(total)},
    )


def gate_tail_risk(result: BacktestResult, cfg: BacktestConfig, gc: GateConfig) -> GateResult:
    """Tail-risk gate — the guard a Sharpe/bootstrap pipeline structurally lacks.

    A strategy with a beautiful Sharpe can still sit on a catastrophic left tail
    (selling crash insurance is the canonical example). This gate fails any
    strategy whose downside is worse than simply holding the benchmark, judged on
    three one-sided measures the standard gates miss:

    * **Max drawdown** must be no deeper than the benchmark's.
    * **Sortino** (downside-only risk-adjusted return) must beat the benchmark's.
    * **CVaR / expected shortfall** (typical loss in the worst tail) must be no
      worse than the benchmark's.
    """
    ppy = cfg.trading_days_per_year
    s_dd = max_drawdown(result.value)
    b_dd = max_drawdown(result.benchmark_value)
    s_sortino = sortino(result.returns, risk_free_annual=cfg.risk_free_annual, periods_per_year=ppy)
    b_sortino = sortino(
        result.benchmark_returns, risk_free_annual=cfg.risk_free_annual, periods_per_year=ppy
    )
    s_cvar = cvar(result.returns, alpha=gc.cvar_alpha)
    b_cvar = cvar(result.benchmark_returns, alpha=gc.cvar_alpha)
    # "Less negative is better" for drawdown and CVaR; higher is better for Sortino.
    dd_ok = bool(s_dd >= b_dd and not np.isnan(s_dd))
    sortino_ok = bool(s_sortino > b_sortino and not np.isnan(s_sortino))
    cvar_ok = bool(s_cvar >= b_cvar and not np.isnan(s_cvar))
    passed = dd_ok and sortino_ok and cvar_ok
    detail = (
        f"Max DD {s_dd:.1%} vs {b_dd:.1%} ({'ok' if dd_ok else 'worse'}); "
        f"Sortino {s_sortino:.2f} vs {b_sortino:.2f} ({'beats' if sortino_ok else 'loses'}); "
        f"CVaR{int(gc.cvar_alpha * 100)} {s_cvar:+.2%} vs {b_cvar:+.2%} "
        f"({'ok' if cvar_ok else 'worse'})."
    )
    return GateResult(
        "Tail-risk (max DD / Sortino / CVaR)",
        passed,
        detail,
        {
            "strategy_max_dd": s_dd,
            "benchmark_max_dd": b_dd,
            "strategy_sortino": s_sortino,
            "benchmark_sortino": b_sortino,
            "strategy_cvar": s_cvar,
            "benchmark_cvar": b_cvar,
        },
    )


def _gate_benchmark_beating(headline: dict[str, float]) -> GateResult:
    cagr_ok = headline["strategy_cagr"] > headline["benchmark_cagr"]
    sharpe_ok = headline["strategy_sharpe"] > headline["benchmark_sharpe"]
    passed = bool(cagr_ok and sharpe_ok)
    detail = (
        f"CAGR {headline['strategy_cagr']:+.2%} vs {headline['benchmark_cagr']:+.2%} "
        f"({'beats' if cagr_ok else 'loses'}); "
        f"Sharpe {headline['strategy_sharpe']:.2f} vs {headline['benchmark_sharpe']:.2f} "
        f"({'beats' if sharpe_ok else 'loses'})."
    )
    return GateResult("Benchmark-beating (CAGR & Sharpe)", passed, detail, {})


def evaluate_gates(
    holdings: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: BacktestConfig,
    gc: GateConfig,
) -> Verdict:
    """Run the base backtest and evaluate all four gates -> a Verdict."""
    result = run_backtest(holdings, prices, cfg)
    headline = _full_sample_metrics(result, cfg)
    headline["excess_cagr"] = headline["strategy_cagr"] - headline["benchmark_cagr"]

    from .rigor import deflated_sharpe_gate

    gates = [
        _gate_expectancy(result, gc),
        _gate_walk_forward(result, cfg, gc),
        _gate_robustness(holdings, prices, cfg, gc),
        _gate_benchmark_beating(headline),
        deflated_sharpe_gate(
            result.excess_returns,
            n_strategies_tried=gc.n_strategies_tried,
            trials_sr_std=gc.trial_sharpe_dispersion,
            threshold=gc.deflated_sharpe_threshold,
            periods_per_year=cfg.trading_days_per_year,
        ),
    ]
    passed = all(g.passed for g in gates)
    return Verdict(passed=passed, gates=gates, headline=headline)
