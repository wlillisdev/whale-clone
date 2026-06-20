"""Can an ML model 'read the charts' and predict trades? Test it honestly.

This builds a machine-learning price-direction predictor — exactly the "AI reads
the graph and makes the call" idea — and runs it through the same five validation
gates as everything else, including the deflated-Sharpe overfitting guard.

The honest design choices that make this a *fair* test (and that most retail
ML-trading demos skip):

* **Strictly walk-forward / no look-ahead.** At each point the model is trained
  only on data whose outcome was already known, then predicts forward. Features
  are causal. A test asserts a future bar cannot change past predictions.
* **Costs charged** on every position change (daily flips are expensive — that is
  part of the honest answer).
* **Deflated Sharpe with a large trial count**, because an ML pipeline implicitly
  searches a huge space (features, model, threshold) — so an in-sample-pretty
  result must clear a high bar to be believed.

Expected outcome (stated up front, per the house rule): the model will look
plausible in-sample and most likely **fail** out-of-sample / the overfitting
guard. If it survives, we will have found something real, rigorously.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from .backtest import BacktestConfig
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
from .metrics import block_bootstrap_mean_ci
from .rigor import deflated_sharpe_gate
from .signal_backtest import SignalBacktestConfig, run_signal_backtest, time_in_market

# An ML pipeline searches a large implicit space (features x model x threshold x
# lookbacks). Be honest about it: hold the edge to a high deflated-Sharpe bar.
_ML_IMPLICIT_TRIALS = 50


def _rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def make_features(prices: pd.Series) -> pd.DataFrame:
    """Causal technical features from a price series (no look-ahead)."""
    p = prices.astype(float)
    r1 = p.pct_change(fill_method=None)
    feats = {
        "ret_1": r1,
        "ret_5": p.pct_change(5, fill_method=None),
        "ret_21": p.pct_change(21, fill_method=None),
        "ret_63": p.pct_change(63, fill_method=None),
        "ret_126": p.pct_change(126, fill_method=None),
        "ret_252": p.pct_change(252, fill_method=None),
        "vol_21": r1.rolling(21).std(),
        "vol_63": r1.rolling(63).std(),
        "ma10": p / p.rolling(10).mean() - 1.0,
        "ma20": p / p.rolling(20).mean() - 1.0,
        "ma50": p / p.rolling(50).mean() - 1.0,
        "ma200": p / p.rolling(200).mean() - 1.0,
        "rsi14": _rsi(p, 14),
        "hi_252": p / p.rolling(252).max() - 1.0,
        "lo_252": p / p.rolling(252).min() - 1.0,
    }
    return pd.DataFrame(feats, index=p.index)


def make_target(prices: pd.Series, horizon: int) -> pd.Series:
    """1 if price is higher ``horizon`` days ahead, else 0 (NaN where unknown)."""
    fwd = prices.shift(-horizon) / prices - 1.0
    return (fwd > 0).astype(float).where(fwd.notna())


def walk_forward_predict(
    prices: pd.Series,
    *,
    horizon: int = 1,
    train_min: int = 750,
    step: int = 21,
    seed: int = 1234,
) -> pd.Series:
    """Expanding-window, leak-free P(up) predictions for each date.

    To predict the block starting at position ``i`` we train only on rows whose
    target was realised strictly before the block (positions ``< i - horizon``).
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    feats = make_features(prices)
    target = make_target(prices, horizon)
    df = pd.concat([feats, target.rename("y")], axis=1).dropna()
    if len(df) < train_min + step:
        return pd.Series(dtype=float)
    feat_cols = list(feats.columns)
    x_all = df[feat_cols].to_numpy(dtype=float)
    y_all = df["y"].to_numpy(dtype=int)

    preds: dict[pd.Timestamp, float] = {}
    i = train_min
    n = len(df)
    while i < n:
        train_end = i - horizon  # avoid overlap leakage
        if train_end < train_min // 2:
            i += step
            continue
        model = HistGradientBoostingClassifier(
            max_depth=3, max_iter=150, learning_rate=0.05, random_state=seed
        )
        model.fit(x_all[:train_end], y_all[:train_end])
        block = slice(i, min(i + step, n))
        prob = model.predict_proba(x_all[block])[:, 1]
        for ts, pr in zip(df.index[block], prob, strict=False):
            preds[ts] = float(pr)
        i += step
    return pd.Series(preds, name="p_up")


def predictions_to_target(
    pred_prob: pd.Series, calendar: pd.DatetimeIndex, *, threshold: float
) -> pd.Series:
    """Long (1) when P(up) > threshold, else flat (0); executed next day (shift 1)."""
    raw = (pred_prob > threshold).astype(float)
    return raw.reindex(calendar).shift(1).fillna(0.0)


def evaluate_ml(prices: pd.Series, settings: Settings) -> tuple[Verdict, dict[str, float]]:
    ppy = settings.trading_days_per_year
    cal = pd.DatetimeIndex(prices.index)
    pred = walk_forward_predict(
        prices,
        horizon=settings.ml_horizon,
        train_min=settings.ml_train_min,
        step=settings.ml_step,
        seed=settings.random_seed,
    )
    if pred.empty:
        raise ValueError("not enough history to train + walk-forward; need a longer series")

    target = predictions_to_target(pred, cal, threshold=settings.ml_threshold)
    sig_cfg = SignalBacktestConfig(
        cost_model=CostModel(0.0, settings.gold_slippage_bps),
        risk_free_annual=settings.risk_free_annual,
        trading_days_per_year=ppy,
    )
    result = run_signal_backtest(prices, prices, target, sig_cfg)
    bc = BacktestConfig(risk_free_annual=settings.risk_free_annual, trading_days_per_year=ppy)
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

    # Out-of-sample directional accuracy (a coin flip is 0.50).
    actual = make_target(prices, settings.ml_horizon).reindex(pred.index)
    pred_label = (pred > settings.ml_threshold).astype(float)
    valid = actual.notna()
    accuracy = float((pred_label[valid] == actual[valid]).mean()) if valid.any() else float("nan")

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
    g3 = _gate_benchmark_beating(headline)
    g4 = deflated_sharpe_gate(
        result.excess_returns,
        n_strategies_tried=_ML_IMPLICIT_TRIALS,
        trials_sr_std=settings.trial_sharpe_dispersion,
        threshold=settings.deflated_sharpe_threshold,
        periods_per_year=ppy,
    )

    verdict = Verdict(all(g.passed for g in (g1, g2, g3, g4)), [g1, g2, g3, g4], headline)
    diagnostics = {
        "oos_accuracy": accuracy,
        "n_predictions": float(len(pred)),
        "time_in_market": time_in_market(target),
        "n_trades": float(len(result.rebalances)),
    }
    return verdict, diagnostics


def load_prices_for_ml(settings: Settings, *, refresh: bool = False) -> pd.Series:
    from .data.prices import load_prices
    from .store import Store

    store = Store(settings.cache_dir)
    panel = load_prices(
        [settings.ml_instrument],
        start=settings.start_date,
        end=settings.end_date,
        source=settings.price_source,
        benchmark=settings.ml_instrument,
        store=store,
        refresh=refresh,
        seed=settings.random_seed,
    )
    return panel[settings.ml_instrument].dropna()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="whale-ml",
        description="Test whether an ML model can predict trades from charts (honest, gated).",
    )
    parser.add_argument("--demo", action="store_true", help="Offline synthetic prices.")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch prices.")
    parser.add_argument("--instrument", help="Ticker to model (default from config).")
    parser.add_argument("--horizon", type=int, help="Prediction horizon in days.")
    args = parser.parse_args(argv)

    overrides: dict[str, object] = {}
    if args.demo:
        overrides["price_source"] = "demo"
    if args.instrument:
        overrides["ml_instrument"] = args.instrument
    if args.horizon:
        overrides["ml_horizon"] = args.horizon
    settings = load_settings(**overrides)

    if settings.price_source == "demo":
        print(
            "[note] DEMO synthetic prices (a random walk) — expect NO edge; that is the point.\n",
            file=sys.stderr,
        )

    try:
        prices = load_prices_for_ml(settings, refresh=args.refresh)
        verdict, diag = evaluate_ml(prices, settings)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if settings.price_source != "demo":
            print("\nIf data hosts are unreachable here, try: whale-ml --demo", file=sys.stderr)
        return 2

    print(verdict.render())
    print(
        f"Diagnostics: out-of-sample directional accuracy {diag['oos_accuracy']:.1%} "
        f"(coin flip = 50.0%) | predictions {int(diag['n_predictions'])} | "
        f"time-in-market {diag['time_in_market']:.0%} | trades {int(diag['n_trades'])} | "
        f"{settings.ml_instrument}, horizon {settings.ml_horizon}d"
    )
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
