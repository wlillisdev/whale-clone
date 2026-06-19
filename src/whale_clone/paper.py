"""Automated quarterly signal + forward paper-trade log.

This is the honest bridge from "a backtested near-miss" to "evidence we can act
on": it computes the current target portfolio for the concentrated 13F clone,
diffs it against the last recorded target to produce a concrete **trade ticket**,
and appends a timestamped entry to a **forward paper-trade log**. Run on a
schedule (see .github/workflows/quarterly-signal.yml), it accumulates genuine
out-of-sample evidence over time — risking zero capital.

No orders are placed. The output is a decision ("here are this quarter's trades")
plus a growing track record. Execution (manual, or a broker paper API later)
stays a separate, deliberate step.

The trade/target math is pure and unit-tested; only data loading and the log
file touch IO.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

import pandas as pd

from .config import Settings, load_settings
from .data.holdings import load_holdings
from .portfolio import holdings_known_on, target_weights
from .store import Store

_LOG_NAME = "paper_log"


@dataclass(frozen=True)
class Trade:
    ticker: str
    action: str  # BUY | SELL | EXIT | NEW
    prev_weight: float
    new_weight: float

    @property
    def delta(self) -> float:
        return self.new_weight - self.prev_weight


def current_target(
    holdings: pd.DataFrame, settings: Settings, *, as_of: pd.Timestamp
) -> dict[str, float]:
    """Target weights for the pre-committed clone, as of ``as_of`` (pure)."""
    visible = holdings_known_on(holdings, as_of)
    return target_weights(
        visible,
        weighting=settings.weighting,
        max_position_weight=settings.max_position_weight,
        top_n=settings.top_n_positions,
    )


def compute_trades(
    prev: dict[str, float], new: dict[str, float], *, threshold: float = 0.005
) -> list[Trade]:
    """Trade ticket to move from ``prev`` weights to ``new`` weights (pure).

    Ignores weight changes smaller than ``threshold`` (no churn on noise).
    """
    trades: list[Trade] = []
    for ticker in sorted(set(prev) | set(new)):
        p = prev.get(ticker, 0.0)
        n = new.get(ticker, 0.0)
        if abs(n - p) < threshold:
            continue
        if p == 0.0 and n > 0.0:
            action = "NEW"
        elif n == 0.0 and p > 0.0:
            action = "EXIT"
        elif n > p:
            action = "BUY"
        else:
            action = "SELL"
        trades.append(Trade(ticker, action, p, n))
    # Largest moves first.
    return sorted(trades, key=lambda t: abs(t.delta), reverse=True)


def _last_target(log: list[dict[str, object]]) -> dict[str, float]:
    if not log:
        return {}
    last = cast("dict[str, float]", log[-1].get("target", {}))
    return {str(k): float(v) for k, v in last.items()}


def render(as_of: pd.Timestamp, target: dict[str, float], trades: list[Trade]) -> str:
    lines = ["=" * 60, "WHALE-CLONE — QUARTERLY SIGNAL (paper / decision only)", "=" * 60]
    lines.append(f"As of latest filing: {as_of:%Y-%m-%d}. NO orders placed.")
    lines.append("-" * 60)
    lines.append("Target portfolio:")
    for t, w in sorted(target.items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"  {t:<8} {w:6.1%}")
    lines.append("-" * 60)
    if not trades:
        lines.append("Trades this run: none (within no-trade band).")
    else:
        lines.append("Trades this run:")
        for tr in trades:
            lines.append(
                f"  {tr.action:<4} {tr.ticker:<8} {tr.prev_weight:5.1%} -> {tr.new_weight:5.1%}"
            )
    lines.append("=" * 60)
    return "\n".join(lines)


def run_signal(settings: Settings, *, refresh: bool = False) -> tuple[str, dict[str, object]]:
    """Compute the current target + trades, append to the paper log, return both."""
    store = Store(settings.cache_dir)
    holdings = load_holdings(
        settings.managers,
        source=settings.holdings_source,
        start=settings.start_date,
        end=settings.end_date,
        store=store,
        refresh=refresh,
        seed=settings.random_seed,
        openfigi_key=settings.openfigi_api_key,
    )
    as_of = pd.to_datetime(holdings["filing_date"]).max()
    target = current_target(holdings, settings, as_of=as_of)

    log: list[dict[str, object]] = store.load_json(_LOG_NAME, []) or []
    prev = _last_target(log)
    trades = compute_trades(prev, target)

    entry: dict[str, object] = {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "as_of": f"{as_of:%Y-%m-%d}",
        "target": {k: round(v, 4) for k, v in target.items()},
        "trades": [
            {"ticker": t.ticker, "action": t.action, "delta": round(t.delta, 4)} for t in trades
        ],
    }
    # Only append a new entry when the target actually changed (new filing or
    # rebalance), so the log stays a clean quarter-by-quarter record.
    if not log or log[-1].get("target") != entry["target"]:
        log.append(entry)
        store.save_json(_LOG_NAME, log)

    return render(as_of, target, trades), entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="whale-signal",
        description="Quarterly clone signal + forward paper-trade log (no orders placed).",
    )
    parser.add_argument("--demo", action="store_true", help="Offline synthetic data.")
    parser.add_argument("--refresh", action="store_true", help="Ignore cache and re-fetch.")
    args = parser.parse_args(argv)

    overrides: dict[str, object] = {}
    if args.demo:
        overrides["holdings_source"] = "demo"
        overrides["price_source"] = "demo"
    settings: Settings = load_settings(**overrides)
    if settings.holdings_source == "demo":
        print("[note] DEMO synthetic holdings — not a real signal.\n", file=sys.stderr)

    try:
        report, _ = run_signal(settings, refresh=args.refresh)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
