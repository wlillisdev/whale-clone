"""CLI: turn the quarterly clone signal into (paper) orders. Dry-run by default.

Safety model — trading requires ALL of:
  * ``--paper`` (use the Alpaca paper broker instead of the offline dry-run one),
  * ``--execute`` (explicit opt-in to actually send orders), and
  * the market being open, no kill-switch, and no guardrail violation.

With no flags it previews the orders and sends nothing. There is no live broker.
"""

from __future__ import annotations

import argparse
import sys
from typing import cast

from .config import Settings, load_settings
from .execution import DryRunBroker, ExecConfig, ExecutionClient, rebalance_to_target
from .paper import run_signal


def _exec_config(settings: Settings) -> ExecConfig:
    return ExecConfig(
        cash_buffer=settings.exec_cash_buffer,
        no_trade_band=settings.exec_no_trade_band,
        max_orders_per_run=settings.exec_max_orders,
        max_order_notional_pct=settings.exec_max_order_notional_pct,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="whale-trade",
        description="Execute the clone signal as PAPER orders. Dry-run unless --paper --execute.",
    )
    parser.add_argument("--demo", action="store_true", help="Use offline synthetic holdings.")
    parser.add_argument("--paper", action="store_true", help="Use the Alpaca PAPER broker.")
    parser.add_argument("--execute", action="store_true", help="Actually send paper orders.")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch holdings.")
    args = parser.parse_args(argv)

    overrides: dict[str, object] = {}
    if args.demo:
        overrides["holdings_source"] = "demo"
        overrides["price_source"] = "demo"
    if args.paper:
        overrides["broker_mode"] = "paper"
    if args.execute:
        overrides["execute"] = True
    settings: Settings = load_settings(**overrides)

    # 1) Get the target portfolio from the (unchanged) signal engine.
    try:
        _, entry = run_signal(settings, refresh=args.refresh)
    except Exception as exc:
        print(f"ERROR loading signal: {exc}", file=sys.stderr)
        return 2
    raw_target = cast("dict[str, float]", entry["target"])
    target = {str(k): float(v) for k, v in raw_target.items()}
    run_id = str(entry.get("as_of", "")).replace("-", "")

    # 2) Pick a broker. Dry-run (offline) unless explicitly paper.
    broker: ExecutionClient
    if settings.broker_mode == "paper":
        try:
            from .broker_alpaca import AlpacaPaperBroker

            broker = AlpacaPaperBroker(execute=settings.execute)
        except Exception as exc:
            print(f"ERROR connecting to Alpaca paper: {exc}", file=sys.stderr)
            return 2
        submit = settings.execute
    else:
        broker = DryRunBroker()
        submit = False
        print("[note] dry-run (offline) — no broker connection, no orders.\n", file=sys.stderr)

    # 3) Plan + (maybe) submit, fully guarded.
    report = rebalance_to_target(
        broker,
        target,
        _exec_config(settings),
        run_id=run_id,
        submit=submit,
        cache_dir=settings.cache_dir,
    )
    print(report.render())
    if report.violations:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
