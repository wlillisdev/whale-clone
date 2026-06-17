"""CLI entry point: ``python -m whale_clone`` (or ``whale-clone``).

Pulls data, runs the backtest, and prints the verdict against the four gates.
"""

from __future__ import annotations

import argparse
import sys

from .config import load_settings
from .pipeline import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="whale-clone",
        description="Backtest a low-turnover 13F clone vs the index, after costs.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use offline synthetic data (no network). NOT a market claim — a pipeline smoke test.",
    )
    parser.add_argument("--refresh", action="store_true", help="Ignore cache and re-fetch data.")
    parser.add_argument(
        "--price-source", choices=["stooq", "yahoo", "demo"], help="Override price source."
    )
    parser.add_argument(
        "--holdings-source", choices=["dataroma", "demo"], help="Override holdings source."
    )
    args = parser.parse_args(argv)

    overrides: dict[str, object] = {}
    if args.demo:
        overrides["price_source"] = "demo"
        overrides["holdings_source"] = "demo"
    if args.price_source:
        overrides["price_source"] = args.price_source
    if args.holdings_source:
        overrides["holdings_source"] = args.holdings_source

    settings = load_settings(**overrides)

    if settings.price_source == "demo" or settings.holdings_source == "demo":
        print(
            "[note] Running with DEMO (synthetic) data — this exercises the full "
            "pipeline and gates but is NOT a claim about any real strategy.\n",
            file=sys.stderr,
        )

    try:
        verdict = run(settings, refresh=args.refresh)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if settings.price_source != "demo":
            print(
                "\nIf data hosts are unreachable in this environment, try: whale-clone --demo",
                file=sys.stderr,
            )
        return 2

    print(verdict.render())
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
