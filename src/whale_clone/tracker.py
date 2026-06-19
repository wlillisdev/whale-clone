"""Superinvestor holdings tracker — a useful by-product, not a market bet.

Reuses the verified EDGAR 13F loader to answer concrete, factual questions:
what does each manager hold now, what did they change since last quarter, and
which names are held across multiple managers (consensus). This is a reporting
tool — its output is sourced fact, with no edge claim and no overfitting risk.

The analysis functions are pure (operate on a validated holdings frame); the CLI
is the only IO.
"""

from __future__ import annotations

import argparse
import sys
from typing import cast

import pandas as pd

from .config import Settings, load_settings
from .data.holdings import load_holdings
from .store import Store


def _latest_two_periods(manager_rows: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp | None]:
    dates = sorted(pd.to_datetime(manager_rows["filing_date"]).unique())
    latest = dates[-1]
    prev = dates[-2] if len(dates) >= 2 else None
    return latest, prev


def latest_holdings(holdings: pd.DataFrame, manager: str) -> pd.DataFrame:
    """Current holdings for one manager: ticker, value, weight %, rank (pure)."""
    rows = holdings[holdings["manager"] == manager]
    if rows.empty:
        return rows.assign(weight=[], rank=[])
    latest, _ = _latest_two_periods(rows)
    cur = cast(
        pd.DataFrame,
        rows[rows["filing_date"] == latest].groupby("ticker", as_index=False)["value"].sum(),
    )
    total = cur["value"].sum()
    cur["weight"] = cur["value"] / total if total > 0 else 0.0
    cur = cur.sort_values("value", ascending=False).reset_index(drop=True)
    cur["rank"] = cur.index + 1
    return cur


def changes(holdings: pd.DataFrame, manager: str) -> pd.DataFrame:
    """Position changes between a manager's two latest filings (pure).

    Returns ticker, prev_value, cur_value, and an action label:
    NEW (initiated), EXIT (sold out), ADD (increased), TRIM (reduced), HOLD.
    """
    rows = holdings[holdings["manager"] == manager]
    if rows.empty:
        return pd.DataFrame(columns=["ticker", "prev_value", "cur_value", "action"])
    latest, prev = _latest_two_periods(rows)
    cur = rows[rows["filing_date"] == latest].groupby("ticker")["value"].sum()
    if prev is None:
        out = cur.reset_index().rename(columns={"value": "cur_value"})
        out["prev_value"] = 0.0
        out["action"] = "NEW"
        return out.loc[:, ["ticker", "prev_value", "cur_value", "action"]]
    pre = rows[rows["filing_date"] == prev].groupby("ticker")["value"].sum()
    merged = pd.concat([pre.rename("prev_value"), cur.rename("cur_value")], axis=1).fillna(0.0)
    merged = merged.reset_index().rename(columns={"index": "ticker"})

    def label(r: pd.Series) -> str:
        p, c = r["prev_value"], r["cur_value"]
        if p == 0 and c > 0:
            return "NEW"
        if c == 0 and p > 0:
            return "EXIT"
        if c > p * 1.05:
            return "ADD"
        if c < p * 0.95:
            return "TRIM"
        return "HOLD"

    merged["action"] = merged.apply(label, axis=1)
    order = {"NEW": 0, "ADD": 1, "TRIM": 2, "EXIT": 3, "HOLD": 4}
    merged["_o"] = merged["action"].map(order)
    return (
        merged.sort_values(["_o", "cur_value"], ascending=[True, False])
        .drop(columns="_o")
        .reset_index(drop=True)
    )


def consensus(holdings: pd.DataFrame) -> pd.DataFrame:
    """Names held by multiple managers in their latest filings (pure)."""
    frames = []
    for m in holdings["manager"].unique():
        cur = latest_holdings(holdings, m)[["ticker"]].copy()
        cur["manager"] = m
        frames.append(cur)
    if not frames:
        return pd.DataFrame(columns=["ticker", "n_managers", "managers"])
    allcur = pd.concat(frames, ignore_index=True)

    def _join(s: pd.Series) -> str:
        return ", ".join(sorted(set(s)))

    g = allcur.groupby("ticker")["manager"]
    grp = pd.DataFrame({"n_managers": g.nunique(), "managers": g.agg(_join)})
    return grp[grp["n_managers"] >= 2].sort_values("n_managers", ascending=False).reset_index()


def build_report(holdings: pd.DataFrame, *, top_n: int = 10) -> str:
    """Human-readable markdown report of current holdings, changes, consensus."""
    lines = ["# Superinvestor holdings tracker", ""]
    asof = pd.to_datetime(holdings["filing_date"]).max()
    lines.append(f"_Latest filing in data: {asof:%Y-%m-%d}. Source: SEC EDGAR 13F._")
    lines.append("")
    for m in sorted(holdings["manager"].unique()):
        cur = latest_holdings(holdings, m)
        chg = changes(holdings, m)
        lines.append(f"## {m}")
        latest, _ = _latest_two_periods(holdings[holdings["manager"] == m])
        lines.append(f"As of filing {latest:%Y-%m-%d} — {len(cur)} positions.")
        lines.append("")
        lines.append(f"**Top {min(top_n, len(cur))} holdings:**")
        for _, r in cur.head(top_n).iterrows():
            lines.append(f"- {r['ticker']}: {r['weight']:.1%}")
        moves = chg[chg["action"].isin(["NEW", "EXIT", "ADD", "TRIM"])]
        if not moves.empty:
            lines.append("")
            lines.append("**Recent changes vs prior filing:**")
            for _, r in moves.head(top_n).iterrows():
                lines.append(f"- {r['action']}: {r['ticker']}")
        lines.append("")
    con = consensus(holdings)
    lines.append("## Consensus (held by 2+ managers)")
    if con.empty:
        lines.append("_None._")
    else:
        for _, r in con.iterrows():
            lines.append(f"- {r['ticker']}: {int(r['n_managers'])} managers ({r['managers']})")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="whale-track",
        description="Report what the tracked 13F managers hold now and recently changed.",
    )
    parser.add_argument("--demo", action="store_true", help="Offline synthetic data (no network).")
    parser.add_argument("--refresh", action="store_true", help="Ignore cache and re-fetch.")
    parser.add_argument("--top", type=int, default=10, help="Top N holdings to show per manager.")
    parser.add_argument("--csv", help="Also write current holdings to this CSV path.")
    args = parser.parse_args(argv)

    overrides: dict[str, object] = {}
    if args.demo:
        overrides["holdings_source"] = "demo"
        overrides["price_source"] = "demo"
    settings: Settings = load_settings(**overrides)

    if settings.holdings_source == "demo":
        print("[note] DEMO synthetic holdings — not real filings.\n", file=sys.stderr)

    store = Store(settings.cache_dir)
    try:
        holdings = load_holdings(
            settings.managers,
            source=settings.holdings_source,
            start=settings.start_date,
            end=settings.end_date,
            store=store,
            refresh=args.refresh,
            seed=settings.random_seed,
            openfigi_key=settings.openfigi_api_key,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(build_report(holdings, top_n=args.top))

    if args.csv:
        frames = [
            latest_holdings(holdings, m).assign(manager=m) for m in holdings["manager"].unique()
        ]
        out = pd.concat(frames, ignore_index=True)
        path = store.export_csv(out.set_index("manager"), args.csv)
        print(f"\n[wrote current holdings to {path}]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
