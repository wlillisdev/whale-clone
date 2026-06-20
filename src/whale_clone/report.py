"""Publishable report generator — the tracker as a product.

Turns the verified EDGAR holdings into a self-contained static HTML page (plus a
CSV) showing what each tracked manager holds now, what they changed last quarter,
and which names they hold in common. No edge claim — it is sourced fact (SEC
13F), the one model with a proven business in this space.

``build_html`` is pure (holdings frame -> HTML string); only the CLI touches IO.
"""

from __future__ import annotations

import argparse
import html
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .config import Settings, load_settings
from .data.holdings import load_holdings
from .store import Store
from .tracker import changes, consensus, latest_holdings

_ACTION_LABEL = {"NEW": "new", "ADD": "add", "TRIM": "trim", "EXIT": "exit"}

_CSS = """
:root{--bg:#0b0d12;--card:#151921;--ink:#e8edf4;--muted:#93a1b5;--line:#222a36;
--up:#36d399;--down:#f87272;--accent:#7aa2f7}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:920px;margin:0 auto;padding:28px 18px 80px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:20px;margin:28px 0 10px}
.sub{color:var(--muted);font-size:14px;margin-bottom:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;
padding:18px 20px;margin:14px 0}
table{width:100%;border-collapse:collapse;font-size:15px}
th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:600;font-size:13px;
text-transform:uppercase;letter-spacing:.03em}
td.w{text-align:right;font-variant-numeric:tabular-nums;color:var(--muted)}
.bar{height:6px;background:var(--accent);border-radius:4px;
display:inline-block;vertical-align:middle}
.badge{display:inline-block;font-size:12px;padding:2px 8px;border-radius:999px;
margin:2px 4px 2px 0;border:1px solid var(--line)}
.new{color:var(--up);border-color:var(--up)}.add{color:var(--up)}
.trim{color:var(--down)}.exit{color:var(--down);border-color:var(--down)}
.foot{color:var(--muted);font-size:13px;margin-top:36px;
border-top:1px solid var(--line);padding-top:14px}
.tag{display:inline-block;background:#1e2633;color:var(--accent);
border-radius:6px;padding:1px 7px;font-size:12px;margin-left:6px}
"""


def _holdings_table(cur: pd.DataFrame, top_n: int) -> str:
    rows = []
    for _, r in cur.head(top_n).iterrows():
        w = float(r["weight"])
        bar = f'<span class="bar" style="width:{max(2, round(w * 160))}px"></span>'
        rows.append(
            f"<tr><td>{html.escape(str(r['ticker']))}</td>"
            f'<td class="w">{w:.1%}</td><td>{bar}</td></tr>'
        )
    return (
        "<table><thead><tr><th>Ticker</th><th>Weight</th><th></th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _changes_badges(chg: pd.DataFrame, top_n: int) -> str:
    moves = chg[chg["action"].isin(_ACTION_LABEL)]
    if moves.empty:
        return '<div class="sub">No reported changes vs prior filing.</div>'
    badges = []
    for _, r in moves.head(top_n * 2).iterrows():
        cls = _ACTION_LABEL[str(r["action"])]
        badges.append(
            f'<span class="badge {cls}">{html.escape(str(r["action"]))} '
            f"{html.escape(str(r['ticker']))}</span>"
        )
    return "<div>" + "".join(badges) + "</div>"


def build_html(holdings: pd.DataFrame, *, top_n: int = 10) -> str:
    """Self-contained HTML report (no external assets) from a holdings frame."""
    as_of = pd.to_datetime(holdings["filing_date"]).max()
    managers = sorted(holdings["manager"].unique())

    sections = []
    for m in managers:
        cur = latest_holdings(holdings, m)
        chg = changes(holdings, m)
        sections.append(
            f'<div class="card"><h2>{html.escape(m)}'
            f'<span class="tag">{len(cur)} positions</span></h2>'
            f"{_holdings_table(cur, top_n)}"
            f'<div style="margin-top:12px">{_changes_badges(chg, top_n)}</div></div>'
        )

    con = consensus(holdings)
    if con.empty:
        consensus_html = '<div class="sub">No names held by 2+ managers.</div>'
    else:
        crows = "".join(
            f"<tr><td>{html.escape(str(r['ticker']))}</td>"
            f"<td class='w'>{int(r['n_managers'])}</td>"
            f"<td>{html.escape(str(r['managers']))}</td></tr>"
            for _, r in con.iterrows()
        )
        consensus_html = (
            "<table><thead><tr><th>Ticker</th><th>Held by</th><th>Managers</th></tr>"
            f"</thead><tbody>{crows}</tbody></table>"
        )

    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Superinvestor Holdings Tracker</title><style>{_CSS}</style></head>
<body><div class="wrap">
<h1>Superinvestor Holdings Tracker</h1>
<div class="sub">What a few low-turnover managers hold now, and what they changed
last quarter. Source: SEC EDGAR 13F. Latest filing in data: {as_of:%Y-%m-%d}.</div>
{"".join(sections)}
<h2>Consensus — held by 2+ managers</h2>
<div class="card">{consensus_html}</div>
<div class="foot">Generated {generated}. Informational only; sourced from public
SEC filings (~45-day disclosure lag). Not investment advice. Built with
whale-clone.</div>
</div></body></html>"""


def export_holdings_csv(holdings: pd.DataFrame) -> pd.DataFrame:
    """Flat current-holdings table for CSV export."""
    frames = []
    for m in sorted(holdings["manager"].unique()):
        cur = latest_holdings(holdings, m).assign(manager=m)
        frames.append(cur)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="whale-report",
        description="Generate a publishable HTML + CSV superinvestor holdings report.",
    )
    parser.add_argument("--demo", action="store_true", help="Offline synthetic holdings.")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch holdings.")
    parser.add_argument("--out", default="site", help="Output directory (default: site/).")
    parser.add_argument("--top", type=int, default=10, help="Top N holdings per manager.")
    args = parser.parse_args(argv)

    overrides: dict[str, object] = {}
    if args.demo:
        overrides["holdings_source"] = "demo"
        overrides["price_source"] = "demo"
    settings: Settings = load_settings(**overrides)
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

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(build_html(holdings, top_n=args.top))
    export_holdings_csv(holdings).to_csv(out / "holdings.csv", index=False)
    print(f"Wrote {out / 'index.html'} and {out / 'holdings.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
