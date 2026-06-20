"""The information product: a curated weekly insider cluster-buy digest.

The research fan-out's verdict on "making money" was blunt: the durable business
in this space is the *information product*, not betting your own capital. The raw
SEC data is free (OpenInsider et al. dump it unanalysed); what nobody sells is a
**curated, honest** read — "here are this week's cluster buys that matter, why,
and the caveats." That editorial + credibility layer is the only defensible moat
when the data is a commodity.

This renders that digest from the same Form-4 pipeline the backtester uses, in
three formats: Markdown (Substack/email), self-contained HTML (web), and a
ready-to-post X/Twitter thread (the distribution flywheel). Pure render
functions; only the CLI touches IO. Nothing is posted anywhere — it produces
content for a human to review and publish.

Honest framing baked in: cluster buys do have real academic support (Lakonishok
& Lee; Cohen, Malloy & Pomorski on *opportunistic* buys; cluster buys ~2x
single-buy excess), but the edge has decayed and lives in small caps — so every
artifact carries the caveat and a not-advice disclaimer.
"""

from __future__ import annotations

import argparse
import html
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings, load_settings
from .insiders import cluster_events, load_insider_data

DIGEST_COLUMNS = [
    "ticker",
    "event_date",
    "n_buyers",
    "n_officers",
    "total_value",
    "titles",
    "conviction",
]

_DISCLAIMER = (
    "Informational only, compiled from public SEC Form 4 filings. Not investment "
    "advice and not a recommendation. The author may hold positions in names "
    "mentioned. Insider-cluster signals have decayed and carry real risk — do "
    "your own research."
)

_NOTE = (
    "Why cluster buys: multiple insiders (especially officers) buying on the open "
    "market in a short window has historically preceded outperformance — cluster "
    "buys run roughly twice the excess of single-insider buys. The edge has "
    "weakened out of sample and is strongest in small caps."
)


def summarize_clusters(
    buys: pd.DataFrame, events: pd.DataFrame, *, window_days: int
) -> pd.DataFrame:
    """Enrich each (ticker, event_date) with buyer/officer counts, value, titles (pure)."""
    if events.empty:
        return pd.DataFrame(columns=DIGEST_COLUMNS)
    win = pd.Timedelta(days=window_days)
    rows: list[dict[str, object]] = []
    for _, ev in events.iterrows():
        t = str(ev["ticker"])
        d = pd.Timestamp(ev["event_date"])
        w = buys[
            (buys["ticker"] == t) & (buys["filing_date"] > d - win) & (buys["filing_date"] <= d)
        ]
        officers = w[w["is_officer"]]
        titles = sorted({str(x).strip() for x in officers["officer_title"] if str(x).strip()})
        total_value = float(w["value"].sum())
        rows.append(
            {
                "ticker": t,
                "event_date": d,
                "n_buyers": int(w["owner"].nunique()),
                "n_officers": int(officers["owner"].nunique()),
                "total_value": total_value,
                "titles": ", ".join(titles[:3]),
            }
        )
    df = pd.DataFrame(rows)
    # Conviction: more distinct buyers and officers, larger dollars = stronger.
    df["conviction"] = (
        df["n_buyers"] + 2.0 * df["n_officers"] + np.log10(df["total_value"].clip(lower=1.0))
    )
    return (
        df.loc[:, DIGEST_COLUMNS].sort_values("conviction", ascending=False).reset_index(drop=True)
    )


def recent_clusters(
    buys: pd.DataFrame, settings: Settings, *, asof: date, weeks: int, top_n: int
) -> pd.DataFrame:
    """Top cluster buys whose event fired within the last ``weeks`` of ``asof``."""
    events = cluster_events(
        buys,
        min_buyers=settings.insider_min_buyers,
        require_officer=settings.insider_require_officer,
        min_value=settings.insider_min_value,
        window_days=settings.insider_window_days,
    )
    summary = summarize_clusters(buys, events, window_days=settings.insider_window_days)
    if summary.empty:
        return summary
    # Cap the lookback so a "show everything" request can't overflow Timedelta.
    cutoff = pd.Timestamp(asof) - pd.Timedelta(weeks=min(weeks, 5000))
    recent = summary[summary["event_date"] >= cutoff]
    return recent.head(top_n).reset_index(drop=True)


def _money(v: float) -> str:
    if v >= 1e6:
        return f"${v / 1e6:.1f}M"
    if v >= 1e3:
        return f"${v / 1e3:.0f}k"
    return f"${v:.0f}"


def build_digest_md(clusters: pd.DataFrame, *, asof: date, weeks: int) -> str:
    """Markdown digest (for Substack / email)."""
    lines = [
        f"# Insider Cluster-Buy Digest — week of {asof:%Y-%m-%d}",
        "",
        f"_{_NOTE}_",
        "",
    ]
    if clusters.empty:
        lines.append(f"No qualifying insider cluster buys in the last {weeks} week(s).")
    else:
        lines.append("| Ticker | Buyers | Officers | ~Value | Roles | Filed |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in clusters.iterrows():
            lines.append(
                f"| **{r['ticker']}** | {int(r['n_buyers'])} | {int(r['n_officers'])} "
                f"| {_money(float(r['total_value']))} | {html.escape(str(r['titles']) or '—')} "
                f"| {pd.Timestamp(r['event_date']):%Y-%m-%d} |"
            )
    lines += ["", "---", "", f"_{_DISCLAIMER}_", "", "Built with whale-clone."]
    return "\n".join(lines)


_CSS = """
:root{--bg:#0b0d12;--card:#151921;--ink:#e8edf4;--muted:#93a1b5;--line:#222a36;
--accent:#7aa2f7}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:820px;margin:0 auto;padding:28px 18px 80px}
h1{font-size:25px;margin:0 0 6px}
.sub{color:var(--muted);font-size:14px;margin-bottom:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;
padding:6px 18px;margin:14px 0}
table{width:100%;border-collapse:collapse;font-size:15px}
th,td{text-align:left;padding:8px;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:600;font-size:13px;text-transform:uppercase;
letter-spacing:.03em}
td.n{text-align:right;font-variant-numeric:tabular-nums}
.tk{font-weight:700;color:var(--accent)}
.foot{color:var(--muted);font-size:13px;margin-top:30px;
border-top:1px solid var(--line);padding-top:14px}
"""


def build_digest_html(clusters: pd.DataFrame, *, asof: date, weeks: int) -> str:
    """Self-contained HTML digest (no external assets)."""
    if clusters.empty:
        body = (
            f'<div class="card"><p>No qualifying insider cluster buys in the last '
            f"{weeks} week(s).</p></div>"
        )
    else:
        rows = "".join(
            f'<tr><td class="tk">{html.escape(str(r["ticker"]))}</td>'
            f'<td class="n">{int(r["n_buyers"])}</td>'
            f'<td class="n">{int(r["n_officers"])}</td>'
            f'<td class="n">{_money(float(r["total_value"]))}</td>'
            f"<td>{html.escape(str(r['titles']) or '—')}</td>"
            f"<td>{pd.Timestamp(r['event_date']):%Y-%m-%d}</td></tr>"
            for _, r in clusters.iterrows()
        )
        body = (
            '<div class="card"><table><thead><tr><th>Ticker</th><th>Buyers</th>'
            "<th>Officers</th><th>~Value</th><th>Roles</th><th>Filed</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
        )
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Insider Cluster-Buy Digest</title><style>{_CSS}</style></head>
<body><div class="wrap">
<h1>Insider Cluster-Buy Digest</h1>
<div class="sub">Week of {asof:%Y-%m-%d}. {html.escape(_NOTE)}</div>
{body}
<div class="foot">Generated {generated}. {html.escape(_DISCLAIMER)} Built with whale-clone.</div>
</div></body></html>"""


def build_x_thread(clusters: pd.DataFrame, *, asof: date, max_items: int = 5) -> list[str]:
    """A ready-to-post X/Twitter thread (list of <=280-char posts). Posts nothing."""
    header = (
        f"🐳 Insider cluster-buy digest — week of {asof:%Y-%m-%d}\n\n"
        "Where multiple insiders (incl. officers) bought their own stock on the "
        "open market this week. A thread 🧵"
    )
    posts = [header[:280]]
    if clusters.empty:
        posts.append("No qualifying insider cluster buys this week.")
    else:
        for i, (_, r) in enumerate(clusters.head(max_items).iterrows(), start=1):
            roles = str(r["titles"]) or "insiders"
            posts.append(
                (
                    f"{i}/ ${r['ticker']}: {int(r['n_buyers'])} insiders "
                    f"({int(r['n_officers'])} officers — {roles}) bought ~"
                    f"{_money(float(r['total_value']))} over the cluster window. "
                    f"Filed {pd.Timestamp(r['event_date']):%b %d}."
                )[:280]
            )
    posts.append(
        (
            "Not advice; from public SEC Form 4 filings. Cluster buys have real "
            "academic support but the edge has decayed and is strongest in small "
            "caps. DYOR."
        )[:280]
    )
    return posts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="whale-digest",
        description="Generate a curated insider cluster-buy digest (Markdown + HTML + X thread).",
    )
    parser.add_argument("--demo", action="store_true", help="Offline synthetic insider data.")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch data.")
    parser.add_argument("--out", default="digest", help="Output directory (default: digest/).")
    parser.add_argument("--weeks", type=int, default=1, help="Lookback window in weeks.")
    parser.add_argument("--top", type=int, default=10, help="Max cluster buys to feature.")
    args = parser.parse_args(argv)

    overrides: dict[str, object] = {}
    if args.demo:
        overrides["insider_source"] = "demo"
        overrides["price_source"] = "demo"
    settings = load_settings(**overrides)
    if settings.insider_source == "demo":
        print("[note] DEMO synthetic insider data — NOT a market claim.\n", file=sys.stderr)

    try:
        buys, _prices = load_insider_data(settings, refresh=args.refresh)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if settings.insider_source != "demo":
            print("\nIf data hosts are unreachable here, try: whale-digest --demo", file=sys.stderr)
        return 2

    asof = settings.end_date
    # In demo the synthetic events cluster anywhere in the sample, so widen the
    # window to the whole sample to render a representative digest.
    weeks = args.weeks if settings.insider_source != "demo" else 100_000
    clusters = recent_clusters(buys, settings, asof=asof, weeks=weeks, top_n=args.top)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "digest.md").write_text(build_digest_md(clusters, asof=asof, weeks=args.weeks))
    (out / "index.html").write_text(build_digest_html(clusters, asof=asof, weeks=args.weeks))
    (out / "thread.txt").write_text("\n\n---\n\n".join(build_x_thread(clusters, asof=asof)))
    print(
        f"Wrote {out / 'digest.md'}, {out / 'index.html'}, {out / 'thread.txt'} "
        f"({len(clusters)} cluster buys featured)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
