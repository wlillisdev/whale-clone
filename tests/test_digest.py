"""Digest product tests: cluster summary, render formats, disclaimer presence."""

from __future__ import annotations

from datetime import date

import pandas as pd

from whale_clone.config import load_settings
from whale_clone.digest import (
    build_digest_html,
    build_digest_md,
    build_x_thread,
    recent_clusters,
    summarize_clusters,
)


def _buys():
    rows = [
        ("X", "2024-03-02", "a", True, "CEO", 5e5),
        ("X", "2024-03-03", "b", True, "CFO", 4e5),
        ("X", "2024-03-04", "c", False, "", 3e5),  # 3 buyers, 2 officers -> strong
        ("Y", "2024-03-02", "d", False, "", 1e5),
        ("Y", "2024-03-03", "e", False, "", 1e5),
        ("Y", "2024-03-04", "f", True, "Director", 1e5),  # 3 buyers, 1 officer
    ]
    df = pd.DataFrame(
        rows, columns=["ticker", "filing_date", "owner", "is_officer", "officer_title", "value"]
    )
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df["shares"] = 1.0
    df["price"] = df["value"]
    return df


def test_summarize_clusters_counts_and_conviction():
    from whale_clone.insiders import cluster_events

    buys = _buys()
    events = cluster_events(buys, min_buyers=3, require_officer=True, min_value=0, window_days=30)
    summary = summarize_clusters(buys, events, window_days=30)
    assert set(summary["ticker"]) == {"X", "Y"}
    # X has 2 officers and more dollars -> higher conviction, sorted first.
    assert summary.iloc[0]["ticker"] == "X"
    x = summary[summary["ticker"] == "X"].iloc[0]
    assert x["n_buyers"] == 3
    assert x["n_officers"] == 2


def test_recent_clusters_respects_lookback():
    buys = _buys()
    s = load_settings(insider_min_buyers=3, insider_min_value=0, insider_window_days=30)
    # asof far after the events with a 1-week window -> nothing recent.
    empty = recent_clusters(buys, s, asof=date(2024, 6, 1), weeks=1, top_n=10)
    assert empty.empty
    # A very wide window catches them.
    got = recent_clusters(buys, s, asof=date(2024, 6, 1), weeks=100, top_n=10)
    assert not got.empty


def test_render_formats_carry_disclaimer():
    buys = _buys()
    s = load_settings(insider_min_buyers=3, insider_min_value=0, insider_window_days=30)
    clusters = recent_clusters(buys, s, asof=date(2024, 6, 1), weeks=100, top_n=10)
    md = build_digest_md(clusters, asof=date(2024, 6, 1), weeks=100)
    html_doc = build_digest_html(clusters, asof=date(2024, 6, 1), weeks=100)
    thread = build_x_thread(clusters, asof=date(2024, 6, 1))

    assert "Not investment advice" in md or "Not investment" in md
    assert "X" in md and "Y" in md
    assert "<table" in html_doc and "&" not in html_doc.replace("&amp;", "")  # escaped
    assert isinstance(thread, list) and len(thread) >= 2
    assert all(len(post) <= 280 for post in thread)  # tweet length bound
    assert "DYOR" in thread[-1]


def test_digest_handles_no_clusters():
    empty = summarize_clusters(pd.DataFrame(), pd.DataFrame(), window_days=30)
    md = build_digest_md(empty, asof=date(2024, 6, 1), weeks=1)
    assert "No qualifying" in md
    thread = build_x_thread(empty, asof=date(2024, 6, 1))
    assert all(len(post) <= 280 for post in thread)
