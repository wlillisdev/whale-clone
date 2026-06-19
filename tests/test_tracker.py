"""Tracker analysis tests (pure functions on a holdings frame)."""

from __future__ import annotations

import pandas as pd

from whale_clone.portfolio import validate_holdings
from whale_clone.tracker import build_report, changes, consensus, latest_holdings


def test_latest_holdings_weights_and_rank(simple_holdings):
    h = validate_holdings(simple_holdings)
    cur = latest_holdings(h, "M1")  # latest filing 2020-08-14: AAA 150, BBB 50
    assert list(cur["ticker"]) == ["AAA", "BBB"]
    assert cur.iloc[0]["weight"] == 0.75
    assert cur.iloc[0]["rank"] == 1
    assert abs(cur["weight"].sum() - 1.0) < 1e-9


def test_changes_labels_add_trim_exit(simple_holdings):
    h = validate_holdings(simple_holdings)
    m1 = changes(h, "M1").set_index("ticker")["action"].to_dict()
    assert m1["AAA"] == "ADD"  # 100 -> 150
    assert m1["BBB"] == "TRIM"  # 100 -> 50
    m2 = changes(h, "M2").set_index("ticker")["action"].to_dict()
    assert m2["BBB"] == "EXIT"  # 100 -> 0
    assert m2["CCC"] == "ADD"  # 300 -> 400


def test_changes_all_new_when_only_one_filing():
    rows = [("M1", "2020-Q1", "2020-05-15", "AAA", 100.0)]
    h = validate_holdings(
        pd.DataFrame(rows, columns=["manager", "period", "filing_date", "ticker", "value"])
    )
    chg = changes(h, "M1")
    assert (chg["action"] == "NEW").all()


def test_consensus_detects_shared_names():
    rows = [
        ("M1", "2020-Q1", "2020-05-15", "AAA", 100.0),
        ("M1", "2020-Q1", "2020-05-15", "SHARED", 100.0),
        ("M2", "2020-Q1", "2020-05-15", "SHARED", 100.0),
        ("M2", "2020-Q1", "2020-05-15", "BBB", 100.0),
    ]
    h = validate_holdings(
        pd.DataFrame(rows, columns=["manager", "period", "filing_date", "ticker", "value"])
    )
    con = consensus(h)
    assert list(con["ticker"]) == ["SHARED"]
    assert int(con.iloc[0]["n_managers"]) == 2


def test_build_report_has_sections(simple_holdings):
    h = validate_holdings(simple_holdings)
    report = build_report(h, top_n=5)
    assert "Superinvestor holdings tracker" in report
    assert "## M1" in report
    assert "Consensus" in report
