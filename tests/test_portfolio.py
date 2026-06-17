from __future__ import annotations

import pandas as pd
import pytest

from whale_clone.portfolio import (
    holdings_known_on,
    target_weights,
    validate_holdings,
)


def test_validate_rejects_missing_columns():
    bad = pd.DataFrame({"manager": ["M1"], "ticker": ["AAA"]})
    try:
        validate_holdings(bad)
    except ValueError as e:
        assert "missing columns" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_validate_uppercases_and_drops_nonpositive(simple_holdings):
    h = simple_holdings.copy()
    h.loc[0, "ticker"] = "aaa"
    h.loc[1, "value"] = 0.0
    clean = validate_holdings(h)
    assert (clean["ticker"] == clean["ticker"].str.upper()).all()
    assert (clean["value"] > 0).all()


def test_holdings_known_on_uses_latest_visible_filing(simple_holdings):
    h = validate_holdings(simple_holdings)
    # Before any filing is public: empty.
    assert holdings_known_on(h, pd.Timestamp("2020-05-14")).empty
    # On Q1 filing date: only Q1 visible.
    q1 = holdings_known_on(h, pd.Timestamp("2020-05-15"))
    assert set(q1["period"]) == {"2020-Q1"}
    # Between Q1 and Q2 filings: still Q1.
    mid = holdings_known_on(h, pd.Timestamp("2020-07-01"))
    assert set(mid["period"]) == {"2020-Q1"}
    # On Q2 filing date: latest per manager is Q2.
    q2 = holdings_known_on(h, pd.Timestamp("2020-08-14"))
    assert set(q2["period"]) == {"2020-Q2"}


def test_target_weights_sum_to_one(simple_holdings):
    h = validate_holdings(simple_holdings)
    visible = holdings_known_on(h, pd.Timestamp("2020-05-15"))
    w = target_weights(visible, weighting="value", max_position_weight=1.0)
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_cap_binds_and_redistributes():
    # One manager, four names, one dominant -> 0.25 cap is feasible (4*0.25=1.0).
    rows = [
        ("M1", "2020-Q1", "2020-05-15", "AAA", 700.0),
        ("M1", "2020-Q1", "2020-05-15", "BBB", 100.0),
        ("M1", "2020-Q1", "2020-05-15", "CCC", 100.0),
        ("M1", "2020-Q1", "2020-05-15", "DDD", 100.0),
    ]
    h = validate_holdings(
        pd.DataFrame(rows, columns=["manager", "period", "filing_date", "ticker", "value"])
    )
    visible = holdings_known_on(h, pd.Timestamp("2020-05-15"))
    w = target_weights(visible, weighting="value", max_position_weight=0.25)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert max(w.values()) <= 0.25 + 1e-9
    # AAA was 70% pre-cap; it should be pinned at the cap.
    assert w["AAA"] == pytest.approx(0.25)


def test_equal_weighting_within_manager(simple_holdings):
    h = validate_holdings(simple_holdings)
    visible = holdings_known_on(h, pd.Timestamp("2020-05-15"))
    w = target_weights(visible, weighting="equal", max_position_weight=1.0)
    # M1 holds AAA,BBB equally (0.5 each /2 managers); M2 holds BBB,CCC equally.
    # BBB appears in both managers: 0.25 + 0.25 = 0.5 before normalisation -> stays 0.5.
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert w["BBB"] > w["AAA"]


def test_empty_holdings_gives_empty_weights():
    empty = pd.DataFrame(columns=["manager", "period", "filing_date", "ticker", "value"])
    assert target_weights(empty) == {}
