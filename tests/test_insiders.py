"""Insider cluster-buying tests: parser, signal causality, basket no-look-ahead."""

from __future__ import annotations

import numpy as np
import pandas as pd

from whale_clone.config import load_settings
from whale_clone.costs import CostModel
from whale_clone.data.insiders import parse_form4
from whale_clone.insiders import (
    build_target_schedule,
    cluster_events,
    run_basket_backtest,
)


def test_parse_form4_keeps_purchase_drops_sale(fixtures_dir):
    rows = parse_form4((fixtures_dir / "form4_buy.xml").read_bytes())
    assert len(rows) == 1  # only the P, not the S
    r = rows[0]
    assert r["ticker"] == "AAPL"
    assert r["is_officer"] is True
    assert "Executive" in str(r["officer_title"])
    assert r["shares"] == 5000.0
    assert r["price"] == 150.0
    assert r["value"] == 750000.0


def _buys(rows):
    df = pd.DataFrame(rows, columns=["ticker", "filing_date", "owner", "is_officer", "value"])
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df["shares"] = 1.0
    df["price"] = df["value"]
    df["officer_title"] = ""
    return df


def test_cluster_events_requires_min_buyers_and_officer():
    rows = [
        ("X", "2020-03-02", "a", True, 1e5),
        ("X", "2020-03-03", "b", False, 1e5),
        ("X", "2020-03-04", "c", False, 1e5),  # 3 distinct buyers, one officer -> event
        ("Y", "2020-03-02", "a", False, 1e5),
        ("Y", "2020-03-03", "a", False, 1e5),  # same buyer twice -> not a cluster
    ]
    ev = cluster_events(
        _buys(rows), min_buyers=3, require_officer=True, min_value=0, window_days=30
    )
    assert set(ev["ticker"]) == {"X"}

    # Require officer: an all-non-officer cluster is rejected.
    rows2 = [
        ("Z", "2020-03-02", "a", False, 1e5),
        ("Z", "2020-03-03", "b", False, 1e5),
        ("Z", "2020-03-04", "c", False, 1e5),
    ]
    ev2 = cluster_events(
        _buys(rows2), min_buyers=3, require_officer=True, min_value=0, window_days=30
    )
    assert ev2.empty


def test_build_target_schedule_is_causal_and_equal_weight():
    cal = pd.bdate_range("2020-01-01", "2020-03-01")
    events = pd.DataFrame({"ticker": ["X"], "event_date": [pd.Timestamp("2020-01-15")]})
    tgt = build_target_schedule(events, cal, hold_days=10)
    # Not held on/before the event date (causal: enter the day AFTER).
    assert tgt.loc[pd.Timestamp("2020-01-15"), "X"] == 0.0
    nxt = cal[cal.searchsorted(pd.Timestamp("2020-01-15"), side="right")]
    assert tgt.loc[nxt, "X"] == 1.0
    # After the hold window it is dropped (2020-02-14 is a business day well past).
    assert tgt.loc[pd.Timestamp("2020-02-14"), "X"] == 0.0


def _panel(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    cols = {a: 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))) for a in ("X", "Y", "SPY")}
    return pd.DataFrame(cols, index=idx)


def test_basket_backtest_no_lookahead():
    panel = _panel()
    cal = pd.DatetimeIndex(panel.index)
    events = pd.DataFrame({"ticker": ["X", "Y"], "event_date": [cal[50], cal[120]]})
    tgt = build_target_schedule(events, cal, hold_days=60)
    cfg = CostModel(0.0, 5.0)
    base = run_basket_backtest(panel, tgt, "SPY", cost_model=cfg)

    tampered = panel.copy()
    tampered.iloc[-1] *= 1.5
    after = run_basket_backtest(tampered, tgt, "SPY", cost_model=cfg)

    cutoff = panel.index[-3]
    pd.testing.assert_series_equal(
        base.value[base.value.index <= cutoff], after.value[after.value.index <= cutoff]
    )


def test_evaluate_insiders_demo_smoke():
    from whale_clone.insiders import evaluate_insiders, load_insider_data

    s = load_settings(insider_source="demo", price_source="demo", bootstrap_iterations=200)
    buys, prices = load_insider_data(s)
    verdict, diag = evaluate_insiders(buys, prices, s)
    assert len(verdict.gates) == 5
    assert "excess_cagr" in verdict.headline
    assert diag["n_events"] >= 0
