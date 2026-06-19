"""Tests for the quarterly signal / paper-trade engine."""

from __future__ import annotations

import pandas as pd

from whale_clone.config import load_settings
from whale_clone.paper import compute_trades, current_target, run_signal
from whale_clone.portfolio import validate_holdings


def test_compute_trades_from_cash_are_all_new():
    trades = compute_trades({}, {"AAA": 0.6, "BBB": 0.4})
    assert {t.ticker for t in trades} == {"AAA", "BBB"}
    assert all(t.action == "NEW" for t in trades)
    # Sorted by absolute move, largest first.
    assert trades[0].ticker == "AAA"


def test_compute_trades_detects_exit_buy_sell():
    prev = {"AAA": 0.5, "BBB": 0.5}
    new = {"AAA": 0.7, "CCC": 0.3}  # add AAA, exit BBB, new CCC
    actions = {t.ticker: t.action for t in compute_trades(prev, new)}
    assert actions["AAA"] == "BUY"
    assert actions["BBB"] == "EXIT"
    assert actions["CCC"] == "NEW"


def test_compute_trades_ignores_tiny_drift():
    prev = {"AAA": 0.5, "BBB": 0.5}
    new = {"AAA": 0.503, "BBB": 0.497}  # within default 0.5% band
    assert compute_trades(prev, new) == []


def test_current_target_uses_top_n(simple_holdings):
    h = validate_holdings(simple_holdings)
    s = load_settings(top_n_positions=1)
    as_of = pd.to_datetime(h["filing_date"]).max()
    target = current_target(h, s, as_of=as_of)
    assert abs(sum(target.values()) - 1.0) < 1e-9
    # top_n=1 per manager -> at most one name per manager survives.
    assert len(target) <= h["manager"].nunique()


def test_run_signal_demo_appends_log(tmp_path):
    s = load_settings(holdings_source="demo", price_source="demo", cache_dir=str(tmp_path))
    report, entry = run_signal(s)
    assert "QUARTERLY SIGNAL" in report
    assert entry.get("target")
    # Log file written and contains the entry.
    import json

    log = json.loads((tmp_path / "paper_log.json").read_text())
    assert len(log) == 1
    # Re-running with the same data does not duplicate the entry.
    run_signal(s)
    log2 = json.loads((tmp_path / "paper_log.json").read_text())
    assert len(log2) == 1
