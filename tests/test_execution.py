"""Execution-layer tests — all offline via DryRunBroker. No network, no money."""

from __future__ import annotations

import pytest

from whale_clone.execution import (
    Account,
    DryRunBroker,
    ExecConfig,
    OrderIntent,
    Position,
    check_guardrails,
    plan_rebalance,
    rebalance_to_target,
    reconcile,
)

CFG = ExecConfig()


def test_plan_from_cash_buys_to_target():
    intents = plan_rebalance({"AAA": 0.5, "BBB": 0.5}, 100_000.0, {}, CFG, run_id="Q1")
    assert {i.symbol for i in intents} == {"AAA", "BBB"}
    assert all(i.side == "buy" for i in intents)
    # 2% cash buffer -> ~49k each, not 50k.
    assert intents[0].notional == pytest.approx(49_000.0, rel=1e-6)


def test_empty_target_never_liquidates():
    # Fail-closed: a broken/empty target must NOT sell the existing book.
    assert plan_rebalance({}, 100_000.0, {"AAA": 50_000.0}, CFG, run_id="Q1") == []


def test_nan_target_produces_no_orders():
    assert plan_rebalance({"AAA": float("nan")}, 100_000.0, {}, CFG, run_id="Q1") == []


def test_oversum_target_rejected():
    # Weights summing to >1 (leverage) -> no orders.
    assert plan_rebalance({"AAA": 0.8, "BBB": 0.8}, 100_000.0, {}, CFG, run_id="Q1") == []


def test_rounding_oversum_is_tolerated_and_renormalised():
    # A target that sums to 1.0001 (rounding) must still trade, not fail-closed.
    intents = plan_rebalance({"AAA": 0.5001, "BBB": 0.5}, 100_000.0, {}, CFG, run_id="Q1")
    assert len(intents) == 2
    # Deployed notional stays within the cash-buffered investable amount.
    assert sum(i.notional for i in intents) <= 98_000.0 + 1.0


def test_no_trade_band_skips_tiny_drift():
    # Already at target within the band -> no orders.
    intents = plan_rebalance({"AAA": 0.49}, 100_000.0, {"AAA": 48_000.0}, CFG, run_id="Q1")
    assert intents == []


def test_sells_ordered_before_buys():
    intents = plan_rebalance(
        {"AAA": 0.7, "BBB": 0.0}, 100_000.0, {"AAA": 10_000.0, "BBB": 40_000.0}, CFG, run_id="Q1"
    )
    sides = [i.side for i in intents]
    assert sides == sorted(sides, key=lambda s: s != "sell")  # sells first
    exit_order = next(i for i in intents if i.symbol == "BBB")
    assert exit_order.reason == "exit"


def test_guardrails_block_too_many_orders():
    acct = Account(100_000.0, 100_000.0, 100_000.0, blocked=False, is_paper=True)
    intents = [OrderIntent(f"S{i}", "buy", 100.0, f"c{i}", "buy") for i in range(30)]
    cfg = ExecConfig(max_orders_per_run=20)
    assert any("max_orders" in v for v in check_guardrails(intents, acct, cfg))


def test_guardrails_block_non_paper_account():
    live = Account(100_000.0, 100_000.0, 100_000.0, blocked=False, is_paper=False)
    v = check_guardrails([], live, CFG)
    assert any("not a paper" in s for s in v)


def test_dry_run_never_submits_even_with_orders():
    broker = DryRunBroker(equity=100_000.0)
    report = rebalance_to_target(
        broker, {"AAA": 0.5, "BBB": 0.5}, CFG, run_id="Q1", submit=False, cache_dir="/tmp/wc-x"
    )
    assert not report.submitted
    assert all(r.status == "dry_run" for r in report.results)
    assert broker.submitted == []  # nothing reached the broker


def test_kill_switch_blocks_everything(tmp_path, monkeypatch):
    monkeypatch.setenv("WHALE_EXEC_KILL", "1")
    broker = DryRunBroker()
    report = rebalance_to_target(
        broker, {"AAA": 1.0}, CFG, run_id="Q1", submit=True, cache_dir=str(tmp_path)
    )
    assert not report.submitted
    assert "KILL SWITCH" in report.note


def test_reconcile_reports_drift():
    drift = reconcile({"AAA": 0.5, "BBB": 0.5}, [Position("AAA", 0.0, 60_000.0)], 100_000.0)
    assert drift["AAA"] == pytest.approx(0.1)  # 60% actual vs 50% target
    assert drift["BBB"] == pytest.approx(-0.5)  # held nothing vs 50% target


def test_trade_cli_dry_run_demo():
    from whale_clone.trade import main

    assert main(["--demo"]) == 0  # offline, no orders, clean exit
