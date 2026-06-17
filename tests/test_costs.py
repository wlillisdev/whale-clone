from __future__ import annotations

import pytest

from whale_clone.costs import CostModel, one_way_turnover, traded_notional


def test_total_bps():
    assert CostModel(commission_bps=1.0, slippage_bps=4.0).total_bps == 5.0


def test_cost_for_traded_linear():
    cm = CostModel(commission_bps=0.0, slippage_bps=10.0)  # 10 bps
    assert cm.cost_for_traded(0.0) == 0.0
    assert cm.cost_for_traded(1.0) == pytest.approx(0.001)  # 10bps of a full book traded
    assert cm.cost_for_traded(0.5) == pytest.approx(0.0005)


def test_negative_traded_rejected():
    with pytest.raises(ValueError):
        CostModel().cost_for_traded(-0.1)


def test_traded_notional_from_cash_is_one():
    # Buying a full book from cash trades 1.0 of notional (all buys).
    assert traded_notional({}, {"A": 0.5, "B": 0.5}) == pytest.approx(1.0)


def test_one_way_turnover_from_cash_is_half():
    # Conventional one-way turnover is half the gross.
    assert one_way_turnover({}, {"A": 0.5, "B": 0.5}) == pytest.approx(0.5)


def test_one_way_turnover_no_change_is_zero():
    w = {"A": 0.6, "B": 0.4}
    assert one_way_turnover(w, w) == pytest.approx(0.0)


def test_one_way_turnover_partial():
    prev = {"A": 0.5, "B": 0.5}
    new = {"A": 0.7, "B": 0.3}
    # |0.2| + |0.2| = 0.4, half = 0.2
    assert one_way_turnover(prev, new) == pytest.approx(0.2)


def test_one_way_turnover_disjoint_names():
    prev = {"A": 1.0}
    new = {"B": 1.0}
    # sell all A, buy all B -> one-way turnover 1.0
    assert one_way_turnover(prev, new) == pytest.approx(1.0)
