"""Combined-system tests: blend math, causal vol targeting, demo smoke."""

from __future__ import annotations

import numpy as np
import pandas as pd

from whale_clone.config import load_settings
from whale_clone.system import build_sleeves, combine_sleeves, evaluate_system


def _sleeves(n=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-31", periods=n, freq="ME")
    return pd.DataFrame(
        {
            "index": rng.normal(0.008, 0.04, n),
            "vrp": rng.normal(0.006, 0.025, n),
            "insider": rng.normal(0.007, 0.05, n),
        },
        index=idx,
    )


def test_combine_weights_are_a_convex_blend_when_unlevered():
    s = _sleeves()
    w = {"index": 0.5, "vrp": 0.3, "insider": 0.2}
    blended = combine_sleeves(
        s, w, vol_target=0.0, max_leverage=1.5, borrow_spread=0.0, rf_annual=0.0
    )
    expected = 0.5 * s["index"] + 0.3 * s["vrp"] + 0.2 * s["insider"]
    pd.testing.assert_series_equal(blended, expected, check_names=False)


def test_missing_sleeve_months_are_treated_as_cash():
    s = _sleeves(n=12)
    s.loc[s.index[:3], "insider"] = np.nan  # first 3 months: no insider signal
    w = {"index": 0.0, "vrp": 0.0, "insider": 1.0}
    blended = combine_sleeves(
        s, w, vol_target=0.0, max_leverage=1.5, borrow_spread=0.0, rf_annual=0.0
    )
    assert (blended.iloc[:3] == 0.0).all()  # cash, not NaN


def test_vol_targeting_is_causal():
    s = _sleeves()
    w = {"index": 0.4, "vrp": 0.4, "insider": 0.2}
    base = combine_sleeves(
        s, w, vol_target=0.12, max_leverage=2.0, borrow_spread=0.0, rf_annual=0.0
    )

    tampered = s.copy()
    tampered.iloc[-1] *= 3.0  # change only the last month
    after = combine_sleeves(
        tampered, w, vol_target=0.12, max_leverage=2.0, borrow_spread=0.0, rf_annual=0.0
    )
    # Leverage uses only trailing data (shifted), so all but the last month match.
    pd.testing.assert_series_equal(base.iloc[:-1], after.iloc[:-1], check_names=False)


def test_vol_targeting_respects_leverage_cap():
    # A very low-vol blend with a high target should clamp at max_leverage.
    idx = pd.date_range("2015-01-31", periods=60, freq="ME")
    s = pd.DataFrame(
        {"index": np.full(60, 0.001), "vrp": np.full(60, 0.001), "insider": np.full(60, 0.001)},
        index=idx,
    )
    w = {"index": 1.0, "vrp": 0.0, "insider": 0.0}
    blended = combine_sleeves(
        s, w, vol_target=0.50, max_leverage=1.5, borrow_spread=0.0, rf_annual=0.0
    )
    # Returns are constant -> trailing vol ~0 -> leverage would explode but is capped at 1.5x.
    assert blended.max() <= 0.001 * 1.5 + 1e-9


def test_evaluate_system_demo_smoke():
    s = load_settings(price_source="demo", insider_source="demo", bootstrap_iterations=200)
    sleeves, index_ret = build_sleeves(s)
    verdict, diag = evaluate_system(sleeves, index_ret, s)
    assert len(verdict.gates) == 6
    assert any("Tail-risk" in g.name for g in verdict.gates)
    assert diag["n_months"] > 0
