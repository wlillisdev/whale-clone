"""Tests for the ML price-predictor experiment, incl. no-look-ahead."""

from __future__ import annotations

import numpy as np
import pandas as pd

from whale_clone.config import load_settings
from whale_clone.ml import (
    make_features,
    make_target,
    predictions_to_target,
    walk_forward_predict,
)


def _prices(n=1100, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))), index=idx)


def test_make_features_are_causal_shape():
    p = _prices(400)
    f = make_features(p)
    assert len(f) == len(p)
    assert "rsi14" in f.columns and "ma200" in f.columns
    # Early rows have NaNs (rolling windows) — that's expected and dropped later.
    assert f["ma200"].isna().iloc[0]


def test_make_target_direction():
    p = pd.Series([10.0, 11.0, 10.5, 12.0], index=pd.bdate_range("2020-01-01", periods=4))
    y = make_target(p, horizon=1)
    assert y.iloc[0] == 1.0  # 10 -> 11 up
    assert y.iloc[1] == 0.0  # 11 -> 10.5 down
    assert pd.isna(y.iloc[-1])  # no future for the last point


def test_walk_forward_predict_no_lookahead():
    p = _prices(1000, seed=1)
    base = walk_forward_predict(p, horizon=1, train_min=300, step=60, seed=7)
    assert not base.empty

    tampered = p.copy()
    tampered.iloc[-1] *= 1.5  # change only the very last bar
    after = walk_forward_predict(tampered, horizon=1, train_min=300, step=60, seed=7)

    # Predictions before the final block must be byte-identical: a future bar
    # cannot leak into earlier predictions.
    cutoff = base.index[-60]
    a = base[base.index < cutoff]
    b = after[after.index < cutoff]
    common = a.index.intersection(b.index)
    assert len(common) > 50
    pd.testing.assert_series_equal(a.loc[common], b.loc[common])


def test_predictions_to_target_is_long_flat_and_shifted():
    cal = pd.bdate_range("2020-01-01", periods=5)
    pred = pd.Series([0.6, 0.4, 0.7], index=cal[1:4])
    tgt = predictions_to_target(pred, cal, threshold=0.5)
    assert set(pd.unique(tgt)).issubset({0.0, 1.0})
    assert len(tgt) == len(cal)


def test_ml_on_noise_has_no_edge_and_fails():
    # Synthetic random walk -> the model must NOT find a real edge.
    from whale_clone.ml import evaluate_ml

    s = load_settings(price_source="demo", bootstrap_iterations=300, ml_train_min=400, ml_step=60)
    prices = _prices(1500, seed=3)
    verdict, diag = evaluate_ml(prices, s)
    assert len(verdict.gates) == 4
    assert not verdict.passed  # noise must not pass
    # Directional accuracy on noise is ~50% (coin flip).
    assert 0.42 <= diag["oos_accuracy"] <= 0.58
