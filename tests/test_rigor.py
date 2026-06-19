"""Tests for the anti-overfitting rigor layer (deflated Sharpe, holdout)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from whale_clone.metrics import deflated_sharpe, expected_max_sharpe, probabilistic_sharpe
from whale_clone.rigor import deflated_sharpe_gate, holdout_split


def _returns(mu, n=1500, sigma=0.01, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sigma, n))


def test_probabilistic_sharpe_high_for_strong_track_record():
    psr = probabilistic_sharpe(_returns(0.0008), sr_benchmark=0.0)
    assert 0.5 < psr <= 1.0  # clearly positive mean -> likely true SR > 0


def test_probabilistic_sharpe_near_half_for_zero_mean_noise():
    psr = probabilistic_sharpe(_returns(0.0), sr_benchmark=0.0)
    assert 0.2 < psr < 0.8  # indistinguishable from zero


def test_expected_max_sharpe_grows_with_trials():
    a = expected_max_sharpe(2, sr_std=0.5)
    b = expected_max_sharpe(50, sr_std=0.5)
    assert b > a > 0  # more trials -> higher noise hurdle
    assert expected_max_sharpe(1, sr_std=0.5) == 0.0


def test_deflated_sharpe_drops_when_more_trials():
    r = _returns(0.0006)
    few = deflated_sharpe(r, n_trials=2, trials_sr_std=0.5)
    many = deflated_sharpe(r, n_trials=200, trials_sr_std=0.5)
    assert many < few  # accounting for more trials lowers confidence


def test_deflated_sharpe_gate_fails_on_noise():
    gate = deflated_sharpe_gate(
        _returns(0.0), n_strategies_tried=8, trials_sr_std=0.5, threshold=0.95
    )
    assert not gate.passed
    assert "deflated" in gate.name.lower()


def test_holdout_split_chronological():
    idx = pd.bdate_range("2010-01-01", periods=1000)
    train_end, holdout_start = holdout_split(idx, holdout_fraction=0.30)
    assert train_end < holdout_start
    # ~70% of dates are on/before train_end.
    frac = (idx <= train_end).mean()
    assert 0.68 < frac < 0.72


def test_holdout_split_rejects_bad_fraction():
    idx = pd.bdate_range("2010-01-01", periods=100)
    with pytest.raises(ValueError):
        holdout_split(idx, holdout_fraction=1.5)
