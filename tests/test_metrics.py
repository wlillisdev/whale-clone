from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from whale_clone.metrics import (
    annualised_vol,
    bootstrap_mean_ci,
    cagr,
    max_drawdown,
    sharpe,
)


def test_cagr_doubling_in_one_year():
    # 252 trading days, value doubles -> CAGR ~100%.
    value = pd.Series(np.linspace(1.0, 2.0, 253))
    assert cagr(value, periods_per_year=252) == pytest.approx(1.0, rel=1e-6)


def test_max_drawdown_simple():
    value = pd.Series([1.0, 1.2, 0.6, 0.9])
    # peak 1.2 -> trough 0.6 = -50%
    assert max_drawdown(value) == pytest.approx(-0.5)


def test_sharpe_zero_vol_is_nan():
    r = pd.Series([0.01, 0.01, 0.01])
    assert np.isnan(sharpe(r))


def test_sharpe_positive_for_positive_low_vol_returns():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.005, 500))
    assert sharpe(r) > 0


def test_annualised_vol_scales():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0, 0.01, 1000))
    v = annualised_vol(r, periods_per_year=252)
    assert v == pytest.approx(0.01 * np.sqrt(252), rel=0.15)


def test_bootstrap_ci_brackets_mean_and_is_deterministic():
    rng = np.random.default_rng(2)
    sample = pd.Series(rng.normal(0.002, 0.01, 400))
    ci1 = bootstrap_mean_ci(sample, iterations=2000, seed=7)
    ci2 = bootstrap_mean_ci(sample, iterations=2000, seed=7)
    assert ci1 == ci2  # deterministic given seed
    assert ci1.lower < ci1.mean < ci1.upper


def test_bootstrap_ci_lower_below_zero_for_zero_mean_noise():
    rng = np.random.default_rng(3)
    sample = pd.Series(rng.normal(0.0, 0.01, 400))
    ci = bootstrap_mean_ci(sample, iterations=2000, seed=11)
    assert ci.lower < 0  # cannot claim an edge from zero-mean noise
