"""Shared fixtures: tiny, hand-built holdings + price panels.

These are deliberately small and deterministic so the pure engine can be tested
without any network or randomness.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def simple_holdings() -> pd.DataFrame:
    """Two managers, two quarters, with explicit filing dates (45-day lag)."""
    rows = [
        # Q1 filing public 2020-05-15
        ("M1", "2020-Q1", "2020-05-15", "AAA", 100.0),
        ("M1", "2020-Q1", "2020-05-15", "BBB", 100.0),
        ("M2", "2020-Q1", "2020-05-15", "BBB", 100.0),
        ("M2", "2020-Q1", "2020-05-15", "CCC", 300.0),
        # Q2 filing public 2020-08-14
        ("M1", "2020-Q2", "2020-08-14", "AAA", 150.0),
        ("M1", "2020-Q2", "2020-08-14", "BBB", 50.0),
        ("M2", "2020-Q2", "2020-08-14", "CCC", 400.0),
    ]
    return pd.DataFrame(rows, columns=["manager", "period", "filing_date", "ticker", "value"])


@pytest.fixture
def simple_prices() -> pd.DataFrame:
    """Daily price panel covering the holdings window incl. SPY benchmark."""
    dates = pd.bdate_range("2020-05-01", "2020-12-31")
    n = len(dates)
    # Deterministic, monotonic-ish series so behaviour is predictable.
    data = {
        "AAA": [100.0 * (1.0006**i) for i in range(n)],
        "BBB": [50.0 * (1.0004**i) for i in range(n)],
        "CCC": [200.0 * (1.0008**i) for i in range(n)],
        "SPY": [300.0 * (1.0005**i) for i in range(n)],
    }
    return pd.DataFrame(data, index=dates)
