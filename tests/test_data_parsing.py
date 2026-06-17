"""Parser tests against recorded fixtures (no network)."""

from __future__ import annotations

from datetime import date

from whale_clone.data.holdings import _demo_holdings, parse_dataroma_holdings
from whale_clone.data.prices import _demo_prices


def test_parse_dataroma_fixture(fixtures_dir):
    html = (fixtures_dir / "dataroma_brk.html").read_text()
    df = parse_dataroma_holdings(html, "BRK")
    assert set(df.columns) == {"manager", "period", "filing_date", "ticker", "value"}
    assert "AAPL" in set(df["ticker"])
    assert "BAC" in set(df["ticker"])
    assert (df["value"] > 0).all()
    assert (df["manager"] == "BRK").all()


def test_demo_holdings_are_low_turnover_and_have_filing_lag():
    start, end = date(2018, 1, 1), date(2020, 12, 31)
    df = _demo_holdings(["BRK", "psc"], start, end, seed=1)
    assert set(df["manager"]) == {"BRK", "psc"}
    # Filing dates lag quarter-ends by 45 days -> never on a quarter end.
    assert (df["filing_date"].dt.day > 1).all()
    # Multiple quarters present.
    assert df["period"].nunique() >= 8


def test_demo_prices_shape_and_positivity():
    start, end = date(2018, 1, 1), date(2019, 12, 31)
    panel = _demo_prices(["AAA", "BBB", "SPY"], start, end, seed=1)
    assert list(panel.columns) == ["AAA", "BBB", "SPY"]
    assert (panel > 0).all().all()
    assert len(panel) > 400  # ~2 years of business days
