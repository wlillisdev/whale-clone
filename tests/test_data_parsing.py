"""Parser tests against recorded fixtures (no network)."""

from __future__ import annotations

from datetime import date

from whale_clone.data.holdings import (
    MANAGER_REGISTRY,
    _demo_holdings,
    _parse_info_table_xml,
    _pick_ticker,
    _resolve,
    parse_dataroma_holdings,
)
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


def test_parse_edgar_info_table_namespace_agnostic(fixtures_dir):
    xml_bytes = (fixtures_dir / "edgar_infotable.xml").read_bytes()
    df = _parse_info_table_xml(xml_bytes)
    assert list(df.columns) == ["cusip", "value"]
    assert len(df) == 2
    # CUSIPs parsed despite the XML namespace prefixes.
    assert "037833100" in set(df["cusip"])
    assert "060505104" in set(df["cusip"])
    # Values are positive floats.
    assert (df["value"] > 0).all()
    assert df.set_index("cusip").loc["037833100", "value"] == 174348000.0


def test_resolve_registry_key_and_raw_cik():
    assert _resolve("berkshire").cik == "0001067983"
    # A raw numeric CIK is accepted and zero-padded.
    assert _resolve("1067983").cik == "0001067983"
    assert "berkshire" in MANAGER_REGISTRY


def test_pick_ticker_prefers_us_equity_and_rejects_junk():
    data = [
        {"ticker": "ATVIEUR", "marketSector": "Equity"},  # foreign (Frankfurt) listing
        {"ticker": "ATVI", "marketSector": "Equity"},  # the US common stock
    ]
    assert _pick_ticker(data) == "ATVI"
    # Falls back to a US-looking symbol even if not tagged Equity.
    assert _pick_ticker([{"ticker": "ABC", "marketSector": "Corp"}]) == "ABC"
    # Pure junk / foreign / derivative tickers are rejected -> None (dropped).
    assert _pick_ticker([{"ticker": "HHC*", "marketSector": "Equity"}]) is None
    assert _pick_ticker([{"ticker": "0VVB", "marketSector": "Equity"}]) is None
    assert _pick_ticker([{"ticker": "TWTRUSD", "marketSector": "Equity"}]) is None
    assert _pick_ticker(None) is None
    assert _pick_ticker([]) is None


def test_us_equity_ticker_allows_share_classes():
    from whale_clone.data.holdings import _is_us_equity_ticker

    assert _is_us_equity_ticker("AAPL")
    assert _is_us_equity_ticker("BRK/B")
    assert _is_us_equity_ticker("BF.B")
    assert not _is_us_equity_ticker("ATVIEUR")
    assert not _is_us_equity_ticker("LM03")
    assert not _is_us_equity_ticker("")


def test_resolve_unknown_raises():
    try:
        _resolve("not_a_manager")
    except KeyError as e:
        assert "unknown manager" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected KeyError")


def test_demo_prices_shape_and_positivity():
    start, end = date(2018, 1, 1), date(2019, 12, 31)
    panel = _demo_prices(["AAA", "BBB", "SPY"], start, end, seed=1)
    assert list(panel.columns) == ["AAA", "BBB", "SPY"]
    assert (panel > 0).all().all()
    assert len(panel) > 400  # ~2 years of business days
