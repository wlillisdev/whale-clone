"""Report generator tests (pure HTML build + CLI smoke)."""

from __future__ import annotations

from whale_clone.portfolio import validate_holdings
from whale_clone.report import build_html, export_holdings_csv


def test_build_html_contains_managers_and_sections(simple_holdings):
    h = validate_holdings(simple_holdings)
    page = build_html(h, top_n=5)
    assert page.startswith("<!doctype html>")
    assert "Superinvestor Holdings Tracker" in page
    assert "M1" in page and "M2" in page
    assert "Consensus" in page
    # A held ticker shows up in a table.
    assert "AAA" in page


def test_build_html_escapes_manager_names():
    import pandas as pd

    rows = [("<script>", "2020-Q1", "2020-05-15", "AAA", 100.0)]
    h = validate_holdings(
        pd.DataFrame(rows, columns=["manager", "period", "filing_date", "ticker", "value"])
    )
    page = build_html(h)
    assert "<script>" not in page  # raw tag must be escaped
    assert "&lt;script&gt;" in page


def test_export_holdings_csv_shape(simple_holdings):
    h = validate_holdings(simple_holdings)
    df = export_holdings_csv(h)
    assert {"ticker", "weight", "manager"} <= set(df.columns)
    assert len(df) > 0


def test_report_cli_demo_writes_files(tmp_path):
    from whale_clone.report import main

    rc = main(["--demo", "--out", str(tmp_path), "--top", "5"])
    assert rc == 0
    assert (tmp_path / "index.html").exists()
    assert (tmp_path / "holdings.csv").exists()
    assert "<!doctype html>" in (tmp_path / "index.html").read_text()
