"""13F holdings loaders.

Output schema (long form), one row per (manager, period, ticker)::

    manager | period | filing_date | ticker | value

``filing_date`` is when the holding became *public* — the only date we are
allowed to act on (acting on quarter-end is look-ahead bias).

Sources:
* ``dataroma`` — scrapes Dataroma's per-manager holdings grid. Dataroma exposes
  the *current* snapshot cleanly; it is the easiest start and good for a live
  "what do they hold now" view. Reconstructing a long filing-date history is
  best done from EDGAR (:func:`load_edgar_history`), the source of truth.
* ``demo``     — deterministic synthetic multi-quarter history with realistic
  45-day filing lags and low turnover, for offline / CI runs. NOT real data.

Network note: ``dataroma`` needs outbound HTTPS to dataroma.com; EDGAR needs
data.sec.gov / www.sec.gov. In sandboxed environments where those are blocked,
use ``source="demo"``.
"""

from __future__ import annotations

import io
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

from ..portfolio import HOLDINGS_COLUMNS, validate_holdings
from ..store import Store

_HEADERS = {"User-Agent": "whale-clone/0.1 (research; contact via repo)"}
_TIMEOUT = 30

# A small, fixed universe used by the demo source so prices + holdings line up.
_DEMO_UNIVERSE = ["AAPL", "AXP", "BAC", "KO", "CVX", "OXY", "KHC", "MCO", "V", "MA", "COST", "JPM"]


def load_holdings(
    managers: list[str],
    *,
    source: str = "dataroma",
    start: date | None = None,
    end: date | None = None,
    store: Store | None = None,
    refresh: bool = False,
    seed: int = 1234,
) -> pd.DataFrame:
    """Load 13F holdings for ``managers`` as a validated long-form frame."""
    cache_key = f"holdings_{source}_{'-'.join(sorted(managers))}"
    if store is not None and store.has(cache_key) and not refresh:
        return store.load(cache_key)

    if source == "demo":
        if start is None or end is None:
            raise ValueError("demo holdings require start and end dates")
        df = _demo_holdings(managers, start, end, seed=seed)
    elif source == "dataroma":
        df = pd.concat([_dataroma_one(m) for m in managers], ignore_index=True)
    else:
        raise ValueError(f"unknown holdings source: {source!r}")

    df = validate_holdings(df)
    if store is not None:
        store.save(cache_key, df)
    return df


# --------------------------------------------------------------------------- #
# Dataroma (current snapshot)
# --------------------------------------------------------------------------- #
def _dataroma_one(manager: str) -> pd.DataFrame:
    """Fetch one manager's current holdings grid from Dataroma."""
    url = f"https://www.dataroma.com/m/holdings.php?m={manager}"
    resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
    resp.raise_for_status()
    return parse_dataroma_holdings(resp.text, manager)


def parse_dataroma_holdings(html: str, manager: str) -> pd.DataFrame:
    """Parse Dataroma's holdings grid HTML into the long-form schema (pure).

    Returns a single-period snapshot. ``filing_date`` is set to today as a
    best-effort (Dataroma does not expose the exact filing date on this page);
    for an honest historical backtest, prefer :func:`load_edgar_history`.

    Separated from fetching so it is unit-testable against a recorded fixture.
    """
    tables = pd.read_html(io.StringIO(html), attrs={"id": "grid"})
    if not tables:
        raise RuntimeError(f"no holdings grid found for manager {manager!r}")
    grid = tables[0]
    grid.columns = [str(c).strip().lower() for c in grid.columns]

    ticker_col = next((c for c in grid.columns if "ticker" in c or "stock" in c), None)
    value_col = next((c for c in grid.columns if "value" in c or "%" in c), None)
    if ticker_col is None or value_col is None:
        raise RuntimeError(f"unexpected Dataroma layout for {manager!r}: {list(grid.columns)}")

    tickers = grid[ticker_col].astype(str).str.split("-").str[0].str.strip()
    values = (
        grid[value_col]
        .astype(str)
        .str.replace(r"[\$,%]", "", regex=True)
        .str.replace(",", "", regex=False)
        .astype(float)
    )
    out = pd.DataFrame(
        {
            "manager": manager,
            "period": pd.Timestamp.today().to_period("Q").strftime("%Y-Q%q"),
            "filing_date": pd.Timestamp.today().normalize(),
            "ticker": tickers,
            "value": values,
        }
    )
    return out.loc[:, HOLDINGS_COLUMNS]


# --------------------------------------------------------------------------- #
# EDGAR (authoritative history) — best-effort, documented
# --------------------------------------------------------------------------- #
def load_edgar_history(cik: str, *, cusip_to_ticker: dict[str, str] | None = None) -> pd.DataFrame:
    """Load a manager's 13F-HR history from SEC EDGAR (source of truth).

    EDGAR is authoritative and gives the **exact filing date** per quarter —
    essential for modelling the 45-day lag honestly. This implementation fetches
    the submissions index and each 13F information table, using ``filingDate``
    from EDGAR. CUSIP->ticker mapping must be supplied (EDGAR reports CUSIPs, not
    tickers); names without a mapping are dropped.

    Note: validated against the live SEC endpoints requires network access to
    data.sec.gov / www.sec.gov.
    """
    cik10 = str(cik).zfill(10)
    sub_url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
    resp = requests.get(sub_url, headers=_HEADERS, timeout=_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    recent = data["filings"]["recent"]
    frame = pd.DataFrame(recent)
    f13 = frame[frame["form"].isin(["13F-HR", "13F-HR/A"])]

    rows: list[pd.DataFrame] = []
    cmap = cusip_to_ticker or {}
    for _, row in f13.iterrows():
        accession = str(row["accessionNumber"]).replace("-", "")
        filing_date = pd.Timestamp(row["filingDate"])
        period = pd.Timestamp(row["reportDate"]).to_period("Q").strftime("%Y-Q%q")
        info = _edgar_info_table(cik10, accession)
        if info is None:
            continue
        info["ticker"] = info["cusip"].map(cmap)
        info = info.dropna(subset=["ticker"])
        info["manager"] = data.get("name", cik)
        info["period"] = period
        info["filing_date"] = filing_date
        rows.append(info.loc[:, HOLDINGS_COLUMNS])

    if not rows:
        raise RuntimeError(f"no 13F filings parsed for CIK {cik}")
    return validate_holdings(pd.concat(rows, ignore_index=True))


def _edgar_info_table(cik10: str, accession: str) -> pd.DataFrame | None:
    """Fetch and parse the 13F information table XML for one accession."""
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik10)}/{accession}"
    index_url = f"{base}/index.json"
    resp = requests.get(index_url, headers=_HEADERS, timeout=_TIMEOUT)
    resp.raise_for_status()
    items = resp.json().get("directory", {}).get("item", [])
    xml_name = next(
        (
            it["name"]
            for it in items
            if it["name"].lower().endswith(".xml") and "info" in it["name"].lower()
        ),
        None,
    )
    if xml_name is None:
        xml_name = next((it["name"] for it in items if it["name"].lower().endswith(".xml")), None)
    if xml_name is None:
        return None
    xml = requests.get(f"{base}/{xml_name}", headers=_HEADERS, timeout=_TIMEOUT)
    xml.raise_for_status()
    # 13F info tables use a namespaced schema; strip namespaces for simple parsing.
    text = xml.text.replace("ns1:", "").replace("n1:", "")
    parsed = pd.read_xml(io.StringIO(text), xpath=".//infoTable")
    if "cusip" not in parsed.columns or "value" not in parsed.columns:
        return None
    return parsed.loc[:, ["cusip", "value"]].astype({"cusip": str, "value": float})


# --------------------------------------------------------------------------- #
# Demo (offline) source
# --------------------------------------------------------------------------- #
def _demo_holdings(managers: list[str], start: date, end: date, *, seed: int) -> pd.DataFrame:
    """Deterministic synthetic low-turnover holdings with 45-day filing lags.

    Each manager holds a stable subset of ``_DEMO_UNIVERSE`` and only nudges
    weights between quarters (low turnover, as the thesis requires). NOT real
    data — for pipeline/gate smoke tests only.
    """
    rng = np.random.default_rng(seed)
    quarter_ends = pd.date_range(start=start, end=end, freq="QE")
    rows: list[dict[str, object]] = []
    for mi, manager in enumerate(managers):
        # Stable per-manager subset of names.
        k = 6 + (mi % 3)
        names = list(rng.choice(_DEMO_UNIVERSE, size=k, replace=False))
        base_vals = rng.uniform(5e8, 5e10, size=k)
        for qe in quarter_ends:
            # Low turnover: small multiplicative drift per quarter.
            base_vals = base_vals * rng.uniform(0.95, 1.06, size=k)
            filing_date = qe + timedelta(days=45)  # honest 45-day disclosure lag
            for name, val in zip(names, base_vals, strict=True):
                rows.append(
                    {
                        "manager": manager,
                        "period": pd.Timestamp(qe).to_period("Q").strftime("%Y-Q%q"),
                        "filing_date": filing_date,
                        "ticker": name,
                        "value": float(val),
                    }
                )
    return pd.DataFrame(rows, columns=HOLDINGS_COLUMNS)
