"""SEC Form 4 insider-transaction loader (open-market purchases).

A "smart money" signal like 13F, but filed within ~2 business days instead of
45 — so it is much fresher. Form 4 XML is un-namespaced and carries the issuer
ticker directly (no CUSIP mapping needed).

Output schema (long form), one row per insider purchase::

    ticker | filing_date | owner | is_officer | officer_title | shares | price | value

Only open-market PURCHASES (transactionCode 'P') are kept — insider *sales* are
documented to be uninformative (liquidity/tax/diversification driven).

Network: the real loader needs www.sec.gov / data.sec.gov. Use ``source="demo"``
offline. Parsing is a pure function, fixture-tested.
"""

from __future__ import annotations

import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
from lxml import etree

from ..store import Store

_HEADERS = {"User-Agent": "whale-clone research (williamlillis100@gmail.com)"}
_TIMEOUT = 30
_SEC_PAUSE = 0.15

INSIDER_COLUMNS = [
    "ticker",
    "filing_date",
    "owner",
    "is_officer",
    "officer_title",
    "shares",
    "price",
    "value",
]

_DEMO_UNIVERSE = ["AAPL", "MSFT", "JPM", "XOM", "KO", "PFE", "CAT", "WMT"]


def _text(el: object, default: str = "") -> str:
    return el.text.strip() if el is not None and el.text else default  # type: ignore[attr-defined]


def parse_form4(xml_bytes: bytes) -> list[dict[str, object]]:
    """Parse a Form 4 ownership document into open-market purchase rows (pure).

    Returns one dict per non-derivative PURCHASE transaction. Robust to the
    common Form 4 layout; unknown/!=P transactions are skipped.
    """
    try:
        root = etree.fromstring(xml_bytes)
    except etree.XMLSyntaxError:
        return []

    def find_local(node: object, name: str) -> object:
        for el in node.iter():  # type: ignore[attr-defined]
            if etree.QName(el.tag).localname == name:
                return el
        return None

    def local_text(node: object, name: str) -> str:
        el = find_local(node, name)
        if el is None:
            return ""
        # transaction sub-fields wrap the scalar in a <value> child.
        val = find_local(el, "value")
        return _text(val) if val is not None else _text(el)

    ticker = ""
    owner = ""
    is_officer = False
    officer_title = ""
    for el in root.iter():
        ln = etree.QName(el.tag).localname
        if ln == "issuerTradingSymbol" and el.text:
            ticker = el.text.strip().upper()
        elif ln == "rptOwnerName" and el.text:
            owner = el.text.strip()
        elif ln == "isOfficer" and el.text:
            is_officer = el.text.strip() in {"1", "true", "Y"}
        elif ln == "officerTitle" and el.text:
            officer_title = el.text.strip()

    rows: list[dict[str, object]] = []
    for el in root.iter():
        if etree.QName(el.tag).localname != "nonDerivativeTransaction":
            continue
        code = local_text(el, "transactionCode")
        if code != "P":  # open-market purchase only
            continue
        try:
            shares = float(local_text(el, "transactionShares") or "nan")
            price = float(local_text(el, "transactionPricePerShare") or "nan")
        except ValueError:
            continue
        if not ticker or np.isnan(shares) or shares <= 0:
            continue
        price = 0.0 if np.isnan(price) else price
        rows.append(
            {
                "ticker": ticker,
                "owner": owner,
                "is_officer": is_officer,
                "officer_title": officer_title,
                "shares": shares,
                "price": price,
                "value": shares * price,
            }
        )
    return rows


def load_insider_buys(
    tickers: list[str],
    *,
    source: str = "edgar",
    start: date | None = None,
    end: date | None = None,
    store: Store | None = None,
    refresh: bool = False,
    seed: int = 1234,
) -> pd.DataFrame:
    """Load open-market insider purchases for a universe of tickers."""
    cache_key = f"insiders_{source}_{'-'.join(sorted(tickers))[:60]}"
    if store is not None and store.has(cache_key) and not refresh:
        return store.load(cache_key)

    if source == "demo":
        if start is None or end is None:
            raise ValueError("demo insider data requires start and end dates")
        df = _demo_insider_buys(tickers, start, end, seed=seed)
    elif source == "edgar":
        df = _edgar_insider_buys(tickers)
    else:
        raise ValueError(f"unknown insider source: {source!r}")

    df = df.loc[:, INSIDER_COLUMNS].copy()
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df["ticker"] = df["ticker"].astype(str).str.upper()
    if start is not None:
        df = df[df["filing_date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["filing_date"] <= pd.Timestamp(end)]
    df = df.reset_index(drop=True)
    if store is not None:
        store.save(cache_key, df)
    return df


# --------------------------------------------------------------------------- #
# EDGAR
# --------------------------------------------------------------------------- #
def _cik_for_ticker(ticker: str, mapping: dict[str, str]) -> str | None:
    return mapping.get(ticker.upper())


def _ticker_cik_map() -> dict[str, str]:
    r = requests.get(
        "https://www.sec.gov/files/company_tickers.json", headers=_HEADERS, timeout=_TIMEOUT
    )
    r.raise_for_status()
    out: dict[str, str] = {}
    for row in r.json().values():
        out[str(row["ticker"]).upper()] = str(row["cik_str"]).zfill(10)
    return out


def _edgar_insider_buys(tickers: list[str]) -> pd.DataFrame:
    mapping = _ticker_cik_map()
    rows: list[dict[str, object]] = []
    for tkr in tickers:
        cik = _cik_for_ticker(tkr, mapping)
        if cik is None:
            continue
        try:
            for accession, fdate in _form4_filings(cik):
                xml = _fetch_form4_xml(cik, accession)
                if xml is None:
                    continue
                for row in parse_form4(xml):
                    row["filing_date"] = fdate
                    rows.append(row)
                time.sleep(_SEC_PAUSE)
        except Exception:  # skip a bad issuer, keep going
            continue
    if not rows:
        return pd.DataFrame(columns=INSIDER_COLUMNS)
    return pd.DataFrame(rows)


def _form4_filings(cik: str) -> list[tuple[str, str]]:
    url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
    resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    blocks = [data["filings"]["recent"]]
    for extra in data["filings"].get("files", []):
        time.sleep(_SEC_PAUSE)
        r = requests.get(
            f"https://data.sec.gov/submissions/{extra['name']}", headers=_HEADERS, timeout=_TIMEOUT
        )
        r.raise_for_status()
        blocks.append(r.json())
    out: list[tuple[str, str]] = []
    for blk in blocks:
        for form, accn, fdate in zip(
            blk.get("form", []),
            blk.get("accessionNumber", []),
            blk.get("filingDate", []),
            strict=False,
        ):
            if form == "4":
                out.append((accn.replace("-", ""), fdate))
    return out


def _fetch_form4_xml(cik: str, accession: str) -> bytes | None:
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}"
    idx = requests.get(f"{base}/index.json", headers=_HEADERS, timeout=_TIMEOUT)
    idx.raise_for_status()
    items = idx.json().get("directory", {}).get("item", [])
    xml_name = next((it["name"] for it in items if it["name"].lower().endswith(".xml")), None)
    if xml_name is None:
        return None
    time.sleep(_SEC_PAUSE)
    x = requests.get(f"{base}/{xml_name}", headers=_HEADERS, timeout=_TIMEOUT)
    x.raise_for_status()
    return x.content


# --------------------------------------------------------------------------- #
# Demo (offline)
# --------------------------------------------------------------------------- #
def _demo_insider_buys(tickers: list[str], start: date, end: date, *, seed: int) -> pd.DataFrame:
    """Synthetic insider purchases: occasional cluster buys. NOT real data."""
    rng = np.random.default_rng(seed)
    universe = [t for t in tickers if t in _DEMO_UNIVERSE] or list(tickers)
    days = pd.bdate_range(start, end)
    rows: list[dict[str, object]] = []
    for tkr in universe:
        # A few cluster-buy episodes per ticker over the sample.
        n_events = rng.integers(2, 6)
        for _ in range(n_events):
            d = days[int(rng.integers(0, len(days)))]
            n_insiders = int(rng.integers(1, 5))
            for j in range(n_insiders):
                rows.append(
                    {
                        "ticker": tkr,
                        "filing_date": d + timedelta(days=int(rng.integers(0, 3))),
                        "owner": f"insider_{j}",
                        "is_officer": bool(j == 0 or rng.random() < 0.4),
                        "officer_title": "CEO" if j == 0 else ("CFO" if j == 1 else ""),
                        "shares": float(rng.integers(1000, 50000)),
                        "price": float(rng.uniform(20, 200)),
                        "value": 0.0,
                    }
                )
    df = pd.DataFrame(rows, columns=INSIDER_COLUMNS)
    if not df.empty:
        df["value"] = df["shares"] * df["price"]
    return df
