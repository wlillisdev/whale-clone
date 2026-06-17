"""13F holdings loaders.

Output schema (long form), one row per (manager, period, ticker)::

    manager | period | filing_date | ticker | value

``filing_date`` is when the holding became *public* — the only date we are
allowed to act on (acting on quarter-end is look-ahead bias).

Sources:
* ``edgar``    — SEC EDGAR 13F-HR filings: the **authoritative** source. Gives
  the exact ``filingDate`` per quarter (so the 45-day lag is modelled honestly)
  and a real multi-year history. EDGAR reports CUSIPs, not tickers, so CUSIPs
  are mapped to tickers via the free OpenFIGI API (cached). This is the source
  that produces a real verdict.
* ``dataroma`` — scrapes Dataroma's per-manager grid. Exposes only the *current*
  snapshot, so it is good for a "what do they hold now" view but cannot drive a
  historical backtest. Kept for convenience.
* ``demo``     — deterministic synthetic multi-quarter history. NOT real data.

Network: ``edgar`` needs data.sec.gov / www.sec.gov and api.openfigi.com.
"""

from __future__ import annotations

import io
import sys
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import cast

import numpy as np
import pandas as pd
import requests
from lxml import etree

from ..portfolio import HOLDINGS_COLUMNS, validate_holdings
from ..store import Store

# SEC requires a descriptive User-Agent with contact info.
_HEADERS = {"User-Agent": "whale-clone research (williamlillis100@gmail.com)"}
_TIMEOUT = 30
_SEC_PAUSE = 0.15  # be polite; SEC allows ~10 req/s
_OPENFIGI_PAUSE = 2.5  # no-key OpenFIGI limit is ~25 req/min

# A small, fixed universe used by the demo source so prices + holdings line up.
_DEMO_UNIVERSE = ["AAPL", "AXP", "BAC", "KO", "CVX", "OXY", "KHC", "MCO", "V", "MA", "COST", "JPM"]


@dataclass(frozen=True)
class ManagerInfo:
    name: str
    cik: str  # 10-digit, zero-padded
    dataroma: str | None = None


# Pre-committed managers: few, concentrated, low-turnover value investors.
# CIKs are from SEC EDGAR. If one is wrong the loader skips it with a warning.
MANAGER_REGISTRY: dict[str, ManagerInfo] = {
    "berkshire": ManagerInfo("Berkshire Hathaway", "0001067983", "BRK"),
    "gates_foundation": ManagerInfo("Bill & Melinda Gates Foundation Trust", "0001166559", "GFT"),
    "pershing_square": ManagerInfo("Pershing Square Capital Management", "0001336528", "psc"),
}


def _warn(msg: str) -> None:
    print(f"[holdings] {msg}", file=sys.stderr)


def _resolve(manager: str) -> ManagerInfo:
    """Resolve a registry key or a raw CIK string to a ManagerInfo."""
    if manager in MANAGER_REGISTRY:
        return MANAGER_REGISTRY[manager]
    digits = manager.lstrip("0") or "0"
    if digits.isdigit():
        return ManagerInfo(name=f"CIK {manager}", cik=manager.zfill(10))
    raise KeyError(
        f"unknown manager {manager!r}; use a registry key "
        f"({', '.join(MANAGER_REGISTRY)}) or a numeric CIK"
    )


def load_holdings(
    managers: list[str],
    *,
    source: str = "edgar",
    start: date | None = None,
    end: date | None = None,
    store: Store | None = None,
    refresh: bool = False,
    seed: int = 1234,
    openfigi_key: str | None = None,
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
        df = pd.concat(
            [_dataroma_one(_resolve(m).dataroma or m) for m in managers], ignore_index=True
        )
    elif source == "edgar":
        df = _edgar_all(managers, store=store, refresh=refresh, openfigi_key=openfigi_key)
        if start is not None:
            df = df[df["filing_date"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["filing_date"] <= pd.Timestamp(end)]
    else:
        raise ValueError(f"unknown holdings source: {source!r}")

    df = validate_holdings(df)
    if df.empty:
        raise RuntimeError("no holdings loaded for any manager")
    if store is not None:
        store.save(cache_key, df)
    return df


# --------------------------------------------------------------------------- #
# EDGAR (authoritative history)
# --------------------------------------------------------------------------- #
def _edgar_all(
    managers: list[str],
    *,
    store: Store | None,
    refresh: bool,
    openfigi_key: str | None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    cusip_map: dict[str, str | None] = {}
    if store is not None:
        cusip_map = store.load_json("cusip_ticker_map", {}) or {}

    for m in managers:
        try:
            info = _resolve(m)
            _warn(f"fetching EDGAR 13F history for {info.name} (CIK {info.cik}) …")
            raw = _edgar_history_raw(info, store=store, refresh=refresh)
            if raw.empty:
                _warn(f"  no 13F filings parsed for {info.name}; skipping")
                continue
            # Map any CUSIPs we have not seen before.
            need = sorted({c for c in raw["cusip"].unique() if c not in cusip_map})
            if need:
                _warn(f"  mapping {len(need)} new CUSIP(s) to tickers via OpenFIGI …")
                cusip_map.update(_map_cusips(need, api_key=openfigi_key))
                if store is not None:
                    store.save_json("cusip_ticker_map", cusip_map)
            raw["ticker"] = raw["cusip"].map(cusip_map)
            mapped = raw.dropna(subset=["ticker"])
            dropped = len(raw) - len(mapped)
            if dropped:
                # Coverage by *value* is what matters, not row count: dropping a
                # handful of tiny option/bond lines is harmless; dropping a big
                # equity position is not.
                total_val = float(raw["value"].sum())
                kept_val = float(mapped["value"].sum())
                cov = kept_val / total_val if total_val > 0 else 0.0
                _warn(
                    f"  dropped {dropped} unmapped row(s); "
                    f"ticker coverage = {cov:.1%} of reported $ value"
                )
                if cov < 0.95:
                    _warn(
                        f"  WARNING: low coverage for {info.name} — the backtest "
                        "for this manager may be distorted by missing positions"
                    )
            frames.append(mapped.loc[:, HOLDINGS_COLUMNS])
        except Exception as exc:  # one manager failing must not kill the rest
            _warn(f"  ERROR for manager {m!r}: {exc}; skipping")
            continue

    if not frames:
        raise RuntimeError("EDGAR returned no usable holdings for any manager")
    return pd.concat(frames, ignore_index=True)


def _edgar_history_raw(info: ManagerInfo, *, store: Store | None, refresh: bool) -> pd.DataFrame:
    """All 13F-HR rows for a manager as (manager, period, filing_date, cusip, value).

    Cached per-CIK so re-runs are offline. Ticker mapping happens later.
    """
    cache_name = f"edgar_raw_{info.cik}"
    if store is not None and store.has(cache_name) and not refresh:
        return store.load(cache_name)

    filings = _edgar_13f_filings(info.cik)
    rows: list[pd.DataFrame] = []
    for i, (accession, filing_date, report_date) in enumerate(filings):
        try:
            info_table = _edgar_info_table(info.cik, accession)
        except Exception as exc:
            _warn(f"  could not parse filing {accession}: {exc}")
            continue
        if info_table is None or info_table.empty:
            continue
        info_table = info_table.copy()
        info_table["manager"] = info.name
        info_table["period"] = pd.Timestamp(report_date).to_period("Q").strftime("%Y-Q%q")
        info_table["filing_date"] = pd.Timestamp(filing_date)
        rows.append(info_table)
        if (i + 1) % 10 == 0:
            _warn(f"  parsed {i + 1}/{len(filings)} filings …")
        time.sleep(_SEC_PAUSE)

    if not rows:
        return pd.DataFrame(columns=["manager", "period", "filing_date", "cusip", "value"])
    combined = pd.concat(rows, ignore_index=True)
    # Sum duplicate CUSIP lines within a filing (multiple share classes).
    keys = ["manager", "period", "filing_date", "cusip"]
    out = cast(pd.DataFrame, combined.groupby(keys, as_index=False)["value"].sum())
    if store is not None:
        store.save(cache_name, out)
    return out


def _edgar_13f_filings(cik: str) -> list[tuple[str, str, str]]:
    """List (accessionNumber, filingDate, reportDate) for all 13F-HR filings.

    Reads the submissions index, including the older paginated files.
    """
    cik10 = cik.zfill(10)
    base = f"https://data.sec.gov/submissions/CIK{cik10}.json"
    resp = requests.get(base, headers=_HEADERS, timeout=_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    blocks = [data["filings"]["recent"]]
    for extra in data["filings"].get("files", []):
        time.sleep(_SEC_PAUSE)
        url = f"https://data.sec.gov/submissions/{extra['name']}"
        r = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        r.raise_for_status()
        blocks.append(r.json())

    out: list[tuple[str, str, str]] = []
    for blk in blocks:
        forms = blk.get("form", [])
        accns = blk.get("accessionNumber", [])
        fdates = blk.get("filingDate", [])
        rdates = blk.get("reportDate", [])
        for form, accn, fdate, rdate in zip(forms, accns, fdates, rdates, strict=False):
            if form in ("13F-HR", "13F-HR/A"):
                out.append((accn.replace("-", ""), fdate, rdate or fdate))
    return out


def _edgar_info_table(cik: str, accession: str) -> pd.DataFrame | None:
    """Fetch and parse the 13F information table for one accession.

    Namespace-agnostic XML parse: we match elements by local tag name so we do
    not depend on the (varying) XML namespace prefixes EDGAR uses.
    """
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}"
    idx = requests.get(f"{base}/index.json", headers=_HEADERS, timeout=_TIMEOUT)
    idx.raise_for_status()
    items = idx.json().get("directory", {}).get("item", [])
    names = [it["name"] for it in items if it["name"].lower().endswith(".xml")]
    # Prefer the information table document.
    xml_name = next((n for n in names if "info" in n.lower() or "table" in n.lower()), None)
    if xml_name is None:
        # Fall back to the largest non-primary xml.
        xml_name = next((n for n in names if "primary_doc" not in n.lower()), None)
    if xml_name is None:
        return None

    time.sleep(_SEC_PAUSE)
    xml = requests.get(f"{base}/{xml_name}", headers=_HEADERS, timeout=_TIMEOUT)
    xml.raise_for_status()
    return _parse_info_table_xml(xml.content)


def _parse_info_table_xml(xml_bytes: bytes) -> pd.DataFrame:
    """Parse a 13F information table into (cusip, value) rows, namespace-agnostic.

    Note: the absolute value scale (thousands vs dollars) is irrelevant here
    because portfolio weights are computed per filing and are scale-invariant.
    """
    root = etree.fromstring(xml_bytes)
    rows: list[dict[str, float | str]] = []
    for el in root.iter():
        if etree.QName(el.tag).localname != "infoTable":
            continue
        cusip: str | None = None
        value: float | None = None
        for child in el.iter():
            ln = etree.QName(child.tag).localname
            if ln == "cusip" and child.text:
                cusip = child.text.strip().upper()
            elif ln == "value" and child.text:
                try:
                    value = float(child.text.replace(",", "").strip())
                except ValueError:
                    value = None
        if cusip and value is not None and value > 0:
            rows.append({"cusip": cusip, "value": value})
    return pd.DataFrame(rows, columns=["cusip", "value"])


def _pick_ticker(data: list[dict[str, str]] | None) -> str | None:
    """Choose the best ticker from OpenFIGI candidates, preferring equities."""
    if not data:
        return None
    for d in data:
        if d.get("ticker") and d.get("marketSector") == "Equity":
            return d["ticker"]
    return data[0].get("ticker")


def _openfigi_post(jobs: list[dict[str, str]], headers: dict[str, str]) -> list[dict[str, object]]:
    for attempt in range(6):
        resp = requests.post(
            "https://api.openfigi.com/v3/mapping", json=jobs, headers=headers, timeout=_TIMEOUT
        )
        if resp.status_code == 429:  # rate limited — back off and retry
            time.sleep(_OPENFIGI_PAUSE * (attempt + 2))
            continue
        resp.raise_for_status()
        result: list[dict[str, object]] = resp.json()
        return result
    raise RuntimeError("OpenFIGI rate limit kept failing")


def _map_cusips(cusips: list[str], *, api_key: str | None = None) -> dict[str, str | None]:
    """Map CUSIPs to tickers via OpenFIGI (batched, rate-limited).

    Two passes: first constrained to US exchanges (the common case), then any
    misses are retried without the exchange filter. This recovers names the
    strict filter drops while still preferring US equity tickers.
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-OPENFIGI-APIKEY"] = api_key
    batch_size = 100 if api_key else 10
    result: dict[str, str | None] = {}
    misses = list(cusips)

    for extra in ({"exchCode": "US"}, {}):  # pass 1: US only; pass 2: unconstrained
        if not misses:
            break
        still_missing: list[str] = []
        for start_i in range(0, len(misses), batch_size):
            batch = misses[start_i : start_i + batch_size]
            jobs = [{"idType": "ID_CUSIP", "idValue": c, **extra} for c in batch]
            try:
                entries = _openfigi_post(jobs, headers)
            except RuntimeError:
                _warn("  OpenFIGI rate limit kept failing; leaving remaining CUSIPs unmapped")
                still_missing.extend(batch)
                continue
            for c, entry in zip(batch, entries, strict=False):
                data = entry.get("data") if isinstance(entry, dict) else None
                ticker = _pick_ticker(data)  # type: ignore[arg-type]
                if ticker:
                    result[c] = ticker
                else:
                    still_missing.append(c)
            if not api_key:
                time.sleep(_OPENFIGI_PAUSE)
        misses = still_missing

    for c in misses:  # genuinely unmappable (options, bonds, defunct CUSIPs)
        result[c] = None
    return result


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
    for an honest historical backtest, use the ``edgar`` source instead.

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
        k = 6 + (mi % 3)
        names = list(rng.choice(_DEMO_UNIVERSE, size=k, replace=False))
        base_vals = rng.uniform(5e8, 5e10, size=k)
        for qe in quarter_ends:
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
