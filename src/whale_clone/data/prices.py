"""Daily adjusted-close price loader + benchmark.

Sources:
* ``stooq``  — free CSV download per ticker (adjusted), no key required.
* ``yahoo``  — Yahoo chart JSON API (adjusted close).
* ``demo``   — deterministic synthetic prices for offline / CI runs.

Returns a wide panel: index = trading days, columns = tickers (incl. benchmark),
values = adjusted close. Results are cached as Parquet via :class:`Store`.

Network note: the real sources require outbound HTTPS to stooq.com /
query*.finance.yahoo.com. In sandboxed environments where those hosts are
blocked, use ``source="demo"``.
"""

from __future__ import annotations

import io
import sys
from datetime import date

import numpy as np
import pandas as pd
import requests

from ..store import Store

_HEADERS = {"User-Agent": "whale-clone/0.1 (research; contact via repo)"}
_TIMEOUT = 30


def _warn(msg: str) -> None:
    print(f"[prices] {msg}", file=sys.stderr)


def load_prices(
    tickers: list[str],
    *,
    start: date,
    end: date,
    source: str = "stooq",
    benchmark: str = "SPY",
    store: Store | None = None,
    refresh: bool = False,
    seed: int = 1234,
) -> pd.DataFrame:
    """Load an adjusted-close panel for ``tickers`` + ``benchmark``."""
    universe = sorted({*(t.upper() for t in tickers), benchmark.upper()})
    cache_key = f"prices_{source}_{start:%Y%m%d}_{end:%Y%m%d}_{hash(tuple(universe)) & 0xFFFFFF:x}"

    if store is not None and store.has(cache_key) and not refresh:
        return store.load(cache_key)

    if source == "demo":
        panel = _demo_prices(universe, start, end, seed=seed)
    elif source == "stooq":
        panel = _stooq_panel(universe, start, end)
    elif source == "yahoo":
        panel = _yahoo_panel(universe, start, end)
    else:
        raise ValueError(f"unknown price source: {source!r}")

    if benchmark.upper() not in panel.columns:
        raise RuntimeError(
            f"benchmark {benchmark!r} has no price data from source {source!r} — "
            "cannot compare against it"
        )
    panel = panel.sort_index()
    panel = panel.loc[(panel.index >= pd.Timestamp(start)) & (panel.index <= pd.Timestamp(end))]
    if store is not None:
        store.save(cache_key, panel)
    return panel


# --------------------------------------------------------------------------- #
# Real sources
# --------------------------------------------------------------------------- #
def _stooq_panel(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    series: dict[str, pd.Series] = {}
    skipped: list[str] = []
    for t in tickers:
        try:
            df = _stooq_one(t, start, end)
        except Exception as exc:  # network / parse error for one ticker
            skipped.append(f"{t} ({exc})")
            continue
        if df is not None and not df.empty:
            series[t] = df
        else:
            skipped.append(f"{t} (no data — delisted or throttled?)")
    if skipped:
        _warn(f"stooq skipped {len(skipped)} ticker(s): {', '.join(skipped)}")
    if not series:
        raise RuntimeError(
            "stooq returned no data for any ticker. Stooq throttles its free CSV "
            "downloads per IP (shared hosts are often over the limit). Try "
            "--price-source yahoo."
        )
    return pd.DataFrame(series)


def _stooq_one(ticker: str, start: date, end: date) -> pd.Series | None:
    # Stooq uses '-' for share classes (BRK-B), not '/'.
    symbol = f"{ticker.lower().replace('/', '-').replace('.', '-')}.us"
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d&d1={start:%Y%m%d}&d2={end:%Y%m%d}"
    resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
    resp.raise_for_status()
    text = resp.text.strip()
    if not text or text.lower().startswith("<"):
        return None
    df = pd.read_csv(io.StringIO(text))
    if "Date" not in df.columns or "Close" not in df.columns:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")["Close"].rename(ticker).astype(float)


def _yahoo_panel(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    series: dict[str, pd.Series] = {}
    skipped: list[str] = []
    for t in tickers:
        try:
            s = _yahoo_one(t, start, end)
        except Exception as exc:  # one bad/delisted ticker must not kill the run
            skipped.append(f"{t} ({exc})")
            continue
        if s is not None and not s.empty:
            series[t] = s
        else:
            skipped.append(f"{t} (no data — delisted or not yet listed)")
    if skipped:
        _warn(f"yahoo skipped {len(skipped)} ticker(s): {', '.join(skipped)}")
    if not series:
        raise RuntimeError("yahoo returned no data for any ticker (network blocked?)")
    return pd.DataFrame(series)


def _yahoo_one(ticker: str, start: date, end: date) -> pd.Series | None:
    p1 = int(pd.Timestamp(start).timestamp())
    p2 = int(pd.Timestamp(end).timestamp())
    # Yahoo uses '-' for share classes (BRK-B), not '/' as 13F/OpenFIGI report it.
    yf_symbol = ticker.replace("/", "-").replace(".", "-")
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{yf_symbol}"
        f"?period1={p1}&period2={p2}&interval=1d&events=div%2Csplit"
    )
    resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()
    result = payload.get("chart", {}).get("result")
    if not result:
        return None
    r0 = result[0]
    ts = r0.get("timestamp")
    adj = r0.get("indicators", {}).get("adjclose")
    if not ts or not adj:
        return None
    closes = adj[0].get("adjclose")
    idx = pd.to_datetime(ts, unit="s").normalize()
    # Keep the original ticker as the series name so it matches the holdings.
    return pd.Series(closes, index=idx, name=ticker).astype(float).dropna()


# --------------------------------------------------------------------------- #
# Demo (offline) source
# --------------------------------------------------------------------------- #
def _demo_prices(tickers: list[str], start: date, end: date, *, seed: int) -> pd.DataFrame:
    """Deterministic synthetic adjusted-close prices (business days).

    Geometric random walk per ticker. This is NOT real market data and any
    verdict produced on it is a pipeline smoke test, never a claim about a real
    strategy. Used so the engine + gates can run with no network.
    """
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start=start, end=end)
    n = len(cal)
    cols: dict[str, np.ndarray] = {}
    for i, t in enumerate(tickers):
        mu = 0.0003 + 0.00002 * i  # tiny per-ticker drift differences
        sigma = 0.011
        shocks = rng.normal(mu, sigma, size=n)
        cols[t] = 100.0 * np.exp(np.cumsum(shocks))
    return pd.DataFrame(cols, index=cal)
