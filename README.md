# whale-clone

> Copy the disclosed stock holdings of a few skilled, **low-turnover** investors
> (Buffett-style), rebalance after each quarterly **13F** disclosure, and **prove
> with a backtest whether it beats just holding the index, after costs.**
> One strategy. One market. End to end. A true yes/no answer.

This repo is the *tester*, built to tell the truth. It is not a get-rich-quick
machine — it is a machine that decides, honestly, whether copying low-turnover
smart money beats SPY after realistic costs.

## The thesis (and the catch)

US institutional managers (>$100M) must disclose holdings quarterly via SEC
Form 13F, with up to a **45-day lag**. That lag kills *fast* strategies — but it
does **not** kill slow, low-turnover value managers who hold for years. There is
documented evidence (Martin & Puthenpurackal, 2008) that a portfolio mimicking
Berkshire's 13F holdings — bought a full month *after* disclosure — still beat
the market over ~1976–2006.

The catch, built in with eyes open: over-diversification destroys the edge (so
we use **few** concentrated managers), manager selection can be overfit (so the
list is **pre-committed** in `config.py`), and survivorship bias is real. The
honest expected outcome is "match or modestly beat the index with less effort,"
not riches.

## How to run

```bash
pip install -e ".[dev]"

# Full empirical run (needs network to Stooq/Yahoo + Dataroma/EDGAR):
make backtest                 # == python -m whale_clone

# Offline pipeline smoke test (no network, synthetic data — NOT a market claim):
make demo                     # == python -m whale_clone --demo

make test                     # pytest
make check                    # ruff + mypy + pytest
```

One command (`python -m whale_clone`) pulls data, runs the backtest, and prints
a PASS/FAIL verdict against the four validation gates.

### Data sources

| Layer     | Source                                    | Notes |
|-----------|-------------------------------------------|-------|
| Prices    | Yahoo (default) / Stooq                    | daily adjusted close; SPY benchmark. Dead/delisted tickers are skipped, not fatal. |
| Holdings  | **SEC EDGAR (default)** / Dataroma          | EDGAR gives the real multi-year history with exact **filing dates**; CUSIPs are mapped to tickers via OpenFIGI (cached). Dataroma is a current-snapshot-only fallback. |
| Offline   | `--demo` synthetic generator               | deterministic; for CI / sandboxes |

> **First EDGAR run is slow** (a few minutes): it downloads every 13F filing for
> each manager and maps CUSIPs→tickers through OpenFIGI's free tier (~25 req/min).
> Everything is cached to `.cache/` as Parquet/JSON, so subsequent runs are fast
> and offline. Set `WHALE_OPENFIGI_API_KEY` to raise the OpenFIGI rate limit.

**Honesty rule on dates:** the backtest acts on the **actual 13F filing date**
(when data became public), never the quarter-end. A test
(`tests/test_backtest_no_lookahead.py`) proves decisions use only
filing-date-available data.

## The four validation gates

A strategy is not "real" until it survives all four (all numbers **after costs**):

1. **Cost-adjusted expectancy** — bootstrap 95% CI lower bound on per-period
   excess return vs SPY is `> 0`.
2. **Walk-forward** — the edge appears in the majority of ≥3 sequential windows,
   with no single window carrying the whole result.
3. **Robustness** — survives parameter variation (manager count, slippage, cap,
   weighting) as a *plateau*, not a single spike.
4. **Benchmark-beating** — net CAGR **and** Sharpe both beat buy-and-hold SPY.

## Current verdict

> **PENDING REAL-DATA RUN.** The engine, gates, and no-look-ahead guarantee are
> implemented and unit-tested (`make test`), and the real data path (SEC EDGAR
> 13F history → OpenFIGI ticker mapping → Yahoo prices) is wired up. The verdict
> below is filled in by running `make backtest` from an environment with
> outbound internet (the original build sandbox blocked the data hosts).
>
> Pre-committed managers: Berkshire Hathaway, Gates Foundation Trust, Pershing
> Square. Window: 2014–2024, most recent 30% reserved out-of-sample.
>
> _Paste the verdict box from `make backtest` here once it runs._

## Architecture

```
src/whale_clone/
├── config.py      # pydantic-settings: pre-committed managers, costs, dates
├── data/
│   ├── holdings.py # Dataroma snapshot + EDGAR history + demo (by filing date)
│   └── prices.py   # Stooq/Yahoo adjusted close + benchmark + demo
├── portfolio.py    # holdings -> capped target weights (pure)
├── costs.py        # commission + slippage on turnover (pure)
├── backtest.py     # quarterly rebalance loop on filing dates, net of costs (pure)
├── metrics.py      # CAGR, Sharpe, drawdown, bootstrap CI (pure)
├── gates.py        # the 4 gates -> verdict (pure given data)
├── store.py        # Parquet cache (system of record)
└── pipeline.py     # the only IO<->engine glue
```

The strategy logic is **pure functions** with no IO, so it is unit-testable with
fixtures. Parquet is the system of record; CSV is for human-readable export only.

## License

MIT.
