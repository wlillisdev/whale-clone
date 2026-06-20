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
a PASS/FAIL verdict against the five validation gates.

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

## The five validation gates

A strategy is not "real" until it survives all of them (all numbers **after costs**):

1. **Cost-adjusted expectancy** — bootstrap 95% CI lower bound on per-period
   excess return vs the benchmark is `> 0` (block bootstrap for timing strategies).
2. **Walk-forward** — the edge appears in the majority of ≥3 sequential windows,
   with no single window carrying the whole *positive* result.
3. **Robustness** — survives parameter variation (manager count, slippage, cap,
   weighting, lookback) as a *plateau*, not a single spike.
4. **Benchmark-beating** — net CAGR **and** Sharpe both beat buy-and-hold.
5. **Overfitting guard (deflated Sharpe)** — the edge must beat what the *best of
   N tried strategies* would produce by luck (Bailey & López de Prado). This is
   the guard against hunting strategies until one passes by chance; see
   `rigor.py` and `docs/SWOT.md`.

## Current verdict

> **Note:** the recorded figures below were produced *before* a post-audit
> hardening pass (excluding 13F-HR/A amendments, deduping filings, a corrected
> walk-forward share denominator). The headline conclusions are unchanged, but
> the exact numbers should be regenerated with `--refresh`. See `docs/SWOT.md`
> for the full audit and the planned rigor layer (sealed holdout + deflated
> Sharpe). A `CHANGELOG`-style summary of the fixes is in the git history.

**FINAL VERDICT: FAIL — but a robust near-miss.** Cloning the **top-5
highest-conviction positions** of three low-turnover managers over 2014–2024
beat SPY on CAGR *and* Sharpe, held up across time, and survived parameter
variation — **3 of the 4 original gates pass**. It fails the
statistical-significance gate (the 95% CI on excess return dips just below zero),
so we still cannot call the edge *proven*. We hold that line rather than loosen
the gate. (A 5th gate — the deflated-Sharpe overfitting guard — was added in the
hardening pass; a near-miss like this does not clear it either, which only
reinforces the verdict.)

Run on real data (SEC EDGAR 13F history → OpenFIGI tickers → Yahoo adjusted
close), pre-committed managers **Berkshire Hathaway, Gates Foundation Trust,
Pershing Square**, top-5 positions each, 2014–2024:

```
Strategy CAGR +15.06% | Benchmark CAGR +13.30% | Excess +1.76%
Strategy Sharpe 0.88  | Benchmark Sharpe 0.81  | Avg turnover/rebalance 10.4% | Total costs 0.77%
----------------------------------------------------------------
[FAIL] Cost-adjusted expectancy
        Mean daily excess +0.0065%; 95% CI [-0.0073%, +0.0201%]; lower bound <= 0.
[PASS] Walk-forward / out-of-sample
        2/3 windows beat benchmark; max single-window share 54% (limit 70%).
[PASS] Robustness (parameter plateau)
        9/9 variants beat benchmark. Concentration gradient:
        top 3 +2.07% > top 5 +1.76% > top 8 +1.19% > top 10 +1.07%.
[PASS] Benchmark-beating (CAGR & Sharpe)
        CAGR +15.06% vs +13.30%; Sharpe 0.88 vs 0.81 — both beat.
----------------------------------------------------------------
FINAL VERDICT: FAIL — does not clear the gates
```

**How to read this.** Concentration validated the brief's central thesis: the
monotonic gradient (more concentration → more edge) is real, directional, and a
plateau rather than a spike. The strategy now clears three gates that the
full-book clone (+0.85% excess) only partly cleared. But the cost-adjusted
expectancy gate — *is the edge statistically distinguishable from zero after
costs?* — still fails by a hair. That is the difference between *"beat the index
in this sample"* and *"has a provable edge"*, and only the latter is a green
light. A robust, repeatable near-miss is an honest, useful result; it is not a
PASS.

**Data caveats (stated, not hidden):**
- Holdings coverage by reported $ value: Berkshire 96.8%, Gates 99.9%, Pershing
  96.3%. Unmapped lines are mostly options/notes and a few foreign listings.
- A handful of **delisted / acquired** names (e.g. TWTR, VIAB, TMK) have no
  history on free Yahoo and are dropped. These are disproportionately
  merger-arb / acquired positions, so the real strategy's exposure to takeout
  outcomes is under-represented — a known limitation of free price data, not the
  engine.
- Numbers are net of a 7.5 bps/side slippage model; results are reproducible
  (seeded bootstrap). Re-run with `make backtest` to regenerate.

## Gold timing verdict (v2)

**FINAL VERDICT: FAIL — decisively. Timing gold was strictly worse than holding
it.** A pre-committed 12-month time-series momentum signal on GLD (long/flat,
monthly, 5 bps/side), benchmarked against buy-and-hold GLD over 2014–2024:

```
Strategy CAGR +2.30% | Buy-and-hold GLD +7.07% | Excess -4.77%
Strategy Sharpe 0.25 | Benchmark Sharpe 0.56   | Time-in-market 57% | 18 trades
----------------------------------------------------------------
[FAIL] Cost-adjusted expectancy (block bootstrap)
        Mean daily excess -0.0197%; 95% CI [-0.0419%, -0.0029%] — entirely below 0.
[FAIL] Walk-forward            0/3 windows beat buy-and-hold.
[FAIL] Robustness (lookback)   0/5 lookbacks beat (126..378 days all negative).
[FAIL] Benchmark-beating       CAGR 2.30% vs 7.07% and Sharpe 0.25 vs 0.56 — both lose.
----------------------------------------------------------------
FINAL VERDICT: FAIL — does not clear the gates
```

Run with `python -m whale_clone.gold`. This is a *clean* fail (unlike the 13F
near-miss): the timer underperforms at **every** lookback, so it is a robust
negative, not an unlucky configuration. The cause is visible in the
diagnostics — being flat ~43% of the time made it miss gold's strong 2019–20 and
2024 runs while paying costs. The honest lesson, twice over: **a simple
buy-and-hold benchmark is hard to beat after costs.** The expectancy gate here
uses a *block* bootstrap, because a low-turnover timer's daily returns are
autocorrelated and the IID bootstrap would overstate confidence.

## Diversification verdict (v3)

**FINAL VERDICT: FAIL on the edge gate — but a smoother ride.** A diversified
portfolio (SPY 40 / IEF 25 / GLD 15 / DBC 10 / SHY 10, quarterly rebalance,
5 bps/side) vs a 60/40 benchmark, ~2006–2024, judged on *risk-adjusted* terms:

```
Diversified: CAGR +7.16% | vol 8.0% | Sharpe 0.90 | maxDD -15.2%
60/40:       CAGR +8.72% | vol 9.9% | Sharpe 0.89 | maxDD -21.3%
----------------------------------------------------------------
[FAIL] Risk-adjusted edge   Sharpe diff +0.01; 95% CI [-0.21, +0.22] — spans 0.
[PASS] Walk-forward         2/3 windows positive Sharpe edge.
[PASS] Robustness           4/6 weight/timing variants positive.
[PASS] Sharpe & drawdown    Sharpe 0.90 vs 0.89; maxDD -15.2% vs -21.3%.
----------------------------------------------------------------
FINAL VERDICT: FAIL — no statistically significant risk-adjusted edge
```

Run with `python -m whale_clone.allocation`. The honest reading: the diversified
book had **essentially the same Sharpe** as 60/40 (the difference is
indistinguishable from zero), with **lower volatility and a shallower worst
drawdown** but lower return. So it is a *smoother ride at the same risk-adjusted
return*, not a better one — real value for crash-tolerance, but not a provable
edge. (2006–2024 was unusually kind to 60/40; both legs rose for most of it.)

## Scoreboard — three honest verdicts

| Strategy | Verdict | One line |
|----------|---------|----------|
| 13F clone (concentrated top-5) | FAIL (near-miss) | +1.76% excess, fails significance |
| Gold momentum timing | FAIL (decisive) | loses to holding gold at every lookback |
| Diversified vs 60/40 | FAIL (edge), smoother | same Sharpe, lower drawdown — no provable edge |

The repeated lesson, and the point of the project: **a simple buy-and-hold
benchmark is very hard to beat after costs.** The machine told the truth three
times instead of selling a curve fit.

## Holdings tracker (the useful by-product)

Since "beat the market" did not survive honest testing, the same verified EDGAR
pipeline powers something that *is* unambiguously useful and makes no edge claim:
a tracker that reports **what the tracked managers hold now, what they changed
last quarter (NEW / ADD / TRIM / EXIT), and which names they hold in common.**

```bash
python -m whale_clone.tracker            # markdown report to stdout
python -m whale_clone.tracker --csv holdings.csv   # also export current holdings
python -m whale_clone.tracker --demo     # offline sample
```

It's sourced fact (SEC 13F), deterministic, and unit-tested — a reporting tool,
not a probabilistic bet.

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
├── metrics.py      # CAGR, Sharpe, drawdown, IID + block bootstrap CI (pure)
├── gates.py        # the 4 gates -> verdict (pure given data)
├── store.py        # Parquet cache (system of record)
├── pipeline.py     # 13F clone: the only IO<->engine glue
├── signals.py      # gold: momentum/SMA signals + causal monthly targets (pure)
├── signal_backtest.py # gold: single-asset long/flat loop, cash earns rf (pure)
├── gold.py         # gold timing pipeline + adapted gates (block bootstrap)
├── allocation.py   # diversified vs 60/40, risk-adjusted gates (pure engine + glue)
├── rigor.py        # holdout split + deflated-Sharpe overfitting gate (pure)
├── tracker.py      # superinvestor holdings report (current / changes / consensus)
├── paper.py        # quarterly clone signal + forward paper-trade log (no orders)
├── execution.py    # target weights -> orders; guardrails; DryRunBroker (pure/offline)
├── broker_alpaca.py# Alpaca PAPER broker (optional dep; the only networked client)
├── trade.py        # whale-trade CLI: dry-run by default, paper-only execution
├── report.py       # whale-report: publishable HTML + CSV holdings tracker (the product)
└── ml.py           # whale-ml: ML chart-predictor, walk-forward, run through the gates
```

### Can an AI predict trades by "reading the charts"?

Common intuition: AI should crush trading by reading price graphs. `whale-ml`
tests it honestly — a gradient-boosted model on technical features, trained
**strictly walk-forward** (no look-ahead, proven by test), then run through the
same five gates including the deflated-Sharpe overfitting guard.

```bash
python -m whale_clone.ml --demo            # on a random walk: expect NO edge
python -m whale_clone.ml --instrument SPY  # real data
```

On synthetic random-walk data the verdict is the honest lesson in one screen:
**~49.8% directional accuracy (a coin flip), all gates FAIL, and ~26% lost to
trading costs from churn.** Markets are adversarial, near-efficient, and
non-stationary — unlike image recognition, a pattern that works gets arbitraged
away the moment it's exploited. Run it on real SPY and judge for yourself; the
overfitting guard exists precisely to catch a model that looks brilliant
in-sample and is really fitting noise.

### The product: a published holdings tracker

The one model with a proven business in this space is the **information product**,
not running a fund. `whale-report` renders a self-contained HTML page + CSV from
the verified EDGAR data — current holdings, last-quarter changes (new/add/trim/
exit), and cross-manager consensus. Sourced fact, no edge claim.

```bash
python -m whale_clone.report --demo --out site   # offline sample -> site/index.html + holdings.csv
python -m whale_clone.report --out site          # real data
```

`.github/workflows/publish-report.yml` regenerates and publishes it to GitHub
Pages after each quarterly 13F deadline (enable Pages → Source: GitHub Actions).

### Paper execution (no live trading, ever)

`whale-trade` turns the quarterly signal into orders against an Alpaca **paper**
account. Safety is layered and fail-closed:

- **Dry-run by default.** With no flags it previews the order ticket and sends
  nothing. Real submission needs **both** `--paper` and `--execute`, plus an open
  market and a clean guardrail check.
- **Paper-only.** There is no live-broker class in the codebase; trading real
  money would require deliberately writing one. Paper API keys can't touch a
  live account anyway.
- **Fail-closed guardrails:** an empty/NaN target never liquidates the book;
  no-leverage + per-order cap + max-orders limits; a kill-switch
  (`WHALE_EXEC_KILL=1` or a `.halt` file); idempotent client order ids.

```bash
python -m whale_clone.trade --demo          # offline preview, no orders
pip install -e ".[broker]"                   # add the Alpaca SDK
export ALPACA_API_KEY=... ALPACA_SECRET_KEY=...   # PAPER keys
python -m whale_clone.trade --paper                # connect to paper, still preview
python -m whale_clone.trade --paper --execute      # actually place paper orders
```

The whole engine is unit-tested offline through `DryRunBroker` — no network, no
money. See `docs/PLAYBOOK.md` for how this fits the core/satellite plan.

**Putting it to work:** `docs/PLAYBOOK.md` is the calculated core/satellite plan
(broad equity core + the concentrated clone as an *earned* satellite). The clone
runs hands-off via `python -m whale_clone.paper` (`whale-signal`) and a scheduled
GitHub Action (`.github/workflows/quarterly-signal.yml`) that emits each
quarter's trade ticket and appends to a forward paper-trade log — building real
out-of-sample evidence with zero capital at risk. No orders are ever placed.

The strategy logic is **pure functions** with no IO, so it is unit-testable with
fixtures. Parquet is the system of record; CSV is for human-readable export only.

## License

MIT.
