# whale-clone — SWOT analysis

A grounded, honest assessment produced from a multi-agent review of the codebase,
tests, and results. The central achievement of this project is *not* a profitable
strategy — all three it tested failed the gates — but a **machine that refuses to
lie**, plus a genuinely useful by-product (the holdings tracker).

## Strengths

- **Honesty enforced in code, not prose.** Three strategies tested, three FAILs
  reported plainly. The expectancy gate is a hard CI-lower-bound-above-zero test
  with no escape hatch; parameters are pre-committed in `config.py`.
- **No-look-ahead is proven, not asserted.** The 13F engine acts on the real
  filing date; signal causality is a single `shift(1)`. Each engine now has a
  test that perturbs a *future* bar and asserts the past is byte-identical
  (13F, signal/gold, and — added in hardening — allocation).
- **Clean architecture.** Pure-function engine (`metrics`, `costs`, `portfolio`,
  `backtest`, `signals`, `signal_backtest`, `gates`); IO isolated to the glue
  modules (`pipeline`, `gold`, `allocation`, `tracker`). Unit-testable with tiny
  fixtures.
- **Above-hobby-grade statistics.** Costs charged on traded notional both legs,
  all numbers net. IID *and* moving-block bootstrap, with the timer correctly
  using the block version (autocorrelated returns).
- **Plateau-not-spike discipline.** Robustness varies parameters; walk-forward
  caps any single window's share of the positive excess.
- **A real, shippable by-product.** The tracker reports current holdings,
  NEW/ADD/TRIM/EXIT changes, and cross-manager consensus — sourced fact, no edge
  claim, unit-tested.
- **Operational hygiene.** Seeded/reproducible, Parquet cache, CI runs
  ruff + mypy(strict) + pytest on 3.11/3.12 plus an offline demo smoke test.

## Weaknesses

- **Free-data survivorship/delisting gap.** Delisted/acquired names (TWTR, VIAB,
  TMK) have no free history and are dropped; the engine treats a missing daily
  return as 0.0 ("holds flat"), which is not perfectly neutral. Documented, not
  fully corrected.
- **No deflated-Sharpe / multiple-testing correction yet.** Several strategies
  have been tested against the same data; the 13F "near-miss" is exactly where
  multiple-comparison inflation matters. `out_of_sample_fraction` is defined but
  not yet wired into a sealed holdout.
- **Tiny, hardcoded manager universe.** Three managers with CIKs in source;
  adding one means editing code. A thin basis for any statistical claim.
- **US-only, single asset-class lens.** SEC 13F + a US-equity ticker filter;
  international exposure is silently dropped.
- **Coverage-driven renormalisation.** Unmapped lines (options/notes/foreign,
  ~1–4% by value) are dropped and weights renormalised over the rest.
- **No live/paper execution, no market-impact or tax modelling.** A flat linear
  bps cost model — fine for a research verdict, a wall for "could I run this."

## Opportunities

- **Lead with the tracker.** A scheduled (post-13F-deadline) GitHub Action that
  regenerates the report/CSV is a small lift and genuinely useful.
- **Build the rigor layer the project already gestures at:** a sealed holdout +
  deflated-Sharpe / multiple-testing gate that knows how many strategies were
  tried. This is the single highest-leverage move for the core thesis.
- **Broaden the manager set via a data-driven registry** (CIKs in config/CSV) —
  paired with the multiple-testing guard so it doesn't become a strategy hunt.
- **Paper-trading / forward log** for the 13F near-miss — the only fully honest
  way to learn whether it is signal or noise.
- **Unify the gate machinery** so the three strategies share one parameterised
  gate suite instead of three drifting implementations.

## Threats

- **Data-source fragility / ToS.** The empirical pipeline depends on unofficial
  free endpoints (Yahoo chart JSON, throttled Stooq, Dataroma scraping, OpenFIGI
  free tier). Any format/ToS change degrades or breaks the real verdict.
- **Look-ahead creeping back in.** The invariant is airtight today but spans
  several modules; every new strategy needs its own no-look-ahead proof.
- **Overfitting by attrition.** The biggest threat is the project's own
  momentum: hunting strategy #4, #5, … until one clears the gate by luck. Build
  the multiple-testing guard *before* testing more.
- **Overselling.** The 13F near-miss is one selective paragraph from being
  marketed as a winner. The honest framing is load-bearing and fragile.
- **Reproducibility decay.** Upstream adjusted-close series get silently
  re-adjusted over time, so exact figures aren't guaranteed reproducible even
  with a fixed seed (the seed only fixes the bootstrap).

## Verdict

Merge it — the engineering and the honesty are the asset. But hold the framing:
this is **a rigor framework that has correctly rejected three strategies**, not a
strategy that is "almost working." The danger isn't the code; it's the near-miss
tempting a hunt for strategy #4 without first building the multiple-testing
guard. Build the guard first.
