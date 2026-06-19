# The Playbook — a calculated plan to actually make money

Profile this is written for: **moderate risk (can stomach a 20–30% drawdown),
meaningful capital, multi-year horizon.** That profile is the *best* setup for
honest returns — long horizons are the only place a calculated edge pays.

This is not a get-rich system. It is the highest expected-value plan consistent
with what this project *proved* (free-data systematic alpha is hard) and with how
money is actually made over multi-year horizons.

## The one idea
Returns come from exactly three sources: **risk, an edge, or leverage.** We use
the first (deliberately), pursue the second only after it earns trust, and avoid
the third. Everything below follows from that.

## Core / satellite

**Core (~85–90%) — the ROI engine.** Broad, low-cost equity, held for years.
This is where the compounding lives: ~7–10%/yr historically, doubling roughly
every 8 years. It needs no validation — it is just owning the market cheaply and
not selling in a panic. At a 20–30% drawdown tolerance and a multi-year horizon,
an equity-heavy core is the correct, boring, highest-probability choice.

**Satellite (~10–15%) — the edge, earned not assumed.** The concentrated 13F
clone (top-5 highest-conviction names of low-turnover managers). It is our one
mechanism-backed lead — a *robust near-miss* (+1.76% excess, 3/4 gates, clean
concentration gradient). It does **not** get real capital until it proves itself
forward, out-of-sample. Even if it works, expect ~1–2%/yr added — real over a
decade (~20% more terminal wealth), but the topping, not the cake.

## The staged shot (capital-preserving)
1. **Deploy the core now.** No reason to wait — it is market beta, cheaply held.
2. **Run the satellite in paper mode** (`whale-signal`, automated quarterly). It
   emits the exact trades and logs a forward track record, risking nothing.
3. **Validate before funding it:** re-check it through the deflated-Sharpe +
   sealed-holdout guard, and watch the paper log for ~3–4 quarters. Only if it
   holds up out-of-sample does it graduate to the satellite sleeve, sized small.
4. **Never** lever an unvalidated edge, chase yield products, or auto-fire live
   orders on a strategy that hasn't earned it. That is how calculated turns into
   broke.

## Automation (what and why)
- **Automated now (zero risk):** the quarterly signal + paper-trade log
  (`.github/workflows/quarterly-signal.yml`). It runs after each 13F deadline,
  produces the trade ticket, and commits the growing track record. This removes
  the #1 destroyer of returns — undisciplined human behavior — from the loop.
- **Manual execution is fine here.** The clone is quarterly and low-turnover
  (~a handful of trades a year). Auto-firing orders adds operational risk for
  almost no benefit; automate the *decision*, click the *button* yourself.
- **Later, optional:** wire a broker *paper* account (e.g. Alpaca) to dry-run
  execution before any live trading.

## Honest expectations
- The core does ~85–90% of the work. The satellite is a small, validated tilt.
- A multi-year, moderate-risk plan compounding equity returns is how "big ROI"
  actually accrues — through time and discipline, not a secret signal.
- If the satellite never validates, you still win: a disciplined, low-cost,
  long-horizon core beats the large majority of active investors after fees.

The sniper's discipline: one core position (the market), one earned satellite
(the clone), no wild shots. Patience is the edge.
