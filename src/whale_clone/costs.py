"""Transaction cost model (pure).

Costs are charged on the **notional actually traded** at a rebalance. Slippage
and commission are paid on every leg — both the sells and the buys — so the
cost base is the gross sum of absolute weight changes, ``traded_notional``.
"Turnover" is reported one-way (half of that), the usual convention.

Every number this project reports is net of these costs; gross numbers are
never reported on their own (brief, section 5).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Linear cost model in basis points of traded notional.

    ``commission_bps`` and ``slippage_bps`` are charged per unit of traded
    notional. A rebalance that trades a fraction ``f`` of the book (sells +
    buys) costs ``f * (commission + slippage) bps``.
    """

    commission_bps: float = 0.0
    slippage_bps: float = 7.5

    @property
    def total_bps(self) -> float:
        return self.commission_bps + self.slippage_bps

    def cost_for_traded(self, traded_fraction: float) -> float:
        """Cost (fraction of portfolio value) for trading ``traded_fraction``.

        ``traded_fraction`` is gross notional traded / portfolio value: 0 means
        no trading; buying a full book from cash is 1.0; a fully-invested book
        that completely swaps its names is 2.0 (sell 1.0 + buy 1.0).
        """
        if traded_fraction < 0:
            raise ValueError("traded fraction cannot be negative")
        return traded_fraction * self.total_bps / 10_000.0


def traded_notional(prev_weights: dict[str, float], new_weights: dict[str, float]) -> float:
    """Gross notional traded (sum of absolute weight changes), both legs.

    Buying a brand-new 100% book from cash trades 1.0. Each unit of this pays
    the cost model — selling X and buying Y are two separate trades.
    """
    tickers = set(prev_weights) | set(new_weights)
    return sum(abs(new_weights.get(t, 0.0) - prev_weights.get(t, 0.0)) for t in tickers)


def one_way_turnover(prev_weights: dict[str, float], new_weights: dict[str, float]) -> float:
    """One-way turnover (half the gross), the conventional reporting figure."""
    return traded_notional(prev_weights, new_weights) / 2.0
