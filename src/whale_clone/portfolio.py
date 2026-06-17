"""Build clone target weights from 13F holdings (pure functions).

Input holdings are in long form with one row per (manager, period, ticker):

    manager | period | filing_date | ticker | value

``filing_date`` is the date the holding became *public*. Nothing in this module
ever looks at a date later than the one it is asked about — that is enforced by
``holdings_known_on`` and tested in ``test_backtest_no_lookahead``.
"""

from __future__ import annotations

import pandas as pd

HOLDINGS_COLUMNS = ["manager", "period", "filing_date", "ticker", "value"]


def validate_holdings(holdings: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalise a holdings frame; return a clean copy."""
    missing = set(HOLDINGS_COLUMNS) - set(holdings.columns)
    if missing:
        raise ValueError(f"holdings frame missing columns: {sorted(missing)}")
    out = holdings.loc[:, HOLDINGS_COLUMNS].copy()
    out["filing_date"] = pd.to_datetime(out["filing_date"])
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out["value"] = out["value"].astype(float)
    out = out[out["value"] > 0]
    return out.reset_index(drop=True)


def filing_dates(holdings: pd.DataFrame) -> list[pd.Timestamp]:
    """Sorted unique filing dates — the only dates we are allowed to act on."""
    return sorted(pd.to_datetime(holdings["filing_date"]).unique())


def holdings_known_on(holdings: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """The most recent filing *per manager* whose filing_date <= ``as_of``.

    This is the no-look-ahead core: on ``as_of`` we may only see filings already
    public. For each manager we take their latest such filing (one period).
    """
    visible = holdings[holdings["filing_date"] <= as_of]
    if visible.empty:
        return visible.iloc[0:0]
    # Latest filing_date per manager, then that whole filing's rows.
    latest = visible.groupby("manager")["filing_date"].max().rename("latest")
    merged = visible.merge(latest, on="manager")
    return merged[merged["filing_date"] == merged["latest"]].drop(columns="latest")


def target_weights(
    visible_holdings: pd.DataFrame,
    *,
    weighting: str = "value",
    max_position_weight: float = 0.25,
) -> dict[str, float]:
    """Combine visible per-manager holdings into capped target weights.

    Managers are pooled with equal weight (each manager contributes the same
    total budget), so one large manager does not dominate. Within a manager,
    ``weighting`` chooses value-proportional or equal weights. Single names are
    capped at ``max_position_weight`` and the book is renormalised to sum to 1.
    """
    if visible_holdings.empty:
        return {}
    if weighting not in {"value", "equal"}:
        raise ValueError(f"unknown weighting: {weighting!r}")

    df = visible_holdings.copy()
    if weighting == "equal":
        df["w_in_mgr"] = df.groupby("manager")["ticker"].transform(lambda s: 1.0 / len(s))
    else:
        totals = df.groupby("manager")["value"].transform("sum")
        df["w_in_mgr"] = df["value"] / totals

    # Equal weight across managers: divide each manager's internal weights by
    # the number of managers, then sum across managers per ticker.
    n_managers = df["manager"].nunique()
    df["w"] = df["w_in_mgr"] / n_managers
    grouped = df.groupby("ticker")["w"].sum()
    weights = {str(ticker): float(w) for ticker, w in grouped.items()}

    return _cap_and_normalise(weights, max_position_weight)


def _cap_and_normalise(weights: dict[str, float], cap: float) -> dict[str, float]:
    """Cap each weight at ``cap`` and renormalise so the book sums to 1.

    Capping is applied iteratively: capped names are frozen and the excess is
    redistributed proportionally across the uncapped names until no name exceeds
    the cap (or everything is capped).
    """
    if not weights:
        return {}
    total = sum(weights.values())
    if total <= 0:
        return {}
    w = {k: v / total for k, v in weights.items()}
    if cap >= 1.0:
        return w

    frozen: dict[str, float] = {}
    while True:
        over = {k: v for k, v in w.items() if k not in frozen and v > cap + 1e-12}
        if not over:
            break
        for k in over:
            frozen[k] = cap
        free = {k: v for k, v in w.items() if k not in frozen}
        budget_left = 1.0 - sum(frozen.values())
        free_total = sum(free.values())
        if free_total <= 0 or budget_left <= 0:
            # Everything hit the cap; distribute the remaining budget evenly.
            n = len(w)
            return dict.fromkeys(w, 1.0 / n)
        w = {**frozen, **{k: free[k] / free_total * budget_left for k in free}}
    return w
