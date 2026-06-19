"""Anti-overfitting rigor layer (pure).

Two guards recommended by the multi-agent protocol review:

* :func:`holdout_split` — a sealed train/holdout split by date, so a verdict can
  be confirmed on data that never influenced any choice.
* :func:`deflated_sharpe_gate` — a Deflated-Sharpe gate (Bailey & López de
  Prado) that discounts a strategy's Sharpe by the expected best-of-N-trials a
  pure-noise search would produce. This is what makes testing many strategies
  *cost* something, so a "PASS" is not just the luckiest of many tries.

These are pure functions over plain pandas/dataclasses, unit-tested offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .gates import GateResult
from .metrics import deflated_sharpe, expected_max_sharpe


def holdout_split(
    index: pd.DatetimeIndex, *, holdout_fraction: float = 0.30
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return ``(train_end, holdout_start)`` cut points for a chronological split.

    The most recent ``holdout_fraction`` of the timeline is the sealed holdout.
    ``train_end`` is the last train day; ``holdout_start`` is the first holdout
    day (the next index entry).
    """
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in (0, 1)")
    cal = pd.DatetimeIndex(index).sort_values()
    if len(cal) < 2:
        raise ValueError("need at least two dates to split")
    cut = int(len(cal) * (1.0 - holdout_fraction))
    cut = min(max(cut, 1), len(cal) - 1)
    return cal[cut - 1], cal[cut]


def deflated_sharpe_gate(
    returns: pd.Series,
    *,
    n_strategies_tried: int,
    trials_sr_std: float,
    threshold: float = 0.95,
    periods_per_year: int = 252,
) -> GateResult:
    """Deflated-Sharpe gate: does the true Sharpe beat the best-of-N noise hurdle?

    ``trials_sr_std`` is the spread of Sharpe ratios across the configurations we
    tried (estimated from the robustness variants). Passes only if the Deflated
    Sharpe exceeds ``threshold``.
    """
    dsr = deflated_sharpe(
        returns,
        n_trials=n_strategies_tried,
        trials_sr_std=trials_sr_std,
        periods_per_year=periods_per_year,
    )
    hurdle = expected_max_sharpe(
        n_strategies_tried, sr_std=trials_sr_std, periods_per_year=periods_per_year
    )
    passed = bool(dsr > threshold and not np.isnan(dsr))
    detail = (
        f"Deflated Sharpe {dsr:.2f} vs threshold {threshold:.2f} "
        f"(tried ~{n_strategies_tried} strategies; noise Sharpe hurdle "
        f"{hurdle:.2f}, trial Sharpe spread {trials_sr_std:.2f})."
    )
    return GateResult("Overfitting guard (deflated Sharpe)", passed, detail, {"dsr": dsr})
