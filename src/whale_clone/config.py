"""Typed, env-driven configuration.

Every knob that could be used to overfit the strategy lives here and is
**pre-committed** (see the build brief, section 10). Change these only with
eyes open: tuning them to sneak past a validation gate is the overfitting trap
the whole project exists to avoid.

Settings load from environment variables prefixed ``WHALE_`` (or a ``.env``
file), so a run is reproducible and self-documenting.
"""

from __future__ import annotations

from datetime import date

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Pre-committed strategy + engine parameters.

    The defaults here ARE the strategy. They were chosen from the brief before
    looking at any results.
    """

    model_config = SettingsConfigDict(
        env_prefix="WHALE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Managers (small, concentrated, low-turnover value) -----------------
    # Registry keys (see data/holdings.py MANAGER_REGISTRY) or raw SEC CIKs.
    # Pre-commit the list; do not cherry-pick winners.
    #   berkshire        = Berkshire Hathaway (Warren Buffett)   — the documented case
    #   gates_foundation = Gates Foundation Trust                — very low turnover
    #   pershing_square  = Pershing Square (Bill Ackman)         — concentrated
    managers: list[str] = Field(
        default_factory=lambda: ["berkshire", "gates_foundation", "pershing_square"]
    )

    # --- Rebalance + weighting ---------------------------------------------
    # We act on the actual 13F *filing date*, never the quarter-end. Acting on
    # quarter-end prices is look-ahead bias and fakes an edge.
    max_position_weight: float = 0.25  # cap any single name to avoid one-stock domination
    weighting: str = "value"  # "value" (manager's own $ weights) or "equal"

    # --- Costs (all results reported NET) ----------------------------------
    commission_bps: float = 0.0  # modern brokers ~0
    slippage_bps: float = 7.5  # per-trade slippage assumption (brief: 5-10 bps)
    slippage_stress_multiplier: float = 2.0  # robustness: stress-test at 2x slippage

    # --- Sample window ------------------------------------------------------
    start_date: date = date(2014, 1, 1)
    end_date: date = date(2024, 12, 31)
    # Reserve the most recent fraction as out-of-sample (untouched until the end).
    out_of_sample_fraction: float = 0.30

    # --- Benchmark ----------------------------------------------------------
    benchmark: str = "SPY"  # "just hold the index" baseline
    risk_free_annual: float = 0.0  # annual risk-free rate for Sharpe

    # --- Validation gates ---------------------------------------------------
    bootstrap_iterations: int = 5000
    bootstrap_confidence: float = 0.95
    walk_forward_windows: int = 3
    # A single window must not carry the whole result: its share of total excess
    # return must stay under this fraction for the walk-forward gate to pass.
    max_single_window_share: float = 0.70

    # --- Data / engine ------------------------------------------------------
    price_source: str = "yahoo"  # "yahoo" | "stooq" | "demo" (yahoo is sturdiest)
    holdings_source: str = "edgar"  # "edgar" (authoritative) | "dataroma" | "demo"
    # Optional OpenFIGI API key (CUSIP->ticker). Without it the free tier limit
    # (~25 req/min) is used; set WHALE_OPENFIGI_API_KEY to go faster.
    openfigi_api_key: str | None = None
    cache_dir: str = ".cache"
    random_seed: int = 1234
    trading_days_per_year: int = 252

    @field_validator("managers")
    @classmethod
    def _non_empty_managers(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("at least one manager must be configured")
        return v

    @field_validator("out_of_sample_fraction")
    @classmethod
    def _valid_oos(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("out_of_sample_fraction must be in (0, 1)")
        return v


def load_settings(**overrides: object) -> Settings:
    """Load settings from env/.env, with optional explicit overrides."""
    return Settings(**overrides)  # type: ignore[arg-type]
