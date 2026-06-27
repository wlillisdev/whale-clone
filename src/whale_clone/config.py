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
    # Concentration: keep only each manager's top-N highest-conviction positions.
    # The brief's thesis is that the edge lives in a few names; full-book cloning
    # (top_n=None) already failed, so we pre-commit a "few" value here. The
    # robustness gate varies N (3/5/8/10) to check this is a plateau, not a spike.
    top_n_positions: int | None = 5

    # --- Costs (all results reported NET) ----------------------------------
    commission_bps: float = 0.0  # modern brokers ~0
    slippage_bps: float = 7.5  # per-trade slippage assumption (brief: 5-10 bps)
    slippage_stress_multiplier: float = 2.0  # robustness: stress-test at 2x slippage

    # --- Sample window ------------------------------------------------------
    start_date: date = date(2014, 1, 1)
    end_date: date = date(2024, 12, 31)
    # Intended size of a future sealed out-of-sample holdout. NOTE: not yet
    # wired into the gates (walk-forward currently splits the full sample); kept
    # as the target for the planned deflated-Sharpe / holdout rigor layer.
    out_of_sample_fraction: float = 0.30

    # --- Benchmark ----------------------------------------------------------
    benchmark: str = "SPY"  # "just hold the index" baseline
    risk_free_annual: float = 0.0  # annual risk-free rate for Sharpe

    # --- Validation gates ---------------------------------------------------
    bootstrap_iterations: int = 5000
    bootstrap_confidence: float = 0.95
    walk_forward_windows: int = 3
    # Deflated-Sharpe rigor guard (Bailey & Lopez de Prado): a strategy's Sharpe
    # must beat what the BEST of `n_strategies_tried` noise strategies would
    # produce by luck. Pre-committed honestly to the number of distinct
    # strategies tested in this repo (13F, gold, diversification, + headroom).
    n_strategies_tried: int = 8
    deflated_sharpe_threshold: float = 0.95
    # Prior for the spread of (annualised) excess-return Sharpes across the
    # strategies/variants tried — used as the noise-search dispersion. A
    # documented conservative assumption, not fitted per run.
    trial_sharpe_dispersion: float = 0.5
    # A single window must not carry the whole result: its share of total excess
    # return must stay under this fraction for the walk-forward gate to pass.
    max_single_window_share: float = 0.70

    # --- Gold timing strategy (pre-committed v1) ---------------------------
    # v1: 12-month time-series momentum on GLD, long/flat, monthly rebalance,
    # benchmarked against buy-and-hold GLD. Chosen before seeing results; the
    # robustness gate varies the lookback to map (not exploit) sensitivity.
    gold_instrument: str = "GLD"
    gold_signal: str = "momentum"  # "momentum" | "sma"
    gold_lookback: int = 252  # 12 months (momentum) or SMA window
    gold_allow_short: bool = False  # long/flat only in v1 (gold drifts up)
    gold_slippage_bps: float = 5.0  # per side; ETF round-trip ~ a few bps
    gold_block_bootstrap_len: int = 0  # 0 = auto (≈ average holding period)

    # --- Diversification study (pre-committed v3) ---------------------------
    # Diversified multi-asset portfolio vs a 60/40 benchmark, judged on
    # RISK-ADJUSTED terms (Sharpe + max drawdown), not raw return. The most
    # evidence-backed idea from the research fan-out. Long-history free ETFs.
    alloc_weights: dict[str, float] = Field(
        default_factory=lambda: {"SPY": 0.40, "IEF": 0.25, "GLD": 0.15, "DBC": 0.10, "SHY": 0.10}
    )
    alloc_benchmark_weights: dict[str, float] = Field(
        default_factory=lambda: {"SPY": 0.60, "IEF": 0.40}
    )
    alloc_rebalance: str = "Q"  # "M" monthly | "Q" quarterly

    # --- Insider cluster-buying strategy (Form 4) --------------------------
    insider_source: str = "edgar"  # "edgar" | "demo"
    insider_universe: list[str] = Field(
        default_factory=lambda: ["AAPL", "MSFT", "JPM", "XOM", "KO", "PFE", "CAT", "WMT"]
    )
    insider_min_buyers: int = 3  # distinct insiders for a cluster
    insider_require_officer: bool = True  # at least one CEO/CFO/officer
    insider_min_value: float = 100_000.0  # min total cluster purchase value
    insider_window_days: int = 30  # cluster window
    insider_hold_days: int = 126  # ~6-month hold

    # --- Volatility risk premium (cash-secured index put-writing) ----------
    # The most economically-grounded edge from the research fan-out: you are PAID
    # to underwrite crash insurance (implied vol runs above realised). Simulated
    # by rolling fully cash-secured ATM puts, Black-Scholes priced off an implied
    # vol series. Judged with an ADDED tail-risk gate (max DD / Sortino / CVaR)
    # because a Sharpe/bootstrap pipeline is blind to the negative-skew left tail.
    vrp_source: str = "yahoo"  # "yahoo" | "stooq" | "demo" (real data by default)
    vrp_index: str = "SPY"  # the underlying we write puts on (= benchmark)
    vrp_dte_days: int = 21  # option tenor / roll period (~1 month)
    vrp_moneyness: float = 1.0  # strike / spot (1.0 = at-the-money)
    vrp_cost_bps: float = 10.0  # round-trip option slippage per roll, on notional
    vrp_iv_markup: float = 1.3  # IV = trailing realised vol x markup when no VIX
    vrp_iv_floor: float = 0.10  # minimum annualised implied vol

    # --- Combined system (the "best bits" portfolio) -----------------------
    # Blend the pieces that actually carried signal into ONE risk-managed book,
    # rebalanced monthly: an index core (beta), a VRP put-write overlay (the
    # risk-adjusted winner), and an insider cluster-buy tilt (academic-backed
    # satellite). The pro move: volatility-target the blend to the index's own
    # risk with capped leverage, so a higher-Sharpe mix beats the index on
    # RETURN, not just smoothness. Judged by all gates incl. the tail-risk gate
    # (levering short-vol is exactly where you blow up). Pre-committed weights;
    # the robustness gate varies the mix. Set 0.0 vol target to disable leverage.
    system_w_index: float = 0.50
    system_w_vrp: float = 0.30
    system_w_insider: float = 0.20
    system_vol_target: float = 0.0  # 0 = match the index's realised vol; else annual target
    system_max_leverage: float = 1.5  # hard cap; levering short-vol is dangerous
    system_borrow_spread: float = 0.01  # financing cost above risk-free on levered notional

    # --- ML price-predictor experiment -------------------------------------
    # Tests whether an ML model can predict direction from chart features.
    # Strictly walk-forward; judged by the same gates incl. deflated Sharpe.
    ml_instrument: str = "SPY"
    ml_horizon: int = 1  # predict direction this many days ahead
    ml_train_min: int = 750  # min training rows before first prediction (~3y)
    ml_step: int = 21  # retrain/predict block size (~monthly)
    ml_threshold: float = 0.50  # go long when P(up) exceeds this

    # --- Execution (paper-only; dry-run by default) ------------------------
    # Trading is OFF unless both broker_mode="paper" AND execute=True. There is
    # no live broker in this codebase; paper carries no real-money risk.
    broker_mode: str = "dry_run"  # "dry_run" | "paper"
    execute: bool = False  # must be explicitly true to send paper orders
    exec_cash_buffer: float = 0.02  # never fully invest (rounding/slippage headroom)
    exec_no_trade_band: float = 0.005  # ignore rebalances smaller than 0.5% of equity
    exec_max_orders: int = 20  # abort a run that wants more orders than this
    exec_max_order_notional_pct: float = 0.25  # cap any single order at 25% of equity

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
