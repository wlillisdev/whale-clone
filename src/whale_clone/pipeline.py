"""Glue: load data -> backtest -> evaluate gates -> verdict.

This is the one place IO (network/cache) meets the pure engine. Everything it
calls downstream is pure and unit-tested; this module just wires it together.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .backtest import BacktestConfig
from .config import Settings
from .costs import CostModel
from .data.holdings import load_holdings
from .data.prices import load_prices
from .gates import GateConfig, Verdict, evaluate_gates
from .store import Store


@dataclass
class PipelineData:
    holdings: pd.DataFrame
    prices: pd.DataFrame


def backtest_config(settings: Settings) -> BacktestConfig:
    return BacktestConfig(
        benchmark=settings.benchmark,
        weighting=settings.weighting,
        max_position_weight=settings.max_position_weight,
        cost_model=CostModel(
            commission_bps=settings.commission_bps,
            slippage_bps=settings.slippage_bps,
        ),
        risk_free_annual=settings.risk_free_annual,
        trading_days_per_year=settings.trading_days_per_year,
    )


def gate_config(settings: Settings) -> GateConfig:
    return GateConfig(
        bootstrap_iterations=settings.bootstrap_iterations,
        bootstrap_confidence=settings.bootstrap_confidence,
        walk_forward_windows=settings.walk_forward_windows,
        max_single_window_share=settings.max_single_window_share,
        random_seed=settings.random_seed,
        trading_days_per_year=settings.trading_days_per_year,
    )


def load_data(settings: Settings, *, refresh: bool = False) -> PipelineData:
    """Load holdings then prices for exactly the tickers held."""
    store = Store(settings.cache_dir)
    holdings = load_holdings(
        settings.managers,
        source=settings.holdings_source,
        start=settings.start_date,
        end=settings.end_date,
        store=store,
        refresh=refresh,
        seed=settings.random_seed,
        openfigi_key=settings.openfigi_api_key,
    )
    tickers = sorted(holdings["ticker"].unique())
    prices = load_prices(
        tickers,
        start=settings.start_date,
        end=settings.end_date,
        source=settings.price_source,
        benchmark=settings.benchmark,
        store=store,
        refresh=refresh,
        seed=settings.random_seed,
    )
    return PipelineData(holdings=holdings, prices=prices)


def run(settings: Settings, *, refresh: bool = False) -> Verdict:
    data = load_data(settings, refresh=refresh)
    return evaluate_gates(
        data.holdings,
        data.prices,
        backtest_config(settings),
        gate_config(settings),
    )
