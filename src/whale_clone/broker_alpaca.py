"""Alpaca **paper** broker — the only networked execution client.

Isolated so the core never imports ``alpaca``: install with the optional extra
``pip install -e ".[broker]"``. This class talks exclusively to Alpaca's paper
endpoint (``paper=True`` -> paper-api.alpaca.markets); paper API keys are not
authorised for a live account, so no real money is ever at risk. There is
deliberately **no live broker** in this package.

Credentials come from the environment: ``ALPACA_API_KEY`` / ``ALPACA_SECRET_KEY``
(from your Alpaca *Paper* dashboard).

Not exercised by the default test suite (it needs network + the SDK); the engine
is fully tested through :class:`whale_clone.execution.DryRunBroker`.
"""

from __future__ import annotations

import os

from .execution import Account, ExecutionClient, OrderIntent, OrderResult, Position


class LiveAccountError(RuntimeError):
    """Raised if the connected account is not a paper account."""


class AlpacaPaperBroker(ExecutionClient):
    def __init__(self, *, execute: bool = False) -> None:
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.trading.client import TradingClient
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "alpaca-py is required for live paper trading: pip install -e '.[broker]'"
            ) from exc

        key = os.environ.get("ALPACA_API_KEY")
        secret = os.environ.get("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise RuntimeError("set ALPACA_API_KEY and ALPACA_SECRET_KEY (paper keys)")

        # paper=True hard-routes to the paper endpoint. Never read this from config.
        self._trading = TradingClient(key, secret, paper=True)
        self._data = StockHistoricalDataClient(key, secret)
        self._execute = execute

        acct = self._trading.get_account()
        # Defence in depth: refuse to proceed unless this really is paper.
        if not bool(getattr(acct, "is_paper", False)):
            raise LiveAccountError("connected account is not a paper account — aborting")

    @property
    def is_paper(self) -> bool:
        return True

    def get_account(self) -> Account:
        a = self._trading.get_account()
        return Account(
            equity=float(a.equity),
            cash=float(a.cash),
            buying_power=float(a.buying_power),
            blocked=bool(a.trading_blocked or a.account_blocked),
            is_paper=bool(getattr(a, "is_paper", True)),
        )

    def get_positions(self) -> list[Position]:
        out: list[Position] = []
        for p in self._trading.get_all_positions():
            out.append(Position(p.symbol, float(p.qty), float(p.market_value)))
        return out

    def is_market_open(self) -> bool:
        return bool(self._trading.get_clock().is_open)

    def submit(self, intent: OrderIntent) -> OrderResult:
        if not self._execute:
            return OrderResult(
                intent.client_order_id,
                intent.symbol,
                intent.side,
                "skipped",
                intent.notional,
                "execute flag off",
            )
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        side = OrderSide.BUY if intent.side == "buy" else OrderSide.SELL
        req = MarketOrderRequest(
            symbol=intent.symbol,
            notional=round(intent.notional, 2),
            side=side,
            time_in_force=TimeInForce.DAY,
            client_order_id=intent.client_order_id,
        )
        try:
            o = self._trading.submit_order(req)
        except Exception as exc:  # duplicate client_order_id (422) == already done
            msg = str(exc)
            status = "submitted" if "client_order_id" in msg or "422" in msg else "rejected"
            return OrderResult(
                intent.client_order_id,
                intent.symbol,
                intent.side,
                status,
                intent.notional,
                msg[:80],
            )
        return OrderResult(
            intent.client_order_id,
            intent.symbol,
            intent.side,
            "submitted",
            intent.notional,
            str(getattr(o, "status", "")),
        )
