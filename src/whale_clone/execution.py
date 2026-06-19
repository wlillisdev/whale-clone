"""Execution layer: turn target weights into orders — paper-only, fail-closed.

Design principles (do not weaken):
* **Dry-run by default.** Nothing is ever submitted unless the caller explicitly
  opts in (``mode="paper"`` AND ``execute=True``). The scheduled/CI path never
  trades.
* **Paper-only.** There is no live-broker class in this codebase. Trading real
  money would require deliberately writing one — it cannot happen by flipping a
  flag.
* **Fail-closed.** An empty/NaN target never liquidates the book; any guardrail
  violation aborts the whole run and submits nothing.
* **Broker-agnostic + offline-testable.** The engine talks to an
  :class:`ExecutionClient` protocol; :class:`DryRunBroker` implements it with no
  network so the full path is unit-tested. The Alpaca wrapper lives in
  ``broker_alpaca.py`` behind the same protocol.

The order-planning math is pure (no IO) and mirrors the weight-diff logic in
``paper.compute_trades`` — now expressed in dollars.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

_KILL_ENV = "WHALE_EXEC_KILL"
_KILL_FILE = ".halt"


# --------------------------------------------------------------------------- #
# Value objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Account:
    equity: float
    cash: float
    buying_power: float
    blocked: bool
    is_paper: bool


@dataclass(frozen=True)
class Position:
    symbol: str
    qty: float
    market_value: float


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: str  # "buy" | "sell"
    notional: float  # positive dollar amount
    client_order_id: str
    reason: str


@dataclass(frozen=True)
class OrderResult:
    client_order_id: str
    symbol: str
    side: str
    status: str  # dry_run | submitted | filled | skipped | rejected
    notional: float = 0.0
    detail: str = ""


@dataclass
class ExecReport:
    submitted: bool
    intents: list[OrderIntent]
    results: list[OrderResult]
    violations: list[str] = field(default_factory=list)
    drift: dict[str, float] = field(default_factory=dict)
    note: str = ""

    def render(self) -> str:
        lines = ["=" * 60, "WHALE-CLONE — EXECUTION", "=" * 60]
        lines.append(f"Mode: {'SUBMIT' if self.submitted else 'DRY-RUN (no orders sent)'}")
        if self.note:
            lines.append(self.note)
        if self.violations:
            lines.append("ABORTED — guardrail violations:")
            lines.extend(f"  - {v}" for v in self.violations)
        lines.append("-" * 60)
        if not self.intents:
            lines.append("No orders (within no-trade band, or fail-closed).")
        else:
            for r in self.results:
                lines.append(
                    f"  [{r.status:<9}] {r.side:<4} {r.symbol:<8} ${r.notional:,.2f} {r.detail}"
                )
        lines.append("=" * 60)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Broker protocol + offline dry-run implementation
# --------------------------------------------------------------------------- #
@runtime_checkable
class ExecutionClient(Protocol):
    def get_account(self) -> Account: ...
    def get_positions(self) -> list[Position]: ...
    def is_market_open(self) -> bool: ...
    def submit(self, intent: OrderIntent) -> OrderResult: ...


@dataclass
class DryRunBroker:
    """Fully offline broker: records intended orders, sends nothing."""

    equity: float = 100_000.0
    cash: float = 100_000.0
    positions: dict[str, float] = field(default_factory=dict)  # symbol -> market value
    market_open: bool = True
    submitted: list[OrderIntent] = field(default_factory=list)

    @property
    def is_paper(self) -> bool:
        return True

    def get_account(self) -> Account:
        return Account(self.equity, self.cash, self.cash, blocked=False, is_paper=True)

    def get_positions(self) -> list[Position]:
        return [Position(s, 0.0, mv) for s, mv in self.positions.items()]

    def is_market_open(self) -> bool:
        return self.market_open

    def submit(self, intent: OrderIntent) -> OrderResult:
        self.submitted.append(intent)
        return OrderResult(
            intent.client_order_id, intent.symbol, intent.side, "dry_run", intent.notional
        )


# --------------------------------------------------------------------------- #
# Pure planning + guardrails
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExecConfig:
    cash_buffer: float = 0.02
    no_trade_band: float = 0.005  # fraction of equity
    min_order_notional: float = 1.0
    max_orders_per_run: int = 20
    max_order_notional_pct: float = 0.25  # of equity
    max_total_weight: float = 1.0


# Reject genuine leverage, but tolerate weight vectors that sum slightly over 1
# from rounding (the logged target is rounded to 4dp).
_SUM_TOLERANCE = 0.02


def _valid_target(target: dict[str, float], max_total: float) -> bool:
    if not target:
        return False
    total = 0.0
    for w in target.values():
        if w is None or math.isnan(w) or math.isinf(w) or w < 0:
            return False
        total += w
    return 0.0 < total <= max_total + _SUM_TOLERANCE


def plan_rebalance(
    target: dict[str, float],
    equity: float,
    current_mv: dict[str, float],
    cfg: ExecConfig,
    *,
    run_id: str,
) -> list[OrderIntent]:
    """Target weights + account state -> minimal costed orders (pure).

    Fail-closed: an empty / NaN / non-normalised target produces **zero** orders
    (never a liquidation). Sells are ordered before buys so cash is freed first.
    """
    if not _valid_target(target, cfg.max_total_weight) or equity <= 0:
        return []

    # Renormalise away rounding so we never deploy more than intended.
    total_w = sum(target.values())
    if total_w > 1.0:
        target = {k: v / total_w for k, v in target.items()}

    investable = equity * (1.0 - cfg.cash_buffer)
    floor = max(cfg.no_trade_band * equity, cfg.min_order_notional)
    intents: list[OrderIntent] = []
    for sym in sorted(set(target) | set(current_mv)):
        target_dollars = investable * target.get(sym, 0.0)
        delta = target_dollars - current_mv.get(sym, 0.0)
        if abs(delta) < floor:
            continue
        side = "buy" if delta > 0 else "sell"
        reason = "exit" if target.get(sym, 0.0) == 0.0 else side
        intents.append(
            OrderIntent(sym, side, round(abs(delta), 2), f"wc-{run_id}-{sym}-{side}", reason)
        )
    # Sells first, largest first.
    return sorted(intents, key=lambda i: (i.side != "sell", -i.notional))


def check_guardrails(intents: list[OrderIntent], account: Account, cfg: ExecConfig) -> list[str]:
    """Return a list of violations; non-empty means abort and submit nothing."""
    v: list[str] = []
    if account.blocked:
        v.append("account is blocked")
    if not account.is_paper:
        v.append("account is not a paper account — refusing to trade")
    if len(intents) > cfg.max_orders_per_run:
        v.append(f"{len(intents)} orders exceeds max_orders_per_run={cfg.max_orders_per_run}")
    cap = cfg.max_order_notional_pct * account.equity
    for i in intents:
        if i.notional > cap:
            v.append(
                f"{i.symbol} order ${i.notional:,.0f} exceeds {cfg.max_order_notional_pct:.0%} cap"
            )
    return v


def kill_switch_engaged(cache_dir: str = ".cache") -> bool:
    if os.environ.get(_KILL_ENV) in {"1", "true", "TRUE"}:
        return True
    return (Path(cache_dir) / _KILL_FILE).exists() or Path(_KILL_FILE).exists()


def reconcile(
    target: dict[str, float], positions: list[Position], equity: float
) -> dict[str, float]:
    """Per-name drift = actual weight - target weight (pure)."""
    if equity <= 0:
        return {}
    actual = {p.symbol: p.market_value / equity for p in positions}
    return {
        s: round(actual.get(s, 0.0) - target.get(s, 0.0), 4)
        for s in sorted(set(target) | set(actual))
    }


def rebalance_to_target(
    broker: ExecutionClient,
    target: dict[str, float],
    cfg: ExecConfig,
    *,
    run_id: str,
    submit: bool,
    cache_dir: str = ".cache",
) -> ExecReport:
    """Plan and (optionally) submit a rebalance. Submission is opt-in + guarded."""
    if kill_switch_engaged(cache_dir):
        return ExecReport(False, [], [], note="KILL SWITCH engaged — no action.")

    account = broker.get_account()
    positions = broker.get_positions()
    current_mv = {p.symbol: p.market_value for p in positions}
    intents = plan_rebalance(target, account.equity, current_mv, cfg, run_id=run_id)

    violations = check_guardrails(intents, account, cfg)
    if violations:
        return ExecReport(False, intents, [], violations=violations)

    do_submit = submit and broker.is_market_open()
    note = "" if do_submit else "(dry-run: market closed or execute not enabled)"
    results: list[OrderResult] = []
    for i in intents:
        if do_submit:
            results.append(broker.submit(i))
        else:
            results.append(OrderResult(i.client_order_id, i.symbol, i.side, "dry_run", i.notional))

    drift = reconcile(target, broker.get_positions(), account.equity)
    return ExecReport(do_submit, intents, results, drift=drift, note=note)
