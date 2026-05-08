"""Persist Decisions, Orders, OracleCall traces, Markets, and Events.

Keeps SQLAlchemy session lifecycles in one place so the rest of the agent
deals with plain dicts/dataclasses.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Iterable

from sqlalchemy import select

from polyclaude.clob.executor import PlaceResult
from polyclaude.ingest.normalizer import RawEvent, to_event
from polyclaude.ledger.db import (
    Decision, Event, Market, OracleCall, Order, OrderStatus, OrderType,
    RiskState, Side, get_session,
)
from polyclaude.ledger.db import BookSnapshot
from polyclaude.markets.gamma import MarketSummary
from polyclaude.markets.india import is_india_market
from polyclaude.oracle.claude import CallTrace
from polyclaude.strategy.value import StrategyDecision


def upsert_market(m: MarketSummary) -> None:
    with get_session() as session:
        existing = session.get(Market, m.id)
        if existing is None:
            session.add(
                Market(
                    id=m.id, condition_id=m.condition_id,
                    yes_token_id=m.yes_token_id, no_token_id=m.no_token_id,
                    question=m.question, description=m.description,
                    resolution_rules=m.resolution_rules,
                    resolution_source=m.resolution_source,
                    category=m.category, tags=list(m.tags),
                    is_india=is_india_market(m), resolves_at=m.end_date,
                )
            )
        else:
            existing.question = m.question
            existing.description = m.description or existing.description
            existing.resolution_rules = m.resolution_rules or existing.resolution_rules
            existing.resolution_source = m.resolution_source or existing.resolution_source
            existing.tags = list(m.tags)
            existing.category = m.category
            existing.is_india = is_india_market(m)
            existing.resolves_at = m.end_date or existing.resolves_at
            existing.updated_at = datetime.now(timezone.utc)
        session.commit()


def save_snapshot(snap: BookSnapshot | None) -> None:
    if snap is None:
        return
    with get_session() as session:
        session.add(snap)
        session.commit()


def save_event(raw: RawEvent) -> Event | None:
    ev = to_event(raw)
    with get_session() as session:
        if session.get(Event, ev.id) is not None:
            return None
        session.add(ev)
        session.commit()
        return ev


def save_oracle_calls(traces: Iterable[CallTrace], market_id: str, signal_id: int | None = None) -> None:
    rows = []
    for t in traces:
        if not t:
            continue
        rows.append(
            OracleCall(
                decision_group_id=t.decision_group_id,
                market_id=market_id,
                signal_id=signal_id,
                call_type=t.call_type,
                model=t.model,
                prompt_system=t.prompt_system,
                prompt_user=t.prompt_user,
                raw_response=t.raw_response,
                parsed_response=t.parsed,
                p=float(t.parsed.get("p")) if isinstance(t.parsed.get("p"), (int, float)) else None,
                confidence=float(t.parsed.get("confidence")) if isinstance(t.parsed.get("confidence"), (int, float)) else None,
                clarity=float(t.parsed.get("clarity")) if isinstance(t.parsed.get("clarity"), (int, float)) else None,
                input_tokens=t.input_tokens,
                output_tokens=t.output_tokens,
                latency_ms=t.latency_ms,
                error=t.error,
            )
        )
    if not rows:
        return
    with get_session() as session:
        session.add_all(rows)
        session.commit()


def save_decision(d: StrategyDecision, *, bankroll_usd: Decimal, daily_used_usd: Decimal,
                  is_live: bool) -> None:
    with get_session() as session:
        if session.get(Decision, d.decision_group_id) is not None:
            return
        session.add(
            Decision(
                id=d.decision_group_id,
                market_id=d.market_id,
                strategy=d.strategy,
                committed_p=float(d.committed_p),
                market_p_at_decision=float(d.market_p),
                edge_bps=int(d.sizing.edge_bps),
                side=Side(d.sizing.side) if d.sizing.side in ("YES", "NO") else Side.NONE,
                kelly_fraction=float(d.sizing.kelly_fraction),
                scaled_fraction=float(d.sizing.scaled_fraction),
                intended_size_usd=Decimal(d.sizing.notional_usd),
                intended_price=Decimal(d.sizing.price),
                action=d.action,
                skip_reason=d.skip_reason,
                bankroll_usd=bankroll_usd,
                daily_notional_used_usd=daily_used_usd,
                is_live=is_live,
            )
        )
        session.commit()


def save_order(*, decision_id: str, market_id: str, token_id: str, side: str,
               price: Decimal, size_shares: Decimal, notional: Decimal,
               result: PlaceResult, order_type: str = "GTC") -> None:
    with get_session() as session:
        order_id = result.order_id or result.idempotency_key
        if session.get(Order, order_id) is not None:
            return
        session.add(
            Order(
                id=order_id,
                idempotency_key=result.idempotency_key,
                decision_id=decision_id,
                market_id=market_id,
                token_id=token_id,
                side=Side(side) if side in ("YES", "NO") else Side.NONE,
                order_type=OrderType(order_type),
                price=price,
                size_shares=size_shares,
                notional_usd=notional,
                status=OrderStatus.open if result.ok else OrderStatus.error,
                error=result.error,
            )
        )
        session.commit()


# --- daily risk state ----------------------------------------------------


def get_or_create_risk_state(daily_cap_usd: Decimal) -> RiskState:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with get_session() as session:
        rs = session.get(RiskState, today)
        if rs is None:
            rs = RiskState(date_utc=today, daily_notional_used_usd=Decimal("0"),
                           daily_notional_cap_usd=daily_cap_usd)
            session.add(rs)
            session.commit()
            session.refresh(rs)
        return rs


def increment_daily_used(amount: Decimal) -> Decimal:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with get_session() as session:
        rs = session.get(RiskState, today)
        if rs is None:
            return Decimal("0")
        rs.daily_notional_used_usd = (rs.daily_notional_used_usd or Decimal("0")) + amount
        rs.trades_today = (rs.trades_today or 0) + 1
        session.commit()
        return rs.daily_notional_used_usd
