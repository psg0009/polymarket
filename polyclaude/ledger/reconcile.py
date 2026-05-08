"""Nightly reconcile job.

1. Pull resolved markets from Gamma. Update Market.resolved_outcome.
2. For each Decision on a resolved market without a CalibrationPoint, create one.
3. Cross-check open Orders against client.get_trades(); insert missing Fill rows.
4. Emit a daily Brier summary into the log.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from typing import Iterable

from sqlalchemy import desc, select

from polyclaude.clob.client import ClobWrapper
from polyclaude.ledger.db import (
    CalibrationPoint, Decision, Fill, Market, OracleCall, Order, OrderStatus, get_session,
)
from polyclaude.logging_setup import get_logger

log = get_logger(__name__)


def _brier(p: float, y: int) -> float:
    return (p - y) ** 2


def _log_loss(p: float, y: int, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


async def pull_resolved_outcomes(limit: int = 200) -> int:
    """Poll Gamma for resolved markets we know about and write the outcome.

    For each Market in our DB without resolved_outcome, query Gamma and check
    `closed=true` + the outcome flag on its tokens.

    Returns the number of markets we just marked resolved.
    """
    from polyclaude.markets.gamma import GammaClient

    updated = 0
    with get_session() as session:
        # Pick markets that should be resolvable: end_date in the past, not yet resolved.
        from datetime import datetime, timezone as _tz

        candidates = session.execute(
            select(Market.id)
            .where(Market.resolved_outcome.is_(None))
            .where((Market.resolves_at.is_(None)) | (Market.resolves_at < datetime.now(_tz.utc)))
            .limit(limit)
        ).all()
        market_ids = [row[0] for row in candidates]

    if not market_ids:
        return 0

    async with GammaClient() as gamma:
        for mid in market_ids:
            try:
                m = await gamma.get_market(mid)
            except Exception as e:  # pragma: no cover
                log.warning("reconcile.gamma_lookup_failed", market=mid, err=str(e))
                continue
            if m is None or not m.closed:
                continue
            outcome = _extract_outcome(m.raw)
            if outcome is None:
                continue
            with get_session() as session:
                row = session.get(Market, mid)
                if row is None:
                    continue
                row.resolved_outcome = outcome
                session.commit()
                updated += 1
    log.info("reconcile.resolved_pulled", updated=updated, candidates=len(market_ids))
    return updated


def _extract_outcome(raw: dict | None) -> str | None:
    """Inspect a Gamma `market` payload and decide which side resolved YES."""
    if not raw:
        return None
    # Most common shape: tokens=[{outcome:"Yes", winner:true}, {outcome:"No", winner:false}]
    tokens = raw.get("tokens") or []
    for t in tokens:
        if t.get("winner") is True:
            return (t.get("outcome") or "").upper() or None
    # Fallback: outcomePrices array — last price 1.0 wins
    prices = raw.get("outcomePrices")
    outcomes = raw.get("outcomes")
    if isinstance(prices, list) and isinstance(outcomes, list) and len(prices) == len(outcomes):
        try:
            best = max(range(len(prices)), key=lambda i: float(prices[i]))
            if float(prices[best]) >= 0.99:
                return str(outcomes[best]).upper()
        except (TypeError, ValueError):
            pass
    return None


def fill_calibration_points(market_outcomes: dict[str, str] | None = None) -> int:
    """Create CalibrationPoint rows for every Decision whose market is resolved.

    `market_outcomes` is an optional override map of market_id -> "YES"/"NO".
    When None, we trust Market.resolved_outcome already set in the DB.
    """
    created = 0
    with get_session() as session:
        if market_outcomes:
            for mid, outcome in market_outcomes.items():
                m = session.get(Market, mid)
                if m and not m.resolved_outcome:
                    m.resolved_outcome = outcome
            session.commit()

        # Decisions with a resolved market and no calibration point yet
        stmt = (
            select(Decision, Market)
            .join(Market, Decision.market_id == Market.id)
            .where(Market.resolved_outcome.isnot(None))
        )
        existing = {
            row[0]
            for row in session.execute(select(CalibrationPoint.decision_id)).all()
        }
        for decision, market in session.execute(stmt).all():
            if decision.id in existing:
                continue
            outcome_yes = (market.resolved_outcome or "").upper() == "YES"
            y = 1 if outcome_yes else 0
            p = float(decision.committed_p)
            # Pull confidence from the probability OracleCall in the same decision group.
            conf_row = session.execute(
                select(OracleCall.confidence)
                .where(
                    OracleCall.decision_group_id == decision.id,
                    OracleCall.call_type == "probability",
                )
                .order_by(desc(OracleCall.ts))
                .limit(1)
            ).first()
            confidence = float(conf_row[0]) if conf_row and conf_row[0] is not None else 0.0
            cp = CalibrationPoint(
                decision_id=decision.id,
                market_id=market.id,
                strategy=decision.strategy,
                p_predicted=p,
                confidence=confidence,
                outcome_yes=outcome_yes,
                brier=_brier(p, y),
                log_loss=_log_loss(p, y),
                resolved_at=datetime.now(timezone.utc),
            )
            session.add(cp)
            created += 1
        session.commit()
    log.info("reconcile.calibration", created=created)
    return created


def reconcile_fills(clob: ClobWrapper) -> int:
    """For every open/partial order, fetch trades and write missing Fill rows."""
    inserted = 0
    with get_session() as session:
        open_orders: Iterable[Order] = session.execute(
            select(Order).where(
                Order.status.in_([OrderStatus.pending, OrderStatus.open, OrderStatus.partially_filled])
            )
        ).scalars().all()
        seen_trade_ids = {
            tid for (tid,) in session.execute(select(Fill.trade_id)).all()
        }
        by_market: dict[str, list[Order]] = defaultdict(list)
        for o in open_orders:
            by_market[o.market_id].append(o)
        for market_id, orders in by_market.items():
            try:
                trades = clob.get_trades(market=market_id)
            except Exception as e:  # pragma: no cover
                log.warning("reconcile.trades_failed", market=market_id, err=str(e))
                continue
            order_by_id = {o.id: o for o in orders}
            for t in trades or []:
                trade_id = str(t.get("id") or t.get("trade_id") or "")
                order_id = str(t.get("orderID") or t.get("order_id") or "")
                if not trade_id or trade_id in seen_trade_ids:
                    continue
                if order_id not in order_by_id:
                    continue
                o = order_by_id[order_id]
                price = Decimal(str(t.get("price", 0)))
                size = Decimal(str(t.get("size", 0)))
                fee = Decimal(str(t.get("fee", 0) or 0))
                ts_raw = t.get("timestamp") or t.get("ts")
                try:
                    ts = datetime.fromtimestamp(int(ts_raw), tz=timezone.utc) if ts_raw else datetime.now(timezone.utc)
                except (TypeError, ValueError):
                    ts = datetime.now(timezone.utc)
                f = Fill(
                    order_id=o.id, trade_id=trade_id, price=price, size_shares=size,
                    fee_usd=fee, ts=ts, raw=t,
                )
                session.add(f)
                inserted += 1
                # Naive status: if size matches order size, mark filled.
                if size >= o.size_shares:
                    o.status = OrderStatus.filled
                    o.closed_at = ts
                else:
                    o.status = OrderStatus.partially_filled
        session.commit()
    log.info("reconcile.fills", inserted=inserted)
    return inserted


def daily_brier_summary() -> dict[str, float]:
    """Return aggregate Brier and per-strategy Brier across all CalibrationPoints."""
    with get_session() as session:
        rows = session.execute(
            select(CalibrationPoint.strategy, CalibrationPoint.brier)
        ).all()
    if not rows:
        return {"overall": float("nan"), "n": 0}
    by: dict[str, list[float]] = defaultdict(list)
    for strat, brier in rows:
        by[strat].append(float(brier))
    out: dict[str, float] = {"n": float(len(rows))}
    for k, v in by.items():
        out[f"brier_{k}"] = sum(v) / len(v)
    out["overall"] = sum(b for v in by.values() for b in v) / len(rows)
    log.info("reconcile.brier_summary", **{k: round(v, 4) for k, v in out.items()})
    return out
