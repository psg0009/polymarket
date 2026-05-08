"""Walk-forward replay of historical BookSnapshots.

The replay reuses the LIVE pipeline modules — same ValueStrategy, same Claude
oracle, same sizing — so any backtest improvement is a live improvement.

Sources:
- BookSnapshot rows for market state (yes price, depth) at each timestep.
- Event rows joined to Signal rows for evidence at each timestep.
- Resolved Market.resolved_outcome to score each closed decision.

This implementation is deliberately simple: one decision per market per day,
filled at the snapshot midpoint, no slippage model. Extend in subclasses.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Iterable

from sqlalchemy import select

from polyclaude.config import Settings, get_settings
from polyclaude.ledger.db import (
    BookSnapshot, Event, Market, Signal, get_session,
)
from polyclaude.logging_setup import get_logger
from polyclaude.oracle.claude import ClaudeOracle
from polyclaude.strategy.value import ValueStrategy

log = get_logger(__name__)


@dataclass
class TradeRow:
    ts: datetime
    market_id: str
    side: str
    price: float
    notional: float
    p: float
    confidence: float
    edge_bps: int
    realized_pnl: float = 0.0


@dataclass
class ReplayResult:
    trades: list[TradeRow] = field(default_factory=list)
    by_strategy_pnl: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)
    brier_points: list[tuple[float, int]] = field(default_factory=list)  # (p, outcome_yes)


def _evidence_for(session, market_id: str, ts: datetime) -> list[dict]:
    rows: Iterable = session.execute(
        select(Event, Signal)
        .join(Signal, Event.id == Signal.event_id)
        .where(Signal.market_id == market_id, Event.ts <= ts)
        .order_by(Event.ts.desc())
        .limit(20)
    ).all()
    out: list[dict] = []
    for ev, sig in rows:
        out.append(
            {
                "ts": ev.ts.isoformat(),
                "source": ev.source,
                "text": ev.text[:600],
                "finbert": sig.finbert_polarity,
                "vader": sig.vader_compound,
            }
        )
    return out


def _market_dict(m: Market, snap: BookSnapshot) -> dict:
    hours = (
        (m.resolves_at - snap.ts).total_seconds() / 3600.0
        if m.resolves_at and m.resolves_at > snap.ts
        else 0.0
    )
    return {
        "id": m.id,
        "question": m.question,
        "description": m.description,
        "resolution_rules": m.resolution_rules,
        "resolution_source": m.resolution_source,
        "resolves_at": m.resolves_at.isoformat() if m.resolves_at else "unknown",
        "category": m.category,
        "tags": m.tags,
        "hours_to_resolution": hours,
    }


def replay(
    *,
    start: datetime,
    end: datetime,
    capital: Decimal,
    strategy_name: str = "value",
    settings: Settings | None = None,
    use_oracle: bool = False,
) -> ReplayResult:
    """Walk every market that has snapshots in [start, end].

    `use_oracle=False` runs the strategy without Claude calls — useful for fast
    iteration when only sizing/gating logic changed. Set `use_oracle=True` for
    a true live-equivalent replay (much slower, costs API tokens).
    """
    s = settings or get_settings()
    oracle = ClaudeOracle(s) if use_oracle else None
    strat = ValueStrategy(oracle, s) if oracle is not None else None

    result = ReplayResult()
    bankroll = float(capital)
    daily_used = Decimal("0")
    last_day = None

    with get_session() as session:
        # one snapshot per (market, day) — pick the latest within day
        stmt = (
            select(BookSnapshot, Market)
            .join(Market, BookSnapshot.market_id == Market.id)
            .where(BookSnapshot.ts >= start, BookSnapshot.ts <= end)
            .order_by(BookSnapshot.ts)
        )
        seen: set[tuple[str, str]] = set()
        for snap, market in session.execute(stmt).all():
            day = snap.ts.strftime("%Y-%m-%d")
            if last_day != day:
                last_day = day
                daily_used = Decimal("0")
                result.equity_curve.append((snap.ts, bankroll))
            key = (market.id, day)
            if key in seen:
                continue
            seen.add(key)

            yes_price = float(snap.yes_midpoint)
            depth_yes = Decimal(str(snap.depth_yes_within_2c))
            depth_no = Decimal(str(snap.depth_no_within_2c))

            if strat is None:
                # No-oracle mode: skip — caller must set use_oracle=True for full replay.
                continue

            evidence = _evidence_for(session, market.id, snap.ts)
            decision, prob, _ = strat.decide(
                market=_market_dict(market, snap), evidence=evidence,
                market_yes_price=yes_price,
                book_depth_yes_usd=depth_yes, book_depth_no_usd=depth_no,
                bankroll_usd=Decimal(str(bankroll)), daily_used_usd=daily_used,
            )

            if decision.action.name != "place" or decision.sizing.notional_usd <= 0:
                continue
            shares = float(decision.sizing.notional_usd) / float(decision.sizing.price)
            outcome_yes = (market.resolved_outcome or "").upper() == "YES"
            if not market.resolved_outcome:
                # Position remains open at end of replay window — treat as flat.
                continue
            wins_yes = decision.sizing.side == "YES" and outcome_yes
            wins_no = decision.sizing.side == "NO" and not outcome_yes
            payoff_per_share = 1.0 if (wins_yes or wins_no) else 0.0
            cost_per_share = float(decision.sizing.price)
            pnl = (payoff_per_share - cost_per_share) * shares
            bankroll += pnl
            daily_used += decision.sizing.notional_usd

            result.trades.append(
                TradeRow(
                    ts=snap.ts, market_id=market.id, side=decision.sizing.side,
                    price=cost_per_share, notional=float(decision.sizing.notional_usd),
                    p=decision.committed_p, confidence=decision.confidence,
                    edge_bps=decision.sizing.edge_bps, realized_pnl=pnl,
                )
            )
            result.by_strategy_pnl[strategy_name] += pnl
            result.brier_points.append((decision.committed_p, 1 if outcome_yes else 0))

    result.equity_curve.append((end, bankroll))
    return result
