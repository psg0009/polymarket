"""Fractional Kelly sizing with hard caps.

For a binary bet at price `q` (cost per YES share), the gross payout per share
on a YES win is 1 (you get $1 per share); the net profit per share is (1-q),
and the loss on a NO outcome is q. With probability `p` of YES:

    Kelly fraction = p - (1-p) * q / (1-q)
                   = (p - q) / (1 - q)            (canonical form for binary YES)

For the NO side, swap p ↔ 1-p and q ↔ 1-q.

We scale Kelly by a small fraction (default 0.25) and then clip by per-trade
and per-day notional caps.

The ambiguity gate can independently apply a `size_multiplier` (1.0 / 0.5 / 0.0).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class RiskCaps:
    max_notional_per_trade: Decimal = Decimal("25")
    max_notional_per_day: Decimal = Decimal("100")
    kelly_fraction: float = 0.25
    min_edge_bps: int = 300                         # 3 cents
    min_book_depth_usd: Decimal = Decimal("50")


@dataclass
class SizingResult:
    side: str                                       # "YES" / "NO" / "NONE"
    edge_bps: int                                   # signed; positive = YES edge
    kelly_fraction: float                           # raw Kelly, pre-scaling
    scaled_fraction: float                          # post-Kelly-multiplier and ambiguity multiplier
    notional_usd: Decimal                           # final $ to deploy
    price: Decimal                                  # price you'd pay (YES price for YES side, 1-YES for NO side)
    skip_reason: str | None = None


def kelly_binary(p: float, q: float) -> float:
    """Canonical Kelly fraction for binary YES at cost q with prob p.

    Negative result means NO side has the edge.
    """
    if q <= 0 or q >= 1:
        return 0.0
    if p >= 1:
        return 1.0
    if p <= 0:
        return -1.0
    return (p - q) / (1.0 - q)


def size_trade(
    *,
    p: float,
    yes_price: float,
    bankroll_usd: Decimal,
    daily_used_usd: Decimal,
    book_depth_yes_usd: Decimal,
    book_depth_no_usd: Decimal,
    caps: RiskCaps,
    ambiguity_multiplier: float = 1.0,
) -> SizingResult:
    if not (0.01 <= yes_price <= 0.99):
        return SizingResult("NONE", 0, 0.0, 0.0, Decimal("0"), Decimal(str(yes_price)),
                            skip_reason=f"price {yes_price:.3f} outside (.01,.99)")

    edge = p - yes_price                            # signed
    edge_bps = int(round(edge * 10_000))

    if abs(edge_bps) < caps.min_edge_bps:
        return SizingResult("NONE", edge_bps, 0.0, 0.0, Decimal("0"), Decimal(str(yes_price)),
                            skip_reason=f"|edge| {abs(edge_bps)}bps < {caps.min_edge_bps}bps")

    if edge > 0:
        side = "YES"
        kelly = kelly_binary(p, yes_price)
        depth = book_depth_yes_usd
        price = Decimal(str(yes_price))
    else:
        side = "NO"
        # NO is symmetric: cost = 1-yes_price, p_no = 1-p
        kelly = kelly_binary(1 - p, 1 - yes_price)
        depth = book_depth_no_usd
        price = Decimal(str(1 - yes_price))

    if depth < caps.min_book_depth_usd:
        return SizingResult(side, edge_bps, kelly, 0.0, Decimal("0"), price,
                            skip_reason=f"book depth ${depth} < ${caps.min_book_depth_usd}")

    if kelly <= 0:
        return SizingResult(side, edge_bps, kelly, 0.0, Decimal("0"), price,
                            skip_reason="non-positive Kelly")

    scaled = max(0.0, kelly) * caps.kelly_fraction * ambiguity_multiplier
    raw_notional = Decimal(str(scaled)) * bankroll_usd

    # Apply caps
    notional = min(raw_notional, caps.max_notional_per_trade)
    remaining_day = max(Decimal("0"), caps.max_notional_per_day - daily_used_usd)
    notional = min(notional, remaining_day)

    if notional <= Decimal("0"):
        return SizingResult(side, edge_bps, kelly, scaled, Decimal("0"), price,
                            skip_reason="capped to $0 (daily/per-trade limits)")

    return SizingResult(side, edge_bps, kelly, scaled, notional, price)
