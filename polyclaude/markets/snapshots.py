"""Persist orderbook snapshots — the fuel for the backtester."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from polyclaude.clob.client import ClobWrapper
from polyclaude.ledger.db import BookSnapshot
from polyclaude.logging_setup import get_logger

log = get_logger(__name__)


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def parse_book(book: Any) -> dict[str, float]:
    """Take a py-clob-client OrderBook (or dict) and compute summary stats.

    Returns dict with: yes_midpoint, yes_best_bid, yes_best_ask,
    depth_yes_within_2c, depth_no_within_2c.
    """
    bids: list[Any] = []
    asks: list[Any] = []
    if hasattr(book, "bids"):
        bids = list(book.bids or [])
        asks = list(book.asks or [])
    elif isinstance(book, dict):
        bids = list(book.get("bids") or [])
        asks = list(book.get("asks") or [])

    def _row(row: Any) -> tuple[float, float]:
        if hasattr(row, "price"):
            return _to_float(row.price), _to_float(row.size)
        if isinstance(row, dict):
            return _to_float(row.get("price")), _to_float(row.get("size"))
        return 0.0, 0.0

    bid_rows = [_row(b) for b in bids]
    ask_rows = [_row(a) for a in asks]
    bid_rows.sort(key=lambda x: -x[0])  # highest bid first
    ask_rows.sort(key=lambda x: x[0])  # lowest ask first

    best_bid = bid_rows[0][0] if bid_rows else 0.0
    best_ask = ask_rows[0][0] if ask_rows else 0.0
    mid = (best_bid + best_ask) / 2 if (best_bid and best_ask) else (best_bid or best_ask)

    # Depth within 2 cents of midpoint, in $ notional (price * size).
    depth_yes = sum(p * s for p, s in bid_rows if mid - p <= 0.02)
    depth_no = sum(p * s for p, s in ask_rows if p - mid <= 0.02)
    return {
        "yes_midpoint": mid,
        "yes_best_bid": best_bid,
        "yes_best_ask": best_ask,
        "depth_yes_within_2c": depth_yes,
        "depth_no_within_2c": depth_no,
    }


def snapshot_market(clob: ClobWrapper, market_id: str, yes_token_id: str) -> BookSnapshot | None:
    if not yes_token_id:
        return None
    try:
        book = clob.get_orderbook(yes_token_id)
    except Exception as e:
        log.warning("snapshot.fetch_failed", market=market_id, err=str(e))
        return None
    stats = parse_book(book)
    raw = book if isinstance(book, dict) else getattr(book, "__dict__", {"_repr": str(book)})
    return BookSnapshot(
        market_id=market_id,
        ts=datetime.now(timezone.utc),
        yes_midpoint=Decimal(str(stats["yes_midpoint"])),
        yes_best_bid=Decimal(str(stats["yes_best_bid"])),
        yes_best_ask=Decimal(str(stats["yes_best_ask"])),
        depth_yes_within_2c=Decimal(str(stats["depth_yes_within_2c"])),
        depth_no_within_2c=Decimal(str(stats["depth_no_within_2c"])),
        raw_book=raw,
    )
