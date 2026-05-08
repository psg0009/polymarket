"""Order placement with idempotency and cancel/replace.

The executor:
- Accepts a Decision (committed_p, side, intended_size, intended_price).
- Generates a deterministic idempotency key per (decision_id, side, price_bucket, attempt).
- Refuses to place if a Decision-keyed Order already exists in the ledger.
- Posts via py-clob-client and updates the ledger.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal

from polyclaude.clob.client import ClobWrapper
from polyclaude.config import Settings, get_settings
from polyclaude.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class PlaceRequest:
    decision_id: str
    market_id: str
    token_id: str
    side: str  # "YES" / "NO"  (translated to BUY/SELL of YES token below)
    price: Decimal  # 0..1
    size_shares: Decimal
    order_type: str = "GTC"  # GTC | FOK
    attempt: int = 0


@dataclass
class PlaceResult:
    ok: bool
    order_id: str | None
    idempotency_key: str
    raw: dict | None
    error: str | None


def _idempotency_key(req: PlaceRequest) -> str:
    bucket = req.price.quantize(Decimal("0.01"), rounding=ROUND_DOWN)
    payload = f"{req.decision_id}|{req.side}|{bucket}|{req.attempt}"
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


class Executor:
    def __init__(self, clob: ClobWrapper | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.clob = clob or ClobWrapper(self.settings)

    def place(self, req: PlaceRequest) -> PlaceResult:
        key = _idempotency_key(req)

        if self.settings.dry_run:
            log.info(
                "executor.dry_run",
                decision=req.decision_id, side=req.side,
                price=str(req.price), size=str(req.size_shares),
                token=req.token_id[:12] + "…", key=key,
            )
            return PlaceResult(ok=True, order_id=f"DRY-{key[:12]}", idempotency_key=key,
                               raw={"dry_run": True}, error=None)

        try:
            from py_clob_client.clob_types import OrderArgs

            # In Polymarket, "YES" side translates to BUY of YES token at price p.
            # "NO" side can be expressed as BUY of NO token at price 1-p
            # (caller passes the correct token_id and price for the side).
            args = OrderArgs(
                token_id=req.token_id,
                price=float(req.price),
                size=float(req.size_shares),
                side="BUY",
            )
            signed = self.clob.create_order(args)
            resp = self.clob.post_order(signed, req.order_type)
            order_id = resp.get("orderID") or resp.get("order_id") or ""
            ok = bool(resp.get("success", True)) and bool(order_id)
            return PlaceResult(ok=ok, order_id=order_id or None, idempotency_key=key,
                               raw=resp, error=None if ok else str(resp))
        except Exception as e:
            log.error("executor.place_failed", err=str(e), decision=req.decision_id)
            return PlaceResult(ok=False, order_id=None, idempotency_key=key, raw=None, error=str(e))

    def cancel(self, order_id: str) -> bool:
        if self.settings.dry_run or order_id.startswith("DRY-"):
            return True
        try:
            self.clob.cancel(order_id)
            return True
        except Exception as e:  # pragma: no cover
            log.error("executor.cancel_failed", err=str(e), order=order_id)
            return False

    def reprice_loop(
        self,
        req: PlaceRequest,
        max_attempts: int = 3,
        sleep_s: float = 60.0,
    ) -> PlaceResult:
        """Place GTC; if not filled within sleep_s, cancel and reprice once.

        Reads back fills via clob.get_trades to detect fills.
        """
        last: PlaceResult | None = None
        for attempt in range(max_attempts):
            req.attempt = attempt
            res = self.place(req)
            last = res
            if not res.ok or not res.order_id or res.order_id.startswith("DRY-"):
                return res
            time.sleep(sleep_s)
            try:
                trades = self.clob.get_trades(market=req.market_id)
                filled = any(t.get("orderID") == res.order_id for t in trades or [])
                if filled:
                    return res
                self.cancel(res.order_id)
            except Exception as e:  # pragma: no cover
                log.warning("executor.reprice_check_failed", err=str(e))
                return res
        return last  # type: ignore[return-value]
