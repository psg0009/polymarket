"""Thin wrapper around py-clob-client.ClobClient with retries.

The official SDK is sync. We dispatch its calls via asyncio.to_thread when used
from the async scheduler.
"""

from __future__ import annotations

from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from polyclaude.config import Settings, get_settings
from polyclaude.logging_setup import get_logger

log = get_logger(__name__)


class ClobUnavailable(RuntimeError):
    pass


class ClobWrapper:
    """Lazy wrapper. The SDK is only imported on first use so dry-runs without
    creds (and without the package installed in some envs) still work for
    inspection / tests."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client: Any | None = None
        self._read_only_client: Any | None = None

    # --- lazy construction ---------------------------------------------------

    def _build_client(self, read_only: bool = False) -> Any:
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.constants import POLYGON
        except ImportError as e:  # pragma: no cover
            raise ClobUnavailable(
                "py-clob-client not installed. `pip install py-clob-client`."
            ) from e

        s = self.settings
        if read_only or not s.has_trading_creds():
            return ClobClient(host=s.clob_host, chain_id=s.chain_id or POLYGON)

        client = ClobClient(
            host=s.clob_host,
            key=s.private_key,
            chain_id=s.chain_id or POLYGON,
            signature_type=s.signature_type,
            funder=s.funder,
        )
        # Derive API creds: create or fetch existing
        try:
            creds = client.create_or_derive_api_creds()
            client.set_api_creds(creds)
            log.info("clob.api_creds_ready", funder=s.funder, sig_type=s.signature_type)
        except Exception as e:  # pragma: no cover - only fires with live creds
            log.error("clob.api_creds_failed", err=str(e))
            raise
        return client

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._build_client(read_only=False)
        return self._client

    @property
    def read_client(self) -> Any:
        if self._read_only_client is None:
            self._read_only_client = self._build_client(read_only=True)
        return self._read_only_client

    # --- read-only helpers ---------------------------------------------------

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(min=1, max=8),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def get_orderbook(self, token_id: str) -> Any:
        return self.read_client.get_order_book(token_id)

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=8), reraise=True)
    def get_midpoint(self, token_id: str) -> float:
        resp = self.read_client.get_midpoint(token_id)
        if isinstance(resp, dict):
            return float(resp.get("mid", 0.0))
        return float(resp)

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=8), reraise=True)
    def get_price(self, token_id: str, side: str = "BUY") -> float:
        resp = self.read_client.get_price(token_id, side)
        if isinstance(resp, dict):
            return float(resp.get("price", 0.0))
        return float(resp)

    # --- authenticated -------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    def get_trades(self, market: str | None = None) -> list[dict]:
        params = {"market": market} if market else None
        return self.client.get_trades(params)  # type: ignore[no-any-return]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    def post_order(self, signed_order: Any, order_type: str = "GTC") -> dict:
        from py_clob_client.clob_types import OrderType

        otype = getattr(OrderType, order_type, OrderType.GTC)
        return self.client.post_order(signed_order, otype)  # type: ignore[no-any-return]

    def create_order(self, args: Any) -> Any:
        return self.client.create_order(args)

    def cancel(self, order_id: str) -> dict:
        return self.client.cancel(order_id=order_id)

    def cancel_all(self) -> dict:
        return self.client.cancel_all()
