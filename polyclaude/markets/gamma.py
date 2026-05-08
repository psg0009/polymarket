"""Gamma Markets API client.

Gamma is the public read-only API for browsing markets:
    GET /markets?closed=false&order=volume&ascending=false&limit=50

We normalize a subset of fields into MarketSummary/MarketDetail Pydantic models
so the rest of the agent doesn't depend on Gamma's loose JSON shape.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from polyclaude.config import Settings, get_settings
from polyclaude.logging_setup import get_logger

log = get_logger(__name__)


class MarketSummary(BaseModel):
    id: str
    condition_id: str = ""
    yes_token_id: str = ""
    no_token_id: str = ""
    question: str
    description: str = ""
    resolution_rules: str = ""
    resolution_source: str | None = None
    category: str | None = None
    tags: list[str] = Field(default_factory=list)
    volume: float = 0.0
    liquidity: float = 0.0
    closed: bool = False
    active: bool = True
    end_date: datetime | None = None
    yes_price: float | None = None
    raw: dict = Field(default_factory=dict)

    @property
    def hours_to_resolution(self) -> float:
        if not self.end_date:
            return 1e9
        return max(0.0, (self.end_date - datetime.now(timezone.utc)).total_seconds() / 3600.0)


def _parse_market(m: dict) -> MarketSummary:
    """Gamma API field shapes have shifted over time. Be defensive."""

    tokens: list[Any] = m.get("tokens") or []
    yes_id = ""
    no_id = ""
    yes_price: float | None = None
    for t in tokens:
        outcome = (t.get("outcome") or "").lower()
        if outcome == "yes":
            yes_id = t.get("token_id") or t.get("tokenId") or ""
            if t.get("price") is not None:
                try:
                    yes_price = float(t["price"])
                except (TypeError, ValueError):
                    yes_price = None
        elif outcome == "no":
            no_id = t.get("token_id") or t.get("tokenId") or ""

    # clobTokenIds is sometimes a JSON-encoded list-of-strings
    if not yes_id and m.get("clobTokenIds"):
        ids = m["clobTokenIds"]
        if isinstance(ids, str):
            try:
                import json

                ids = json.loads(ids)
            except Exception:
                ids = []
        if isinstance(ids, list) and len(ids) >= 2:
            yes_id, no_id = str(ids[0]), str(ids[1])

    end = m.get("endDate") or m.get("end_date_iso") or m.get("end_date")
    end_dt: datetime | None = None
    if end:
        try:
            end_dt = datetime.fromisoformat(str(end).replace("Z", "+00:00"))
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=timezone.utc)
        except ValueError:
            end_dt = None

    tags = m.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]

    def _f(key: str, default: float = 0.0) -> float:
        try:
            return float(m.get(key, default) or default)
        except (TypeError, ValueError):
            return default

    return MarketSummary(
        id=str(m.get("id") or m.get("conditionId") or m.get("condition_id") or ""),
        condition_id=str(m.get("conditionId") or m.get("condition_id") or ""),
        yes_token_id=yes_id,
        no_token_id=no_id,
        question=m.get("question") or m.get("title") or "",
        description=m.get("description") or "",
        resolution_rules=(
            m.get("resolution_rules")
            or m.get("resolutionRules")
            or m.get("description")
            or ""
        ),
        resolution_source=m.get("resolutionSource") or m.get("resolution_source"),
        category=m.get("category"),
        tags=tags if isinstance(tags, list) else [],
        volume=_f("volume"),
        liquidity=_f("liquidity"),
        closed=bool(m.get("closed", False)),
        active=bool(m.get("active", True)),
        end_date=end_dt,
        yes_price=yes_price,
        raw=m,
    )


class GammaClient:
    def __init__(self, settings: Settings | None = None, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = client
        self._owns_client = client is None

    async def __aenter__(self) -> "GammaClient":
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.settings.gamma_host, timeout=15.0)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _ensure(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.settings.gamma_host, timeout=15.0)
        return self._client

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(min=1, max=8),
        retry=retry_if_exception_type((httpx.HTTPError,)),
        reraise=True,
    )
    async def list_markets(
        self,
        limit: int = 50,
        closed: bool = False,
        order: str = "volume",
        offset: int = 0,
    ) -> list[MarketSummary]:
        client = await self._ensure()
        params = {
            "limit": limit,
            "closed": str(closed).lower(),
            "order": order,
            "ascending": "false",
            "offset": offset,
            "active": "true",
        }
        resp = await client.get("/markets", params=params)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "markets" in data:
            data = data["markets"]
        markets = [_parse_market(m) for m in data if isinstance(m, dict)]
        log.info("gamma.list", count=len(markets), limit=limit, offset=offset)
        return markets

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    async def get_market(self, market_id: str) -> MarketSummary | None:
        client = await self._ensure()
        resp = await client.get(f"/markets/{market_id}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return _parse_market(resp.json())
