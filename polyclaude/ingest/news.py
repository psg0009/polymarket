"""Pull news from RSS feeds + (optional) NewsAPI."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import AsyncIterator

import httpx

from polyclaude.config import Settings, get_settings
from polyclaude.ingest.normalizer import RawEvent
from polyclaude.logging_setup import get_logger

log = get_logger(__name__)


def _domain(url: str) -> str:
    try:
        from urllib.parse import urlparse

        return urlparse(url).netloc
    except Exception:
        return ""


def _parse_rss_dt(entry: dict) -> datetime:
    for k in ("published_parsed", "updated_parsed"):
        v = entry.get(k)
        if v:
            try:
                import time as _t

                return datetime.fromtimestamp(_t.mktime(v), tz=timezone.utc)
            except Exception:
                pass
    return datetime.now(timezone.utc)


async def from_rss(feed_urls: list[str], is_india: bool = False) -> list[RawEvent]:
    try:
        import feedparser
    except ImportError:  # pragma: no cover
        log.warning("ingest.feedparser_unavailable")
        return []

    out: list[RawEvent] = []

    def _fetch(url: str) -> list[RawEvent]:
        try:
            parsed = feedparser.parse(url)
        except Exception as e:
            log.warning("ingest.rss_failed", url=url, err=str(e))
            return []
        events: list[RawEvent] = []
        for entry in parsed.entries[:30]:
            link = entry.get("link", "")
            text = entry.get("summary", "") or entry.get("description", "") or entry.get("title", "")
            events.append(
                RawEvent(
                    source=_domain(link) or "rss",
                    text=text,
                    title=entry.get("title"),
                    url=link,
                    ts=_parse_rss_dt(entry),
                    is_india=is_india,
                )
            )
        return events

    results = await asyncio.gather(
        *(asyncio.to_thread(_fetch, u) for u in feed_urls), return_exceptions=False
    )
    for batch in results:
        out.extend(batch)
    log.info("ingest.rss", feeds=len(feed_urls), events=len(out))
    return out


async def from_newsapi(query: str, settings: Settings | None = None) -> list[RawEvent]:
    s = settings or get_settings()
    if not s.newsapi_key:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {"q": query, "sortBy": "publishedAt", "pageSize": 50, "apiKey": s.newsapi_key}
    out: list[RawEvent] = []
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        log.warning("ingest.newsapi_failed", err=str(e))
        return out
    for art in data.get("articles", [])[:50]:
        link = art.get("url", "")
        try:
            ts = datetime.fromisoformat(art["publishedAt"].replace("Z", "+00:00"))
        except Exception:
            ts = datetime.now(timezone.utc)
        out.append(
            RawEvent(
                source=(art.get("source") or {}).get("name") or _domain(link) or "newsapi",
                text=(art.get("description") or "") + "\n\n" + (art.get("content") or ""),
                title=art.get("title"),
                url=link,
                ts=ts,
            )
        )
    return out
