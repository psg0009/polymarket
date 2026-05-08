"""Normalize raw ingest payloads to the Event Pydantic model + DB row."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone

from polyclaude.ledger.db import Event, Modality


SOURCE_TIERS: dict[str, int] = {
    # 1 = official; 2 = major wire; 3 = national; 4 = regional; 5 = social
    "eci.gov.in": 1, "rbi.org.in": 1, "pib.gov.in": 1, "whitehouse.gov": 1,
    "federalreserve.gov": 1, "ec.europa.eu": 1, "un.org": 1,
    "reuters.com": 2, "ap.org": 2, "bloomberg.com": 2, "afp.com": 2, "ptinews.com": 2,
    "nytimes.com": 3, "wsj.com": 3, "bbc.co.uk": 3, "thehindu.com": 3,
    "indianexpress.com": 3, "ndtv.com": 3, "timesofindia.indiatimes.com": 3,
    "news18.com": 4, "thequint.com": 4, "scroll.in": 4,
    "twitter.com": 5, "x.com": 5, "reddit.com": 5,
}


def source_tier(source: str) -> int:
    s = (source or "").lower()
    for k, v in SOURCE_TIERS.items():
        if k in s:
            return v
    return 4


@dataclass
class RawEvent:
    source: str
    text: str
    title: str | None = None
    url: str | None = None
    ts: datetime | None = None
    audio_url: str | None = None
    transcript_meta: dict | None = None
    modality: Modality = Modality.text
    lang: str = "en"
    is_india: bool = False
    extra: dict = field(default_factory=dict)


def event_id_for(source: str, url: str | None, ts: datetime, text: str) -> str:
    h = hashlib.sha256()
    h.update((source or "").encode())
    h.update(b"|")
    h.update((url or "").encode())
    h.update(b"|")
    h.update(ts.replace(microsecond=0).isoformat().encode())
    h.update(b"|")
    h.update(text[:128].encode("utf-8", errors="ignore"))
    return h.hexdigest()[:32]


def to_event(raw: RawEvent) -> Event:
    ts = raw.ts or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    eid = event_id_for(raw.source, raw.url, ts, raw.text)
    return Event(
        id=eid,
        ts=ts,
        modality=raw.modality,
        source=raw.source,
        source_tier=source_tier(raw.source),
        url=raw.url,
        title=raw.title,
        text=raw.text,
        audio_url=raw.audio_url,
        transcript_meta=raw.transcript_meta,
        lang=raw.lang,
        is_india=raw.is_india,
    )
