"""Poll a list of audio sources, transcribe new items, persist as Events.

The list of sources lives in `audio_sources.txt` (one URL per line, optional
`# tier=N` and `# india=true` comments per line). The scheduler fetches each
source's RSS / Atom feed, downloads any audio enclosure that's newer than the
last successful run, transcribes it via faster-whisper, and saves an Event
(modality=audio).
"""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import httpx

from polyclaude.config import get_settings
from polyclaude.ingest.audio import transcribe, transcript_to_event
from polyclaude.ingest.normalizer import RawEvent
from polyclaude.ledger.persist import save_event
from polyclaude.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class AudioSource:
    url: str
    name: str
    is_india: bool = False


def parse_sources(path: str | Path) -> list[AudioSource]:
    """Read newline-separated URL config.

    Format per line:
        https://example.com/feed.xml [# tier=1] [# india=true]

    Lines starting with `#` are comments. Empty lines ignored.
    """
    out: list[AudioSource] = []
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        url, _, comment = line.partition("#")
        url = url.strip()
        is_india = "india=true" in comment.lower()
        name = re.sub(r"https?://", "", url).split("/")[0]
        out.append(AudioSource(url=url, name=name, is_india=is_india))
    return out


async def _fetch_feed(client: httpx.AsyncClient, url: str) -> list[dict]:
    try:
        import feedparser
    except ImportError:  # pragma: no cover
        return []
    try:
        resp = await client.get(url, timeout=15.0)
        resp.raise_for_status()
    except Exception as e:
        log.warning("audio_sched.fetch_failed", url=url, err=str(e))
        return []
    parsed = feedparser.parse(resp.text)
    items: list[dict] = []
    for entry in parsed.entries[:10]:
        encs = [e for e in (entry.get("links") or []) if (e.get("rel") == "enclosure" or e.get("type", "").startswith("audio"))]
        if not encs:
            continue
        items.append({
            "audio_url": encs[0].get("href"),
            "title": entry.get("title"),
            "published": entry.get("published"),
        })
    return items


async def _download_audio(client: httpx.AsyncClient, url: str) -> str | None:
    try:
        resp = await client.get(url, timeout=120.0)
        resp.raise_for_status()
    except Exception as e:
        log.warning("audio_sched.download_failed", url=url, err=str(e))
        return None
    suffix = os.path.splitext(url.split("?")[0])[1] or ".mp3"
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="polyclaude_aud_")
    with os.fdopen(fd, "wb") as f:
        f.write(resp.content)
    return path


def _processed_marker_dir() -> Path:
    s = get_settings()
    base = Path(".cache/audio_marker")
    base.mkdir(parents=True, exist_ok=True)
    return base


def _is_seen(marker: str) -> bool:
    return (_processed_marker_dir() / marker).exists()


def _mark_seen(marker: str) -> None:
    (_processed_marker_dir() / marker).touch()


def _safe_marker(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", s)[:200]


async def run_once(sources_path: str | Path = "audio_sources.txt") -> int:
    sources = parse_sources(sources_path)
    if not sources:
        log.info("audio_sched.no_sources", path=str(sources_path))
        return 0
    transcribed = 0
    async with httpx.AsyncClient() as client:
        for src in sources:
            items = await _fetch_feed(client, src.url)
            for item in items:
                aud_url = item.get("audio_url")
                if not aud_url:
                    continue
                marker = _safe_marker(aud_url)
                if _is_seen(marker):
                    continue
                local = await _download_audio(client, aud_url)
                if local is None:
                    continue
                tx = await asyncio.to_thread(transcribe, local)
                try:
                    os.unlink(local)
                except OSError:
                    pass
                if tx is None or not tx.text.strip():
                    _mark_seen(marker)
                    continue
                ev = transcript_to_event(
                    audio_url=aud_url, source=src.name, transcript=tx,
                    ts=datetime.now(timezone.utc), is_india=src.is_india,
                )
                save_event(ev)
                _mark_seen(marker)
                transcribed += 1
    log.info("audio_sched.done", transcribed=transcribed, sources=len(sources))
    return transcribed


async def run_forever(interval_s: int = 1800, sources_path: str | Path = "audio_sources.txt") -> None:
    while True:
        try:
            await run_once(sources_path)
        except Exception as e:  # pragma: no cover
            log.error("audio_sched.cycle_failed", err=str(e))
        await asyncio.sleep(interval_s)


def cli_run_once() -> int:
    return asyncio.run(run_once())
