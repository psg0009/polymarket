"""Discovery pipeline: ingest → score → link → persist Events+Signals.

Runs once per scan cycle. Produces a `dict[market_id, list[evidence_dict]]` that
the scheduler feeds into `strategy.decide(...)`.

Each evidence dict carries `source_tier` so the oracle prompt can weight
ECI/PIB > Reuters/Bloomberg > regional > social explicitly.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from sqlalchemy import select

from polyclaude.config import Settings, get_settings
from polyclaude.ingest import news as news_ingest
from polyclaude.ingest.india_sources import DEFAULT_INDIA_RSS
from polyclaude.ingest.normalizer import RawEvent, source_tier
from polyclaude.ledger.db import Event, Signal, get_session
from polyclaude.ledger.persist import save_event
from polyclaude.logging_setup import get_logger
from polyclaude.markets.gamma import MarketSummary
from polyclaude.markets.india import is_india_market
from polyclaude.nlp import ensemble, finbert, finvader, linker

log = get_logger(__name__)


@dataclass
class EvidenceRow:
    market_id: str
    event_id: str
    text: str
    source: str
    source_tier: int
    ts: datetime
    finbert_polarity: float
    vader_compound: float
    ensemble_score: float
    z: float
    link_score: float

    def to_oracle_dict(self) -> dict:
        return {
            "ts": self.ts.isoformat(),
            "source": f"{self.source} (tier {self.source_tier})",
            "tier": self.source_tier,
            "text": self.text[:600],
            "finbert": self.finbert_polarity,
            "vader": self.vader_compound,
        }


async def gather_raw_events(settings: Settings | None = None) -> list[RawEvent]:
    s = settings or get_settings()
    feeds = list(s.rss_feeds_list)
    india_feeds = list(s.india_rss_feeds_list) or list(DEFAULT_INDIA_RSS)
    raws: list[RawEvent] = []
    if feeds:
        raws.extend(await news_ingest.from_rss(feeds, is_india=False))
    if india_feeds:
        raws.extend(await news_ingest.from_rss(india_feeds, is_india=True))
    log.info("pipeline.ingested", events=len(raws))
    return raws


def _score_event(text: str) -> tuple[dict, dict]:
    fb = finbert.score(text)
    vd = finvader.score(text)
    return fb, vd


def link_and_persist(
    raws: Iterable[RawEvent], markets: list[MarketSummary], min_link_score: float = 0.10,
) -> dict[str, list[EvidenceRow]]:
    """Score each raw event, link to candidate markets, and persist.

    Returns a map of market_id → newest-first list of EvidenceRow.
    """
    by_market: dict[str, list[EvidenceRow]] = defaultdict(list)
    market_baselines: dict[str, list[float]] = defaultdict(list)
    now = datetime.now(timezone.utc)

    # Prime trailing scores from ledger (last 100 per market) for z computation.
    with get_session() as session:
        rows = session.execute(
            select(Signal.market_id, Signal.ensemble_score)
            .order_by(Signal.created_at.desc())
            .limit(2000)
        ).all()
    for mid, sc in rows:
        if len(market_baselines[mid]) < 100:
            market_baselines[mid].append(float(sc))

    for raw in raws:
        try:
            ev = save_event(raw)
        except Exception as e:
            log.warning("pipeline.save_event_failed", source=raw.source, err=str(e))
            continue
        if ev is None:
            continue  # already seen
        try:
            fb, vd = _score_event(raw.text)
        except Exception as e:
            log.warning("pipeline.score_failed", err=str(e))
            continue
        score = ensemble.combine(fb, vd, raw.ts or now, now=now)
        # Link to every market with link_score >= threshold.
        new_signals: list[Signal] = []
        for m in markets:
            mtext = f"{m.question}\n{m.description}\n{(m.resolution_rules or '')[:600]}"
            ls = linker.link_score(raw.text, mtext, m.tags)
            if ls < min_link_score:
                continue
            history = market_baselines.get(m.id, [])
            z = ensemble.z_vs_baseline(score.ensemble_score, history)
            sig = Signal(
                event_id=ev.id, market_id=m.id, link_score=float(ls),
                finbert_polarity=score.finbert_polarity,
                finbert_intensity=score.finbert_intensity,
                vader_compound=score.vader_compound,
                ensemble_score=score.ensemble_score,
                z_vs_baseline=z,
            )
            new_signals.append(sig)
            by_market[m.id].append(
                EvidenceRow(
                    market_id=m.id, event_id=ev.id,
                    text=raw.text, source=raw.source,
                    source_tier=source_tier(raw.source),
                    ts=raw.ts or now,
                    finbert_polarity=score.finbert_polarity,
                    vader_compound=score.vader_compound,
                    ensemble_score=score.ensemble_score,
                    z=z, link_score=float(ls),
                )
            )
            market_baselines[m.id].append(score.ensemble_score)
        if new_signals:
            try:
                with get_session() as session:
                    session.add_all(new_signals)
                    session.commit()
            except Exception as e:
                log.warning("pipeline.signals_commit_failed", err=str(e))

    # Sort newest first per market
    for mid in by_market:
        by_market[mid].sort(key=lambda r: r.ts, reverse=True)

    log.info("pipeline.linked", markets=len(by_market),
             signals=sum(len(v) for v in by_market.values()))
    return by_market


def evidence_for_oracle(
    rows: list[EvidenceRow], top_k: int = 20,
) -> list[dict]:
    """Pick the most decision-relevant rows: prefer high tier (low number) and recent."""
    # Sort by (tier asc, ts desc) so official sources beat social, ties broken by recency.
    rows = sorted(rows, key=lambda r: (r.source_tier, -r.ts.timestamp()))
    return [r.to_oracle_dict() for r in rows[:top_k]]


def hot_markets(by_market: dict[str, list[EvidenceRow]], z_threshold: float = 3.0) -> list[str]:
    """Return market_ids whose most recent signal exceeded z_threshold — news-event path."""
    out: list[str] = []
    for mid, rows in by_market.items():
        if any(abs(r.z) >= z_threshold for r in rows):
            out.append(mid)
    return out


async def run_discover(
    markets: list[MarketSummary], settings: Settings | None = None,
) -> dict[str, list[EvidenceRow]]:
    """Async entry point for the scheduler."""
    s = settings or get_settings()
    raws = await gather_raw_events(s)
    # Mark rss-india feed events as is_india regardless of source domain match.
    # Then run scoring + linking in a thread to avoid blocking the loop.
    def _work() -> dict[str, list[EvidenceRow]]:
        return link_and_persist(raws, markets)

    return await asyncio.to_thread(_work)
