"""Link an Event to candidate Markets.

Two-stage:
1. Cheap keyword/tag overlap as a recall filter.
2. Embedding cosine via sentence-transformers (lazy-loaded; falls back to
   keyword score if the package is unavailable).

A market is considered linked if combined_score >= threshold.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from polyclaude.logging_setup import get_logger

log = get_logger(__name__)

_WORD = re.compile(r"\b[a-zA-Z][a-zA-Z'-]+\b")


def _tokens(s: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(s) if len(w) > 2}


def keyword_score(event_text: str, market_text: str, market_tags: list[str] | None = None) -> float:
    """Jaccard over tokens, with a small bonus for tag overlap."""
    et = _tokens(event_text)
    mt = _tokens(market_text)
    if not et or not mt:
        return 0.0
    inter = len(et & mt)
    union = len(et | mt)
    jaccard = inter / union if union else 0.0
    tag_bonus = 0.0
    if market_tags:
        et_low = {w.lower() for w in et}
        for t in market_tags:
            if t.lower() in et_low:
                tag_bonus += 0.1
    return min(1.0, jaccard + tag_bonus)


@lru_cache(maxsize=1)
def _model() -> Any | None:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    try:
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:  # pragma: no cover
        log.warning("linker.model_load_failed", err=str(e))
        return None


def embedding_score(event_text: str, market_text: str) -> float | None:
    m = _model()
    if m is None:
        return None
    try:
        a, b = m.encode([event_text, market_text], convert_to_numpy=True)
        # Cosine
        import numpy as np

        denom = (float(np.linalg.norm(a)) * float(np.linalg.norm(b))) or 1.0
        return float(np.dot(a, b) / denom)
    except Exception as e:  # pragma: no cover
        log.warning("linker.embed_failed", err=str(e))
        return None


def link_score(event_text: str, market_text: str, market_tags: list[str] | None = None) -> float:
    kw = keyword_score(event_text, market_text, market_tags)
    emb = embedding_score(event_text, market_text)
    if emb is None:
        return kw
    return 0.5 * kw + 0.5 * max(0.0, emb)
