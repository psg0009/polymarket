"""FinBERT (ProsusAI/finbert) sentiment.

The model is heavy (≈440MB). We lazy-load on first call so unit tests and
dry-runs that don't actually score events stay fast.

If `transformers` isn't installed, score() returns a neutral signal — this lets
the rest of the pipeline run end-to-end on minimal installs.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from polyclaude.logging_setup import get_logger

log = get_logger(__name__)

_MODEL_NAME = "ProsusAI/finbert"


@lru_cache(maxsize=1)
def _pipeline() -> Any | None:
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    except ImportError:
        log.info("nlp.finbert.unavailable", reason="transformers not installed")
        return None
    try:
        tok = AutoTokenizer.from_pretrained(_MODEL_NAME)
        mdl = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
        return pipeline("sentiment-analysis", model=mdl, tokenizer=tok, top_k=None)
    except Exception as e:  # pragma: no cover
        log.warning("nlp.finbert.load_failed", err=str(e))
        return None


def score(text: str) -> dict:
    """Return {polarity ∈ [-1,1], intensity ∈ [0,1]}.

    polarity = positive_score - negative_score
    intensity = max class score (a proxy for confidence)
    """
    if not text:
        return {"polarity": 0.0, "intensity": 0.0}
    pipe = _pipeline()
    if pipe is None:
        return {"polarity": 0.0, "intensity": 0.0}
    out = pipe(text[:512])
    rows = out[0] if isinstance(out, list) and out and isinstance(out[0], list) else out
    by = {r["label"].lower(): float(r["score"]) for r in rows}
    polarity = by.get("positive", 0.0) - by.get("negative", 0.0)
    intensity = max(by.values()) if by else 0.0
    return {"polarity": polarity, "intensity": intensity}
