"""Ensemble FinBERT + FinVADER into a single SignalScore with recency decay."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class SignalScore:
    finbert_polarity: float
    finbert_intensity: float
    vader_compound: float
    ensemble_score: float           # in [-1, 1]
    age_hours: float


def _decay(age_hours: float, half_life_h: float = 24.0) -> float:
    return 0.5 ** (max(0.0, age_hours) / half_life_h)


def combine(
    finbert: dict,
    vader: dict,
    event_ts: datetime,
    now: datetime | None = None,
    finbert_weight: float = 0.6,
    vader_weight: float = 0.4,
    half_life_hours: float = 24.0,
) -> SignalScore:
    now = now or datetime.now(timezone.utc)
    age_h = max(0.0, (now - event_ts).total_seconds() / 3600.0)
    pol = float(finbert.get("polarity", 0.0))
    inten = float(finbert.get("intensity", 0.0))
    vc = float(vader.get("compound", 0.0))
    raw = finbert_weight * pol + vader_weight * vc
    decayed = raw * _decay(age_h, half_life_hours)
    return SignalScore(
        finbert_polarity=pol,
        finbert_intensity=inten,
        vader_compound=vc,
        ensemble_score=max(-1.0, min(1.0, decayed)),
        age_hours=age_h,
    )


def z_vs_baseline(score: float, history: list[float]) -> float:
    """Trailing z-score; needs at least 10 historical scores or returns 0."""
    if len(history) < 10:
        return 0.0
    mean = sum(history) / len(history)
    var = sum((x - mean) ** 2 for x in history) / len(history)
    sd = math.sqrt(var) if var > 0 else 1e-6
    return (score - mean) / sd
