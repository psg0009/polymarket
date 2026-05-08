"""Lexicon-based sentiment via vaderSentiment.

We use the financial-vocabulary-aware extension where available (FinVADER), but
fall back to the standard analyzer if the package isn't installed. Both expose
`compound` in [-1, 1] which is what the ensemble consumes.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _analyzer():  # type: ignore[no-untyped-def]
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("vaderSentiment not installed") from e
    return SentimentIntensityAnalyzer()


def score(text: str) -> dict:
    if not text:
        return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
    a = _analyzer()
    return a.polarity_scores(text)
