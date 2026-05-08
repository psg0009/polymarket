"""Detect markets resolving on India-context events.

Heuristic combination of:
1. Tag/category match (e.g., "india", "elections-india", "rbi", "ipl").
2. Keyword match in question/description.
3. Named entities for parties, leaders, regulators, leagues, indices.

The output is conservative: prefer false negatives over false positives so we
don't accidentally trade US-context markets through the India strategy.
"""

from __future__ import annotations

import re

from polyclaude.markets.gamma import MarketSummary

INDIA_TAGS = {
    "india", "indian", "indian-elections", "ipl", "rbi", "nifty", "sensex",
    "modi", "lok-sabha", "rajya-sabha", "eci",
}

INDIA_KEYWORDS = [
    r"\bindia\b", r"\bindian\b", r"\brbi\b", r"\bmpc\b", r"\bnifty\b", r"\bsensex\b",
    r"\blok\s*sabha\b", r"\brajya\s*sabha\b", r"\bbjp\b", r"\binc\b(?!\.)",
    r"\bcongress(?:\s+party)?\b", r"\bmodi\b", r"\bgandhi\b", r"\bkejriwal\b", r"\byogi\b",
    r"\baap\b", r"\baipl\b", r"\bbcci\b", r"\bipo\s+(?:listing|listing\s+price)\b",
    r"\bmumbai\b", r"\bdelhi\b", r"\bkolkata\b", r"\bchennai\b", r"\bbengaluru\b",
    r"\bbangalore\b", r"\bhyderabad\b", r"\beci\b",
]

_KEYWORD_RE = re.compile("|".join(INDIA_KEYWORDS), re.IGNORECASE)


def is_india_market(m: MarketSummary) -> bool:
    tagset = {t.lower() for t in m.tags or []}
    if tagset & INDIA_TAGS:
        return True
    if (m.category or "").lower() in INDIA_TAGS:
        return True
    haystack = f"{m.question}\n{m.description}\n{m.resolution_rules}"
    return bool(_KEYWORD_RE.search(haystack))


_VERT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("india_rbi", re.compile(r"\b(rbi|mpc|repo\s+rate|monetary\s+policy)\b", re.I)),
    ("india_cricket", re.compile(r"\b(ipl|bcci|test\s+match|odi|t20|cricket)\b", re.I)),
    ("india_markets", re.compile(r"\b(nifty|sensex|ipo)\b", re.I)),
    ("india_election", re.compile(r"\b(election|lok\s+sabha|rajya\s+sabha|eci|vote|seat)\b", re.I)),
)


def vertical_for(m: MarketSummary) -> str | None:
    """Map an India market to a sub-vertical for strategy routing.

    Most-specific verticals (RBI, cricket, markets) are checked before the
    catch-all "election" branch so an RBI market doesn't get routed elsewhere.
    """
    text = f"{m.question} {m.description}"
    for name, pat in _VERT_PATTERNS:
        if pat.search(text):
            return name
    return None
