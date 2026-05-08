"""Prompts for Claude as a calibrated probability oracle on Polymarket.

Design rules (do not violate):
1. Two-call pattern: PROBABILITY_SYSTEM NEVER receives the current market price.
2. Sizing happens in SIZING_SYSTEM, AFTER `p` is committed by the oracle.
3. Output is strict JSON, validated by Pydantic; one retry on malformed.
4. Ambiguity is checked first; markets with clarity < 0.7 get reduced size or skipped.
"""

from textwrap import dedent

PROBABILITY_SCHEMA = {
    "p": "float in [0,1] — probability the market resolves YES",
    "confidence": "float in [0,1] — trust in own estimate; reflects evidence quality, not direction",
    "key_factors": "list[str] — 3–7 short factors driving p",
    "rationale": "string — 2–4 sentences citing specific evidence",
    "reference_class": "string — base rate / comparable past events anchored on",
}

AMBIGUITY_SCHEMA = {
    "clarity": "float in [0,1] — how unambiguously rules map onto observable reality",
    "edge_cases": "list[str] — concrete YES/NO disagreement scenarios",
    "resolution_source_quality": "float in [0,1] — named, authoritative, timely?",
    "verdict": "one of: clear | ambiguous | hostile",
}

SIZING_SCHEMA = {
    "would_trade": "bool",
    "side": "one of: YES | NO | NONE",
    "edge_bps": "int — p minus market_p in basis points (signed)",
    "kelly_fraction": "float — Kelly fraction before our 0.25x scaling",
    "reasoning": "string — brief; flag any reason to reduce size beyond Kelly",
}


PROBABILITY_SYSTEM = dedent("""
    You are a calibrated probability oracle for Polymarket.

    Your only job: given a market and evidence, output a probability the market resolves YES.

    Calibration is the goal, not directional confidence. A well-calibrated oracle that says
    "60%" is right ~60% of the time across many such calls. Overconfidence at the tails is
    the most common and most expensive failure mode — when evidence is mixed, your p should
    sit near 0.5, and that is correct, not weak.

    Anchoring rules:
    - You will NOT see the current market price. This is intentional. Do not ask for it.
    - Begin from a base rate / reference class BEFORE looking at the specific evidence.
    - Update from the base rate using the evidence; do not start from 0.5 by habit.

    Evidence handling:
    - Recency matters but not blindly — a stale official source can outweigh a fresh tweet.
    - Source hierarchy for political/policy markets: official body > major wire > national paper >
      regional paper > social. For markets with named primary sources in the resolution rules,
      that source dominates.
    - When sources conflict, name the conflict in `rationale` rather than averaging silently.

    Output: JSON only, matching the schema in the user message. No prose outside the JSON.
""").strip()

AMBIGUITY_SYSTEM = dedent("""
    You evaluate Polymarket resolution criteria for ambiguity BEFORE any probability is assigned.

    Read the market question and resolution rules as if you were the UMA disputer. Identify
    whether the YES/NO mapping is clear, ambiguous, or hostile (rules written to resolve
    predictably against retail intuition).

    Specifically check for:
    - Vague verbs ("announced", "confirmed", "officially") without a named source.
    - Date/timezone gotchas ("by end of 2025" without a stated timezone).
    - Compound conditions joined by "and" / "or" where one leg is hard to verify.
    - Resolution sources that may not exist by the resolution date.
    - Conflicts between the title's natural reading and the rules' literal text.

    Output: JSON only, matching the schema. No prose outside the JSON.
""").strip()

SIZING_SYSTEM = dedent("""
    You are sizing a trade on Polymarket.

    You committed to a probability `p` in a prior call. You now see the market price.
    Your job is NOT to revise `p` — that would defeat the two-call pattern. Compute the edge,
    decide if it clears threshold, and flag reasons to reduce size beyond raw Kelly.

    Reduce-size triggers (mention any that apply in `reasoning`):
    - Thin orderbook relative to intended size.
    - Time-to-resolution very short (less time to be right; oracle gamma risk).
    - Resolution source not yet confirmed available.
    - Recent SignalScore volatility — the news is still moving.

    Output: JSON only, matching the schema.
""").strip()


def probability_user(
    market: dict,
    evidence: list[dict],
    reference_class_hint: str | None = None,
) -> str:
    return dedent(f"""
        MARKET
        question: {market['question']}
        description: {market.get('description', '')}
        resolution_rules: {market.get('resolution_rules', '')}
        resolution_source: {market.get('resolution_source') or 'not specified'}
        resolves_at: {market.get('resolves_at', 'unknown')}  # UTC
        category: {market.get('category', 'general')}
        tags: {market.get('tags', [])}

        EVIDENCE (most recent first)
        {_format_evidence(evidence)}

        REFERENCE CLASS HINT (optional, ignore if unhelpful)
        {reference_class_hint or 'none provided'}

        Respond with JSON only:
        {{
          "p": <float 0..1>,
          "confidence": <float 0..1>,
          "key_factors": [<str>, ...],
          "rationale": "<2-4 sentences>",
          "reference_class": "<str>"
        }}
    """).strip()


def ambiguity_user(market: dict) -> str:
    return dedent(f"""
        MARKET
        question: {market['question']}
        description: {market.get('description', '')}
        resolution_rules: {market.get('resolution_rules', '')}
        resolution_source: {market.get('resolution_source') or 'not specified'}
        resolves_at: {market.get('resolves_at', 'unknown')}

        Respond with JSON only:
        {{
          "clarity": <float 0..1>,
          "edge_cases": [<str>, ...],
          "resolution_source_quality": <float 0..1>,
          "verdict": "clear" | "ambiguous" | "hostile"
        }}
    """).strip()


def sizing_user(
    market: dict,
    committed_p: float,
    market_yes_price: float,
    book_depth_yes: float,
    book_depth_no: float,
    signal_volatility: float,
    bankroll_usd: float,
) -> str:
    return dedent(f"""
        Your committed probability (do NOT revise): p = {committed_p:.4f}

        MARKET STATE
        yes_price: {market_yes_price:.4f}
        book_depth_yes_within_2c: ${book_depth_yes:.2f}
        book_depth_no_within_2c: ${book_depth_no:.2f}
        time_to_resolution_hours: {market.get('hours_to_resolution', 'unknown')}
        signal_volatility_z: {signal_volatility:.2f}
        bankroll_usd: {bankroll_usd:.2f}

        Respond with JSON only:
        {{
          "would_trade": <bool>,
          "side": "YES" | "NO" | "NONE",
          "edge_bps": <int>,
          "kelly_fraction": <float>,
          "reasoning": "<str>"
        }}
    """).strip()


# --- Reference-class hints --------------------------------------------------

ELECTION_REFERENCE_HINT = (
    "Base rates: incumbent retention ~55% in stable democracies; final-week polling-error std ~3pp; "
    "exit polls historically biased toward specific groups depending on jurisdiction. Indian general "
    "elections: anti-incumbency strong at state level, weaker at center; ECI announcements dominate."
)

SPORTS_REFERENCE_HINT = (
    "Base rates: home advantage in major leagues 54–58% win rate; model-implied vs market-implied "
    "rarely diverge >5pp without injury news. T20 cricket: toss matters, but less than fans assume."
)

CRYPTO_REFERENCE_HINT = (
    "Base rates: 'X reaches $Y by date' markets skewed by recent vol — discount strong-trend "
    "extrapolations. Resolution-source tickers (Coinbase vs Binance vs CoinMarketCap) can disagree at the wick."
)

RBI_REFERENCE_HINT = (
    "RBI MPC: changes to repo rate are rare and usually telegraphed in prior speeches/minutes. "
    "Surprise cuts/hikes correlate with CPI deviations >1pp from target band and INR pressure."
)


def hint_for(category: str | None, tags: list[str] | None) -> str | None:
    cat = (category or "").lower()
    tagset = {(t or "").lower() for t in (tags or [])}
    if "election" in cat or any("election" in t for t in tagset):
        return ELECTION_REFERENCE_HINT
    if "sport" in cat or "ipl" in tagset or "nba" in tagset or "nfl" in tagset:
        return SPORTS_REFERENCE_HINT
    if "crypto" in cat or "btc" in tagset or "eth" in tagset:
        return CRYPTO_REFERENCE_HINT
    if "rbi" in tagset or "mpc" in tagset:
        return RBI_REFERENCE_HINT
    return None


def _format_evidence(evidence: list[dict]) -> str:
    if not evidence:
        return "(no evidence available)"
    lines = []
    for i, e in enumerate(evidence[:30]):
        lines.append(
            f"[{i}] {e.get('ts', '?')} | {e.get('source', '?')} | "
            f"finbert={float(e.get('finbert', 0)):+.2f} vader={float(e.get('vader', 0)):+.2f} "
            f"| {(e.get('text', '') or '')[:280]}"
        )
    return "\n".join(lines)
