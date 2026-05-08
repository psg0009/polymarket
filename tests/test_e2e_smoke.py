"""End-to-end smoke against real APIs.

Skipped unless both ANTHROPIC_API_KEY (for the oracle) and POLYCLAUDE_E2E=1
(an explicit opt-in) are set. CI never runs this — it's a manual gate before
flipping --live on.

Asserts:
- Gamma returns at least one market.
- ClaudeOracle.evaluate_probability returns valid JSON for one market.
- The committed_p sits in [0, 1] and confidence in [0, 1].
"""

from __future__ import annotations

import asyncio
import os

import pytest

from polyclaude.config import get_settings
from polyclaude.markets.gamma import GammaClient
from polyclaude.oracle.claude import ClaudeOracle


pytestmark = pytest.mark.skipif(
    os.getenv("POLYCLAUDE_E2E") != "1" or not os.getenv("ANTHROPIC_API_KEY"),
    reason="Set POLYCLAUDE_E2E=1 and ANTHROPIC_API_KEY to run E2E smoke.",
)


def test_gamma_returns_markets():
    async def _call():
        async with GammaClient(get_settings()) as g:
            return await g.list_markets(limit=5)

    markets = asyncio.run(_call())
    assert len(markets) > 0


def test_oracle_round_trip():
    async def _markets():
        async with GammaClient(get_settings()) as g:
            return await g.list_markets(limit=3)

    markets = asyncio.run(_markets())
    assert markets, "no markets returned from Gamma"
    oracle = ClaudeOracle()
    market_dict = {
        "id": markets[0].id,
        "question": markets[0].question,
        "description": markets[0].description,
        "resolution_rules": markets[0].resolution_rules,
        "resolution_source": markets[0].resolution_source,
        "resolves_at": markets[0].end_date.isoformat() if markets[0].end_date else "unknown",
        "category": markets[0].category,
        "tags": markets[0].tags,
        "hours_to_resolution": markets[0].hours_to_resolution,
    }
    resp, trace = oracle.evaluate_probability(market_dict, evidence=[])
    assert resp is not None, f"oracle returned None; raw={trace.raw_response[:200]}"
    assert 0.0 <= resp.p <= 1.0
    assert 0.0 <= resp.confidence <= 1.0
    assert resp.rationale
