"""VCR-recorded integration test for Gamma client.

The cassette file `tests/cassettes/gamma_list_markets.yaml` contains a
synthetic but schema-realistic Gamma response. The test replays it and
asserts the client correctly normalises markets into MarketSummary objects.

To re-record against the live API, set POLYCLAUDE_VCR_RECORD=1 and run pytest.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from polyclaude.markets.gamma import GammaClient
from polyclaude.markets.india import is_india_market

CASSETTE = Path(__file__).parent / "cassettes" / "gamma_list_markets.yaml"
RECORD = os.getenv("POLYCLAUDE_VCR_RECORD") == "1"

vcr = pytest.importorskip("vcr")


@pytest.fixture
def cassette():
    record_mode = "all" if RECORD else "none"
    cfg = vcr.VCR(
        cassette_library_dir=str(CASSETTE.parent),
        record_mode=record_mode,
        match_on=["method", "scheme", "host", "path", "query"],
        decode_compressed_response=True,
    )
    return cfg


def test_gamma_list_markets_replays(cassette):
    with cassette.use_cassette("gamma_list_markets.yaml"):
        async def _call():
            async with GammaClient() as g:
                return await g.list_markets(limit=2)

        markets = asyncio.run(_call())

    assert len(markets) == 2
    m1, m2 = markets
    assert m1.id == "0x_test_market_001"
    assert m1.yes_token_id == "tok_yes_001"
    assert m1.no_token_id == "tok_no_001"
    assert pytest.approx(m1.volume, rel=1e-9) == 12345.6
    assert m1.tags == ["test", "synthetic"]
    assert not is_india_market(m1)

    assert m2.id == "0x_test_market_002"
    assert is_india_market(m2)
    # End-date parsed
    assert m2.end_date is not None
