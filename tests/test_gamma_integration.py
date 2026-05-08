"""VCR-driven integration tests for the Gamma client + outcome parser.

The cassette `tests/cassettes/gamma_list_markets.yaml` covers five market
shapes — see `tests/cassettes/RECORDING.md`.

To re-record against the live API, set POLYCLAUDE_VCR_RECORD=1 and run pytest
from any environment with network access.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from polyclaude.ledger.reconcile import _extract_outcome
from polyclaude.markets.gamma import GammaClient
from polyclaude.markets.india import is_india_market

CASSETTE = Path(__file__).parent / "cassettes" / "gamma_list_markets.yaml"
RECORD = os.getenv("POLYCLAUDE_VCR_RECORD") == "1"

vcr = pytest.importorskip("vcr")


@pytest.fixture
def cassette():
    record_mode = "all" if RECORD else "none"
    return vcr.VCR(
        cassette_library_dir=str(CASSETTE.parent),
        record_mode=record_mode,
        match_on=["method", "scheme", "host", "path", "query"],
        decode_compressed_response=True,
    )


def test_gamma_list_markets_replays_active(cassette):
    with cassette.use_cassette("gamma_list_markets.yaml"):

        async def _call():
            async with GammaClient() as g:
                return await g.list_markets(limit=5)

        markets = asyncio.run(_call())

    assert len(markets) == 5
    assert markets[0].id == "0x_test_market_001"
    assert markets[0].yes_token_id == "tok_yes_001"
    assert markets[0].no_token_id == "tok_no_001"
    assert markets[0].volume == pytest.approx(12345.6)
    assert not is_india_market(markets[0])


def test_gamma_list_markets_replays_india(cassette):
    with cassette.use_cassette("gamma_list_markets.yaml"):

        async def _call():
            async with GammaClient() as g:
                return await g.list_markets(limit=5)

        markets = asyncio.run(_call())

    india = next(m for m in markets if m.id == "0x_test_market_002")
    assert is_india_market(india)
    assert india.end_date is not None


def test_extract_outcome_winner_flag(cassette):
    with cassette.use_cassette("gamma_list_markets.yaml"):

        async def _call():
            async with GammaClient() as g:
                return await g.list_markets(limit=5)

        markets = asyncio.run(_call())

    resolved = next(m for m in markets if m.id == "0x_test_resolved_yes")
    assert resolved.closed
    assert _extract_outcome(resolved.raw) == "YES"


def test_extract_outcome_price_form(cassette):
    with cassette.use_cassette("gamma_list_markets.yaml"):

        async def _call():
            async with GammaClient() as g:
                return await g.list_markets(limit=5)

        markets = asyncio.run(_call())

    resolved = next(m for m in markets if m.id == "0x_test_resolved_price")
    assert _extract_outcome(resolved.raw) == "YES"


def test_extract_outcome_jsonstr_form(cassette):
    with cassette.use_cassette("gamma_list_markets.yaml"):

        async def _call():
            async with GammaClient() as g:
                return await g.list_markets(limit=5)

        markets = asyncio.run(_call())

    resolved = next(m for m in markets if m.id == "0x_test_jsonstr_outcomes")
    # The clobTokenIds JSON-string was parsed into yes/no token ids.
    assert resolved.yes_token_id == "tok_yes_005"
    assert resolved.no_token_id == "tok_no_005"
    assert _extract_outcome(resolved.raw) == "YES"
