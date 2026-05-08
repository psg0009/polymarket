"""Hard daily Anthropic-spend ceiling.

Tests:
1. call_cost_usd applies the right per-million-token rates per model.
2. _today_spend_usd reads OracleCall rows from the ledger and sums correctly.
3. ClaudeOracle._call short-circuits with a budget-exceeded CallTrace when
   today's spend already exceeds the configured budget.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from polyclaude.ledger.db import OracleCall, get_session, init_db
from polyclaude.oracle.claude import ClaudeOracle, _today_spend_usd, call_cost_usd


def test_call_cost_opus_pricing():
    # 1k input, 0.5k output → 1000 * 15/1M + 500 * 75/1M = 0.015 + 0.0375 = 0.0525
    cost = call_cost_usd("claude-opus-4-7", 1000, 500)
    assert cost == pytest.approx(0.0525, rel=1e-9)


def test_call_cost_haiku_pricing():
    # Haiku: 1000 in × $1/M + 500 out × $5/M = 0.001 + 0.0025 = 0.0035
    cost = call_cost_usd("claude-haiku-4-5", 1000, 500)
    assert cost == pytest.approx(0.0035, rel=1e-9)


def test_call_cost_unknown_model_is_conservative():
    # Unknown models get charged Opus rates (under-budget on the safe side).
    cost = call_cost_usd("some-future-model", 1000, 500)
    assert cost == pytest.approx(0.0525, rel=1e-9)


def test_today_spend_zero_on_empty_ledger():
    init_db()
    assert _today_spend_usd() == 0.0


def test_today_spend_sums_oracle_calls(monkeypatch):
    init_db()
    now = datetime.now(timezone.utc)

    # 1 Opus call (3000 in, 500 out) and 2 Haiku calls (1500 in, 300 out each)
    with get_session() as s:
        s.add(OracleCall(
            decision_group_id="d1", market_id="m1", call_type="probability",
            model="claude-opus-4-7", prompt_system="x", prompt_user="x",
            raw_response="{}", parsed_response={},
            input_tokens=3000, output_tokens=500, latency_ms=100, ts=now,
        ))
        s.add(OracleCall(
            decision_group_id="d1", market_id="m1", call_type="ambiguity",
            model="claude-haiku-4-5-20251001", prompt_system="x", prompt_user="x",
            raw_response="{}", parsed_response={},
            input_tokens=1500, output_tokens=300, latency_ms=100, ts=now,
        ))
        s.add(OracleCall(
            decision_group_id="d2", market_id="m2", call_type="ambiguity",
            model="claude-haiku-4-5-20251001", prompt_system="x", prompt_user="x",
            raw_response="{}", parsed_response={},
            input_tokens=1500, output_tokens=300, latency_ms=100, ts=now,
        ))
        s.commit()

    # Opus: 3000 * 15/1M + 500 * 75/1M = 0.045 + 0.0375 = 0.0825
    # Haiku ×2: 2 * (1500 * 1/1M + 300 * 5/1M) = 2 * 0.003 = 0.006
    # Total: 0.0885
    assert _today_spend_usd() == pytest.approx(0.0885, rel=1e-9)


def test_today_spend_excludes_yesterday():
    init_db()
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    with get_session() as s:
        s.add(OracleCall(
            decision_group_id="d_old", market_id="m1", call_type="probability",
            model="claude-opus-4-7", prompt_system="x", prompt_user="x",
            raw_response="{}", parsed_response={},
            input_tokens=10000, output_tokens=2000, latency_ms=100, ts=yesterday,
        ))
        s.commit()
    assert _today_spend_usd() == 0.0


def test_oracle_call_short_circuits_when_over_budget(monkeypatch):
    """When today's spend ≥ budget, _call returns immediately without invoking
    the Anthropic SDK. The returned CallTrace carries an explicit error string."""
    init_db()
    now = datetime.now(timezone.utc)
    # Insert one call that pushes today's spend to ~$1.05 worth so a $1 budget is breached.
    # Opus 100_000 input + 10_000 output = 100_000*15/1M + 10_000*75/1M = 1.5 + 0.75 = 2.25
    with get_session() as s:
        s.add(OracleCall(
            decision_group_id="d", market_id="m", call_type="probability",
            model="claude-opus-4-7", prompt_system="x", prompt_user="x",
            raw_response="{}", parsed_response={},
            input_tokens=100_000, output_tokens=10_000, latency_ms=100, ts=now,
        ))
        s.commit()

    monkeypatch.setenv("ANTHROPIC_DAILY_USD_BUDGET", "1.0")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    from polyclaude import config as cfg

    cfg.get_settings.cache_clear()
    oracle = ClaudeOracle()

    # Sanity-check the budget read from settings.
    assert float(oracle.settings.anthropic_daily_usd_budget) == 1.0

    # Track whether we'd have hit the SDK.
    sdk_called = {"value": False}

    class _FakeMessages:
        def create(self, **_):
            sdk_called["value"] = True
            raise AssertionError("SDK should not be reached when over budget")

    class _FakeClient:
        messages = _FakeMessages()

    oracle._client = _FakeClient()

    raw, parsed, trace = oracle._call(
        "sys", "user", "decision-id", "probability", model="claude-opus-4-7",
    )

    assert sdk_called["value"] is False
    assert raw == ""
    assert parsed == {}
    assert "budget exceeded" in (trace.error or "")
