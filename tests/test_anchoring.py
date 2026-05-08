"""Verify that the probability call NEVER receives the market price.

This is the single most important calibration rule in the system. We test it
two ways:
1. Directly: prompts.probability_user() with a market dict containing yes_price
   must not surface that price in the rendered string.
2. Indirectly: oracle.evaluate_probability sanitizes the market dict before
   calling the API.
"""

from polyclaude.oracle import prompts as P


def test_probability_user_does_not_render_price_field():
    market = {
        "id": "x", "question": "Will X happen?", "description": "...",
        "resolution_rules": "...", "resolves_at": "2026-01-01",
        "yes_price": 0.42, "midpoint": 0.42, "market_p": 0.42,
        "category": "general", "tags": [],
    }
    out = P.probability_user(market, evidence=[])
    assert "yes_price" not in out
    assert "0.42" not in out


def test_sizing_user_does_render_price():
    """The sizing call SHOULD see the price — just not the probability call."""
    market = {"hours_to_resolution": 24}
    out = P.sizing_user(
        market, committed_p=0.6, market_yes_price=0.42,
        book_depth_yes=100, book_depth_no=100, signal_volatility=0.5, bankroll_usd=100,
    )
    assert "0.4200" in out or "0.42" in out


def test_oracle_strips_price_from_market_dict_before_probability_call(monkeypatch):
    from polyclaude.oracle.claude import ClaudeOracle

    oracle = ClaudeOracle()
    captured = {}

    def fake_call(self, system, user, gid, call_type, model=None, max_tokens=1024):
        captured.update({"system": system, "user": user, "call_type": call_type})
        return "{}", {}, type("T", (), {
            "decision_group_id": gid, "call_type": call_type, "model": model or "x",
            "prompt_system": system, "prompt_user": user, "raw_response": "{}",
            "parsed": {}, "input_tokens": None, "output_tokens": None, "latency_ms": 0,
            "error": None,
        })()

    monkeypatch.setattr(ClaudeOracle, "_call", fake_call, raising=True)
    market = {
        "id": "x", "question": "Will X?", "description": "...", "resolution_rules": "...",
        "resolves_at": "2026-01-01", "yes_price": 0.42, "midpoint": 0.42, "market_p": 0.42,
        "category": "general", "tags": [],
    }
    oracle.evaluate_probability(market, evidence=[])
    assert "0.42" not in captured["user"]
    assert "yes_price" not in captured["user"]
