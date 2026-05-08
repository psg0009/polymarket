"""JSON parser robustness for the oracle."""

from polyclaude.oracle.claude import (
    AmbiguityResponse, ProbabilityResponse, SizingResponse, extract_json,
)


def test_extract_json_plain():
    s = '{"p": 0.42, "confidence": 0.7, "key_factors": [], "rationale": "x"}'
    assert extract_json(s)["p"] == 0.42


def test_extract_json_fenced():
    s = "Here is the JSON:\n```json\n{\"p\": 0.5, \"confidence\": 0.5, \"key_factors\": [], \"rationale\": \"x\"}\n```"
    assert extract_json(s)["p"] == 0.5


def test_extract_json_with_prose():
    s = "Sure thing — {\"p\": 0.33, \"confidence\": 0.4, \"key_factors\": [], \"rationale\": \"x\"} that's it."
    assert extract_json(s)["p"] == 0.33


def test_extract_json_malformed_returns_none():
    assert extract_json("not json at all") is None


def test_probability_response_clamps_and_validates():
    r = ProbabilityResponse.model_validate({
        "p": 0.7, "confidence": 0.8, "key_factors": ["a", "b"],
        "rationale": "x", "reference_class": "y",
    })
    assert r.p == 0.7
    assert r.confidence == 0.8


def test_probability_response_rejects_out_of_range():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ProbabilityResponse.model_validate({
            "p": 1.5, "confidence": 0.5, "key_factors": [],
            "rationale": "x", "reference_class": "y",
        })


def test_ambiguity_verdict_enum():
    import pytest
    from pydantic import ValidationError

    AmbiguityResponse.model_validate({
        "clarity": 0.8, "edge_cases": [], "resolution_source_quality": 0.9, "verdict": "clear",
    })
    with pytest.raises(ValidationError):
        AmbiguityResponse.model_validate({
            "clarity": 0.8, "edge_cases": [], "resolution_source_quality": 0.9, "verdict": "fuzzy",
        })


def test_sizing_side_enum():
    SizingResponse.model_validate({
        "would_trade": True, "side": "YES", "edge_bps": 200,
        "kelly_fraction": 0.1, "reasoning": "ok",
    })
