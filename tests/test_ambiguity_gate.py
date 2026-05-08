from polyclaude.oracle.ambiguity import AmbiguityGate
from polyclaude.oracle.claude import AmbiguityResponse


def _resp(clarity, verdict="clear"):
    return AmbiguityResponse(
        clarity=clarity, edge_cases=[], resolution_source_quality=0.9, verdict=verdict
    )


def test_gate_blocks_below_min_clarity():
    gate = AmbiguityGate(min_clarity=0.7)
    allow, mult, reason = gate.decision(_resp(0.5))
    assert not allow
    assert mult == 0.0


def test_gate_halves_in_middle_band():
    gate = AmbiguityGate(min_clarity=0.7, halve_below=0.85)
    allow, mult, reason = gate.decision(_resp(0.8))
    assert allow
    assert mult == 0.5


def test_gate_full_size_when_clear():
    gate = AmbiguityGate()
    allow, mult, _ = gate.decision(_resp(0.95))
    assert allow and mult == 1.0


def test_hostile_blocked_even_with_high_clarity():
    gate = AmbiguityGate()
    allow, _, _ = gate.decision(_resp(0.99, verdict="hostile"))
    assert not allow


def test_none_response_blocks():
    gate = AmbiguityGate()
    allow, mult, _ = gate.decision(None)
    assert not allow and mult == 0.0
