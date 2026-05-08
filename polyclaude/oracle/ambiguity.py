"""Ambiguity gate logic, separate from the Claude call so it's easy to test."""

from __future__ import annotations

from dataclasses import dataclass

from polyclaude.oracle.claude import AmbiguityResponse


@dataclass
class AmbiguityGate:
    min_clarity: float = 0.7
    halve_below: float = 0.85   # clarity in [0.7, 0.85) → halve size

    def decision(self, resp: AmbiguityResponse | None) -> tuple[bool, float, str]:
        """Return (allow_trade, size_multiplier, reason)."""
        if resp is None:
            return False, 0.0, "ambiguity oracle returned no response"
        if resp.verdict == "hostile":
            return False, 0.0, "ambiguity verdict: hostile"
        if resp.clarity < self.min_clarity:
            return False, 0.0, f"clarity {resp.clarity:.2f} < {self.min_clarity:.2f}"
        if resp.clarity < self.halve_below:
            return True, 0.5, f"clarity {resp.clarity:.2f}, halving size"
        return True, 1.0, "ok"
