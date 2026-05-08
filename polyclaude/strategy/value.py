"""The simplest strategy: trade if |edge| > threshold and book depth is sane.

Composes the ambiguity gate + sizing module + executor into one decide() call.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from decimal import Decimal

from polyclaude.config import Settings, get_settings
from polyclaude.ledger.db import DecisionAction, Side
from polyclaude.oracle.ambiguity import AmbiguityGate
from polyclaude.oracle.claude import ClaudeOracle, ProbabilityResponse
from polyclaude.strategy.sizing import RiskCaps, SizingResult, size_trade


@dataclass
class StrategyDecision:
    decision_group_id: str
    market_id: str
    strategy: str
    committed_p: float
    confidence: float
    market_p: float
    sizing: SizingResult
    action: DecisionAction
    ambiguity_clarity: float | None
    skip_reason: str | None


class ValueStrategy:
    name = "value"

    def __init__(
        self,
        oracle: ClaudeOracle,
        settings: Settings | None = None,
        ambiguity_gate: AmbiguityGate | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.oracle = oracle
        self.gate = ambiguity_gate or AmbiguityGate(min_clarity=self.settings.min_ambiguity_clarity)

    def caps(self) -> RiskCaps:
        s = self.settings
        return RiskCaps(
            max_notional_per_trade=s.max_notional_per_trade,
            max_notional_per_day=s.max_notional_per_day,
            kelly_fraction=s.kelly_fraction,
            min_edge_bps=s.min_edge_bps,
            min_book_depth_usd=s.min_book_depth_usd,
        )

    def decide(
        self,
        market: dict,
        evidence: list[dict],
        market_yes_price: float,
        book_depth_yes_usd: Decimal,
        book_depth_no_usd: Decimal,
        bankroll_usd: Decimal,
        daily_used_usd: Decimal,
        signal_volatility: float = 0.0,
    ) -> tuple[StrategyDecision, ProbabilityResponse | None, list]:
        gid = str(uuid.uuid4())
        traces = []

        # 1) Ambiguity gate
        amb_resp, amb_trace = self.oracle.evaluate_ambiguity(market, decision_group_id=gid)
        traces.append(amb_trace)
        allowed, mult, amb_reason = self.gate.decision(amb_resp)
        if not allowed:
            return (
                StrategyDecision(
                    decision_group_id=gid,
                    market_id=market["id"], strategy=self.name,
                    committed_p=0.0, confidence=0.0, market_p=market_yes_price,
                    sizing=SizingResult("NONE", 0, 0.0, 0.0, Decimal("0"), Decimal(str(market_yes_price)),
                                        skip_reason=amb_reason),
                    action=DecisionAction.skip_ambiguous,
                    ambiguity_clarity=amb_resp.clarity if amb_resp else None,
                    skip_reason=amb_reason,
                ),
                None,
                traces,
            )

        # 2) Probability call (no anchoring on price)
        prob_resp, prob_trace = self.oracle.evaluate_probability(market, evidence, decision_group_id=gid)
        traces.append(prob_trace)
        if prob_resp is None:
            return (
                StrategyDecision(
                    decision_group_id=gid,
                    market_id=market["id"], strategy=self.name,
                    committed_p=0.0, confidence=0.0, market_p=market_yes_price,
                    sizing=SizingResult("NONE", 0, 0.0, 0.0, Decimal("0"), Decimal(str(market_yes_price)),
                                        skip_reason="probability oracle failed"),
                    action=DecisionAction.skip_ambiguous,
                    ambiguity_clarity=amb_resp.clarity if amb_resp else None,
                    skip_reason="probability oracle failed",
                ),
                None,
                traces,
            )

        # 3) Sizing
        sizing = size_trade(
            p=prob_resp.p, yes_price=market_yes_price,
            bankroll_usd=bankroll_usd, daily_used_usd=daily_used_usd,
            book_depth_yes_usd=book_depth_yes_usd, book_depth_no_usd=book_depth_no_usd,
            caps=self.caps(), ambiguity_multiplier=mult,
        )

        if sizing.side == "NONE":
            action = DecisionAction.skip_low_edge
        elif sizing.notional_usd <= 0:
            action = (DecisionAction.skip_thin_book
                      if (sizing.skip_reason or "").startswith("book depth")
                      else DecisionAction.skip_risk_cap)
        else:
            action = DecisionAction.place

        return (
            StrategyDecision(
                decision_group_id=gid,
                market_id=market["id"], strategy=self.name,
                committed_p=prob_resp.p, confidence=prob_resp.confidence,
                market_p=market_yes_price, sizing=sizing, action=action,
                ambiguity_clarity=amb_resp.clarity if amb_resp else None,
                skip_reason=sizing.skip_reason,
            ),
            prob_resp,
            traces,
        )

    @staticmethod
    def to_side_enum(side: str) -> Side:
        return Side(side) if side in ("YES", "NO") else Side.NONE
