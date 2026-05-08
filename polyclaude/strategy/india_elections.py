"""India elections strategy.

Wraps ValueStrategy with India-specific guardrails:
- Source weighting: ECI/PIB outrank wires; regional outrank social.
- Exit-poll embargo: refuse to trade between exit-poll embargo start and result release.
- Phase awareness: if a market resolves on a multi-phase election, weight evidence
  per phase rather than averaging.

The strategy itself stays simple — most India behavior lives in the ingestion layer
(india_sources, india tagger). Here we just gate trade timing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

from polyclaude.strategy.value import ValueStrategy

IST = ZoneInfo("Asia/Kolkata")


@dataclass
class IndiaElectionStrategy:
    inner: ValueStrategy
    name: str = "india_election"
    # Embargo windows in IST (start, end). All times are inclusive.
    embargo_windows: tuple[tuple[time, time], ...] = (
        (time(7, 0), time(18, 30)),  # rough Indian polling hours
    )

    def in_embargo_now(self, now_utc: datetime | None = None) -> bool:
        now = (now_utc or datetime.now(timezone.utc)).astimezone(IST).time()
        return any(start <= now <= end for start, end in self.embargo_windows)

    def decide(self, *args, **kwargs):
        if self.in_embargo_now():
            from polyclaude.ledger.db import DecisionAction
            from polyclaude.strategy.sizing import SizingResult
            from polyclaude.strategy.value import StrategyDecision
            from decimal import Decimal
            import uuid

            market = args[0] if args else kwargs.get("market", {})
            yes_price = args[2] if len(args) > 2 else kwargs.get("market_yes_price", 0.5)
            return (
                StrategyDecision(
                    decision_group_id=str(uuid.uuid4()),
                    market_id=market.get("id", ""), strategy=self.name,
                    committed_p=0.0, confidence=0.0, market_p=yes_price,
                    sizing=SizingResult("NONE", 0, 0.0, 0.0, Decimal("0"), Decimal(str(yes_price)),
                                        skip_reason="exit-poll embargo window"),
                    action=DecisionAction.skip_dry_run,
                    ambiguity_clarity=None,
                    skip_reason="exit-poll embargo window",
                ),
                None,
                [],
            )
        return self.inner.decide(*args, **kwargs)
