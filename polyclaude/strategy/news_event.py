"""News-event strategy: re-evaluate on SignalScore spikes.

Fires when a Signal arrives with z_vs_baseline above threshold. Inherits the
ValueStrategy decide() pipeline but reduces gating thresholds — we expect more
edge and less time to capture it.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from polyclaude.strategy.sizing import RiskCaps
from polyclaude.strategy.value import ValueStrategy


@dataclass
class NewsEventStrategy:
    inner: ValueStrategy
    z_threshold: float = 3.0
    aggressive_edge_bps: int = 500
    name: str = "news_event"

    def caps(self) -> RiskCaps:
        base = self.inner.caps()
        # On news-event paths we tolerate slightly thinner books (latency-sensitive)
        return RiskCaps(
            max_notional_per_trade=base.max_notional_per_trade,
            max_notional_per_day=base.max_notional_per_day,
            kelly_fraction=base.kelly_fraction,
            min_edge_bps=base.min_edge_bps,
            min_book_depth_usd=Decimal(str(float(base.min_book_depth_usd) * 0.6)),
        )

    def should_fire(self, z: float) -> bool:
        return abs(z) >= self.z_threshold

    def decide(self, *args, **kwargs):
        return self.inner.decide(*args, **kwargs)
