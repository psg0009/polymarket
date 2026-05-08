"""Shadow-mode CI re-scoring.

Re-runs the live oracle pipeline against the last N days of resolved markets
and asserts that overall Brier hasn't regressed below a checked-in baseline.

Used by .github/workflows/shadow.yml to guard prompt / strategy changes.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from sqlalchemy import select

from polyclaude.backtest.metrics import brier, calibration_bins, hit_rate, log_loss
from polyclaude.backtest.replay import replay
from polyclaude.ledger.db import CalibrationPoint, get_session
from polyclaude.logging_setup import get_logger

log = get_logger(__name__)


def run(days: int = 7) -> dict:
    """Compute Brier / log-loss / hit-rate for resolved markets in the last N days.

    Two sources are considered:
    1. CalibrationPoints already in the ledger (cheap, no API calls).
    2. If `days` is in the past and the ledger has no CalibrationPoints, fall back
       to running `backtest.replay` on the same window so CI still produces a number.
    """
    since = datetime.now(timezone.utc) - timedelta(days=days)
    with get_session() as session:
        rows = session.execute(
            select(CalibrationPoint.p_predicted, CalibrationPoint.outcome_yes,
                   CalibrationPoint.confidence)
            .where(CalibrationPoint.resolved_at >= since)
        ).all()
    points = [(float(p), 1 if y else 0) for p, y, _ in rows]
    if points:
        out = {
            "n": len(points),
            "brier": brier(points),
            "log_loss": log_loss(points),
            "hit_rate": hit_rate(points),
            "calibration_bins": [
                {"lo": b.lo, "hi": b.hi, "n": b.n, "mean_p": b.mean_p, "yes_rate": b.empirical_yes_rate}
                for b in calibration_bins(points)
            ],
            "source": "calibration_points",
        }
        log.info("shadow.from_ledger", **{k: v for k, v in out.items() if k != "calibration_bins"})
        return out

    end = datetime.now(timezone.utc)
    res = replay(start=since, end=end, capital=Decimal("1000"), use_oracle=False)
    out = {
        "n": len(res.brier_points),
        "brier": brier(res.brier_points),
        "log_loss": log_loss(res.brier_points),
        "hit_rate": hit_rate(res.brier_points),
        "source": "backtest_replay",
    }
    log.info("shadow.from_replay", **out)
    return out
