"""Top-level backtest entry point used by `polyclaude backtest`."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from polyclaude.backtest.metrics import (
    brier, calibration_bins, hit_rate, log_loss, max_drawdown,
)
from polyclaude.backtest.replay import replay


def run(*, start: datetime, end: datetime, strategy: str, capital: Decimal) -> dict:
    res = replay(start=start, end=end, capital=capital, strategy_name=strategy, use_oracle=True)
    eq = [v for _, v in res.equity_curve]
    out = {
        "trades": len(res.trades),
        "final_bankroll": eq[-1] if eq else float(capital),
        "max_drawdown": max_drawdown(eq),
        "brier": brier(res.brier_points),
        "log_loss": log_loss(res.brier_points),
        "hit_rate": hit_rate(res.brier_points),
        "calibration_bins": [
            {"lo": b.lo, "hi": b.hi, "n": b.n, "mean_p": b.mean_p, "yes_rate": b.empirical_yes_rate}
            for b in calibration_bins(res.brier_points)
        ],
        "pnl_by_strategy": dict(res.by_strategy_pnl),
    }
    return out
