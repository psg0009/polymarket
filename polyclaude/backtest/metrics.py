"""Backtest metrics: Sharpe, Brier, calibration bins, max drawdown."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class CalibrationBin:
    lo: float
    hi: float
    n: int
    mean_p: float
    empirical_yes_rate: float


def brier(points: list[tuple[float, int]]) -> float:
    if not points:
        return float("nan")
    return sum((p - y) ** 2 for p, y in points) / len(points)


def log_loss(points: list[tuple[float, int]], eps: float = 1e-6) -> float:
    if not points:
        return float("nan")
    s = 0.0
    for p, y in points:
        p = min(max(p, eps), 1 - eps)
        s += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return s / len(points)


def calibration_bins(points: list[tuple[float, int]], n_bins: int = 10) -> list[CalibrationBin]:
    bins: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for p, y in points:
        idx = min(n_bins - 1, max(0, int(p * n_bins)))
        bins[idx].append((p, y))
    out: list[CalibrationBin] = []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        rows = bins.get(i, [])
        if not rows:
            out.append(CalibrationBin(lo, hi, 0, (lo + hi) / 2, 0.0))
            continue
        mean_p = sum(p for p, _ in rows) / len(rows)
        emp = sum(y for _, y in rows) / len(rows)
        out.append(CalibrationBin(lo, hi, len(rows), mean_p, emp))
    return out


def sharpe(returns: list[float]) -> float:
    if not returns:
        return float("nan")
    mu = sum(returns) / len(returns)
    var = sum((r - mu) ** 2 for r in returns) / len(returns)
    sd = math.sqrt(var)
    if sd == 0:
        return float("nan")
    return mu / sd * math.sqrt(252)


def max_drawdown(equity: list[float]) -> float:
    if not equity:
        return 0.0
    peak = equity[0]
    dd = 0.0
    for v in equity:
        peak = max(peak, v)
        if peak > 0:
            dd = min(dd, (v - peak) / peak)
    return dd  # negative number


def hit_rate(points: list[tuple[float, int]]) -> float:
    """Fraction of trades whose direction matched outcome."""
    if not points:
        return 0.0
    correct = sum(1 for p, y in points if (p > 0.5) == bool(y))
    return correct / len(points)
