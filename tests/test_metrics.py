from polyclaude.backtest.metrics import (
    brier, calibration_bins, hit_rate, log_loss, max_drawdown,
)


def test_brier_perfect_calibration_zero():
    assert brier([(1.0, 1), (0.0, 0)]) == 0.0


def test_brier_worst_case_one():
    assert brier([(1.0, 0), (0.0, 1)]) == 1.0


def test_brier_uniform_quarter():
    # All 0.5 predictions: brier = 0.25 regardless of outcomes
    assert brier([(0.5, 1), (0.5, 0), (0.5, 1)]) == 0.25


def test_log_loss_finite_at_extremes():
    val = log_loss([(0.999, 0)])
    assert val > 0


def test_calibration_bins_basic():
    points = [(0.05, 0), (0.15, 0), (0.55, 1), (0.95, 1)]
    bins = calibration_bins(points, n_bins=10)
    assert len(bins) == 10


def test_max_drawdown_negative_or_zero():
    assert max_drawdown([100, 90, 110, 80, 120]) <= 0
    assert max_drawdown([]) == 0.0


def test_hit_rate():
    points = [(0.7, 1), (0.3, 0), (0.6, 0), (0.4, 1)]
    # Correct: (0.7,1) and (0.3,0) → 2/4 = 0.5
    assert hit_rate(points) == 0.5
