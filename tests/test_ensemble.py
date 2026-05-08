from datetime import datetime, timedelta, timezone

from polyclaude.nlp.ensemble import combine, z_vs_baseline


def test_combine_decay_reduces_score_over_time():
    now = datetime.now(timezone.utc)
    fresh = combine(
        finbert={"polarity": 0.8, "intensity": 0.9},
        vader={"compound": 0.6},
        event_ts=now, now=now, half_life_hours=24,
    )
    stale = combine(
        finbert={"polarity": 0.8, "intensity": 0.9},
        vader={"compound": 0.6},
        event_ts=now - timedelta(hours=48), now=now, half_life_hours=24,
    )
    assert abs(stale.ensemble_score) < abs(fresh.ensemble_score)


def test_z_vs_baseline_zero_when_history_short():
    assert z_vs_baseline(1.0, [0.0] * 5) == 0.0


def test_z_vs_baseline_high_when_score_outlier():
    history = [0.0] * 50
    z = z_vs_baseline(1.0, history)
    assert z > 5
