"""Sizing math: Kelly edge cases, cap enforcement, side selection."""

from decimal import Decimal

import pytest

from polyclaude.strategy.sizing import RiskCaps, kelly_binary, size_trade


def test_kelly_basic():
    # p=0.6, q=0.5 → kelly = (0.6-0.5)/(1-0.5) = 0.2
    assert kelly_binary(0.6, 0.5) == pytest.approx(0.2, abs=1e-9)


def test_kelly_negative_when_yes_overpriced():
    # Market is overpriced YES; canonical kelly is negative.
    k = kelly_binary(0.4, 0.6)
    assert k < 0


def test_kelly_extremes_safe():
    assert kelly_binary(0.5, 0.0) == 0.0  # invalid q
    assert kelly_binary(0.5, 1.0) == 0.0  # invalid q
    assert kelly_binary(1.0, 0.5) == 1.0


def _caps(**kw):
    base = dict(
        max_notional_per_trade=Decimal("25"),
        max_notional_per_day=Decimal("100"),
        kelly_fraction=0.25,
        min_edge_bps=300,
        min_book_depth_usd=Decimal("50"),
    )
    base.update(kw)
    return RiskCaps(**base)


def test_size_trade_below_edge_skips():
    res = size_trade(
        p=0.52, yes_price=0.50,
        bankroll_usd=Decimal("1000"), daily_used_usd=Decimal("0"),
        book_depth_yes_usd=Decimal("1000"), book_depth_no_usd=Decimal("1000"),
        caps=_caps(),
    )
    assert res.side == "NONE"
    assert res.notional_usd == Decimal("0")
    assert "edge" in (res.skip_reason or "")


def test_size_trade_thin_book_skips():
    res = size_trade(
        p=0.70, yes_price=0.50,
        bankroll_usd=Decimal("1000"), daily_used_usd=Decimal("0"),
        book_depth_yes_usd=Decimal("10"), book_depth_no_usd=Decimal("10"),
        caps=_caps(),
    )
    assert res.notional_usd == Decimal("0")
    assert "book depth" in (res.skip_reason or "")


def test_size_trade_takes_yes_side_when_edge_positive():
    res = size_trade(
        p=0.70, yes_price=0.50,
        bankroll_usd=Decimal("1000"), daily_used_usd=Decimal("0"),
        book_depth_yes_usd=Decimal("500"), book_depth_no_usd=Decimal("500"),
        caps=_caps(),
    )
    assert res.side == "YES"
    assert res.edge_bps == 2000
    assert res.notional_usd > 0
    assert res.price == Decimal("0.5")


def test_size_trade_takes_no_side_when_edge_negative():
    res = size_trade(
        p=0.30, yes_price=0.50,
        bankroll_usd=Decimal("1000"), daily_used_usd=Decimal("0"),
        book_depth_yes_usd=Decimal("500"), book_depth_no_usd=Decimal("500"),
        caps=_caps(),
    )
    assert res.side == "NO"
    assert res.edge_bps == -2000
    assert res.notional_usd > 0
    # Cost basis for NO is 1 - yes_price
    assert res.price == Decimal("0.5")


def test_per_trade_cap_enforced():
    caps = _caps(max_notional_per_trade=Decimal("5"))
    res = size_trade(
        p=0.95, yes_price=0.50,
        bankroll_usd=Decimal("10000"), daily_used_usd=Decimal("0"),
        book_depth_yes_usd=Decimal("9999"), book_depth_no_usd=Decimal("9999"),
        caps=caps,
    )
    assert res.notional_usd == Decimal("5")


def test_daily_cap_clips_remaining():
    caps = _caps(max_notional_per_day=Decimal("20"), max_notional_per_trade=Decimal("100"))
    res = size_trade(
        p=0.95, yes_price=0.50,
        bankroll_usd=Decimal("10000"), daily_used_usd=Decimal("15"),
        book_depth_yes_usd=Decimal("9999"), book_depth_no_usd=Decimal("9999"),
        caps=caps,
    )
    # 20 cap - 15 used = 5 remaining
    assert res.notional_usd == Decimal("5")


def test_zero_bankroll_zero_size():
    res = size_trade(
        p=0.95, yes_price=0.50,
        bankroll_usd=Decimal("0"), daily_used_usd=Decimal("0"),
        book_depth_yes_usd=Decimal("9999"), book_depth_no_usd=Decimal("9999"),
        caps=_caps(),
    )
    assert res.notional_usd == Decimal("0")


def test_ambiguity_multiplier_halves_size():
    res_full = size_trade(
        p=0.70, yes_price=0.50,
        bankroll_usd=Decimal("1000"), daily_used_usd=Decimal("0"),
        book_depth_yes_usd=Decimal("9999"), book_depth_no_usd=Decimal("9999"),
        caps=_caps(),
    )
    res_half = size_trade(
        p=0.70, yes_price=0.50,
        bankroll_usd=Decimal("1000"), daily_used_usd=Decimal("0"),
        book_depth_yes_usd=Decimal("9999"), book_depth_no_usd=Decimal("9999"),
        caps=_caps(),
        ambiguity_multiplier=0.5,
    )
    # Half should be roughly half-ish (clamped by caps when relevant)
    assert res_half.scaled_fraction == pytest.approx(res_full.scaled_fraction * 0.5, rel=1e-9)


def test_extreme_price_skipped():
    for q in (0.001, 0.999):
        res = size_trade(
            p=0.5, yes_price=q,
            bankroll_usd=Decimal("1000"), daily_used_usd=Decimal("0"),
            book_depth_yes_usd=Decimal("9999"), book_depth_no_usd=Decimal("9999"),
            caps=_caps(),
        )
        assert res.side == "NONE"
