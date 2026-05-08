from datetime import datetime, timezone

from polyclaude.markets.gamma import MarketSummary
from polyclaude.markets.india import is_india_market, vertical_for


def _m(**kw):
    base = dict(
        id="m1", question="", description="", resolution_rules="",
        category=None, tags=[], end_date=datetime.now(timezone.utc),
    )
    base.update(kw)
    return MarketSummary(**base)


def test_tag_match():
    assert is_india_market(_m(tags=["india", "elections"]))


def test_keyword_match_modi():
    assert is_india_market(_m(question="Will Modi win the next general election?"))


def test_keyword_match_rbi():
    assert is_india_market(_m(question="Will RBI cut the repo rate in December?"))


def test_no_false_positive_us_market():
    assert not is_india_market(
        _m(question="Will the SP500 close above 6000?", description="US equities", tags=["us"])
    )


def test_vertical_election():
    assert vertical_for(_m(question="Lok Sabha seat count for BJP")) == "india_election"


def test_vertical_rbi():
    assert vertical_for(_m(question="RBI MPC repo rate decision")) == "india_rbi"


def test_vertical_cricket():
    assert vertical_for(_m(question="Will CSK win the IPL final?")) == "india_cricket"
