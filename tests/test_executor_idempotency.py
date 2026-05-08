"""Idempotency keys must collapse retries of the same logical request."""

from decimal import Decimal

from polyclaude.clob.executor import PlaceRequest, _idempotency_key


def _req(**kw):
    base = dict(
        decision_id="d1", market_id="m1", token_id="t1",
        side="YES", price=Decimal("0.501"), size_shares=Decimal("10"),
    )
    base.update(kw)
    return PlaceRequest(**base)


def test_same_request_same_key():
    r1 = _req(attempt=0)
    r2 = _req(attempt=0)
    assert _idempotency_key(r1) == _idempotency_key(r2)


def test_different_attempt_different_key():
    assert _idempotency_key(_req(attempt=0)) != _idempotency_key(_req(attempt=1))


def test_price_bucket_is_one_cent():
    """0.501 and 0.509 round into the same 1-cent bucket and share a key."""
    a = _idempotency_key(_req(price=Decimal("0.501"), attempt=0))
    b = _idempotency_key(_req(price=Decimal("0.509"), attempt=0))
    assert a == b
    c = _idempotency_key(_req(price=Decimal("0.520"), attempt=0))
    assert a != c


def test_different_side_different_key():
    assert _idempotency_key(_req(side="YES")) != _idempotency_key(_req(side="NO"))
