"""Round-trip: create schema, insert one of each row, query back."""

from datetime import datetime, timezone
from decimal import Decimal

from polyclaude.ledger.db import (
    Decision, DecisionAction, Market, OracleCall, Order, OrderStatus,
    OrderType, Side, get_session, init_db,
)


def test_schema_round_trip():
    init_db()
    with get_session() as s:
        m = Market(
            id="m1", condition_id="c1", yes_token_id="y1", no_token_id="n1",
            question="Q?", description="...", resolution_rules="...",
            tags=["india"], is_india=True, resolves_at=datetime.now(timezone.utc),
        )
        s.add(m)
        s.commit()
        d = Decision(
            id="d1", market_id="m1", strategy="value",
            committed_p=0.6, market_p_at_decision=0.5, edge_bps=1000,
            side=Side.YES, kelly_fraction=0.2, scaled_fraction=0.05,
            intended_size_usd=Decimal("10.00"), intended_price=Decimal("0.50"),
            action=DecisionAction.place, bankroll_usd=Decimal("100"),
            daily_notional_used_usd=Decimal("0"), is_live=False,
        )
        s.add(d)
        oc = OracleCall(
            decision_group_id="d1", market_id="m1", call_type="probability",
            model="claude-test", prompt_system="sys", prompt_user="usr",
            raw_response='{"p":0.6}', parsed_response={"p": 0.6},
            p=0.6, confidence=0.8, latency_ms=100,
        )
        s.add(oc)
        o = Order(
            id="o1", idempotency_key="k1", decision_id="d1", market_id="m1",
            token_id="y1", side=Side.YES, order_type=OrderType.GTC,
            price=Decimal("0.50"), size_shares=Decimal("20"),
            notional_usd=Decimal("10"), status=OrderStatus.open,
        )
        s.add(o)
        s.commit()

    with get_session() as s:
        assert s.get(Market, "m1").question == "Q?"
        assert s.get(Decision, "d1").side == Side.YES
        assert s.get(Order, "o1").price == Decimal("0.50000000")
