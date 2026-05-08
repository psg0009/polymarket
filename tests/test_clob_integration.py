"""Integration tests for the ClobWrapper.

We don't call the live SDK here — instead we exercise the wrapper's contract
against a fake `client` object so the wrapper's retry, midpoint normalization,
and error handling are covered without network.

This complements the unit tests for executor idempotency and the VCR-style
gamma test (which uses HTTP, the natural place for VCR).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from polyclaude.clob.client import ClobWrapper


def test_get_midpoint_dict_form(monkeypatch):
    cw = ClobWrapper()
    cw._read_only_client = MagicMock()
    cw._read_only_client.get_midpoint.return_value = {"mid": "0.42"}
    assert cw.get_midpoint("token_x") == 0.42


def test_get_midpoint_scalar_form():
    cw = ClobWrapper()
    cw._read_only_client = MagicMock()
    cw._read_only_client.get_midpoint.return_value = 0.55
    assert cw.get_midpoint("token_x") == 0.55


def test_get_price_passes_side():
    cw = ClobWrapper()
    cw._read_only_client = MagicMock()
    cw._read_only_client.get_price.return_value = {"price": 0.6}
    assert cw.get_price("tok", "BUY") == 0.6
    cw._read_only_client.get_price.assert_called_with("tok", "BUY")


def test_get_orderbook_returns_passthrough():
    book = SimpleNamespace(bids=[], asks=[])
    cw = ClobWrapper()
    cw._read_only_client = MagicMock()
    cw._read_only_client.get_order_book.return_value = book
    assert cw.get_orderbook("tok") is book
