"""Tests for the dashboard's HTTP-only Turso client.

The dashboard uses this client instead of sqlalchemy-libsql so it can run on
Streamlit Cloud's Python 3.14 environment (where libsql-experimental has no
prebuilt wheel).
"""

from __future__ import annotations

import pytest

from polyclaude.ui.turso_http import TursoEndpoint, _columns_and_rows


def test_endpoint_from_libsql_url():
    e = TursoEndpoint.from_url("libsql://my-db.turso.io?authToken=tok123")
    assert e.base_url == "https://my-db.turso.io"
    assert e.auth_token == "tok123"


def test_endpoint_from_sqlalchemy_form():
    e = TursoEndpoint.from_url(
        "sqlite+libsql://my-db.aws-ap-northeast-1.turso.io/?authToken=eyJabc"
    )
    assert e.base_url == "https://my-db.aws-ap-northeast-1.turso.io"
    assert e.auth_token == "eyJabc"


def test_endpoint_rejects_non_libsql_url():
    with pytest.raises(ValueError):
        TursoEndpoint.from_url("postgresql://user:pass@host/db")


def test_endpoint_rejects_url_without_auth_token():
    with pytest.raises(ValueError):
        TursoEndpoint.from_url("libsql://my-db.turso.io")


def test_columns_and_rows_parses_pipeline_response():
    response = {
        "results": [
            {
                "type": "ok",
                "response": {
                    "type": "execute",
                    "result": {
                        "cols": [{"name": "id"}, {"name": "ts"}, {"name": "p"}],
                        "rows": [
                            [
                                {"type": "integer", "value": "42"},
                                {"type": "text", "value": "2026-05-08T10:00:00Z"},
                                {"type": "float", "value": 0.71},
                            ],
                            [
                                {"type": "integer", "value": "43"},
                                {"type": "null", "value": None},
                                {"type": "float", "value": 0.5},
                            ],
                        ],
                    },
                },
            },
            {"type": "ok", "response": {"type": "close"}},
        ]
    }
    cols, rows = _columns_and_rows(response)
    assert cols == ["id", "ts", "p"]
    assert rows == [[42, "2026-05-08T10:00:00Z", 0.71], [43, None, 0.5]]


def test_columns_and_rows_handles_empty_result():
    cols, rows = _columns_and_rows({"results": []})
    assert cols == []
    assert rows == []


def test_columns_and_rows_skips_non_ok_results():
    response = {
        "results": [
            {"type": "error", "error": {"message": "syntax error"}},
        ]
    }
    cols, rows = _columns_and_rows(response)
    assert cols == []
    assert rows == []
