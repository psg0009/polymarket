"""Read-only Turso client over the HTTP/Hrana v3 pipeline.

Streamlit Cloud's Python 3.14 environment has no prebuilt wheel for
`libsql-experimental`, and it can't compile the Rust + cmake build chain
either. Rather than fight that, the dashboard talks directly to Turso's
HTTP API. The API is documented at
    https://docs.turso.tech/sdk/http/reference

This module is dependency-light: only `httpx`. Nothing from
`sqlalchemy-libsql` or `libsql-experimental` is imported.

Parses these URL forms:
    libsql://<host>?authToken=<token>
    sqlite+libsql://<host>?authToken=<token>
    sqlite+libsql://<host>/?authToken=<token>
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlsplit


@dataclass(frozen=True)
class TursoEndpoint:
    base_url: str          # e.g. https://polyclaude-foo.turso.io
    auth_token: str

    @classmethod
    def from_url(cls, url: str) -> "TursoEndpoint":
        u = url
        if u.startswith("sqlite+libsql://"):
            u = "libsql://" + u[len("sqlite+libsql://"):]
        if not u.startswith("libsql://"):
            raise ValueError(f"not a libsql URL: {url!r}")
        parts = urlsplit(u)
        host = parts.hostname or ""
        params = parse_qs(parts.query)
        token = params.get("authToken", [""])[0]
        if not host or not token:
            raise ValueError("libsql URL missing host or authToken")
        return cls(base_url=f"https://{host}", auth_token=token)


def _columns_and_rows(resp_json: dict) -> tuple[list[str], list[list[Any]]]:
    """Pull (column_names, rows_as_lists) out of a Hrana v3 pipeline response."""
    cols: list[str] = []
    rows: list[list[Any]] = []
    results = resp_json.get("results") or []
    for r in results:
        if r.get("type") != "ok":
            continue
        rsp = r.get("response") or {}
        if rsp.get("type") != "execute":
            continue
        result = rsp.get("result") or {}
        cols = [c.get("name") for c in result.get("cols") or []]
        for raw_row in result.get("rows") or []:
            row: list[Any] = []
            for cell in raw_row:
                ctype = cell.get("type")
                value = cell.get("value")
                if ctype == "null":
                    row.append(None)
                elif ctype == "integer":
                    row.append(int(value))
                elif ctype == "float":
                    row.append(float(value))
                elif ctype == "blob":
                    row.append(value)
                else:
                    row.append(value)
            rows.append(row)
    return cols, rows


def query(endpoint: TursoEndpoint, sql: str, args: list[Any] | None = None,
          timeout: float = 15.0) -> tuple[list[str], list[list[Any]]]:
    """Run a single SQL query against Turso. Returns (columns, rows)."""
    import httpx

    body_args = []
    for a in args or []:
        if a is None:
            body_args.append({"type": "null", "value": None})
        elif isinstance(a, bool):
            body_args.append({"type": "integer", "value": str(int(a))})
        elif isinstance(a, int):
            body_args.append({"type": "integer", "value": str(a)})
        elif isinstance(a, float):
            body_args.append({"type": "float", "value": a})
        else:
            body_args.append({"type": "text", "value": str(a)})

    payload = {
        "requests": [
            {"type": "execute", "stmt": {"sql": sql, "args": body_args}},
            {"type": "close"},
        ]
    }
    headers = {
        "Authorization": f"Bearer {endpoint.auth_token}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(f"{endpoint.base_url}/v3/pipeline", json=payload, headers=headers)
        resp.raise_for_status()
        return _columns_and_rows(resp.json())


def query_dicts(endpoint: TursoEndpoint, sql: str, args: list[Any] | None = None) -> list[dict]:
    cols, rows = query(endpoint, sql, args)
    return [dict(zip(cols, r)) for r in rows]


def from_env() -> TursoEndpoint | None:
    """Convenience: pull DATABASE_URL from env / Streamlit secrets."""
    url = os.getenv("DATABASE_URL", "")
    if not url:
        return None
    try:
        return TursoEndpoint.from_url(url)
    except ValueError:
        return None
