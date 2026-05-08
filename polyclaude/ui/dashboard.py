"""Streamlit dashboard.

Reads from a Turso database directly via Turso's HTTP API. This avoids
depending on the Rust-built `libsql-experimental` package, which has no
prebuilt wheel for Streamlit Cloud's Python 3.14 environment and can't be
compiled there either (no cmake).

DATABASE_URL is read from:
1. `st.secrets["DATABASE_URL"]` (Streamlit Cloud)
2. `DATABASE_URL` env var (local)

The dashboard is read-only by design. The agent (running in GitHub Actions
or locally) is what writes Decisions / Orders / OracleCalls / etc.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Streamlit Cloud runs `streamlit run polyclaude/ui/dashboard.py` from the
# repo root but does NOT add that root to sys.path automatically, so the
# `from polyclaude.ui.turso_http import ...` line below fails with
# ModuleNotFoundError. Prepend the project root explicitly.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _wire_secrets() -> None:
    """Bridge Streamlit secrets → env vars before any other import that reads them."""
    try:
        import streamlit as st  # noqa
    except ImportError:  # pragma: no cover
        return
    try:
        for k in ("DATABASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_DAILY_USD_BUDGET"):
            if k in st.secrets and not os.getenv(k):
                os.environ[k] = str(st.secrets[k])
    except Exception:
        pass


_wire_secrets()


# --- Anthropic pricing (mirror of polyclaude.oracle.claude._PRICING_USD_PER_MTOK) ---
# Replicated here so the dashboard can compute today's spend without importing
# the agent module (which transitively pulls in the anthropic SDK we don't need).
_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-7": (15.0, 75.0),
    "claude-opus-4-6": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
}
_DEFAULT_PRICE = (15.0, 75.0)


def _cost_for(model: str | None, in_tok: int | None, out_tok: int | None) -> float:
    in_price, out_price = _PRICING.get(model or "", _DEFAULT_PRICE)
    return (in_tok or 0) * in_price / 1e6 + (out_tok or 0) * out_price / 1e6


def main() -> None:  # pragma: no cover - UI entry
    import pandas as pd
    import streamlit as st

    from polyclaude.ui.turso_http import from_env, query_dicts

    st.set_page_config(page_title="polyclaude", layout="wide")
    st.title("polyclaude — ledger")

    db_url = os.getenv("DATABASE_URL", "")
    st.caption(f"DATABASE_URL = {db_url[:80] + ('…' if len(db_url) > 80 else '')}")

    endpoint = from_env()
    if endpoint is None:
        st.error(
            "DATABASE_URL not set or not a libsql:// URL. Configure it in "
            "Streamlit Cloud → app settings → Secrets, e.g. "
            "`DATABASE_URL = \"libsql://your-db.turso.io?authToken=…\"`."
        )
        st.stop()

    try:
        decisions = pd.DataFrame(query_dicts(
            endpoint,
            "SELECT id, market_id, strategy, committed_p, market_p_at_decision, "
            "edge_bps, side, intended_size_usd, action, skip_reason, ts "
            "FROM decisions ORDER BY ts DESC LIMIT 500",
        ))
        orders = pd.DataFrame(query_dicts(
            endpoint,
            "SELECT id, decision_id, market_id, side, order_type, price, "
            "size_shares, notional_usd, status, placed_at "
            "FROM orders ORDER BY placed_at DESC LIMIT 500",
        ))
        calib = pd.DataFrame(query_dicts(
            endpoint,
            "SELECT decision_id, market_id, strategy, p_predicted, confidence, "
            "outcome_yes, brier, log_loss, resolved_at FROM calibration_points",
        ))
        oracle_today = pd.DataFrame(query_dicts(
            endpoint,
            "SELECT model, input_tokens, output_tokens "
            "FROM oracle_calls "
            "WHERE ts >= datetime('now', 'start of day')",
        ))
    except Exception as e:
        st.error(f"Could not fetch data from Turso: {e}")
        st.stop()

    spend_today = 0.0
    if len(oracle_today):
        for _, row in oracle_today.iterrows():
            spend_today += _cost_for(row.get("model"), row.get("input_tokens"), row.get("output_tokens"))
    budget = float(os.getenv("ANTHROPIC_DAILY_USD_BUDGET", "50"))

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Decisions", len(decisions))
    col2.metric("Orders", len(orders))
    col3.metric(
        "Filled",
        int((orders["status"] == "filled").sum()) if len(orders) else 0,
    )
    if len(calib):
        col4.metric("Brier", f"{calib['brier'].mean():.3f}", help=f"n={len(calib)}")
    else:
        col4.metric("Brier", "n/a")
    col5.metric(
        "Anthropic $ today",
        f"${spend_today:.2f}",
        delta=f"${budget - spend_today:.2f} left" if spend_today < budget else "OVER BUDGET",
        delta_color="normal" if spend_today < budget else "inverse",
        help=f"daily cap: ${budget:.2f}",
    )

    tabs = st.tabs(["Decisions", "Orders", "Calibration", "Live status"])
    with tabs[0]:
        st.dataframe(decisions, use_container_width=True)
    with tabs[1]:
        st.dataframe(orders, use_container_width=True)
    with tabs[2]:
        if len(calib):
            calib = calib.copy()
            calib["bucket"] = (calib["p_predicted"] * 10).round() / 10
            chart = calib.groupby("bucket").agg(
                mean_p=("p_predicted", "mean"),
                yes_rate=("outcome_yes", "mean"),
                n=("outcome_yes", "count"),
            )
            st.dataframe(chart, use_container_width=True)
            st.line_chart(chart[["mean_p", "yes_rate"]])
        else:
            st.info("No resolved markets yet — `polyclaude reconcile` populates this once markets settle.")
    with tabs[3]:
        gate_brier = float(calib["brier"].mean()) if len(calib) else float("nan")
        n = len(calib)
        ready = n >= 100 and gate_brier <= 0.20
        if ready:
            st.success(f"Ready for live: Brier={gate_brier:.3f}, n={n}")
        else:
            st.warning(
                f"Not ready for live yet — Brier={gate_brier:.3f}, n={n}. "
                "Need n ≥ 100 and Brier ≤ 0.20."
            )


if __name__ == "__main__":  # pragma: no cover
    main()
