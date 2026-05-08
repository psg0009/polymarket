"""Streamlit dashboard.

Reads the ledger DB defined by:
- `st.secrets["DATABASE_URL"]` (Streamlit Cloud), or
- `DATABASE_URL` env var (local), or
- the default `.env` value.

Run locally:    streamlit run polyclaude/ui/dashboard.py
Run hosted:     deploy via https://share.streamlit.io against this repo.

Dashboard is intentionally read-only. The agent (running locally or as a
worker) is what writes oracle calls and decisions; this UI only renders them.
"""

from __future__ import annotations

import os


def _wire_secrets() -> None:
    """Bridge Streamlit secrets → env vars before any polyclaude imports."""
    try:
        import streamlit as st  # noqa
    except ImportError:  # pragma: no cover
        return
    try:
        for k in ("DATABASE_URL", "ANTHROPIC_API_KEY"):
            if k in st.secrets and not os.getenv(k):
                os.environ[k] = str(st.secrets[k])
    except Exception:
        pass


_wire_secrets()


def main() -> None:  # pragma: no cover - UI entry
    import pandas as pd
    import streamlit as st
    from sqlalchemy import select

    from polyclaude.ledger.db import (
        CalibrationPoint, Decision, Order, OrderStatus, get_session,
    )

    st.set_page_config(page_title="polyclaude", layout="wide")
    st.title("polyclaude — ledger")
    st.caption(f"DATABASE_URL = {os.getenv('DATABASE_URL', '(default)')[:80]}")

    with get_session() as session:
        decisions = pd.read_sql(select(Decision).order_by(Decision.ts.desc()).limit(500), session.bind)
        orders = pd.read_sql(select(Order).order_by(Order.placed_at.desc()).limit(500), session.bind)
        calib = pd.read_sql(select(CalibrationPoint), session.bind)

    from polyclaude.config import get_settings
    from polyclaude.oracle.claude import _today_spend_usd

    spend_today = _today_spend_usd()
    budget = float(get_settings().anthropic_daily_usd_budget)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Decisions", len(decisions))
    col2.metric("Orders", len(orders))
    col3.metric(
        "Filled",
        int((orders["status"] == OrderStatus.filled.value).sum()) if len(orders) else 0,
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
            st.info("No resolved markets yet — run `polyclaude reconcile` after markets settle.")
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
