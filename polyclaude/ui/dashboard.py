"""Streamlit dashboard.

Run with:  streamlit run polyclaude/ui/dashboard.py
"""

from __future__ import annotations

from sqlalchemy import select

from polyclaude.ledger.db import (
    CalibrationPoint, Decision, Order, OrderStatus, get_session,
)


def main() -> None:  # pragma: no cover - UI entry
    import pandas as pd
    import streamlit as st

    st.set_page_config(page_title="polyclaude", layout="wide")
    st.title("polyclaude — ledger")

    with get_session() as session:
        decisions = pd.read_sql(select(Decision).order_by(Decision.ts.desc()).limit(500), session.bind)
        orders = pd.read_sql(select(Order).order_by(Order.placed_at.desc()).limit(500), session.bind)
        calib = pd.read_sql(select(CalibrationPoint), session.bind)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Decisions", len(decisions))
    col2.metric("Orders", len(orders))
    col3.metric("Filled", int((orders["status"] == OrderStatus.filled.value).sum()) if len(orders) else 0)
    if len(calib):
        col4.metric("Brier", f"{calib['brier'].mean():.3f}", help=f"n={len(calib)}")
    else:
        col4.metric("Brier", "n/a")

    tabs = st.tabs(["Decisions", "Orders", "Calibration"])
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


if __name__ == "__main__":  # pragma: no cover
    main()
