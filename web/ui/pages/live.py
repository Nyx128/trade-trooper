"""Live dashboard page."""

from __future__ import annotations

import time

import streamlit as st

from web.ui import charts
from web.ui.data import (
    latest_report_snapshot,
    load_cycles,
    load_equity_history,
    load_latest_decisions,
    load_trades,
)
from web.ui.runner_control import status as runner_status
from web.ui.theme import kpi_tile, section, status_pill


def render() -> None:
    section("📊", "Live Paper Trading Monitor",
            "Auto-refreshes every 30 s. Data from the SQLite journals + JSONL decision logs.")

    rs = runner_status()
    if rs["running"]:
        status_pill(f"🟢 RUNNER LIVE — {len(rs['universe'])} tickers · PID {rs['pid']}", "running")
    else:
        status_pill("🟡 RUNNER STOPPED — start it in the Runner tab", "stopped")
    st.caption("Market-hour only: 09:30–16:00 ET, Mon–Fri.")

    equity = load_equity_history()
    trades = load_trades()
    cycles = load_cycles()
    decisions = load_latest_decisions(10)

    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    with col_a:
        kpi_tile("Open positions", str(len(load_trades().ticker.unique()) if not trades.empty else 0))
    with col_b:
        kpi_tile("Total trades", str(len(trades)))
    with col_c:
        kpi_tile("Realized P&L", f"${trades['pnl'].sum():,.0f}" if not trades.empty else "$0")
    with col_d:
        wins = trades[trades["pnl"] > 0]
        kpi_tile("Win rate", f"{len(wins) / len(trades) * 100:.1f}%" if not trades.empty else "—")
    with col_e:
        if not trades.empty:
            gw, gl = wins["pnl"].sum(), abs(trades[trades["pnl"] <= 0]["pnl"].sum())
            pf = gw / gl if gl > 0 else float("inf")
            kpi_tile("Profit factor", f"{pf:.2f}" if pf != float("inf") else "∞")
        else:
            kpi_tile("Profit factor", "—")

    if not equity.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(charts.equity_curve(equity), width="stretch", key="live_equity_curve")
        with col2:
            st.plotly_chart(charts.drawdown_timeline(equity), width="stretch", key="live_drawdown_timeline")
        st.plotly_chart(charts.equity_vs_pnl(equity, trades), width="stretch", key="live_equity_vs_pnl")
    else:
        st.info("No equity history yet — the runner will record equity each cycle.")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Latest Decision")
        snap = latest_report_snapshot()
        if snap and snap.get("decision"):
            d = snap["decision"]
            st.markdown(
                f"**{d.get('trade_decision', 'n/a').replace('_', ' ').title()}** "
                f"— bias {d.get('composite_bias')} · confidence {d.get('confidence_score', 0):.2f}"
            )
            st.write(d.get("rationale") or d.get("summary") or "")
            st.write(
                f"Entry ${d.get('entry_price', 0):.2f} · Stop ${d.get('stop_loss', 0):.2f} · "
                f"Target ${d.get('take_profit', 0):.2f}"
            )
        else:
            st.info("No decision yet.")
    with col4:
        st.subheader("LLM Narrative")
        if snap and snap.get("analysis"):
            a = snap["analysis"]
            st.write(a.get("summary") or "No narrative available.")
            st.write(f"Actionability: {a.get('actionability')} · Confidence: {a.get('confidence')}")
        else:
            st.info("LLM synthesis disabled or not yet run.")

        if snap:
            if st.button("🧠 Generate Live Analysis", key="live_gen_ai_btn"):
                with st.spinner("Synthesizing decision & runner logs with Ollama…"):
                    from web.ui.trace import generate_live_narrative
                    rs_curr = runner_status()
                    logs = rs_curr.get("last_log_lines") or []
                    reply = generate_live_narrative(snap, logs=logs)
                    st.session_state["live_ai_narrative"] = reply
            if "live_ai_narrative" in st.session_state:
                st.markdown(st.session_state["live_ai_narrative"])

    if not cycles.empty:
        st.plotly_chart(charts.risk_score_timeline(cycles), width="stretch", key="live_risk_score_timeline")

    if not decisions.empty:
        st.subheader("Recent Decision Log Entries")
        cols = ["ts", "ticker", "decision", "composite_bias", "confidence"]
        st.dataframe(decisions[cols], width="stretch")


def auto_refresh() -> None:
    """Scoped auto-refresh: only touches session state, no global rerun spam."""
    now = time.time()
    last = st.session_state.get("live_last_refresh", now)
    elapsed = now - last
    remaining = max(0, int(30 - elapsed))
    st.caption(f"Refreshing in ~{remaining}s")
    if elapsed >= 30:
        st.session_state["live_last_refresh"] = now
        st.rerun()
