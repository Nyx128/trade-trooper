"""Backtest page — date-range backtests as async jobs + rich charts.

- Pick industries/tickers; the weights DB is shown per ticker.
- Choose any start/end date range.
- The run is a background subprocess (job), so other work continues.
- Results (trades journal) are rendered as KPIs, charts and a trade table with
  one-click Ollama explanations of each decision trace.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from web.ui import charts, jobs
from web.ui.data import (
    fetch_bars_df,
    load_journal_cycles,
    load_journal_trades,
    venv_python,
    weights_db_path,
)
from web.ui.job_widgets import jobs_auto_refresh, render_job
from web.ui.theme import kpi_tile, section
from web.ui.trace import chat_about_trade, decision_trace_text, generate_backtest_analysis
from weights_db import INDUSTRY_STOCKS, TICKER_NAMESPACE, load_weights_db, resolve_industry


def _ticker_selection() -> list[str]:
    industries = sorted(INDUSTRY_STOCKS)
    all_tickers = sorted({t for ts in INDUSTRY_STOCKS.values() for t in ts})
    cols = st.columns(4)
    chosen_inds = []
    for i, ind in enumerate(industries):
        if cols[i % 4].checkbox(ind, value=ind in ("Technology", "Financials"), key=f"bt_ind_{ind}"):
            chosen_inds.append(ind)
    if chosen_inds:
        defaults = sorted({t for ind in chosen_inds for t in INDUSTRY_STOCKS[ind]})
    else:
        defaults = []
    tickers = st.multiselect("Tickers", all_tickers, default=defaults, key="bt_tickers")
    return tickers


def _config_preview(tickers: list[str]) -> None:
    db = load_weights_db(weights_db_path())
    if not db:
        st.caption("weights_db.json not found — runs will use global defaults.")
        return
    rows = []
    for t in tickers:
        stock = (db.get(TICKER_NAMESPACE) or {}).get(t)
        ind = resolve_industry(t)
        rows.append({"ticker": t, "industry": ind, "override": f"stock*{t}" if stock else "—"})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _launch_form() -> tuple[list[str], dict]:
    section("📈", "Date-Range Backtest",
            "Replays the deterministic Phase 2–4 chain over any historical window (news-aware when the cache exists).")

    tickers = _ticker_selection()
    col_start, col_end, col_w = st.columns([2, 2, 1])
    with col_start:
        start = st.date_input("Start date", key="bt_start_date")
    with col_end:
        end = st.date_input("End date", key="bt_end_date")
    with col_w:
        use_weights = st.checkbox("Use weights DB", value=True, key="bt_use_weights",
                                  help="Resolve each ticker's industry/stock config from data/weights_db.json")
    params = {"start": start.isoformat(), "end": end.isoformat(), "use_weights": use_weights}

    if tickers:
        _config_preview(tickers)

    if st.button("▶ Run backtest", type="primary", disabled=not tickers):
        if start >= end:
            st.error("Start date must be before end date.")
        else:
            job_id = jobs.create("backtest", label=f"{len(tickers)} tickers {start}→{end}")
            journal_path = jobs.job_artifact(job_id, "trades.db")
            cmd = [venv_python(), "-m", "scripts.tune", "evaluate",
                   "--universe", ",".join(tickers),
                   "--start-date", start.isoformat(), "--end-date", end.isoformat(),
                   "--journal", str(journal_path)]
            if use_weights:
                cmd += ["--weights-db", str(weights_db_path())]
            jobs.launch(job_id, cmd)
            st.success(f"Backtest job {job_id} launched.")
            st.rerun()
    return tickers, params


def _render_results(job_id: str, params: dict) -> None:
    journal_path = jobs.job_artifact(job_id, "trades.db")
    if not journal_path.exists():
        st.info("Job finished but produced no trades journal.")
        return
    trades = load_journal_trades(journal_path)
    cycles = load_journal_cycles(journal_path)
    if trades.empty:
        st.info("No trades in this range — the strategy held through it.")
        return

    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    gw = wins["pnl"].sum()
    gl = abs(losses["pnl"].sum())
    pf = gw / gl if gl > 0 else float("inf")
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in trades["pnl"]:
        equity += p
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_tile("Trades", str(len(trades)))
    with c2:
        kpi_tile("Win rate", f"{len(wins) / len(trades) * 100:.1f}%")
    with c3:
        kpi_tile("Profit factor", f"{pf:.2f}" if pf != float("inf") else "∞")
    with c4:
        kpi_tile("Expectancy", f"${trades['pnl'].mean():,.2f}")
    with c5:
        kpi_tile("Max drawdown", f"${max_dd:,.0f}")

    with st.expander("🧠 Generate AI Analysis with Ollama (Performance & Log Synthesis)", expanded=False):
        st.caption("Passes the full backtest KPIs, top trades, exit reasons, and execution logs to local Ollama for strategic reasoning.")
        custom_prompt = st.text_input(
            "Analysis instructions",
            value="Synthesize the trading performance, win rate, risk metrics, and what the execution logs reveal about edge and execution quality.",
            key=f"bt_ai_prompt_{job_id}",
        )
        if st.button("⚡ Generate Analysis", type="primary", key=f"bt_ai_btn_{job_id}"):
            with st.spinner("Ollama is analyzing backtest telemetry & execution logs..."):
                kpis = {
                    "trades_count": len(trades),
                    "win_rate": f"{len(wins) / len(trades) * 100:.1f}%",
                    "profit_factor": f"{pf:.2f}" if pf != float("inf") else "∞",
                    "expectancy": f"${trades['pnl'].mean():,.2f}",
                    "max_drawdown": f"${max_dd:,.0f}",
                    "total_pnl": float(trades["pnl"].sum()),
                }
                job_logs = jobs.get_job_log(job_id, max_lines=50)
                analysis_text = generate_backtest_analysis(
                    kpis=kpis,
                    trades=trades,
                    params=params,
                    logs=job_logs,
                    question=custom_prompt,
                )
            st.markdown(analysis_text)

    st.plotly_chart(charts.cumulative_pnl(trades), width="stretch", key=f"bt_cum_pnl_{job_id}")
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(charts.pnl_bar(trades), width="stretch", key=f"bt_pnl_bar_{job_id}")
    with col_b:
        st.plotly_chart(charts.win_by_weekday(trades), width="stretch", key=f"bt_win_by_weekday_{job_id}")
    st.plotly_chart(charts.pnl_by_instrument(trades), width="stretch", key=f"bt_pnl_by_instrument_{job_id}")

    st.subheader("Price + trades")
    ticker = st.selectbox("Ticker for candles", sorted(trades["ticker"].unique()),
                          key=f"candle_{job_id}")
    bars = fetch_bars_df(ticker, params["start"], params["end"])
    if bars.empty:
        st.info("No bar data fetched (check API credentials).")
    else:
        t_ticker = trades[trades["ticker"] == ticker]
        st.plotly_chart(charts.candlestick_chart(bars, t_ticker, f"{ticker} — entries/exits"),
                        width="stretch", key=f"bt_candle_chart_{job_id}_{ticker}")
        st.plotly_chart(charts.volume_chart(bars), width="stretch", key=f"bt_volume_chart_{job_id}_{ticker}")

    st.subheader("Trades")
    view = trades.sort_values("closed_ts", ascending=False)
    cols = ["closed_ts", "ticker", "instrument", "entry_price", "exit_price", "pnl",
            "pnl_pct", "exit_reason"]
    st.dataframe(view[cols].head(50), width="stretch")

    _explain_trades(view, cycles, job_id)


def _explain_trades(trades: pd.DataFrame, cycles: pd.DataFrame, job_id: str) -> None:
    if cycles.empty:
        return
    with st.expander("💬 Explain a trade (Ollama reasoning over the decision trace)"):
        options = trades.apply(
            lambda r: f"{r['ticker']}  {r['opened_ts'][:10]}  pnl=${r['pnl']:.0f}", axis=1
        ).tolist()
        choice = st.selectbox("Trade", options, key=f"explain_sel_{job_id}")
        row = trades.iloc[options.index(choice)]
        ticker = row["ticker"]
        opened = row["opened_ts"]
        sub = cycles[cycles["ticker"] == ticker]
        snapshot = {}
        if not sub.empty:
            sub = sub.copy()
            sub["ts"] = pd.to_datetime(sub["ts"])
            target = pd.to_datetime(opened)
            sub["dist"] = (sub["ts"] - target).abs()
            best = sub.loc[sub["dist"].idxmin()]
            try:
                snapshot = json.loads(best["snapshot"])
            except (ValueError, TypeError):
                snapshot = {}
        question = st.text_input(
            "Ask about this trade",
            value="Explain what happened and whether the decision was sound.",
            key=f"explain_q_{job_id}")
        if st.button("Run", type="primary", key=f"explain_run_{job_id}"):
            with st.spinner("Reasoning over the decision trace and execution logs…"):
                job_logs = jobs.get_job_log(job_id, max_lines=40)
                trace = decision_trace_text(snapshot, row.to_dict(), logs=job_logs)
                reply = chat_about_trade(trace, question)
            st.markdown(reply)
            with st.expander("Decision trace (including execution logs)"):
                st.code(trace)


def render() -> None:
    tickers, params = _launch_form()
    st.divider()
    section("🧪", "Backtest Jobs", "Runs asynchronously — you can tune or run other backtests meanwhile.")
    for job in jobs.list_jobs("backtest")[:6]:
        render_job(job)
        if job.get("status") == "done":
            _render_results(job["id"], params)
        st.divider()
    jobs_auto_refresh()
