"""
dashboard.py
============

Streamlit front-end for the AI Options Bot.

Run with:
    streamlit run src/dashboard.py

This file ensures the project root is on sys.path so that imports like
'from src.data_pipeline import get_price_dataframe' work even when run as a script.
"""

import sys
import os

# ─── Ensure Python can import from the project root ──────────────────────────
# FILE_DIR = ~/ai-options-bot/src
FILE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, os.pardir))  # ~/ai-options-bot
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Now import using the src package namespace
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from joblib import load

from src.data_pipeline import get_price_dataframe

# ─── Configuration ──────────────────────────────────────────────────────────
MODEL = load("models/lr_rsi.pkl")
DEFAULT_TICKER   = "SPY"
INTRADAY_LIMITS = {
    "1m": 7, "2m": 7,
    "5m": 60, "15m": 60, "30m": 60,
    "60m": 730, "90m": 730, "1h": 730,
    "1d": 3650, "1wk": 3650, "1mo": 3650,
}

# ─── Sidebar Controls ───────────────────────────────────────────────────────
st.sidebar.header("Controls")
ticker_input = st.sidebar.text_input("Symbol", DEFAULT_TICKER)
ticker = ticker_input.upper()

interval = st.sidebar.selectbox(
    "Interval",
    options=list(INTRADAY_LIMITS.keys()),
    index=list(INTRADAY_LIMITS.keys()).index("1d"),
    help="Select the candle interval"
)

max_days = INTRADAY_LIMITS[interval]
days = st.sidebar.slider(
    "Look-back (days)",
    min_value=5,
    max_value=max_days,
    value=min(120, max_days),
    help="Window size for the historical data"
)

# ─── Fetch Data & Compute Model Predictions ─────────────────────────────────
try:
    df = get_price_dataframe(ticker, days=days, interval=interval)
except Exception as exc:
    st.error(f"Data error: {exc}")
    st.stop()

# Build feature matrix: [rsi14, price/sma20, price/sma50]
features = np.column_stack([
    df["rsi14"],
    df["Close"] / df["sma20"],
    df["Close"] / df["sma50"],
])
df["pred"] = MODEL.predict(features)

# ─── Display Latest Signal ──────────────────────────────────────────────────
latest = df.iloc[-1]
pred = int(latest["pred"])
action = {
    1:  "⟹ Over-bought – consider bearish spread / buy put 🟥",
    -1: "⟹ Over-sold  – consider bullish spread / buy call 🟩",
    0:  "⟹ Neutral – no action"
}[pred]

st.markdown(
    f"## {ticker} — {interval}  \n"
    f"Close: **{latest['Close']:.2f}**   RSI-14: **{latest['rsi14']:.1f}**  \n"
    f"### Model signal: **{action}**",
    unsafe_allow_html=True
)

# ─── Build Plotly Candlestick Chart with Annotations ────────────────────────
fig = go.Figure()

# 1) Price candles
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    name="Price",
    increasing_line_color="#2ECC71",
    decreasing_line_color="#E74C3C"
))

# 2) SMA-20 (yellow) and SMA-50 (blue)
fig.add_trace(go.Scatter(
    x=df.index,
    y=df["sma20"],
    mode="lines",
    line=dict(width=1, color="#F1C40F"),
    name="SMA-20"
))
fig.add_trace(go.Scatter(
    x=df.index,
    y=df["sma50"],
    mode="lines",
    line=dict(width=1, color="#3498DB"),
    name="SMA-50"
))

# 3) Signal markers: green for buy-call (pred == -1), red for buy-put (pred == 1)
bull_indices = df.index[df["pred"] == -1]
bear_indices = df.index[df["pred"] == 1]

fig.add_trace(go.Scatter(
    x=bull_indices,
    y=df.loc[bull_indices, "Low"] * 0.995,
    mode="markers",
    marker=dict(size=8, color="#2ECC71"),
    name="Buy CALL"
))
fig.add_trace(go.Scatter(
    x=bear_indices,
    y=df.loc[bear_indices, "High"] * 1.005,
    mode="markers",
    marker=dict(size=8, color="#E74C3C"),
    name="Buy PUT"
))

# 4) Layout tweaks
fig.update_layout(
    title=f"{ticker} Price + SMAs + Signals",
    height=600,
    xaxis_rangeslider_visible=False,
    template="plotly_white",
    margin=dict(l=30, r=30, t=40, b=40)
)

st.plotly_chart(fig, use_container_width=True)

# ─── Optional: Raw Data Table ───────────────────────────────────────────────
if st.sidebar.checkbox("Show raw data"):
    st.dataframe(df.tail(200))
