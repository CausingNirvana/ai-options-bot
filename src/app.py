"""
app.py
======

FastAPI service for the AI Options Bot.

Endpoints:
  • GET  /advice/{ticker}       → returns latest RSI-based signal (JSON)
  • GET  /chart/{ticker}        → returns an HTML page containing an interactive Plotly chart
                                   (candlesticks + SMA-20/50 + buy/sell markers)
  • GET  /options/{ticker}      → returns the recommended ATM option contract and its Greeks (JSON)

Query parameters (for /advice and /options):
  • interval : one of ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "1wk", "1mo"]
  • days     : positive int (look-back window, in days)
  • r        : risk-free rate in % (annualized)  [only for /options]
  • implied_vol : implied volatility in %         [only for /options]

Yahoo rules:
  • Intraday intervals < 1h are limited to max_lookback[interval] (see table below).
  • Daily bars (“1d”), weekly (“1wk”), and monthly (“1mo”) are always available.

Usage:
  1) Ensure you have trained the model: `python -m src.train_lr`
  2) Run the server: `uvicorn src.app:app --reload`
  3) Browse Swagger UI at http://127.0.0.1:8000/docs or fetch charts at /chart/{ticker}.
"""

from pathlib import Path
from typing import Literal
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from joblib import load

from src.data_pipeline import get_price_dataframe
from src.options_chain import get_nearest_expiry_and_strike, compute_greeks

# ─────────────────────────────────────────────────────────────────────────────
# Load model (must have been created by `python -m src.train_lr`)
MODEL_PATH = Path("models/lr_rsi.pkl")
if not MODEL_PATH.exists():
    raise RuntimeError("Model file missing. Run: python -m src.train_lr first.")
model = load(MODEL_PATH)

app = FastAPI(title="AI Options Bot", version="0.3.0")

# Allowed intervals and their maximum Yahoo look-back (in days)
ALLOWED_INTERVALS = {
    "1m": 7,   "2m": 7,
    "5m": 60,  "15m": 60, "30m": 60,
    "60m": 730, "90m": 730, "1h": 730,
    "1d": 3650, "1wk": 3650, "1mo": 3650,
}


# ─────────────────────────────────────────────────────────────────────────────
@app.get("/advice/{ticker}", summary="Get RSI-based trading advice (JSON)")
def advice(
    ticker: str,
    interval: Literal[
        "1m", "2m", "5m", "15m", "30m",
        "60m", "90m", "1h", "1d", "1wk", "1mo"
    ] = Query("1d", description="Candle interval"),
    days: int = Query(120, gt=0, description="Look-back window, in days")
):
    """
    Returns a JSON object with the latest RSI-based signal and recommended action.
    """

    # 1) Validate that 'days' does not exceed Yahoo’s limit for the given interval
    max_days = ALLOWED_INTERVALS[interval]
    if days > max_days:
        raise HTTPException(
            status_code=400,
            detail=f"{interval} supports at most {max_days} days; got {days}"
        )

    # 2) Fetch price data + indicators
    try:
        df = get_price_dataframe(ticker.upper(), days=days, interval=interval)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Data error: {exc}")

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="Yahoo returned no rows (symbol/interval may be invalid)."
        )

    # 3) Build feature vector from the latest bar
    last = df.iloc[-1]
    features = [[
        last["rsi14"],
        last["Close"] / last["sma20"],
        last["Close"] / last["sma50"],
    ]]
    pred = int(model.predict(features)[0])

    # 4) Map prediction to human-readable action
    action = {
        1:  "RSI > 70 => over-bought – consider bearish spread / buy put",
        -1: "RSI < 30 => over-sold  – consider bullish spread / buy call",
    }.get(pred, "RSI neutral – no action")

    # 5) Return JSON response
    return {
        "ticker": ticker.upper(),
        "interval": interval,
        "lookback_days": days,
        "close": round(last["Close"], 2),
        "rsi14": round(last["rsi14"], 1),
        "prediction": pred,
        "action": action,
    }


# ─────────────────────────────────────────────────────────────────────────────
@app.get("/chart/{ticker}", summary="Return an HTML page with an interactive Plotly chart")
def chart_page(
    ticker: str,
    days: int = Query(30, gt=0, description="Look-back window, in days"),
    interval: Literal[
        "1m", "2m", "5m", "15m", "30m",
        "60m", "90m", "1h", "1d", "1wk", "1mo"
    ] = Query("1d", description="Candle interval")
) -> HTMLResponse:
    """
    Returns a standalone HTML page containing the Plotly candlestick chart
    with SMA-20/50 and buy/sell markers for each prediction.
    Uses the /advice logic under the hood to fetch and annotate data.
    """

    # Validate look-back limit as in /advice
    max_days = ALLOWED_INTERVALS.get(interval, 0)
    if days > max_days:
        raise HTTPException(
            status_code=400,
            detail=f"{interval} supports at most {max_days} days; got {days}"
        )

    # Fetch data + indicators
    try:
        df = get_price_dataframe(ticker.upper(), days=days, interval=interval)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Data error: {exc}")

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="Yahoo returned no rows (symbol/interval may be invalid)."
        )

    # Build feature matrix for all bars, then predictions
    feats = np.column_stack([
        df["rsi14"],
        df["Close"] / df["sma20"],
        df["Close"] / df["sma50"],
    ])
    df["pred"] = model.predict(feats)

    # Construct the Plotly figure
    fig = go.Figure()

    # 1) Candlesticks
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

    # 2) SMA-20 and SMA-50
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

    # 3) Signal markers (green below candle for buy-call, red above for buy-put)
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

    # 4) Layout settings
    fig.update_layout(
        title=f"{ticker.upper()} | {interval} – Last {days} Days",
        height=650,
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        margin=dict(l=30, r=30, t=50, b=30)
    )

    # Render as standalone HTML
    html_str = fig.to_html(full_html=True)
    return HTMLResponse(html_str)


# ─────────────────────────────────────────────────────────────────────────────
@app.get("/options/{ticker}", summary="Get recommended ATM option & Greeks (JSON)")
def options_info(
    ticker: str,
    days: int = Query(30, gt=0, description="Look-back window for signal"),
    interval: Literal[
        "1m", "2m", "5m", "15m", "30m",
        "60m", "90m", "1h", "1d", "1wk", "1mo"
    ] = Query("1d", description="Candle interval"),
    r: float = Query(1.5, description="Risk-free rate in % (annualized)"),
    implied_vol: float = Query(20.0, description="Assumed IV in %"),
):
    """
    Returns the ATM contract and Greeks for the model’s recommendation.

    Steps:
      1) Run the same logic as /advice to get the latest prediction.
      2) If pred == 0, return early with “no contract recommended.”
      3) Otherwise, pick nearest expiry & ATM strike for the given contract type.
      4) Compute and return Greeks (Delta, Gamma, Theta, Vega, Rho).
    """

    # 1a) Validate look-back limit
    max_days = ALLOWED_INTERVALS[interval]
    if days > max_days:
        raise HTTPException(
            status_code=400,
            detail=f"{interval} supports at most {max_days} days; got {days}"
        )

    # 1b) Fetch price data + indicators
    try:
        df = get_price_dataframe(ticker.upper(), days=days, interval=interval)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Data error: {exc}")

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="Yahoo returned no rows (symbol/interval may be invalid)."
        )

    # 2) Get the latest prediction
    last = df.iloc[-1]
    feats = [[
        last["rsi14"],
        last["Close"] / last["sma20"],
        last["Close"] / last["sma50"],
    ]]
    pred = int(model.predict(feats)[0])

    # 3) If neutral, respond accordingly
    if pred == 0:
        return JSONResponse(
            content={
                "ticker": ticker.upper(),
                "interval": interval,
                "lookback_days": days,
                "prediction": 0,
                "message": "RSI is neutral; no option contract recommended."
            }
        )

    # 4) Determine option type (call if -1 = oversold; put if 1 = overbought)
    option_type = "call" if pred == -1 else "put"

    # 5) Find nearest expiry & ATM strike
    try:
        expiry, strike = get_nearest_expiry_and_strike(
            ticker.upper(),
            option_type=option_type,
            days_lookahead=days
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Options chain error: {exc}"
        )

    # 6) Compute days to expiry (calendar days)
    today = datetime.utcnow().date()
    expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
    days_to_expiry = max((expiry_date - today).days, 0)

    # 7) Compute Greeks via mibian
    try:
        greeks = compute_greeks(
            S=float(last["Close"]),
            K=strike,
            r=r,
            implied_vol=implied_vol,
            days_to_expiry=days_to_expiry,
            option_type=option_type
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Greek computation error: {exc}"
        )

    # 8) Return JSON with all details
    return {
        "ticker": ticker.upper(),
        "interval": interval,
        "lookback_days": days,
        "prediction": pred,
        "option_type": option_type,
        "expiry": expiry,
        "strike": strike,
        "greeks": greeks,
    }
