# ───────────────────────────────────────────────────────────────────────────────────────
# src/app.py

from pathlib import Path
from typing import Literal
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
import yfinance as yf

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from joblib import load

from src.data_pipeline import get_price_dataframe
from src.options_chain import get_nearest_expiry_and_strike, compute_greeks

# ───────────────────────────────────────────────────────────────────────────────────────
# 1) Load pretrained RSI model (created by `python -m src.train_lr`)
MODEL_PATH = Path("models/lr_rsi.pkl")
if not MODEL_PATH.exists():
    raise RuntimeError("Model file missing. Run: python -m src.train_lr first.")
model = load(MODEL_PATH)

app = FastAPI(title="AI Options Bot", version="0.3.0 (r=4% default)")

# Allowed intervals and their maximum Yahoo look-back (in days)
ALLOWED_INTERVALS = {
    "1m":   7,   "2m":   7,
    "5m":   60,  "15m":  60,  "30m":  60,
    "60m":  730, "90m":  730, "1h":   730,
    "1d":   3650, "1wk": 3650, "1mo":  3650,
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
    ticker = ticker.upper()

    # 1) Validate that 'days' does not exceed Yahoo’s limit for the given interval
    max_days = ALLOWED_INTERVALS.get(interval, 0)
    if days > max_days:
        raise HTTPException(
            status_code=400,
            detail=f"{interval} supports at most {max_days} days; got {days}"
        )

    # 2) Fetch price data + indicators
    try:
        df = get_price_dataframe(ticker, days=days, interval=interval)
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
        1:  "RSI > 70 ⇒ over-bought  – consider bearish spread / buy put",
        -1: "RSI < 30 ⇒ over-sold   – consider bullish spread / buy call",
    }.get(pred, "RSI neutral – no action")

    # 5) Return JSON response
    return {
        "ticker": ticker,
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
    """
    ticker = ticker.upper()

    # Validate look-back limit as in /advice
    max_days = ALLOWED_INTERVALS.get(interval, 0)
    if days > max_days:
        raise HTTPException(
            status_code=400,
            detail=f"{interval} supports at most {max_days} days; got {days}"
        )

    # Fetch data + indicators
    try:
        df = get_price_dataframe(ticker, days=days, interval=interval)
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
        title=f"{ticker} | {interval} – Last {days} Days",
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
    r: float = Query(4.0, description="Risk-free rate in % (annualized)")
):
    """
    Returns the ATM contract and Greeks for the model’s recommendation,
    using TODAY’s ATM implied volatility (live) only—no CSV baseline.
    """
    ticker = ticker.upper()

    # 1a) Validate look-back limit
    max_days = ALLOWED_INTERVALS.get(interval, 0)
    if days > max_days:
        raise HTTPException(
            status_code=400,
            detail=f"{interval} supports at most {max_days} days; got {days}"
        )

    # 1b) Fetch price data + indicators
    try:
        df = get_price_dataframe(ticker, days=days, interval=interval)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Data error: {exc}")

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="Yahoo returned no rows (symbol/interval may be invalid)."
        )

    # 2) Get the latest RSI prediction (±1)
    last = df.iloc[-1]
    feats = [[
        last["rsi14"],
        last["Close"] / last["sma20"],
        last["Close"] / last["sma50"],
    ]]
    pred = int(model.predict(feats)[0])

    # 3) If neutral (should not happen with ±1 model, but just in case)
    if pred == 0:
        return JSONResponse(
            content={
                "ticker": ticker,
                "interval": interval,
                "lookback_days": days,
                "prediction": 0,
                "message": "RSI is neutral; no option contract recommended."
            }
        )

    # 4) Determine option type (call if -1 = oversold; put if 1 = overbought)
    option_type = "call" if pred == -1 else "put"

    # 5) Fetch TODAY’s ATM implied volatility via yfinance
    try:
        yf_tkr = yf.Ticker(ticker)
        expiries = yf_tkr.options
        if not expiries:
            raise RuntimeError(f"No option expiration dates found for {ticker}.")
        front_month = expiries[0]

        hist = yf_tkr.history(period="1d")
        if hist.empty:
            raise RuntimeError(f"No price history available for {ticker}.")
        last_close_price = float(hist["Close"].iloc[-1])

        opt_chain = yf_tkr.option_chain(front_month)
        calls_df = opt_chain.calls
        calls_df["dist"] = (calls_df["strike"] - last_close_price).abs()
        atm_row = calls_df.loc[calls_df["dist"].idxmin()]

        iv_today = float(atm_row.get("impliedVolatility", atm_row.get("impliedVol", None)))
        if iv_today is None:
            raise RuntimeError("impliedVol/impliedVolatility not found in option chain.")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch today's IV for {ticker}: {e}"
        )

    # 6) Find nearest expiry & ATM strike (using our helper)
    try:
        expiry, strike = get_nearest_expiry_and_strike(
            ticker,
            option_type=option_type,
            days_lookahead=days
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Options chain error: {exc}"
        )

    # 7) Compute days to expiry (calendar days from now)
    today = datetime.utcnow().date()
    expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
    days_to_expiry = max((expiry_date - today).days, 0)

    # 8) Compute Greeks via mibian
    try:
        greeks = compute_greeks(
            S=float(last["Close"]),
            K=strike,
            r=r,
            implied_vol=iv_today * 100.0,
            days_to_expiry=days_to_expiry,
            option_type=option_type
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Greek computation error: {exc}"
        )

    # 9) Return JSON with all details
    return {
        "ticker": ticker,
        "interval": interval,
        "lookback_days": days,
        "prediction": pred,
        "option_type": option_type,
        "expiry": expiry,
        "strike": strike,
        "live_iv_atm": round(iv_today * 100, 2),   # e.g. 0.2012 → 20.12%
        "greeks": greeks,
    }


# ─────────────────────────────────────────────────────────────────────────────
@app.get("/full/{ticker}", summary="Combined advice + options + chart (HTML)",
         response_class=HTMLResponse)
def full_dashboard(
    ticker: str,
    interval: Literal[
        "1m", "2m", "5m", "15m", "30m",
        "60m", "90m", "1h", "1d", "1wk", "1mo"
    ] = Query("1d", description="Candle interval"),
    days: int = Query(60, gt=0, description="Look-back window, days"),
    r: float = Query(4.0, description="Risk-free rate in % (annualized)")
):
    """
    Returns a single HTML page that contains:
      1) Latest RSI-based “advice” (formatted in HTML)
      2) ATM option contract + Greeks (formatted in HTML, using TODAY’s IV)
      3) The interactive Plotly candlestick chart (embedded via <script> tags)
    """
    ticker = ticker.upper()

    # 1) Validate look-back limit
    max_days = ALLOWED_INTERVALS.get(interval, 0)
    if days > max_days:
        raise HTTPException(
            status_code=400,
            detail=f"{interval} supports at most {max_days} days; got {days}"
        )

    # 2) Fetch price data + indicators
    try:
        df = get_price_dataframe(ticker, days=days, interval=interval)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Data error: {exc}")

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="Yahoo returned no data (symbol/interval may be invalid)."
        )

    # ─── RSI Advice ──────────────────────────────────────────────────────────
    last = df.iloc[-1]
    feats = [[
        last["rsi14"],
        last["Close"] / last["sma20"],
        last["Close"] / last["sma50"],
    ]]
    pred = int(model.predict(feats)[0])

    advice_text = {
        1:  "RSI > 70 ⇒ over-bought – consider bearish spread / buy put",
        -1: "RSI < 30 ⇒ over-sold  – consider bullish spread / buy call",
    }.get(pred, "RSI neutral – no action")

    advice_html = f"""
      <h3>RSI Advice</h3>
      <p>
        <b>Latest Close:</b> {last['Close']:.2f}<br>
        <b>RSI(14):</b> {last['rsi14']:.1f}<br>
        <b>Prediction:</b> {pred}<br>
        <b>Action:</b> {advice_text}
      </p>
    """

    # ─── Options Advice (using TODAY’s live IV) ─────────────────────────────
    if pred == 0:
        # neutral
        option_html = (
            "<h3>Options Advice</h3>"
            "<p>No ATM contract recommended (RSI neutral).</p>"
        )
    else:
        option_type = "call" if pred == -1 else "put"

        # Fetch TODAY’s IV (exact same code as /options)
        try:
            yf_tkr = yf.Ticker(ticker)
            expiries = yf_tkr.options
            if not expiries:
                raise RuntimeError(f"No option expiration dates found for {ticker}.")
            front_month = expiries[0]

            hist = yf_tkr.history(period="1d")
            if hist.empty:
                raise RuntimeError(f"No price history available for {ticker}.")
            last_close_price = float(hist["Close"].iloc[-1])

            opt_chain = yf_tkr.option_chain(front_month)
            calls_df = opt_chain.calls
            calls_df["dist"] = (calls_df["strike"] - last_close_price).abs()
            atm_row = calls_df.loc[calls_df["dist"].idxmin()]

            iv_today = float(atm_row.get("impliedVolatility", atm_row.get("impliedVol", None)))
            if iv_today is None:
                raise RuntimeError("impliedVol/impliedVolatility not found in option chain.")
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch today's IV for {ticker}: {e}"
            )

        # Find nearest expiry & ATM strike
        try:
            expiry, strike = get_nearest_expiry_and_strike(
                ticker, option_type=option_type, days_lookahead=days
            )
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Options chain error: {exc}"
            )

        today = datetime.utcnow().date()
        exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()
        days_to_exp = max((exp_date - today).days, 0)

        # Compute Greeks
        try:
            greeks = compute_greeks(
                S=float(last["Close"]),
                K=strike,
                r=r,
                implied_vol=iv_today * 100.0,
                days_to_expiry=days_to_exp,
                option_type=option_type
            )
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Greek computation error: {exc}"
            )

        option_html = f"""
            <h3>Options Advice</h3>
            <ul>
              <li><b>Prediction:</b> {"Buy CALL" if option_type=="call" else "Buy PUT"}</li>
              <li><b>Expiry:</b> {expiry}</li>
              <li><b>Strike:</b> {strike:.2f}</li>
              <li><b>Live ATM IV:</b> {iv_today * 100:.2f}%</li>
              <li><b>Greeks:</b>
                <ul style="list-style: none; margin-left: 1em;">
                  <li>Delta: {greeks['delta']:.4f}</li>
                  <li>Gamma: {greeks['gamma']:.4f}</li>
                  <li>Theta: {greeks['theta']:.4f}</li>
                  <li>Vega: {greeks['vega']:.4f}</li>
                  <li>Rho:   {greeks['rho']:.4f}</li>
                </ul>
              </li>
            </ul>
        """

    # ─── Build Plotly Chart ───────────────────────────────────────────────────
    df["pred"] = model.predict(
        np.column_stack([
            df["rsi14"],
            df["Close"] / df["sma20"],
            df["Close"] / df["sma50"],
        ])
    )

    fig = go.Figure()
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
    bull_idx = df.index[df["pred"] == -1]
    bear_idx = df.index[df["pred"] == 1]
    fig.add_trace(go.Scatter(
        x=bull_idx,
        y=df.loc[bull_idx, "Low"] * 0.995,
        mode="markers",
        marker=dict(size=8, color="#2ECC71"),
        name="Buy CALL"
    ))
    fig.add_trace(go.Scatter(
        x=bear_idx,
        y=df.loc[bear_idx, "High"] * 1.005,
        mode="markers",
        marker=dict(size=8, color="#E74C3C"),
        name="Buy PUT"
    ))
    fig.update_layout(
        title=f"{ticker} | {interval} – Last {days} Days",
        height=600,
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        margin=dict(l=30, r=30, t=50, b=30)
    )

    chart_div: str = plot(
        fig,
        output_type="div",
        include_plotlyjs="cdn",
        show_link=False,
        auto_open=False
    )

    # ─── Combine into a full HTML response ─────────────────────────────────
    full_page = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8" />
      <title>AI Options Bot · Combined View / {ticker}</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; }}
        .container {{ max-width: 1200px; margin: 20px auto; padding: 10px; }}
        h1 {{ margin-bottom: 0.2em; }}
        h3 {{ margin-top: 1.5em; margin-bottom: 0.2em; }}
        p, ul {{ line-height: 1.4; }}
        .section {{ margin-bottom: 2em; }}
      </style>
    </head>
    <body>
      <div class="container">
        <h1>AI Options Bot · Combined View / {ticker}</h1>

        <div class="section">
          {advice_html}
        </div>

        <div class="section">
          {option_html}
        </div>

        <div class="section">
          <h3>Price Chart (Candlesticks + SMAs + Signals)</h3>
          {chart_div}
        </div>

      </div>
    </body>
    </html>
    """

    return HTMLResponse(content=full_page)
