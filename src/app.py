# src/app.py
# ─────────────────────────────────────────────────────────────────────────────
from pathlib import Path
from datetime import datetime, date
from typing import Literal, Optional, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
import yfinance as yf

from fastapi import FastAPI, HTTPException, Query, Body, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from joblib import load

from src.data_pipeline import get_price_dataframe
from src.options_chain import get_nearest_expiry_and_strike, compute_greeks

# ─────────────────────────────────────────────────────────────────────────────
# Load pretrained RSI model (created by `python -m src.train_lr`)
MODEL_PATH = Path("models/lr_rsi.pkl")
if not MODEL_PATH.exists():
    raise RuntimeError("Model file missing. Run: python -m src.train_lr first.")
model = load(MODEL_PATH)

app = FastAPI(title="AI Options Bot", version="0.4.1")

# Where our Jinja2 templates live:
templates = Jinja2Templates(directory="templates")

# Allowed intervals and their maximum Yahoo look-back (in days)
ALLOWED_INTERVALS = {
    "1m":   7,   "2m":   7,
    "5m":   60,  "15m":  60,  "30m":  60,
    "60m":  730, "90m":  730, "1h":   730,
    "1d":   3650, "1wk": 3650, "1mo":  3650,
}


# ─────────────────────────────────────────────────────────────────────────────
class Position(BaseModel):
    """
    JSON payload for an existing long option position.
      - ticker          : underlying symbol, e.g. "SPY"
      - expiry          : expiration date (YYYY-MM-DD) that appears in yfinance’s chain
      - strike          : strike price as a float, e.g. 595.0
      - purchase_price  : the premium you paid (cost) PER CONTRACT
      - quantity        : number of contracts you bought (integer)
      - purchase_date   : (optional) date you bought it (YYYY-MM-DD), for reference
    """
    ticker: str
    expiry: str
    strike: float
    purchase_price: float
    quantity: int
    purchase_date: Optional[date] = None


# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def redirect_to_docs():
    """
    Redirect “/” to Swagger docs by default.
    """
    return RedirectResponse(url="/docs")


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
        1:  "RSI > 70 ⇒ over-bought – consider bearish spread / buy put",
        -1: "RSI < 30 ⇒ over-sold  – consider bullish spread / buy call",
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
    max_days = ALLOWED_INTERVALS.get(interval, 0)
    if days > max_days:
        raise HTTPException(
            status_code=400,
            detail=f"{interval} supports at most {max_days} days; got {days}"
        )

    try:
        df = get_price_dataframe(ticker, days=days, interval=interval)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Data error: {exc}")

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="Yahoo returned no rows (symbol/interval may be invalid)."
        )

    feats = np.column_stack([
        df["rsi14"],
        df["Close"] / df["sma20"],
        df["Close"] / df["sma50"],
    ])
    df["pred"] = model.predict(feats)

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
    fig.update_layout(
        title=f"{ticker} | {interval} – Last {days} Days",
        height=650,
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        margin=dict(l=30, r=30, t=50, b=30)
    )

    html_str = fig.to_html(full_html=True)
    return HTMLResponse(html_str)


# ─────────────────────────────────────────────────────────────────────────────
@app.get("/options/{ticker}", summary="Get recommended ATM option(s) & Greeks (JSON)")
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
    Returns ATM contract(s) and Greeks for the model’s recommendation,
    checking ALL available expiries.  
    Steps:
      1) Run the same logic as /advice to get the latest prediction.
      2) If pred == 0, return early with “no contract recommended.”
      3) Otherwise, pull every expiration from yfinance.
      4) For each expiry, find the ATM strike, live IV, compute Greeks.
      5) Return all suggestions in a JSON list under "suggestions".
    """
    ticker = ticker.upper()
    max_days = ALLOWED_INTERVALS.get(interval, 0)
    if days > max_days:
        raise HTTPException(
            status_code=400,
            detail=f"{interval} supports at most {max_days} days; got {days}"
        )

    try:
        df = get_price_dataframe(ticker, days=days, interval=interval)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Data error: {exc}")

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="Yahoo returned no rows (symbol/interval may be invalid)."
        )

    # Get latest RSI prediction
    last = df.iloc[-1]
    feats = [[
        last["rsi14"],
        last["Close"] / last["sma20"],
        last["Close"] / last["sma50"],
    ]]
    pred = int(model.predict(feats)[0])

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

    option_type = "call" if pred == -1 else "put"

    # Pull all expiries
    # ─── Replace single‐expiry logic with a loop over expiries[:5] ─────────
    yf_tkr = yf.Ticker(ticker)
    expiries = yf_tkr.options
    if not expiries:
        raise RuntimeError(f"No expirations found for {ticker}.")

    # Pull today’s close once
    hist = yf_tkr.history(period="1d")
    if hist.empty:
        raise RuntimeError(f"No price history for {ticker}.")
    last_close_price = float(hist["Close"].iloc[-1])

    suggestions = []
    for exp_date in expiries[:5]:
        chain = yf_tkr.option_chain(exp_date)
        calls_df = chain.calls
        puts_df  = chain.puts

        calls_df["dist"] = (calls_df["strike"] - last_close_price).abs()
        puts_df["dist"]  = (puts_df["strike"]  - last_close_price).abs()

        if option_type == "call":
            atm_row = calls_df.loc[calls_df["dist"].idxmin()]
        else:
            atm_row = puts_df.loc[puts_df["dist"].idxmin()]

        atm_strike = float(atm_row["strike"])
        iv_today   = float(atm_row.get("impliedVolatility", atm_row.get("impliedVol", None)))
        if iv_today is None:
            raise RuntimeError(f"impliedVol not found for {ticker} exp {exp_date}.")

        today_date = datetime.utcnow().date()
        exp_dt_obj = datetime.strptime(exp_date, "%Y-%m-%d").date()
        days_to_exp = max((exp_dt_obj - today_date).days, 0)

        greeks = compute_greeks(
            S=float(last["Close"]),
            K=atm_strike,
            r=r,
            implied_vol=iv_today * 100.0,
            days_to_expiry=days_to_exp,
            option_type=option_type
        )

        suggestions.append({
            "expiry": exp_date,
            "strike": atm_strike,
            "iv": round(iv_today * 100, 2),
            "greeks": greeks
        })

    return {
        "ticker": ticker,
        "interval": interval,
        "lookback_days": days,
        "prediction": pred,
        "option_type": option_type,
        "suggestions": suggestions
    }


# ─────────────────────────────────────────────────────────────────────────────
@app.post(
    "/position/sell_advice",
    summary="Given a long option, get a ‘sell or hold’ recommendation (JSON)"
)
def position_sell_advice(
    pos: Position = Body(
        ...,
        example={
            "ticker":        "SPY",
            "expiry":        "2025-06-05",
            "strike":        595.0,
            "purchase_price":"2.75",
            "quantity":      2,
            "purchase_date": "2025-05-20"
        },
        description="Details of your existing long option position."
    ),
    profit_target: float = Query(
        0.50,
        description="Profit target as decimal (0.50 = 50% profit). If reached or exceeded → recommend sell."
    ),
    rsi_sell_threshold: float = Query(
        70.0,
        description="Underlying RSI threshold above which to recommend selling the option."
    )
):
    """
    1) Fetch underlying price + compute latest RSI(14) exactly as /advice.
    2) Fetch current mid‐price of the exact option (ticker, expiry, strike) via yfinance.
    3) Compute unrealized P&L = (current_option_price - purchase_price) / purchase_price, then × quantity.
    4) If P&L ≥ profit_target OR underlying RSI ≥ rsi_sell_threshold → action = "sell", else "hold".
    5) Return JSON with all details.
    """
    ticker = pos.ticker.upper()
    expiry = pos.expiry
    strike = pos.strike
    paid = pos.purchase_price
    qty = pos.quantity

    # ─── 1) Compute latest underlying RSI (use 50 days to ensure SMA50 exists) ─
    try:
        df_under = get_price_dataframe(ticker, days=75, interval="1d")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot fetch underlying data: {exc}")

    if df_under.empty:
        raise HTTPException(
            status_code=400,
            detail="Underlying returned no price data (symbol may be invalid)."
        )

    rsi_val = float(df_under.iloc[-1]["rsi14"])

    # ─── 2) Fetch current mid‐price of the specified option ──────────────
    yf_tkr = yf.Ticker(ticker)
    all_exps = yf_tkr.options
    if expiry not in all_exps:
        raise HTTPException(
            status_code=400,
            detail=f"Expiry {expiry} not found for {ticker}. Available expiries: {all_exps}"
        )

    opt_chain = yf_tkr.option_chain(expiry)
    calls_df = opt_chain.calls
    puts_df  = opt_chain.puts

    # Check PUT first, then CALL
    match_put  = puts_df[puts_df["strike"] == strike]
    match_call = calls_df[calls_df["strike"] == strike]
    is_call = False
    if not match_put.empty:
        is_call = False
        row = match_put.iloc[0]
    elif not match_call.empty:
        is_call = True
        row = match_call.iloc[0]
    else:
        raise HTTPException(
            status_code=400,
            detail=f"No option with strike {strike} found for expiry {expiry}."
        )

    bid        = row.get("bid", None)
    ask        = row.get("ask", None)
    last_price = row.get("lastPrice", None)

    if bid and ask and (bid > 0) and (ask > 0):
        current_opt_price = float((bid + ask) / 2.0)
    elif last_price:
        current_opt_price = float(last_price)
    else:
        current_opt_price = float(row.get("mark", 0.0))

    # ─── 3) Compute P&L percentage (per contract), then P&L dollar amount ─────
    profit_pct_single = 0.0
    if paid != 0:
        profit_pct_single = (current_opt_price - paid) / paid

    pnl_dollars = (current_opt_price - paid) * qty * 100

    # ─── 4) Decide “sell” vs. “hold” ─────────────────────────────────────────
    if (profit_pct_single >= profit_target) or (rsi_val >= rsi_sell_threshold):
        action = "sell"
    else:
        action = "hold"

    return {
        "ticker": ticker,
        "expiry": expiry,
        "strike": strike,
        "is_call": is_call,
        "purchase_price": paid,
        "quantity": qty,
        "current_option_price": round(current_opt_price, 4),
        "profit_pct_single": round(profit_pct_single * 100, 2),
        "pnl_dollars": round(pnl_dollars, 2),
        "underlying_rsi14": round(rsi_val, 2),
        "profit_target_pct": profit_target * 100,
        "rsi_sell_threshold": rsi_sell_threshold,
        "action": action,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
@app.get(
    "/full/{ticker}",
    summary="Combined advice + options + chart + (option‐sell advice via form)",
    response_class=HTMLResponse
)
def full_dashboard(
    request: Request,
    ticker: str,
    interval: Literal[
        "1m", "2m", "5m", "15m", "30m",
        "60m", "90m", "1h", "1d", "1wk", "1mo"
    ] = Query("1d", description="Candle interval"),
    days: int = Query(60, gt=0, description="Look-back window, days"),
    r: float = Query(4.0, description="Risk-free rate in % (annualized)"),
    # Optional position parameters. If user does not supply these, we omit position advice.
    pos_expiry: Optional[str] = Query(
        None,
        description="(Optional) Expiration date of your existing position (YYYY-MM-DD)"
    ),
    pos_strike: Optional[float] = Query(
        None,
        description="(Optional) Strike price of your existing position as a float"
    ),
    pos_purchase_price: Optional[float] = Query(
        None,
        description="(Optional) Premium you paid per contract (float)"
    ),
    pos_quantity: Optional[int] = Query(
        None,
        description="(Optional) Number of contracts you purchased (int)"
    ),
    pos_profit_target: float = Query(
        0.50,
        description="(Optional) Profit‐target (0.5 = 50%). If reached → recommend sell"
    ),
    pos_rsi_sell_threshold: float = Query(
        70.0,
        description="(Optional) Underlying RSI threshold (>= this → recommend sell)"
    )
):
    """
    Returns a single HTML page that contains:
      1) Latest RSI‐based advice (HTML)
      2) ATM options advice + Greeks (HTML, using live IV)
      3) The interactive Plotly candlestick chart (HTML)
      4) (Optionally) If pos_* parameters are provided, we also compute “sell vs hold”
         and embed that in a “Position Sell Advice” section.
    """
    ticker = ticker.upper()

    # 1) Validate look‐back limit
    max_days = ALLOWED_INTERVALS.get(interval, 0)
    if days > max_days:
        raise HTTPException(
            status_code=400,
            detail=f"{interval} supports at most {max_days} days; got {days}"
        )

    # 2) Fetch price data + indicators (for RSI & chart & options)
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
        1:  "RSI > 70 ⇒ over‐bought – consider bearish spread / buy put",
        -1: "RSI < 30 ⇒ over‐sold – consider bullish spread / buy call",
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
    # ─── Options Advice (using TODAY’s live IV for the next 5 expirations) ────
    if pred == 0:
        options_html = (
            "<h3>Options Advice</h3>"
            "<p>No ATM contract recommended (RSI neutral).</p>"
        )
    else:
        option_type = "call" if pred == -1 else "put"
        try:
            yf_tkr = yf.Ticker(ticker)
            expiries = yf_tkr.options
            if not expiries:
                raise RuntimeError(f"No expirations found for {ticker}.")

            # Fetch today’s closing price only once:
            hist = yf_tkr.history(period="1d")
            if hist.empty:
                raise RuntimeError(f"No price history for {ticker}.")
            last_close_price = float(hist["Close"].iloc[-1])

            suggestion_items = []
            collected = 0

            # Iterate through expirations until we collect 5 valid (days_to_exp > 0) entries
            for exp_date in expiries:
                if collected >= 5:
                    break

                # 1) Pull option chain for this expiration
                chain = yf_tkr.option_chain(exp_date)
                calls_df = chain.calls.copy()
                puts_df  = chain.puts.copy()

                # 2) Compute a “distance” column so we can pick ATM
                calls_df["dist"] = (calls_df["strike"] - last_close_price).abs()
                puts_df["dist"]  = (puts_df["strike"]  - last_close_price).abs()

                if option_type == "call":
                    atm_row = calls_df.loc[calls_df["dist"].idxmin()]
                else:
                    atm_row = puts_df.loc[puts_df["dist"].idxmin()]

                atm_strike = float(atm_row["strike"])
                iv_today   = atm_row.get("impliedVolatility", atm_row.get("impliedVol", None))
                if iv_today is None:
                    # If neither column exists, skip this expiry
                    continue
                iv_today = float(iv_today)

                # 3) Compute days to expiry
                today_dt   = datetime.utcnow().date()
                exp_dt_obj = datetime.strptime(exp_date, "%Y-%m-%d").date()
                days_to_exp = (exp_dt_obj - today_dt).days

                # Skip any expiry that is today or already expired (days_to_exp <= 0)
                if days_to_exp <= 0:
                    continue

                # 4) Compute Greeks – now days_to_exp is guaranteed > 0
                greeks = compute_greeks(
                    S=float(last["Close"]),
                    K=atm_strike,
                    r=r,
                    implied_vol=iv_today * 100.0,
                    days_to_expiry=days_to_exp,
                    option_type=option_type
                )

                # 5) Build one HTML <li> snippet for this expiration
                suggestion_items.append(f"""
                  <li>
                    <b>Expiry:</b> {exp_date} &nbsp;&nbsp;
                    <b>Strike:</b> {atm_strike:.2f} &nbsp;&nbsp;
                    <b>IV:</b> {iv_today * 100:.2f}% &nbsp;&nbsp;
                    <b>Δ:</b> {greeks['delta']:.3f}, 
                    <b>Γ:</b> {greeks['gamma']:.3f}, 
                    <b>Θ:</b> {greeks['theta']:.3f}, 
                    <b>V:</b> {greeks['vega']:.3f}, 
                    <b>ρ:</b> {greeks['rho']:.3f}
                  </li>
                """)
                collected += 1

            # If we couldn’t find any “future” expirations, show a message:
            if collected == 0:
                options_html = """
                  <h3>Options Advice</h3>
                  <p>No valid future expirations (days_to_expiry &gt; 0) could be computed.</p>
                """
            else:
                joined = "\n".join(suggestion_items)
                options_html = f"""
                    <h3>Options Advice → Next {collected} Expirations</h3>
                    <ul style="list-style: disc; margin-left: 1.5em;">
                        {joined}
                    </ul>
                """
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to build 5‐expiration suggestions for {ticker}: {e}"
            )

    # ─── Plotly Chart ────────────────────────────────────────────────────────
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

    # ─── Position Sell Advice (server‐side) ─────────────────────────────────
    position_html = ""
    if (
        pos_expiry is not None
        and pos_strike is not None
        and pos_purchase_price is not None
        and pos_quantity is not None
    ):
        try:
            # 1) Fetch underlying RSI with at least 50 calendar days so RSI(14)/SMA(50) exist
            df_under = get_price_dataframe(ticker, days=50, interval="1d")
            if df_under.empty:
                raise RuntimeError("Cannot fetch underlying price data.")
            rsi_val = float(df_under.iloc[-1]["rsi14"])
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Cannot fetch underlying data: {exc}")

        ym = yf.Ticker(ticker)
        all_exps = ym.options
        if pos_expiry not in all_exps:
            raise HTTPException(
                status_code=400,
                detail=f"Expiry {pos_expiry} not found. Available: {all_exps}"
            )

        opt_chain = ym.option_chain(pos_expiry)
        calls_df = opt_chain.calls
        puts_df  = opt_chain.puts

        # ─── Now “snap” pos_strike to the nearest existing strike:
        all_strikes = sorted(set(puts_df["strike"].tolist() + calls_df["strike"].tolist()))

        if pos_strike in all_strikes:
            nearest_strike = pos_strike
        else:
            nearest_strike = min(all_strikes, key=lambda s: abs(s - pos_strike))

        # 3) pick whether nearest_strike was a put or call
        if nearest_strike in puts_df["strike"].values:
            is_call = False
            row = puts_df[puts_df["strike"] == nearest_strike].iloc[0]
        elif nearest_strike in calls_df["strike"].values:
            is_call = True
            row = calls_df[calls_df["strike"] == nearest_strike].iloc[0]
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unexpected: strike {nearest_strike} not found in either calls or puts."
            )

        bid        = row.get("bid", None)
        ask        = row.get("ask", None)
        last_price = row.get("lastPrice", None)

        if bid and ask and (bid > 0) and (ask > 0):
            current_opt_price = float((bid + ask) / 2.0)
        elif last_price:
            current_opt_price = float(last_price)
        else:
            current_opt_price = float(row.get("mark", 0.0))

        paid = pos_purchase_price
        qty  = pos_quantity

        profit_pct_single = 0.0
        if paid != 0:
            profit_pct_single = (current_opt_price - paid) / paid

        pnl_dollars = (current_opt_price - paid) * qty * 100

        if (profit_pct_single >= pos_profit_target) or (rsi_val >= pos_rsi_sell_threshold):
            pos_action = "sell"
        else:
            pos_action = "hold"

        position_html = f"""
          <h3>Position Sell Advice</h3>
          <ul>
            <li><b>Nearest strike:</b> {nearest_strike:.2f}</li>
            <li><b>Is Call?</b> {is_call}</li>
            <li><b>Current Option Mid Price:</b> {current_opt_price:.4f}</li>
            <li><b>P&L % (per contract):</b> {profit_pct_single * 100:.2f}%</li>
            <li><b>P&L $ (total):</b> ${pnl_dollars:.2f}</li>
            <li><b>Underlying RSI(14):</b> {rsi_val:.2f}</li>
            <li><b>Profit Target:</b> {pos_profit_target * 100:.1f}%</li>
            <li><b>RSI Sell Threshold:</b> {pos_rsi_sell_threshold:.1f}</li>
            <li><b>Action:</b> {pos_action.upper()}</li>
          </ul>
        """

    # ─── Finally, render template “full.html” ──────────────────────────────
    return templates.TemplateResponse(
        "full.html",
        {
            "request": request,
            "ticker": ticker,
            "advice_html": advice_html,
            "options_html": options_html,
            "chart_div": chart_div,
            "position_html": position_html
        }
    )