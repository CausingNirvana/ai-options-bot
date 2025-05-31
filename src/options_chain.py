"""
options_chain.py

Utilities to fetch the options chain from yfinance, select the ATM contract,
and compute Greeks using Black-Scholes (via mibian).
"""

from __future__ import annotations
from datetime import datetime, timedelta
from typing import Literal

import pandas as pd
import yfinance as yf
import mibian


def get_nearest_expiry_and_strike(
    ticker: str,
    option_type: Literal["call", "put"],
    days_lookahead: int = 30
) -> tuple[str, float]:
    """
    Fetch the list of expiry dates for the given ticker, pick the nearest expiry
    that’s at least 1 calendar day (24 hours) in the future. Then, from that
    expiry’s chain, pick the ATM strike (closest to current underlying price).

    Returns:
      expiry (str)   – 'YYYY-MM-DD'
      atm_strike (float)
    """
    yf_tkr = yf.Ticker(ticker)
    all_expiries = yf_tkr.options  # e.g., ['2025-06-20', '2025-07-18', ...]
    if not all_expiries:
        raise ValueError(f"No options data for {ticker}")

    today = datetime.utcnow().date()
    # Parse to date objects and keep only >= tomorrow
    candidate_dates = [
        datetime.strptime(d, "%Y-%m-%d").date()
        for d in all_expiries
    ]
    future_dates = [d for d in candidate_dates if d >= today + timedelta(days=1)]
    if not future_dates:
        raise ValueError(f"No future expiries for {ticker}")
    nearest_date = min(future_dates)
    expiry_str = nearest_date.strftime("%Y-%m-%d")

    # Download the option chain for that expiry
    opt_chain = yf_tkr.option_chain(expiry_str)
    df_chain = opt_chain.calls if option_type == "call" else opt_chain.puts

    # Get the current underlying price
    hist = yf_tkr.history(period="1d", interval="1m")
    if hist.empty:
        # fallback: use last close from the daily history
        hist = yf_tkr.history(period="5d", interval="1d")
    current_price = float(hist["Close"].iloc[-1])

    # Find the ATM strike (closest to current_price)
    df_chain["diff"] = (df_chain["strike"] - current_price).abs()
    atm_row = df_chain.loc[df_chain["diff"].idxmin()]

    return expiry_str, float(atm_row["strike"])


def compute_greeks(
    S: float,
    K: float,
    r: float,
    implied_vol: float,
    days_to_expiry: int,
    option_type: Literal["call", "put"],
) -> dict[str, float]:
    """
    Compute Black-Scholes Greeks using mibian.
    • S : spot price (float)
    • K : strike price (float)
    • r : risk-free interest rate in % (e.g. 1.5 for 1.5%)
    • implied_vol : implied volatility in % (e.g. 20 for 20%)
    • days_to_expiry : integer days until expiry (calendar days)
    • option_type : "call" or "put"

    Returns a dict with: delta, gamma, theta, vega, rho
    """
    # mibian.BS expects [spot, strike, r, days] and vol in percent
    bs = mibian.BS([S, K, r, days_to_expiry], volatility=implied_vol)
    if option_type == "call":
        return {
            "delta": bs.callDelta,
            "gamma": bs.gamma,
            "theta": bs.callTheta,
            "vega": bs.vega,
            "rho": bs.callRho,
        }
    else:
        return {
            "delta": bs.putDelta,
            "gamma": bs.gamma,
            "theta": bs.putTheta,
            "vega": bs.vega,
            "rho": bs.putRho,
        }
