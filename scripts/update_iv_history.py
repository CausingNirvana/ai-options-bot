# ─────────────────────────────────────────────────────────────────────────────
# scripts/update_iv_history.py

import csv
import os
from datetime import datetime

import yfinance as yf

def fetch_atm_iv_and_append(ticker: str):
    """
    1) Connect to Yahoo via yfinance.
    2) Find the nearest-expiry date (first in .options list).
    3) Download its call chain.
    4) Identify the strike closest to the current price (ATM).
    5) Read impliedVolatility (decimal, e.g. 0.2012 → 20.12%).
    6) Append today’s date + IV to iv_history_{ticker}.csv.
    """

    ticker = ticker.upper()
    csv_filename = f"iv_history_{ticker}.csv"

    # 1) Connect to yfinance and get front-month expiry
    yf_tkr = yf.Ticker(ticker)
    expiries = yf_tkr.options  # e.g. ['2025-06-20', '2025-07-18', …]
    if not expiries:
        raise RuntimeError(f"No option expiration dates found for {ticker}.")
    front_month = expiries[0]

    # 2) Get today’s last close price (for ATM selection)
    hist = yf_tkr.history(period="1d")
    if hist.empty:
        raise RuntimeError(f"No price history available for {ticker}.")
    last_close = float(hist["Close"].iloc[-1])

    # 3) Download the call chain for front_month
    opt_chain = yf_tkr.option_chain(front_month)
    calls_df = opt_chain.calls

    # 4) Find the row whose strike is closest to last_close
    calls_df["dist"] = (calls_df["strike"] - last_close).abs()
    atm_row = calls_df.loc[calls_df["dist"].idxmin()]

    # 5) Read impliedVolatility (yfinance’s column name)
    try:
        iv_today = float(atm_row["impliedVolatility"])  # e.g. 0.2012 → 20.12%
    except KeyError:
        raise RuntimeError(
            "Could not find 'impliedVolatility' column in yfinance output. "
            "Please ensure you have a recent yfinance version (pip install --upgrade yfinance)."
        )

    # 6) Append to CSV: columns “date,atm_iv”
    today_str = datetime.utcnow().date().isoformat()  # YYYY-MM-DD
    file_exists = os.path.isfile(csv_filename)

    # If file doesn’t exist, create and write header
    if not file_exists:
        with open(csv_filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "atm_iv"])
            writer.writerow([today_str, f"{iv_today:.6f}"])
        print(f"Created {csv_filename}, wrote {today_str},{iv_today:.6f}")
    else:
        # Check if today’s date is already in the last row (to avoid duplicates)
        with open(csv_filename, "r", newline="") as f:
            rows = list(csv.reader(f))
            if len(rows) >= 2 and rows[-1][0] == today_str:
                print(f"{csv_filename} already has entry for {today_str}. Skipping append.")
                return
        # Append new row
        with open(csv_filename, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([today_str, f"{iv_today:.6f}"])
        print(f"Appended to {csv_filename}: {today_str},{iv_today:.6f}")


if __name__ == "__main__":
    # Example usage: python update_iv_history.py SPY
    import sys

    if len(sys.argv) != 2:
        print("Usage: python update_iv_history.py <TICKER>")
        sys.exit(1)

    ticker_arg = sys.argv[1]
    try:
        fetch_atm_iv_and_append(ticker_arg)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("Done.")
