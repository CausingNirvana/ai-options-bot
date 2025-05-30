"""
src/data_pipeline.py
Pulls price data with yfinance and adds basic TA indicators.
"""

import yfinance as yf
# src/data_pipeline.py
from datetime import datetime, timedelta
import yfinance as yf
from .indicators import add_basic_indicators
import pandas as pd


def get_price_dataframe(
    ticker: str = "SPY",
    days: int = 5,
    interval: str = "1h",        # 1-hour bars work any time of day
) -> pd.DataFrame:
    end   = datetime.utcnow()
    start = end - timedelta(days=days)

    raw = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column",        # <- keeps single-level columns
    )

    if raw.empty:
        raise ValueError(f"No {interval} data returned for {ticker}. "
                         "Try a coarser interval (e.g. '1h' or '1d').")

    raw.rename(columns=str.title, inplace=True)   # Open, High, …
    raw = add_basic_indicators(raw)

    # make sure index is datetime and sorted
    raw.sort_index(inplace=True)
    return raw
