"""
src/data_pipeline.py
Pulls price data with yfinance and adds basic TA indicators.
"""

import yfinance as yf
from .indicators import add_basic_indicators   # relative import

def get_price_dataframe(ticker: str = "SPY",
                        days: int = 2,
                        interval: str = "30m"):
    """
    Returns a pandas DataFrame with OHLCV + RSI, SMA20/50, ATR.
    """
    raw = yf.download(
        ticker,
        period=f"{days}d",
        interval=interval,
        progress=False
    )
    raw = raw.rename(columns=str.title)         # unify col names
    return add_basic_indicators(raw)
