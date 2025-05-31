"""
make_dataset.py

Builds a simple classification dataset:

Features (X):
    rsi14, price/sma20, price/sma50

Labels (y):
    1  → RSI > 70  (over-bought   → bearish bias)
   -1  → RSI < 30  (over-sold     → bullish bias)
Neutral rows (30 ≤ RSI ≤ 70) are dropped.
"""

import numpy as np
import pandas as pd
from src.data_pipeline import get_price_dataframe

def build_dataset(
    ticker: str = "SPY",
    lookback_days: int = 90,
    interval: str = "1h",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Returns:
      X : numpy array of shape (n_samples, 3) containing [rsi14, p_sma20, p_sma50]
      y : numpy array of shape (n_samples,) containing labels {1, -1}
      df: pandas DataFrame (filtered) with columns ['Open', ..., 'rsi14','sma20','sma50','p_sma20','p_sma50','label']
    """

    # Note: get_price_dataframe now requires keyword args for days and interval
    df = get_price_dataframe(
        ticker=ticker,
        days=lookback_days,
        interval=interval
    )

    # Compute price-to-SMA ratios
    df["p_sma20"] = df["Close"] / df["sma20"]
    df["p_sma50"] = df["Close"] / df["sma50"]

    # Create labels: 1 if rsi14 > 70, -1 if rsi14 < 30, else 0
    conditions = [
        df["rsi14"] > 70,
        df["rsi14"] < 30,
    ]
    df["label"] = np.select(conditions, [1, -1], default=0)

    # Keep only overbought/oversold rows
    df = df[df["label"] != 0]

    X = df[["rsi14", "p_sma20", "p_sma50"]].values
    y = df["label"].values

    return X, y, df
