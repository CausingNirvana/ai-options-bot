"""
data_pipeline.py
Download Yahoo data, flatten any MultiIndex layout, add indicators,
return a tidy DataFrame (no NaNs, single-level columns).
"""

from __future__ import annotations
import pandas as pd
import yfinance as yf
from .indicators import add_basic_indicators

# ───────────────────────────────────────── helpers ──────────────────────────────────────────
def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make columns simple strings (Open, Close, …) regardless of Yahoo layout.
    Handles:
      • ('Price', 'Close')
      • ('Close',  'SPY')
      • ('Close',)  (already flat)
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Case A ─ first level is 'Price'
        if "Price" in df.columns.get_level_values(0):
            df = df.xs("Price", level=0, axis=1)

        # At this point layout may be ('Close','SPY').  Drop the ticker level.
        if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels == 2:
            df.columns = df.columns.get_level_values(0)

    # Title-case all names
    df.columns = [str(c).strip().title() for c in df.columns]
    return df

# ───────────────────────────────────────── public API ───────────────────────────────────────
def get_price_dataframe(
    ticker: str = "SPY",
    *,
    days: int = 120,
    interval: str = "1d",
) -> pd.DataFrame:
    """Return OHLCV + RSI/SMA/ATR with ≥ 50 rows (SMA-50 needs that)."""
    raw = yf.download(
        ticker,
        period=f"{days}d",
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}")

    raw = _flatten(raw)

    if "Close" not in raw.columns:
        raise RuntimeError(f"'Close' column still missing → {raw.columns.tolist()}")

    tidy = add_basic_indicators(raw).sort_index()
    return tidy
