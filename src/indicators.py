import pandas_ta as ta
import pandas as pd

def add_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rsi14"] = ta.rsi(out["Close"], length=14)
    out["sma20"] = ta.sma(out["Close"], length=20)
    out["sma50"] = ta.sma(out["Close"], length=50)
    out["atr14"] = ta.atr(out["High"], out["Low"], out["Close"], length=14)
    return out.dropna()
