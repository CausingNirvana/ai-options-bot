"""
backtest.py
-----------
Walk-forward PnL of a 1-bar option strategy:

• Model prediction  1  -> buy 1 ATM PUT at close, sell at next bar close
• Model prediction -1  -> buy 1 ATM CALL at close, sell at next bar close
• prediction == 0      -> stay flat

The option premium is approximated by Close * 0.01  (1 % of underlying)
so a +1 % move doubles the option value and vice-versa.
THIS IS A TOY BACK-TEST meant only to sanity-check model edge.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load

from .data_pipeline import get_price_dataframe


def simulate(ticker: str = "SPY",
             days: int = 120,
             interval: str = "1d") -> pd.DataFrame:
    """Return a dataframe with equity-curve of the simple option play."""
    model = load("models/lr_rsi.pkl")
    df = get_price_dataframe(ticker, days=days, interval=interval)

    # build feature matrix
    feats = np.column_stack([
        df["rsi14"],
        df["Close"] / df["sma20"],
        df["Close"] / df["sma50"],
    ])
    preds = model.predict(feats)

    df = df.copy()
    df["pred"] = preds

    # toy option payoff: premium = 1 % of underlying price
    premium = df["Close"] * 0.01

    # shift close to get next-bar return
    df["next_close"] = df["Close"].shift(-1)
    pct_move = (df["next_close"] - df["Close"]) / df["Close"]

    # payoff: put profits when price drops; call profits when price rises
    df["pnl"] = np.where(
        df["pred"] == 1,            # buy PUT
        (-pct_move - 0.01) * premium,    # payoff minus premium
        np.where(
            df["pred"] == -1,       # buy CALL
            (pct_move - 0.01) * premium,
            0.0                     # flat
        )
    )
    df["equity"] = df["pnl"].cumsum()
    return df.dropna()


def main():
    df = simulate("SPY", days=365, interval="1d")
    print("Total PnL:", df["equity"].iloc[-1].round(2))

    # plot equity curve
    plt.figure(figsize=(8, 4))
    plt.plot(df.index, df["equity"])
    plt.title("Toy 1-Day Option Strategy – Equity Curve")
    plt.ylabel("Cumulative $ PnL")
    plt.xlabel("Date")
    plt.tight_layout()
    out = Path("backtest_equity.png")
    plt.savefig(out)
    print("Chart saved ->", out)


if __name__ == "__main__":
    main()
