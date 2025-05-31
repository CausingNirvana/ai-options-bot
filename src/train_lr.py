"""
train_lr.py

Train a baseline logistic-regression model on the RSI-labelled dataset
and save it to models/lr_rsi.pkl.

Usage:
    python -m src.train_lr
"""

from pathlib import Path
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.make_dataset import build_dataset

if __name__ == "__main__":
    # 1) Build the data (now using keyword args)
    X, y, _ = build_dataset(
        lookback_days=90,
        interval="1h"
    )

    # 2) Chronological split (80% train, 20% test)
    split = int(0.8 * len(X))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    # 3) Fit logistic-regression
    clf = LogisticRegression(max_iter=600).fit(X_tr, y_tr)

    # 4) Print train/test accuracy
    print(f"train acc: {accuracy_score(y_tr, clf.predict(X_tr)):.3f}")
    print(f"test  acc: {accuracy_score(y_te, clf.predict(X_te)):.3f}")

    # 5) Persist the model
    Path("models").mkdir(exist_ok=True)
    dump(clf, "models/lr_rsi.pkl")
    print("model saved → models/lr_rsi.pkl")
