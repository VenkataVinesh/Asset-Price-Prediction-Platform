"""Walk-forward, one-step-ahead evaluation of LSTM and ARIMA against a naive baseline.

    cd backend && python evaluate.py        # writes results.json and ../docs/evaluation.png

Both models are fitted on the first 80% of each SIMULATED series and then
predict every remaining day using only the prices before it. On a random walk
the honest expectation is that nothing beats "tomorrow = today"; this script
measures whether that holds rather than assuming either way.
"""
from __future__ import annotations

import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import DATA_SOURCE, TICKER_CONFIGS, simulated_prices
from models import arima_model
from models.lstm_model import LSTMForecaster

TRAIN_FRAC = 0.8


def metrics(pred: np.ndarray, actual: np.ndarray, prev: np.ndarray) -> dict:
    err = pred - actual
    # A move smaller than 1e-9 of the price is floating-point noise, not a directional call.
    tol = 1e-9 * np.abs(prev)
    d_pred = np.where(np.abs(pred - prev) > tol, np.sign(pred - prev), 0)
    d_true = np.sign(actual - prev)
    called = d_pred != 0
    n = int(called.sum())
    hits = int((d_pred[called] == d_true[called]).sum())
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "hit_rate": hits / n if n else None,
        "directional_calls": n,
        "margin_95_pp": 196 * math.sqrt(0.25 / n) if n else None,
    }


def evaluate(ticker: str) -> tuple[dict, dict]:
    close = simulated_prices(ticker)["close"].to_numpy()
    train_end = int(len(close) * TRAIN_FRAC)
    idx = np.arange(train_end, len(close))
    actual, prev = close[idx], close[idx - 1]

    order = arima_model.select_order(close[:train_end])
    preds = {
        "naive": prev.copy(),
        "arima": arima_model.one_step_predictions(close, train_end, idx, order),
        "lstm": LSTMForecaster(seed=0).fit(close, train_end).one_step_predictions(close, idx),
    }
    res = {name: metrics(p, actual, prev) for name, p in preds.items()}
    res["arima"]["order"] = list(order)
    res["test_points"] = int(len(idx))
    return res, {"actual": actual, **preds}


def main() -> None:
    out = {"data_source": DATA_SOURCE, "protocol": f"fit on first {int(TRAIN_FRAC*100)}%, one-step walk-forward on the rest", "tickers": {}}
    fig, axes = plt.subplots(1, len(TICKER_CONFIGS), figsize=(15, 4))
    for ax, ticker in zip(axes, TICKER_CONFIGS):
        res, series = evaluate(ticker)
        out["tickers"][ticker] = res
        k = slice(0, 40)
        ax.plot(series["actual"][k], color="black", lw=1.4, label="actual (simulated)")
        ax.plot(series["naive"][k], color="#999", ls="--", label=f"naive  MAE {res['naive']['mae']:.2f}")
        ax.plot(series["arima"][k], color="#c05621", label=f"ARIMA{tuple(res['arima']['order'])}  MAE {res['arima']['mae']:.2f}")
        ax.plot(series["lstm"][k], color="#2b6cb0", label=f"LSTM  MAE {res['lstm']['mae']:.2f}")
        ax.set_title(f"{ticker}: first 40 test days, one step ahead")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("Simulated prices (geometric Brownian motion), 100 walk-forward test days per ticker. Lower MAE is better.", fontsize=10)
    fig.tight_layout()
    os.makedirs("../docs", exist_ok=True)
    fig.savefig("../docs/evaluation.png", dpi=110)
    with open("results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
