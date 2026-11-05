"""FastAPI service: history with indicators, plus LSTM or ARIMA multi-step forecasts.

Every response carries `data_source`. Prices are simulated (see data.py); the
models are real: the LSTM is trained on first use per ticker and the ARIMA order
is chosen by AIC. If a model fails, the request fails. Nothing is invented in
its place.
"""
from __future__ import annotations

import json
import os
from datetime import date, timedelta
from functools import lru_cache

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from data import DATA_SOURCE, TICKER_CONFIGS, simulated_prices
from models import arima_model
from models.lstm_model import LSTMForecaster
from utils import data_pipeline

app = FastAPI(title="Asset Price Prediction API", version="2.0.0",
              description="LSTM vs ARIMA forecasts on SIMULATED prices, with a walk-forward evaluation against a naive baseline.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])

HISTORY_DAYS = 60
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")


@lru_cache(maxsize=8)
def lstm_for(ticker: str) -> LSTMForecaster:
    close = simulated_prices(ticker)["close"].to_numpy()
    return LSTMForecaster(seed=0).fit(close)


def evaluation_for(ticker: str) -> dict | None:
    if not os.path.exists(RESULTS_PATH):
        return None
    with open(RESULTS_PATH) as f:
        return json.load(f)["tickers"].get(ticker)


@app.get("/api/tickers", response_model=list[str])
async def get_tickers():
    return list(TICKER_CONFIGS)


@app.get("/api/evaluation")
async def get_evaluation():
    if not os.path.exists(RESULTS_PATH):
        raise HTTPException(404, "No evaluation yet. Run `python evaluate.py` in backend/.")
    with open(RESULTS_PATH) as f:
        return json.load(f)


@app.get("/api/forecast")
def get_forecast(
    ticker: str = Query("AAPL"),
    model: str = Query("lstm", pattern="^(lstm|arima)$"),
    horizon: int = Query(10, ge=1, le=30),
):
    if ticker not in TICKER_CONFIGS:
        raise HTTPException(404, "Ticker not supported")
    df = simulated_prices(ticker)
    close = df["close"].to_numpy()
    enriched = data_pipeline.enrich_dataset(df.tail(HISTORY_DAYS + 30), price_column="close").tail(HISTORY_DAYS)

    try:
        if model == "lstm":
            m = lstm_for(ticker)
            points = m.forecast(close, horizon)
            detail = {"type": "stacked LSTM (2x32), trained on this series",
                      "interval": "±1.96 × validation residual std of log returns × √h",
                      "validation_residual_std": m.resid_std_}
        else:
            values, intervals, info = arima_model.get_forecast(close.tolist(), horizon)
            points = [{"price": v, **iv} for v, iv in zip(values, intervals)]
            detail = {"type": f"ARIMA{tuple(info['order'])}, order chosen by AIC", "interval": "analytical 95%", **info}
    except Exception as exc:  # surface the failure; do not fabricate a forecast
        raise HTTPException(502, f"{model} forecast failed: {exc}") from exc

    last = date.fromisoformat(df["date"].iloc[-1])
    return {
        "ticker": ticker,
        "model": model,
        "data_source": DATA_SOURCE,
        "simulated": True,
        "model_detail": detail,
        "evaluation": evaluation_for(ticker),
        "history": enriched.to_dict(orient="records"),
        "forecast": [
            {"date": (last + timedelta(days=i + 1)).isoformat(), "price": p["price"],
             "lower_bound": p["lower"], "upper_bound": p["upper"]}
            for i, p in enumerate(points)
        ],
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
