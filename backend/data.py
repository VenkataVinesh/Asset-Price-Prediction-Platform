"""Price data.

There is no market-data feed in this project. Prices are SIMULATED with seeded
geometric Brownian motion, and every API response says so. `load_csv` is the
hook for real data (a file with `date` and `close` columns).
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

DATA_SOURCE = "simulated: seeded geometric Brownian motion (no market data)"

TICKER_CONFIGS = {
    "AAPL": {"start_price": 175.0, "daily_std": 0.012, "drift": 0.0005, "seed": 1},
    "MSFT": {"start_price": 420.0, "daily_std": 0.010, "drift": 0.0008, "seed": 2},
    "BTC-USD": {"start_price": 64000.0, "daily_std": 0.035, "drift": 0.0015, "seed": 3},
}


def simulated_prices(ticker: str, days: int = 500, end: date | None = None) -> pd.DataFrame:
    if ticker not in TICKER_CONFIGS:
        raise ValueError(f"Unknown ticker: {ticker}")
    cfg = TICKER_CONFIGS[ticker]
    rng = np.random.default_rng(cfg["seed"])
    log_ret = rng.normal(cfg["drift"] - 0.5 * cfg["daily_std"] ** 2, cfg["daily_std"], days - 1)
    close = cfg["start_price"] * np.exp(np.concatenate([[0.0], np.cumsum(log_ret)]))
    end = end or date(2026, 11, 4)  # fixed so results are reproducible
    dates = [end - timedelta(days=days - 1 - i) for i in range(days)]
    return pd.DataFrame({"date": [d.isoformat() for d in dates], "close": close})


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"date", "close"} <= set(df.columns):
        raise ValueError("CSV needs 'date' and 'close' columns")
    return df[["date", "close"]].sort_values("date").reset_index(drop=True)
