"""ARIMA(p,1,q) with the order chosen by AIC on the training data."""
from __future__ import annotations

import itertools
import warnings

import numpy as np
from statsmodels.tsa.arima.model import ARIMA


def select_order(prices: np.ndarray, max_p: int = 2, max_q: int = 2) -> tuple[int, int, int]:
    best, best_aic = (0, 1, 0), np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p, q in itertools.product(range(max_p + 1), range(max_q + 1)):
            try:
                aic = ARIMA(prices, order=(p, 1, q)).fit().aic
            except Exception:  # some orders fail to converge; they simply don't compete
                continue
            if aic < best_aic:
                best, best_aic = (p, 1, q), aic
    return best


def fit(prices: np.ndarray, order: tuple[int, int, int] | None = None):
    order = order or select_order(prices)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ARIMA(prices, order=order).fit(), order


def get_forecast(prices: list[float], horizon: int = 10):
    """Multi-step forecast with the model's analytical 95% interval. Raises on failure: no invented fallback."""
    res, order = fit(np.asarray(prices, dtype=float))
    fc = res.get_forecast(steps=horizon)
    ci = np.asarray(fc.conf_int(alpha=0.05))
    values = [float(v) for v in np.asarray(fc.predicted_mean)]
    intervals = [{"lower": float(lo), "upper": float(hi)} for lo, hi in ci]
    return values, intervals, {"order": list(order), "aic": float(res.aic)}


def one_step_predictions(prices: np.ndarray, train_end: int, idx: np.ndarray, order=None) -> np.ndarray:
    """Fit on prices[:train_end] only, then filter forward with fixed parameters (no refit, no look-ahead)."""
    res, order = fit(prices[:train_end], order)
    full = res.apply(prices)
    return np.asarray(full.get_prediction().predicted_mean)[idx]
