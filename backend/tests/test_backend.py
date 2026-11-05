import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from data import simulated_prices
from models import arima_model
from models.lstm_model import LSTMForecaster, features


def test_simulated_prices_are_reproducible_and_positive():
    a = simulated_prices("AAPL")["close"].to_numpy()
    b = simulated_prices("AAPL")["close"].to_numpy()
    assert np.array_equal(a, b) and (a > 0).all() and len(a) == 500


def test_features_are_causal():
    close = simulated_prices("MSFT")["close"].to_numpy()
    full = features(close)
    cut = features(close[:300])
    np.testing.assert_allclose(full[:300], cut, rtol=1e-9, atol=1e-12)  # future prices never change past rows


def test_lstm_scaler_uses_training_rows_only_and_is_deterministic():
    close = simulated_prices("AAPL")["close"].to_numpy()
    m1 = LSTMForecaster(epochs=5, seed=0).fit(close, train_end=300)
    m2 = LSTMForecaster(epochs=5, seed=0).fit(close, train_end=300)
    val_start = int(300 * 0.85)
    np.testing.assert_allclose(m1.mean_, features(close[:300])[:val_start].mean(0))
    assert m1.predict_return(close[:300]) == m2.predict_return(close[:300])


def test_lstm_forecast_bands_widen_and_contain_the_point():
    close = simulated_prices("AAPL")["close"].to_numpy()
    out = LSTMForecaster(epochs=5).fit(close).forecast(close, 5)
    widths = [p["upper"] - p["lower"] for p in out]
    assert all(p["lower"] < p["price"] < p["upper"] for p in out)
    assert widths == sorted(widths)


def test_arima_order_selected_by_aic_and_one_step_is_causal():
    close = simulated_prices("MSFT")["close"].to_numpy()
    order = arima_model.select_order(close[:200])
    assert order[1] == 1
    idx = np.arange(200, 210)
    p1 = arima_model.one_step_predictions(close, 200, idx, order)
    tampered = close.copy()
    tampered[215:] *= 10  # changing the future must not change these predictions
    p2 = arima_model.one_step_predictions(tampered, 200, idx, order)
    np.testing.assert_allclose(p1, p2)


def test_api_labels_simulated_data_and_rejects_bad_input():
    from main import app
    c = TestClient(app)
    r = c.get("/api/forecast", params={"ticker": "MSFT", "model": "arima", "horizon": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["simulated"] is True and "simulated" in body["data_source"]
    assert len(body["forecast"]) == 3 and len(body["history"]) == 60
    assert c.get("/api/forecast", params={"ticker": "NOPE"}).status_code == 404
    assert c.get("/api/forecast", params={"model": "magic"}).status_code == 422
