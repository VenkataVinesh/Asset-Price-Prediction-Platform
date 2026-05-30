from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from typing import List, Tuple, Dict

def get_forecast(prices: List[float], horizon: int = 10) -> Tuple[List[float], List[Dict[str, float]]]:
    """
    Fits an ARIMA(1, 1, 1) model to the historical prices and returns the projected
    future values along with the 95% confidence intervals.
    """
    try:
        # Fit ARIMA model on historical price sequence
        model = ARIMA(prices, order=(1, 1, 1))
        model_fit = model.fit()
        
        # Forecast steps ahead
        forecast_res = model_fit.get_forecast(steps=horizon)
        predictions = forecast_res.predicted_mean
        
        # Get 95% confidence intervals
        conf_int = forecast_res.conf_int(alpha=0.05)
        
        forecast_values = [float(x) for x in predictions]
        intervals = []
        for i in range(horizon):
            intervals.append({
                "lower": float(conf_int[i][0]),
                "upper": float(conf_int[i][1])
            })
            
        return forecast_values, intervals
    except Exception as e:
        # Fallback if ARIMA fitting fails (e.g. insufficient stationarity or data length)
        last_price = prices[-1]
        forecast_values = []
        intervals = []
        drift = (prices[-1] - prices[0]) / len(prices)
        
        for i in range(horizon):
            pred = last_price + drift * (i + 1)
            forecast_values.append(float(pred))
            intervals.append({
                "lower": float(pred - (last_price * 0.02 * (i + 1))),
                "upper": float(pred + (last_price * 0.02 * (i + 1)))
            })
        return forecast_values, intervals
