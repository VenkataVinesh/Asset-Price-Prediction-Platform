import uvicorn
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict

# Import local ML modules
from utils import data_pipeline
from models import lstm_model
from models import arima_model

app = FastAPI(
    title="Asset Price Prediction API",
    description="Asynchronous prediction backend serving LSTM and ARIMA forecasts",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simulated historical data generation parameters
TICKER_CONFIGS = {
    "AAPL": {"start_price": 175.0, "daily_std": 0.012, "drift": 0.0005},
    "MSFT": {"start_price": 420.0, "daily_std": 0.010, "drift": 0.0008},
    "BTC-USD": {"start_price": 64000.0, "daily_std": 0.035, "drift": 0.0015}
}

def generate_historical_data(ticker: str, days: int = 60) -> pd.DataFrame:
    """Generates a realistic random walk pricing dataset for a given ticker."""
    if ticker not in TICKER_CONFIGS:
        raise ValueError(f"Unknown ticker: {ticker}")
        
    config = TICKER_CONFIGS[ticker]
    np.random.seed(42) # Seed to ensure consistency for demonstration
    
    prices = [config["start_price"]]
    for _ in range(days - 1):
        change_pct = np.random.normal(config["drift"], config["daily_std"])
        next_price = prices[-1] * (1.0 + change_pct)
        prices.append(float(next_price))
        
    # Create dates ending today
    today = datetime.now().date()
    dates = [today - timedelta(days=i) for i in range(days)]
    dates.reverse()
    
    df = pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "close": prices
    })
    return df

@app.get("/api/tickers", response_model=List[str])
async def get_tickers():
    """Returns the list of supported asset tickers."""
    return list(TICKER_CONFIGS.keys())

@app.get("/api/forecast")
async def get_forecast(
    ticker: str = Query("AAPL", description="Asset symbol"),
    model: str = Query("lstm", description="Forecasting model to run: lstm or arima"),
    horizon: int = Query(10, description="Forecasting steps ahead", ge=1, le=30)
):
    """
    Simulates feature extraction, runs ML forecast model, and returns
    historical indicators alongside predictions with confidence bounds.
    """
    if ticker not in TICKER_CONFIGS:
        raise HTTPException(status_code=404, detail="Ticker not supported")
        
    model = model.lower()
    if model not in ["lstm", "arima"]:
        raise HTTPException(status_code=400, detail="Invalid model type. Supported: lstm, arima")
        
    # 1. Generate and enrich historical time series
    try:
        df_hist = generate_historical_data(ticker, days=60)
        df_enriched = data_pipeline.enrich_dataset(df_hist, price_column="close")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data pipeline error: {str(e)}")
        
    # Convert dataframe back to dictionary payload
    history_list = df_enriched.to_dict(orient="records")
    close_prices = df_enriched["close"].tolist()
    rsi_list = df_enriched["rsi"].tolist()
    vol_list = df_enriched["volatility"].tolist()
    
    # 2. Run selected forecast inference engine
    forecast_points = []
    confidence_intervals = []
    
    if model == "lstm":
        forecast_prices = lstm_model.get_forecast(close_prices, rsi_list, vol_list, horizon=horizon)
        # For LSTM, simulate confidence intervals widening with time steps
        last_price = close_prices[-1]
        for i, val in enumerate(forecast_prices):
            error_margin = last_price * 0.015 * (i + 1)
            confidence_intervals.append({
                "lower": val - error_margin,
                "upper": val + error_margin
            })
    else: # arima
        forecast_prices, confidence_intervals = arima_model.get_forecast(close_prices, horizon=horizon)
        
    # 3. Format future forecasting dates
    last_date_str = history_list[-1]["date"]
    last_date = datetime.strptime(last_date_str, "%Y-%m-%d").date()
    
    forecast_list = []
    for i in range(horizon):
        next_date = last_date + timedelta(days=i+1)
        forecast_list.append({
            "date": next_date.strftime("%Y-%m-%d"),
            "price": forecast_prices[i],
            "lower_bound": confidence_intervals[i]["lower"],
            "upper_bound": confidence_intervals[i]["upper"]
        })
        
    return {
        "ticker": ticker,
        "model": model,
        "history": history_list,
        "forecast": forecast_list
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
