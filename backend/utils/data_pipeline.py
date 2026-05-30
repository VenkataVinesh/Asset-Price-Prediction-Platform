import pandas as pd
import numpy as np

def calculate_sma(series: pd.Series, window: int = 14) -> pd.Series:
    """Calculates Simple Moving Average (SMA)"""
    return series.rolling(window=window, min_periods=1).mean()

def calculate_ema(series: pd.Series, window: int = 14) -> pd.Series:
    """Calculates Exponential Moving Average (EMA)"""
    return series.ewm(span=window, adjust=False, min_periods=1).mean()

def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Calculates Relative Strength Index (RSI)"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    # Replace initial NaNs
    rsi.iloc[0] = 50.0
    return rsi

def calculate_volatility(series: pd.Series, window: int = 14) -> pd.Series:
    """Calculates historical volatility based on log returns"""
    log_returns = np.log(series / series.shift(1))
    volatility = log_returns.rolling(window=window, min_periods=1).std()
    volatility.iloc[0] = 0.0
    return volatility.fillna(0.0)

def enrich_dataset(df: pd.DataFrame, price_column: str = "close") -> pd.DataFrame:
    """Computes all technical indicators for the dataframe"""
    df = df.copy()
    df["sma"] = calculate_sma(df[price_column])
    df["ema"] = calculate_ema(df[price_column])
    df["rsi"] = calculate_rsi(df[price_column])
    df["volatility"] = calculate_volatility(df[price_column])
    return df
