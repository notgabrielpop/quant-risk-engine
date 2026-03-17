"""Feature engineering for ML baseline models (ARIMA, RF, XGBoost).

Computes technical indicators, rolling statistics, and lagged features
from aligned price and return data for supervised learning.

TODO:
  - Rolling realized volatility (5d, 21d, 63d windows)
  - RSI, MACD, Bollinger Bands
  - Cross-asset correlation features
  - Regime indicators from VIX levels
"""

from pathlib import Path

import numpy as np
import pandas as pd


def compute_rolling_features(
    returns: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute rolling statistics for each asset.

    Args:
        returns: DataFrame of log-returns.
        windows: Rolling window sizes in days.

    Returns:
        DataFrame with rolling mean, std, skew, kurtosis per asset per window.
    """
    if windows is None:
        windows = [5, 21, 63]

    features = pd.DataFrame(index=returns.index)

    for col in returns.columns:
        for w in windows:
            features[f"{col}_mean_{w}d"] = returns[col].rolling(w).mean()
            features[f"{col}_vol_{w}d"] = returns[col].rolling(w).std()
            features[f"{col}_skew_{w}d"] = returns[col].rolling(w).skew()
            features[f"{col}_kurt_{w}d"] = returns[col].rolling(w).kurt()

    return features.dropna()


def compute_lagged_returns(
    returns: pd.DataFrame,
    n_lags: int = 5,
) -> pd.DataFrame:
    """Create lagged return features.

    Args:
        returns: DataFrame of log-returns.
        n_lags: Number of lags to include.

    Returns:
        DataFrame with lagged return columns.
    """
    features = pd.DataFrame(index=returns.index)

    for col in returns.columns:
        for lag in range(1, n_lags + 1):
            features[f"{col}_lag_{lag}"] = returns[col].shift(lag)

    return features.dropna()
