"""Feature engineering for ML baseline models.

Computes lagged returns, rolling statistics, VIX features, and calendar
dummies from aligned return data for supervised next-day return prediction.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from python.utils.config import DATA_PROCESSED

logger = logging.getLogger(__name__)


def build_features(
    returns: pd.DataFrame,
    vix: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build feature matrix for ML models.

    Args:
        returns: DataFrame of daily log-returns (assets as columns).
        vix: Optional VIX price series for additional features.

    Returns:
        DataFrame with features. Index aligned to returns. NaN-leading rows dropped.
    """
    feats = pd.DataFrame(index=returns.index)

    for col in returns.columns:
        r = returns[col]

        # Lagged returns
        for lag in [1, 2, 3, 5, 10, 20]:
            feats[f"{col}_lag{lag}"] = r.shift(lag)

        # Rolling mean
        for w in [5, 20, 60]:
            feats[f"{col}_mean{w}d"] = r.rolling(w).mean()

        # Rolling volatility
        for w in [5, 20, 60]:
            feats[f"{col}_vol{w}d"] = r.rolling(w).std()

        # Rolling skewness & kurtosis
        for w in [20, 60]:
            feats[f"{col}_skew{w}d"] = r.rolling(w).skew()
            feats[f"{col}_kurt{w}d"] = r.rolling(w).kurt()

        # Rolling min/max/range (20d)
        feats[f"{col}_min20d"] = r.rolling(20).min()
        feats[f"{col}_max20d"] = r.rolling(20).max()
        feats[f"{col}_range20d"] = feats[f"{col}_max20d"] - feats[f"{col}_min20d"]

    # VIX features
    if vix is not None:
        vix_aligned = vix.reindex(returns.index)
        if "VIX" in vix_aligned.columns:
            feats["VIX_level"] = vix_aligned["VIX"]
            feats["VIX_change"] = vix_aligned["VIX"].diff()
        elif len(vix_aligned.columns) == 1:
            feats["VIX_level"] = vix_aligned.iloc[:, 0]
            feats["VIX_change"] = vix_aligned.iloc[:, 0].diff()

    # Calendar dummies
    feats["dow"] = returns.index.dayofweek
    for d in range(5):
        feats[f"dow_{d}"] = (feats["dow"] == d).astype(int)
    feats["month"] = returns.index.month
    for m in range(1, 13):
        feats[f"month_{m}"] = (feats["month"] == m).astype(int)
    feats.drop(columns=["dow", "month"], inplace=True)

    # Drop NaN rows from rolling calculations
    n_before = len(feats)
    feats = feats.dropna()
    logger.info(f"Features: {feats.shape[1]} columns, {len(feats)} rows "
                f"({n_before - len(feats)} dropped from rolling windows)")

    return feats


def build_and_save(
    returns_path: Path | None = None,
    vix_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Build features and save to CSV."""
    if returns_path is None:
        returns_path = DATA_PROCESSED / "portfolio_returns.csv"
    if vix_path is None:
        vix_path = DATA_PROCESSED / "vix_daily.csv"
    if output_path is None:
        output_path = DATA_PROCESSED / "ml_features.csv"

    returns = pd.read_csv(returns_path, index_col="Date", parse_dates=True)
    vix = pd.read_csv(vix_path, index_col="Date", parse_dates=True)

    feats = build_features(returns, vix)
    feats.to_csv(output_path)
    return feats
