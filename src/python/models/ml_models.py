"""ML baseline models: Random Forest and XGBoost for next-day return prediction.

Walk-forward evaluation with periodic refitting and pseudo-prediction intervals.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


@dataclass
class MLResult:
    """Container for ML walk-forward results."""
    model_name: str
    ticker: str
    forecasts: pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)


class RFForecaster:
    """Random Forest with walk-forward evaluation and tree-based intervals."""

    def __init__(self, n_estimators: int = 500, max_depth: int = 10,
                 min_samples_leaf: int = 20, refit_every: int = 60) -> None:
        self.params = dict(n_estimators=n_estimators, max_depth=max_depth,
                           min_samples_leaf=min_samples_leaf, random_state=42,
                           n_jobs=-1)
        self.refit_every = refit_every
        self.model_ = None

    def walk_forward(self, features: pd.DataFrame, target: pd.Series,
                     train_end: pd.Timestamp, ticker: str = "") -> MLResult:
        """Walk-forward prediction with periodic refitting."""
        train_mask = features.index < train_end
        test_mask = features.index >= train_end

        X_all = features.values
        y_all = target.values
        dates = features.index

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        rows = []
        last_importance = None

        for step, i in enumerate(test_idx):
            # Refit periodically
            if step % self.refit_every == 0:
                hist_end = i
                X_tr = X_all[:hist_end]
                y_tr = y_all[:hist_end]
                self.model_ = RandomForestRegressor(**self.params)
                self.model_.fit(X_tr, y_tr)
                last_importance = pd.Series(
                    self.model_.feature_importances_, index=features.columns
                ).sort_values(ascending=False)
                if ticker and step % 100 == 0:
                    logger.info(f"  RF {ticker}: {step}/{len(test_idx)}")

            X_test = X_all[i:i + 1]
            pred = self.model_.predict(X_test)[0]

            # Pseudo-PI from individual tree predictions
            tree_preds = np.array([t.predict(X_test)[0]
                                   for t in self.model_.estimators_])
            lo_95, hi_95 = np.percentile(tree_preds, [2.5, 97.5])
            lo_80, hi_80 = np.percentile(tree_preds, [10, 90])

            rows.append({
                "date": dates[i],
                "actual": y_all[i],
                "predicted": pred,
                "lower_80": lo_80,
                "upper_80": hi_80,
                "lower_95": lo_95,
                "upper_95": hi_95,
            })

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

        return MLResult(
            model_name="RandomForest",
            ticker=ticker,
            forecasts=df,
            feature_importance=last_importance.to_frame("importance").reset_index()
            if last_importance is not None else pd.DataFrame(),
        )


class XGBForecaster:
    """XGBoost with walk-forward evaluation."""

    def __init__(self, n_estimators: int = 500, max_depth: int = 6,
                 learning_rate: float = 0.01, refit_every: int = 60) -> None:
        self.params = dict(n_estimators=n_estimators, max_depth=max_depth,
                           learning_rate=learning_rate, subsample=0.8,
                           colsample_bytree=0.8, random_state=42,
                           verbosity=0, n_jobs=-1)
        self.refit_every = refit_every
        self.model_ = None

    def walk_forward(self, features: pd.DataFrame, target: pd.Series,
                     train_end: pd.Timestamp, ticker: str = "") -> MLResult:
        """Walk-forward prediction with periodic refitting."""
        X_all = features.values
        y_all = target.values
        dates = features.index

        test_mask = features.index >= train_end
        test_idx = np.where(test_mask)[0]

        rows = []
        last_importance = None

        for step, i in enumerate(test_idx):
            if step % self.refit_every == 0:
                hist_end = i
                X_tr = X_all[:hist_end]
                y_tr = y_all[:hist_end]
                self.model_ = XGBRegressor(**self.params)
                self.model_.fit(X_tr, y_tr)
                last_importance = pd.Series(
                    self.model_.feature_importances_, index=features.columns
                ).sort_values(ascending=False)
                if ticker and step % 100 == 0:
                    logger.info(f"  XGB {ticker}: {step}/{len(test_idx)}")

            X_test = X_all[i:i + 1]
            pred = self.model_.predict(X_test)[0]

            rows.append({
                "date": dates[i],
                "actual": y_all[i],
                "predicted": pred,
                "lower_80": np.nan,
                "upper_80": np.nan,
                "lower_95": np.nan,
                "upper_95": np.nan,
            })

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

        return MLResult(
            model_name="XGBoost",
            ticker=ticker,
            forecasts=df,
            feature_importance=last_importance.to_frame("importance").reset_index()
            if last_importance is not None else pd.DataFrame(),
        )
