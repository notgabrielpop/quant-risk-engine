"""ARIMA model for return forecasting with prediction intervals.

Uses pmdarima auto_arima for order selection and walk-forward evaluation
with periodic refitting.
"""

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pmdarima as pm
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class ARIMAResult:
    """Container for ARIMA walk-forward results."""
    ticker: str
    order: tuple[int, int, int]
    aic: float
    forecasts: pd.DataFrame = field(default_factory=pd.DataFrame)
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))


class ARIMAForecaster:
    """ARIMA forecaster with walk-forward prediction intervals."""

    def __init__(self, max_p: int = 5, max_q: int = 5,
                 refit_every: int = 20) -> None:
        self.max_p = max_p
        self.max_q = max_q
        self.refit_every = refit_every
        self.model_ = None
        self.order_: tuple[int, int, int] = (1, 0, 1)
        self.aic_: float = np.nan

    def select_order(self, series: pd.Series) -> tuple[int, int, int]:
        """Run auto_arima to select order on training data."""
        try:
            auto = pm.auto_arima(
                series.values, d=0, max_p=self.max_p, max_q=self.max_q,
                seasonal=False, information_criterion="aic",
                stepwise=True, suppress_warnings=True, error_action="ignore",
            )
            self.order_ = auto.order
            self.aic_ = auto.aic()
        except Exception as e:
            logger.warning(f"auto_arima failed: {e}. Falling back to ARIMA(1,0,1)")
            self.order_ = (1, 0, 1)
            self.aic_ = np.nan
        return self.order_

    def walk_forward(self, train: pd.Series, test: pd.Series,
                     ticker: str = "") -> ARIMAResult:
        """Walk-forward forecasting with periodic refitting.

        Returns ARIMAResult with forecasts DataFrame containing:
            date, actual, predicted, lower_80, upper_80, lower_95, upper_95
        """
        order = self.order_
        full = pd.concat([train, test])
        train_end = len(train)
        n_test = len(test)

        rows = []
        resid_list = []

        # Initial fit on training data
        try:
            model = ARIMA(train.values, order=order).fit()
            resid_list = model.resid.tolist()
        except Exception as e:
            logger.warning(f"{ticker} initial ARIMA fit failed: {e}")
            model = None

        for i in range(n_test):
            t = train_end + i
            actual = full.iloc[t]
            date = full.index[t]

            # Refit periodically
            if i > 0 and i % self.refit_every == 0:
                try:
                    history = full.iloc[:t].values
                    model = ARIMA(history, order=order).fit()
                    if ticker and i % 100 == 0:
                        logger.info(f"  {ticker}: {i}/{n_test} predictions")
                except Exception:
                    pass  # keep last model

            if model is not None:
                try:
                    # Forecast 1 step ahead from current model
                    n_history = t - (len(model.resid))  # how many new obs since last fit
                    if n_history > 0:
                        new_obs = full.iloc[t - n_history:t].values
                        model = model.apply(new_obs)

                    fc = model.get_forecast(steps=1)
                    pred = fc.predicted_mean[0]
                    ci_80 = fc.conf_int(alpha=0.20)[0]
                    ci_95 = fc.conf_int(alpha=0.05)[0]
                except Exception:
                    pred = 0.0
                    sigma = train.std()
                    ci_80 = [-1.28 * sigma, 1.28 * sigma]
                    ci_95 = [-1.96 * sigma, 1.96 * sigma]
            else:
                pred = 0.0
                sigma = train.std()
                ci_80 = [-1.28 * sigma, 1.28 * sigma]
                ci_95 = [-1.96 * sigma, 1.96 * sigma]

            rows.append({
                "date": date,
                "actual": actual,
                "predicted": pred,
                "lower_80": ci_80[0],
                "upper_80": ci_80[1],
                "lower_95": ci_95[0],
                "upper_95": ci_95[1],
            })

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

        return ARIMAResult(
            ticker=ticker,
            order=order,
            aic=self.aic_,
            forecasts=df,
            residuals=np.array(resid_list),
        )
