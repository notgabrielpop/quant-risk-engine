"""ARIMA model for return forecasting baseline.

Uses pmdarima auto_arima for automatic order selection.

TODO:
  - Implement fit/predict interface
  - Rolling-window re-estimation
  - AIC/BIC diagnostics
"""

import pandas as pd


class ARIMAForecaster:
    """ARIMA forecaster with automatic order selection."""

    def __init__(self, max_p: int = 5, max_q: int = 5, seasonal: bool = False) -> None:
        self.max_p = max_p
        self.max_q = max_q
        self.seasonal = seasonal
        self.model_ = None
        self.order_: tuple[int, int, int] | None = None

    def fit(self, series: pd.Series) -> "ARIMAForecaster":
        """Fit ARIMA model using auto_arima.

        Args:
            series: Time series of returns.

        Returns:
            self
        """
        # TODO: implement with pmdarima
        pass

    def predict(self, n_periods: int = 1) -> pd.Series:
        """Forecast n_periods ahead.

        Args:
            n_periods: Number of periods to forecast.

        Returns:
            Forecasted values.
        """
        # TODO: implement
        pass
