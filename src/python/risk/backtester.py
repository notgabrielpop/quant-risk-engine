"""Backtesting framework for VaR and ES model validation.

Implements Kupiec POF test, Christoffersen independence test,
and traffic-light framework for VaR model backtesting.

TODO:
  - Rolling-window VaR/ES computation
  - Exception counting and coverage tests
  - P&L attribution
"""


class VaRBacktester:
    """Backtest VaR models using standard regulatory tests."""

    def __init__(self, confidence: float = 0.99, window: int = 252) -> None:
        self.confidence = confidence
        self.window = window

    def run(self, returns: "pd.Series", var_series: "pd.Series") -> dict:
        """Run backtesting suite on a VaR forecast series.

        Args:
            returns: Actual portfolio returns.
            var_series: Predicted VaR values.

        Returns:
            Dict of test results and exception counts.
        """
        # TODO: implement
        pass
