"""Heston model calibration from implied volatility surfaces.

Calibrates (kappa, theta, xi, rho, v0) by minimising the distance between
model-implied and market-observed option prices or implied volatilities.

TODO:
  - Characteristic function pricing (Lewis / Carr-Madan)
  - Levenberg-Marquardt optimisation
  - Regularisation for ill-posed inverse problems
"""


class HestonCalibrator:
    """Calibrate Heston parameters to market implied volatility surface."""

    def __init__(self) -> None:
        self.params_: dict[str, float] | None = None

    def fit(self, market_ivs: "pd.DataFrame", spot: float, r: float) -> "HestonCalibrator":
        """Fit Heston model to a grid of market implied volatilities.

        Args:
            market_ivs: DataFrame with columns (strike, expiry, iv).
            spot: Current spot price.
            r: Risk-free rate.

        Returns:
            self
        """
        # TODO: implement
        pass
