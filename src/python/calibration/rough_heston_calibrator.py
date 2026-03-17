"""Rough Heston model calibration.

Extends HestonCalibrator with the additional Hurst exponent H.
Uses the Adams scheme or hybrid discretisation for the fractional kernel.

TODO:
  - Fractional Riccati ODE solver for characteristic function
  - Joint calibration of (kappa, theta, xi, rho, v0, H)
"""


class RoughHestonCalibrator:
    """Calibrate Rough Heston parameters including Hurst exponent H."""

    def __init__(self) -> None:
        self.params_: dict[str, float] | None = None

    def fit(self, market_ivs: "pd.DataFrame", spot: float, r: float) -> "RoughHestonCalibrator":
        """Fit Rough Heston model.

        Args:
            market_ivs: DataFrame with columns (strike, expiry, iv).
            spot: Current spot price.
            r: Risk-free rate.

        Returns:
            self
        """
        # TODO: implement
        pass
