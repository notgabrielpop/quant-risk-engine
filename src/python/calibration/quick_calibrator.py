"""Fast parameter estimation from historical returns for backtesting.

Estimates GBM, Heston, and Rough Heston parameters from a window of
daily log-returns. No implied vol surface needed — uses realized
volatility and its dynamics.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class CalibratedGBM:
    S0: float
    mu: float
    sigma: float

@dataclass
class CalibratedHeston:
    S0: float
    mu: float
    v0: float
    kappa: float
    theta: float
    sigma_v: float
    rho: float

@dataclass
class CalibratedRoughHeston(CalibratedHeston):
    H: float = 0.1


def calibrate_gbm(returns: np.ndarray, S0: float = 1.0) -> CalibratedGBM:
    """Estimate GBM parameters from historical returns.

    Uses last 504 trading days (2 years) for sigma to balance
    responsiveness with stability.
    """
    mu = np.mean(returns) * 252
    # Cap lookback to ~2 years so sigma reflects recent regime
    lookback = min(len(returns), 504)
    sigma = np.std(returns[-lookback:], ddof=1) * np.sqrt(252)
    return CalibratedGBM(S0=S0, mu=mu, sigma=max(sigma, 0.01))


def _realized_variance_series(returns: np.ndarray, window: int = 20) -> np.ndarray:
    """Compute rolling realized variance (annualized)."""
    n = len(returns)
    if n < window:
        return np.array([np.var(returns, ddof=1) * 252])
    rv = np.empty(n - window + 1)
    for i in range(n - window + 1):
        rv[i] = np.var(returns[i:i + window], ddof=1) * 252
    return rv


def calibrate_heston(returns: np.ndarray, S0: float = 1.0) -> CalibratedHeston:
    """Estimate Heston parameters from historical returns."""
    mu = np.mean(returns) * 252

    rv = _realized_variance_series(returns, window=20)

    # theta: long-run variance (from full sample)
    theta = np.mean(rv)
    theta = max(theta, 1e-6)

    # v0: current variance — EWMA-like blend of recent and medium-term
    # Use max of 20-day and 60-day to avoid underestimation in calm pockets
    if len(returns) >= 60:
        rv_20 = np.var(returns[-20:], ddof=1) * 252
        rv_60 = np.var(returns[-60:], ddof=1) * 252
        v0 = max(rv_20, 0.7 * rv_60)
    elif len(returns) >= 20:
        v0 = np.var(returns[-20:], ddof=1) * 252
    else:
        v0 = np.var(returns, ddof=1) * 252
    # Floor: never let v0 drop below 50% of long-run variance
    v0 = max(v0, 0.5 * theta)

    # kappa: mean reversion speed from AR(1) on realized variance
    if len(rv) > 2:
        rv_lag = rv[:-1]
        rv_curr = rv[1:]
        cov_mat = np.cov(rv_lag, rv_curr)
        if cov_mat[0, 0] > 1e-12:
            phi = cov_mat[0, 1] / cov_mat[0, 0]
            phi = np.clip(phi, 0.01, 0.999)
            kappa = -252.0 / 20.0 * np.log(phi)  # daily AR(1) -> annualized
        else:
            kappa = 2.0
    else:
        kappa = 2.0
    kappa = np.clip(kappa, 0.5, 10.0)

    # sigma_v: vol of vol
    if len(rv) > 5:
        rv_pos = rv[rv > 1e-8]
        if len(rv_pos) > 5:
            sigma_v = np.std(rv_pos, ddof=1) / (2.0 * np.sqrt(np.mean(rv_pos)))
        else:
            sigma_v = 0.3
    else:
        sigma_v = 0.3
    sigma_v = np.clip(sigma_v, 0.05, 1.5)

    # rho: leverage correlation
    if len(returns) > 40:
        rv_full = _realized_variance_series(returns, window=20)
        # Align: returns[20:] with rv change
        n_rv = len(rv_full)
        if n_rv > 2:
            ret_aligned = returns[19:19 + n_rv - 1]
            rv_change = rv_full[1:] - rv_full[:-1]
            min_len = min(len(ret_aligned), len(rv_change))
            if min_len > 10:
                rho = np.corrcoef(ret_aligned[:min_len], rv_change[:min_len])[0, 1]
                if np.isnan(rho):
                    rho = -0.7
            else:
                rho = -0.7
        else:
            rho = -0.7
    else:
        rho = -0.7
    rho = np.clip(rho, -0.95, 0.0)

    return CalibratedHeston(
        S0=S0, mu=mu, v0=v0, kappa=kappa, theta=theta,
        sigma_v=sigma_v, rho=rho
    )


def calibrate_rough_heston(returns: np.ndarray, S0: float = 1.0) -> CalibratedRoughHeston:
    """Estimate Rough Heston parameters. Uses Heston + Hurst estimation."""
    heston = calibrate_heston(returns, S0)

    # Estimate Hurst exponent from log-variogram of realized volatility
    H = 0.1  # default
    rv = _realized_variance_series(returns, window=20)
    if len(rv) > 50:
        log_rv = np.log(np.maximum(rv, 1e-10))
        lags = [1, 2, 3, 5, 8, 13]
        log_lags = []
        log_variogram = []
        for lag in lags:
            if lag < len(log_rv):
                diffs = log_rv[lag:] - log_rv[:-lag]
                var = np.mean(diffs ** 2)
                if var > 1e-15:
                    log_lags.append(np.log(lag))
                    log_variogram.append(np.log(var))
        if len(log_lags) >= 3:
            # Fit slope: variogram ~ C * lag^(2H) -> log(var) = const + 2H*log(lag)
            coeffs = np.polyfit(log_lags, log_variogram, 1)
            H = coeffs[0] / 2.0
    H = np.clip(H, 0.05, 0.45)

    return CalibratedRoughHeston(
        S0=heston.S0, mu=heston.mu, v0=heston.v0, kappa=heston.kappa,
        theta=heston.theta, sigma_v=heston.sigma_v, rho=heston.rho, H=H
    )


def update_v0(params, returns_recent: np.ndarray):
    """Fast update: only refresh v0 from recent returns (no full recalibration)."""
    v0 = np.var(returns_recent[-20:], ddof=1) * 252 if len(returns_recent) >= 20 else np.var(returns_recent, ddof=1) * 252
    params.v0 = max(v0, 1e-6)
    return params
