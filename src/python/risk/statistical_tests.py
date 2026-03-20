"""Statistical tests for VaR and ES backtesting.

Implements Kupiec POF, Christoffersen independence and conditional coverage,
McNeil-Frey ES test, and Basel traffic light classification.
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass


@dataclass
class KupiecResult:
    n_exceedances: int
    n_total: int
    expected_rate: float
    actual_rate: float
    lr_statistic: float
    p_value: float
    reject: bool  # at 5% level


@dataclass
class ChristoffersenResult:
    lr_ind: float
    p_ind: float
    lr_cc: float
    p_cc: float
    pi01: float
    pi11: float
    clustering_ratio: float
    reject_independence: bool
    reject_cc: bool


@dataclass
class McNeilFreyResult:
    mean_residual: float
    t_statistic: float
    p_value: float
    n_exceedances: int
    reject: bool


@dataclass
class BaselResult:
    zone: str  # 'green', 'yellow', 'red'
    exceedances: int
    n_days: int


def kupiec_test(n_exceedances: int, n_total: int,
                expected_rate: float) -> KupiecResult:
    """Kupiec unconditional coverage test.

    H0: true exceedance rate = expected_rate
    LR_uc ~ chi2(1) under H0.
    """
    x = n_exceedances
    T = n_total
    p = expected_rate

    if x == 0:
        # No exceedances — model is conservative
        lr = -2.0 * (T * np.log(1 - p))
        pval = stats.chi2.sf(lr, 1)
        return KupiecResult(x, T, p, 0.0, lr, pval, pval < 0.05)

    if x == T:
        lr = -2.0 * (T * np.log(p))
        pval = stats.chi2.sf(lr, 1)
        return KupiecResult(x, T, p, 1.0, lr, pval, pval < 0.05)

    actual_rate = x / T
    # LR = -2 * [log L(p) - log L(p_hat)]
    log_l0 = (T - x) * np.log(1 - p) + x * np.log(p)
    log_l1 = (T - x) * np.log(1 - actual_rate) + x * np.log(actual_rate)
    lr = -2.0 * (log_l0 - log_l1)
    lr = max(lr, 0.0)
    pval = stats.chi2.sf(lr, 1)

    return KupiecResult(x, T, p, actual_rate, lr, pval, pval < 0.05)


def christoffersen_test(exceedance_series: np.ndarray,
                        expected_rate: float) -> ChristoffersenResult:
    """Christoffersen independence + conditional coverage test.

    Tests both correct coverage AND independence of exceedances.
    LR_cc = LR_kupiec + LR_independence ~ chi2(2) under H0.
    """
    exc = exceedance_series.astype(int)
    T = len(exc)

    # Build transition counts
    n00 = n01 = n10 = n11 = 0
    for i in range(1, T):
        prev, curr = exc[i - 1], exc[i]
        if prev == 0 and curr == 0: n00 += 1
        elif prev == 0 and curr == 1: n01 += 1
        elif prev == 1 and curr == 0: n10 += 1
        elif prev == 1 and curr == 1: n11 += 1

    # Transition probabilities
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(T - 1, 1)  # unconditional

    # Clustering ratio
    clustering = pi11 / pi01 if pi01 > 0 else 0.0

    # Independence LR
    if pi <= 0 or pi >= 1:
        lr_ind = 0.0
    else:
        log_l0 = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)

        log_l1 = 0.0
        if n00 + n01 > 0:
            pi01_safe = max(pi01, 1e-15)
            log_l1 += n00 * np.log(max(1 - pi01_safe, 1e-15)) + n01 * np.log(pi01_safe)
        if n10 + n11 > 0:
            pi11_safe = max(pi11, 1e-15)
            log_l1 += n10 * np.log(max(1 - pi11_safe, 1e-15)) + n11 * np.log(pi11_safe)

        lr_ind = -2.0 * (log_l0 - log_l1)
        lr_ind = max(lr_ind, 0.0)

    p_ind = stats.chi2.sf(lr_ind, 1)

    # Kupiec component
    n_exc = int(np.sum(exc))
    kupiec = kupiec_test(n_exc, T, expected_rate)

    # Combined CC
    lr_cc = kupiec.lr_statistic + lr_ind
    p_cc = stats.chi2.sf(lr_cc, 2)

    return ChristoffersenResult(
        lr_ind=lr_ind, p_ind=p_ind,
        lr_cc=lr_cc, p_cc=p_cc,
        pi01=pi01, pi11=pi11,
        clustering_ratio=clustering,
        reject_independence=p_ind < 0.05,
        reject_cc=p_cc < 0.05,
    )


def mcneil_frey_test(actual_returns: np.ndarray,
                     var_forecasts: np.ndarray,
                     es_forecasts: np.ndarray) -> McNeilFreyResult:
    """McNeil-Frey ES backtest on exceedance days.

    On days where actual loss > VaR:
        residual_t = (-actual_return_t) - ES_t
    H0: E[residual] = 0 (ES is correctly calibrated).
    """
    # Exceedance days: actual return < -VaR (loss exceeds VaR)
    exc_mask = actual_returns < -var_forecasts
    n_exc = np.sum(exc_mask)

    if n_exc < 3:
        return McNeilFreyResult(
            mean_residual=np.nan, t_statistic=np.nan,
            p_value=1.0, n_exceedances=int(n_exc), reject=False
        )

    # Residuals: actual loss minus predicted ES
    actual_losses = -actual_returns[exc_mask]
    predicted_es = es_forecasts[exc_mask]
    residuals = actual_losses - predicted_es

    mean_r = np.mean(residuals)
    se_r = np.std(residuals, ddof=1) / np.sqrt(n_exc)
    t_stat = mean_r / se_r if se_r > 1e-15 else 0.0
    p_value = 2 * stats.t.sf(abs(t_stat), n_exc - 1)

    return McNeilFreyResult(
        mean_residual=mean_r, t_statistic=t_stat,
        p_value=p_value, n_exceedances=int(n_exc),
        reject=p_value < 0.05,
    )


def basel_traffic_light(n_exceedances: int, n_days: int = 250) -> BaselResult:
    """Basel Committee traffic light for VaR at 99% over 250 days.

    Green: 0-4, Yellow: 5-9, Red: 10+
    """
    if n_exceedances <= 4:
        zone = 'green'
    elif n_exceedances <= 9:
        zone = 'yellow'
    else:
        zone = 'red'
    return BaselResult(zone=zone, exceedances=n_exceedances, n_days=n_days)
