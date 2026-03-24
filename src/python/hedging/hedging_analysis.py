"""P&L analysis for dynamic hedging simulations."""

import numpy as np
from dataclasses import dataclass


@dataclass
class HedgingMetrics:
    label: str
    strategy: str
    mean_pnl: float
    std_pnl: float
    skewness: float
    kurtosis: float
    var_99: float
    es_99: float
    prob_loss_2x_premium: float
    option_premium: float


def compute_hedging_metrics(pnl: np.ndarray, premium: float,
                            label: str, strategy: str) -> HedgingMetrics:
    """Compute P&L statistics for a hedging simulation."""
    m = np.mean(pnl)
    s = np.std(pnl, ddof=1)
    centered = (pnl - m) / s if s > 1e-15 else np.zeros_like(pnl)
    skew = np.mean(centered ** 3)
    kurt = np.mean(centered ** 4) - 3.0

    var_99 = -np.percentile(pnl, 1)
    tail = pnl[pnl <= -var_99]
    es_99 = -np.mean(tail) if len(tail) > 0 else var_99

    prob_2x = np.mean(pnl < -2 * premium)

    return HedgingMetrics(
        label=label, strategy=strategy,
        mean_pnl=m, std_pnl=s, skewness=skew, kurtosis=kurt,
        var_99=var_99, es_99=es_99,
        prob_loss_2x_premium=prob_2x, option_premium=premium,
    )
