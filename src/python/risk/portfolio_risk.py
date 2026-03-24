"""Portfolio-level VaR and ES computation from simulated multi-asset returns."""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class PortfolioRiskMetrics:
    var_95: float
    var_99: float
    es_95: float
    es_99: float
    mean_return: float
    std_return: float
    prob_loss_5pct: float
    prob_loss_10pct: float
    prob_loss_20pct: float
    max_loss: float


def portfolio_returns(r_sim: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute portfolio return per scenario: r_p = sum(w_j * r_j)."""
    return r_sim @ weights


def compute_risk_metrics(r_portfolio: np.ndarray) -> PortfolioRiskMetrics:
    """Compute VaR, ES, and related risk metrics from portfolio returns."""
    var_95 = -np.percentile(r_portfolio, 5)
    var_99 = -np.percentile(r_portfolio, 1)

    tail_95 = r_portfolio[r_portfolio <= -var_95]
    tail_99 = r_portfolio[r_portfolio <= -var_99]
    es_95 = -np.mean(tail_95) if len(tail_95) > 0 else var_95
    es_99 = -np.mean(tail_99) if len(tail_99) > 0 else var_99

    return PortfolioRiskMetrics(
        var_95=var_95,
        var_99=var_99,
        es_95=es_95,
        es_99=es_99,
        mean_return=np.mean(r_portfolio),
        std_return=np.std(r_portfolio),
        prob_loss_5pct=np.mean(r_portfolio < -0.05),
        prob_loss_10pct=np.mean(r_portfolio < -0.10),
        prob_loss_20pct=np.mean(r_portfolio < -0.20),
        max_loss=-np.min(r_portfolio),
    )


def compute_es_at_quantiles(r_portfolio: np.ndarray,
                            quantiles: np.ndarray) -> np.ndarray:
    """Compute ES at multiple quantile levels.

    quantiles: array of levels (e.g., [0.5, 0.1, 0.05, 0.01, 0.001])
    Returns array of ES values (positive = loss).
    """
    es_values = np.empty(len(quantiles))
    for i, q in enumerate(quantiles):
        threshold = np.percentile(r_portfolio, q * 100)
        tail = r_portfolio[r_portfolio <= threshold]
        es_values[i] = -np.mean(tail) if len(tail) > 0 else -threshold
    return es_values


def individual_es(r_sim: np.ndarray, weights: np.ndarray,
                  quantile: float = 0.01) -> np.ndarray:
    """Compute standalone ES for each asset (weighted)."""
    n_assets = r_sim.shape[1]
    es_ind = np.empty(n_assets)
    for j in range(n_assets):
        w_r = weights[j] * r_sim[:, j]
        threshold = np.percentile(w_r, quantile * 100)
        tail = w_r[w_r <= threshold]
        es_ind[j] = -np.mean(tail) if len(tail) > 0 else -threshold
    return es_ind


def risk_comparison_table(results: dict) -> pd.DataFrame:
    """Create comparison table from dict of {label: PortfolioRiskMetrics}."""
    rows = []
    for label, m in results.items():
        rows.append({
            'Model': label,
            'VaR_95': m.var_95,
            'VaR_99': m.var_99,
            'ES_95': m.es_95,
            'ES_99': m.es_99,
            'Mean': m.mean_return,
            'Std': m.std_return,
            'P(loss>5%)': m.prob_loss_5pct,
            'P(loss>10%)': m.prob_loss_10pct,
        })
    return pd.DataFrame(rows)
