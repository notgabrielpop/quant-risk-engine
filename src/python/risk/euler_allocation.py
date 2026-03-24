"""Euler allocation of portfolio Expected Shortfall.

ES_i = E[w_i * r_i | r_portfolio <= -VaR]

Each asset's contribution to portfolio tail risk, summing to total ES.
"""

import numpy as np
import pandas as pd


def euler_es_allocation(r_sim: np.ndarray, weights: np.ndarray,
                        quantile: float = 0.01) -> dict:
    """Compute Euler ES allocation.

    Args:
        r_sim: (n_scenarios, n_assets) simulated returns
        weights: (n_assets,) portfolio weights
        quantile: tail quantile (0.01 = worst 1%)

    Returns dict with:
        'contributions': per-asset ES contributions (positive = loss)
        'total_es': portfolio ES
        'pct_contributions': fraction of total ES per asset
        'weights': input weights
    """
    r_portfolio = r_sim @ weights
    threshold = np.percentile(r_portfolio, quantile * 100)
    tail_mask = r_portfolio <= threshold
    n_tail = np.sum(tail_mask)

    if n_tail == 0:
        n_assets = r_sim.shape[1]
        return {
            'contributions': np.zeros(n_assets),
            'total_es': 0.0,
            'pct_contributions': np.zeros(n_assets),
            'weights': weights,
        }

    # Per-asset weighted returns in the tail
    contributions = np.empty(r_sim.shape[1])
    for j in range(r_sim.shape[1]):
        contributions[j] = -np.mean(weights[j] * r_sim[tail_mask, j])

    total_es = -np.mean(r_portfolio[tail_mask])

    pct = contributions / total_es if total_es > 0 else np.zeros_like(contributions)

    return {
        'contributions': contributions,
        'total_es': total_es,
        'pct_contributions': pct,
        'weights': weights,
    }


def allocation_table(allocation: dict, asset_names: list) -> pd.DataFrame:
    """Format Euler allocation as a DataFrame."""
    return pd.DataFrame({
        'Asset': asset_names,
        'Weight': allocation['weights'],
        'ES_Contribution': allocation['contributions'],
        'Pct_of_ES': allocation['pct_contributions'],
    })
