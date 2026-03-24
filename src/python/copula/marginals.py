"""Marginal distribution fitting for portfolio assets.

Fits Student-t and Normal distributions to each asset's log-returns,
validates via KS-test, and transforms to uniform margins for copula estimation.
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass


@dataclass
class MarginalFit:
    asset: str
    dist_name: str  # 't' or 'norm'
    params: tuple   # (df, loc, scale) for t; (loc, scale) for norm
    aic: float
    bic: float
    ks_stat: float
    ks_pvalue: float


def fit_marginals(returns: pd.DataFrame) -> dict:
    """Fit Student-t and Normal to each asset, return best fits.

    Returns dict: asset_name -> {
        't_fit': MarginalFit, 'norm_fit': MarginalFit,
        'best': MarginalFit, 'uniform': np.ndarray
    }
    """
    results = {}
    for col in returns.columns:
        r = returns[col].dropna().values

        # Fit Student-t
        t_params = stats.t.fit(r)
        t_ll = np.sum(stats.t.logpdf(r, *t_params))
        t_aic = -2 * t_ll + 2 * 3  # 3 params: df, loc, scale
        t_bic = -2 * t_ll + 3 * np.log(len(r))
        t_ks = stats.kstest(r, 't', args=t_params)

        t_fit = MarginalFit(
            asset=col, dist_name='t', params=t_params,
            aic=t_aic, bic=t_bic,
            ks_stat=t_ks.statistic, ks_pvalue=t_ks.pvalue
        )

        # Fit Normal
        n_params = stats.norm.fit(r)
        n_ll = np.sum(stats.norm.logpdf(r, *n_params))
        n_aic = -2 * n_ll + 2 * 2  # 2 params: loc, scale
        n_bic = -2 * n_ll + 2 * np.log(len(r))
        n_ks = stats.kstest(r, 'norm', args=n_params)

        norm_fit = MarginalFit(
            asset=col, dist_name='norm', params=n_params,
            aic=n_aic, bic=n_bic,
            ks_stat=n_ks.statistic, ks_pvalue=n_ks.pvalue
        )

        # Select best by AIC
        best = t_fit if t_aic < n_aic else norm_fit

        # Transform to uniform using best fit CDF
        if best.dist_name == 't':
            u = stats.t.cdf(r, *best.params)
        else:
            u = stats.norm.cdf(r, *best.params)

        # Clip to avoid exact 0 or 1
        u = np.clip(u, 1e-10, 1 - 1e-10)

        results[col] = {
            't_fit': t_fit,
            'norm_fit': norm_fit,
            'best': best,
            'uniform': u
        }

    return results


def get_uniform_data(marginal_results: dict, asset_names: list) -> np.ndarray:
    """Stack uniform transforms into (n_obs, n_assets) array."""
    arrays = [marginal_results[name]['uniform'] for name in asset_names]
    return np.column_stack(arrays)


def inverse_marginal_cdf(u: np.ndarray, marginal_results: dict,
                         asset_names: list) -> np.ndarray:
    """Transform uniform samples back to returns using fitted inverse CDFs."""
    n, d = u.shape
    returns = np.empty_like(u)
    for j, name in enumerate(asset_names):
        fit = marginal_results[name]['best']
        if fit.dist_name == 't':
            returns[:, j] = stats.t.ppf(u[:, j], *fit.params)
        else:
            returns[:, j] = stats.norm.ppf(u[:, j], *fit.params)
    return returns


def marginal_fits_table(marginal_results: dict) -> pd.DataFrame:
    """Create summary table of marginal fits."""
    rows = []
    for asset, res in marginal_results.items():
        t = res['t_fit']
        n = res['norm_fit']
        rows.append({
            'Asset': asset,
            't_df': t.params[0],
            't_loc': t.params[1],
            't_scale': t.params[2],
            't_AIC': t.aic,
            't_KS_stat': t.ks_stat,
            't_KS_pvalue': t.ks_pvalue,
            'norm_loc': n.params[0],
            'norm_scale': n.params[1],
            'norm_AIC': n.aic,
            'norm_KS_stat': n.ks_stat,
            'norm_KS_pvalue': n.ks_pvalue,
            'Best': res['best'].dist_name
        })
    return pd.DataFrame(rows)
