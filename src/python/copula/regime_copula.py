"""Regime-conditional copula analysis.

Fits separate vine copulas for calm and crisis periods,
defined by VIX level. Also provides rolling copula analysis.
"""

import numpy as np
import pandas as pd
from .marginals import fit_marginals, get_uniform_data, inverse_marginal_cdf
from .vine_copula import fit_vine_copula, simulate_vine
from .copula_families import tail_dependence_bicop


VIX_CRISIS_THRESHOLD = 25.0  # VIX > 25 = crisis


def load_vix(vix_path: str) -> pd.Series:
    """Load VIX data indexed by date."""
    vix = pd.read_csv(vix_path, parse_dates=['Date'], index_col='Date')
    return vix['VIX']


def classify_regimes(returns: pd.DataFrame,
                     vix: pd.Series,
                     threshold: float = VIX_CRISIS_THRESHOLD) -> tuple:
    """Split returns into calm and crisis periods based on VIX.

    Returns (calm_returns, crisis_returns) DataFrames.
    """
    # Align dates
    common = returns.index.intersection(vix.index)
    returns_aligned = returns.loc[common]
    vix_aligned = vix.loc[common]

    calm_mask = vix_aligned <= threshold
    crisis_mask = vix_aligned > threshold

    return returns_aligned[calm_mask], returns_aligned[crisis_mask]


def fit_regime_copulas(returns: pd.DataFrame,
                       vix: pd.Series,
                       threshold: float = VIX_CRISIS_THRESHOLD) -> dict:
    """Fit separate vine copulas for calm and crisis regimes.

    Returns dict with 'calm' and 'crisis' sub-dicts, each containing:
        'marginals', 'vine', 'n_obs', 'returns'
    """
    calm_ret, crisis_ret = classify_regimes(returns, vix, threshold)
    asset_names = list(returns.columns)

    results = {}
    for label, ret in [('calm', calm_ret), ('crisis', crisis_ret)]:
        if len(ret) < 100:
            results[label] = None
            continue
        marg = fit_marginals(ret)
        u = get_uniform_data(marg, asset_names)
        vine = fit_vine_copula(u, trunc_lvl=5)
        results[label] = {
            'marginals': marg,
            'vine': vine,
            'u_data': u,
            'n_obs': len(ret),
            'returns': ret,
        }

    return results


def regime_tail_dependence(regime_results: dict, d: int) -> pd.DataFrame:
    """Compare tail dependence coefficients between regimes."""
    rows = []
    for regime, res in regime_results.items():
        if res is None:
            continue
        vine = res['vine']
        for edge in range(d - 1):
            pc = vine.get_pair_copula(0, edge)
            lam_L, lam_U = tail_dependence_bicop(pc)
            rows.append({
                'Regime': regime,
                'Tree': 1,
                'Edge': edge + 1,
                'Family': str(pc.family).split('.')[-1],
                'Lambda_L': lam_L,
                'Lambda_U': lam_U,
            })
    return pd.DataFrame(rows)


def rolling_copula_analysis(returns: pd.DataFrame,
                            window: int = 252,
                            step: int = 63) -> pd.DataFrame:
    """Rolling vine copula estimation (quarterly steps).

    At each window, fits a vine copula and records:
    - Portfolio ES_99 (vine)
    - Portfolio ES_99 (Gaussian copula, for comparison)
    - Average tail dependence across Tree 1 edges
    """
    from .copula_families import ALLOWED_FAMILIES
    from scipy.stats import norm as sp_norm
    import pyvinecopulib as pv

    asset_names = list(returns.columns)
    d = len(asset_names)
    dates = returns.index
    results_rows = []

    for start in range(0, len(returns) - window, step):
        end = start + window
        window_ret = returns.iloc[start:end]
        center_date = dates[start + window // 2]

        try:
            marg = fit_marginals(window_ret)
            u = get_uniform_data(marg, asset_names)

            # Vine copula
            vine = fit_vine_copula(u, trunc_lvl=3)
            u_sim = simulate_vine(vine, 100_000, seed=start)
            r_sim = inverse_marginal_cdf(u_sim, marg, asset_names)
            w = np.ones(d) / d
            r_p_vine = r_sim @ w
            es_vine = -np.mean(r_p_vine[r_p_vine <= np.percentile(r_p_vine, 1)])

            # Gaussian copula comparison
            corr = np.corrcoef(u, rowvar=False)
            rng = np.random.default_rng(start)
            z = rng.multivariate_normal(np.zeros(d), corr, size=100_000)
            u_gauss = sp_norm.cdf(z)
            u_gauss = np.clip(u_gauss, 1e-10, 1 - 1e-10)
            r_sim_g = inverse_marginal_cdf(u_gauss, marg, asset_names)
            r_p_gauss = r_sim_g @ w
            es_gauss = -np.mean(r_p_gauss[r_p_gauss <= np.percentile(r_p_gauss, 1)])

            # Average tail dependence (Tree 1)
            avg_lam_L = 0
            n_edges = 0
            for edge in range(d - 1):
                try:
                    pc = vine.get_pair_copula(0, edge)
                    lam_L, _ = tail_dependence_bicop(pc)
                    avg_lam_L += lam_L
                    n_edges += 1
                except Exception:
                    pass
            avg_lam_L = avg_lam_L / n_edges if n_edges > 0 else 0

            results_rows.append({
                'Date': center_date,
                'ES_99_Vine': es_vine,
                'ES_99_Gaussian': es_gauss,
                'Avg_Tail_Dep': avg_lam_L,
            })
        except Exception as e:
            results_rows.append({
                'Date': center_date,
                'ES_99_Vine': np.nan,
                'ES_99_Gaussian': np.nan,
                'Avg_Tail_Dep': np.nan,
            })

    return pd.DataFrame(results_rows)
