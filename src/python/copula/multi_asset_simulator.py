"""Multi-asset simulation pipeline using vine copulas.

Three-stage pipeline:
1. Fit marginals + vine copula on historical data
2. Simulate correlated uniforms from the vine
3. Transform back to returns via inverse marginal CDFs
"""

import numpy as np
import pandas as pd
from .marginals import fit_marginals, get_uniform_data, inverse_marginal_cdf
from .vine_copula import fit_vine_copula, simulate_vine


class MultiAssetSimulator:
    """Simulate correlated multi-asset returns via vine copulas."""

    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        self.asset_names = list(returns.columns)
        self.n_assets = len(self.asset_names)
        self.marginal_results = None
        self.vine_copula = None
        self.u_data = None

    def fit(self, trunc_lvl: int = 5):
        """Stage 1: fit marginals and vine copula."""
        self.marginal_results = fit_marginals(self.returns)
        self.u_data = get_uniform_data(self.marginal_results, self.asset_names)
        self.vine_copula = fit_vine_copula(self.u_data, trunc_lvl=trunc_lvl)
        return self

    def simulate(self, n_scenarios: int = 500_000,
                 seed: int = 42) -> np.ndarray:
        """Stages 2-3: simulate correlated returns.

        Returns (n_scenarios, n_assets) array of simulated log-returns.
        """
        u_sim = simulate_vine(self.vine_copula, n_scenarios, seed=seed)
        r_sim = inverse_marginal_cdf(u_sim, self.marginal_results,
                                     self.asset_names)
        return r_sim

    def simulate_gaussian(self, n_scenarios: int = 500_000,
                          seed: int = 42) -> np.ndarray:
        """Simulate using Gaussian copula (for comparison)."""
        rng = np.random.default_rng(seed)
        corr = np.corrcoef(self.u_data, rowvar=False)
        from scipy.stats import norm
        z = rng.multivariate_normal(np.zeros(self.n_assets), corr,
                                    size=n_scenarios)
        u_gauss = norm.cdf(z)
        u_gauss = np.clip(u_gauss, 1e-10, 1 - 1e-10)
        return inverse_marginal_cdf(u_gauss, self.marginal_results,
                                    self.asset_names)

    def simulate_independent(self, n_scenarios: int = 500_000,
                             seed: int = 42) -> np.ndarray:
        """Simulate with independent margins (no copula)."""
        rng = np.random.default_rng(seed)
        u_indep = rng.uniform(size=(n_scenarios, self.n_assets))
        u_indep = np.clip(u_indep, 1e-10, 1 - 1e-10)
        return inverse_marginal_cdf(u_indep, self.marginal_results,
                                    self.asset_names)

    def simulate_comonotonic(self, n_scenarios: int = 500_000,
                             seed: int = 42) -> np.ndarray:
        """Simulate with perfect dependence (comonotonic copula)."""
        rng = np.random.default_rng(seed)
        u_common = rng.uniform(size=n_scenarios)
        u_como = np.tile(u_common[:, None], (1, self.n_assets))
        u_como = np.clip(u_como, 1e-10, 1 - 1e-10)
        return inverse_marginal_cdf(u_como, self.marginal_results,
                                    self.asset_names)
