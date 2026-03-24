"""C++ engine wrapper with pure Python fallback."""

import os
import sys
import numpy as np
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'build', 'src', 'cpp'))


@st.cache_resource
def get_engine():
    """Get engine wrapper singleton."""
    return EngineWrapper()


class EngineWrapper:
    def __init__(self):
        try:
            import quant_engine_py as qe
            self.qe = qe
            self.live = True
        except ImportError:
            self.qe = None
            self.live = False

    def simulate_gbm(self, S0, mu, sigma, T, n_paths, n_steps):
        if not self.live:
            from visualization.components.py_simulator import simulate_gbm
            return simulate_gbm(S0, mu, sigma, T, min(n_paths, 50_000), n_steps)
        config = self.qe.SimConfig()
        config.n_paths = n_paths
        config.n_steps = n_steps
        config.seed = 42
        config.antithetic = True
        p = self.qe.GBMParams()
        p.S0, p.mu, p.sigma, p.T, p.n_steps = S0, mu, sigma, T, n_steps
        prices = np.array(self.qe.simulate_gbm_paths(p, config))
        returns = np.log(prices[:, -1] / prices[:, 0])
        return {'prices': prices, 'terminal': prices[:, -1], 'returns': returns}

    def simulate_heston(self, S0, mu, v0, kappa, theta, sigma_v, rho, T, n_paths, n_steps):
        if not self.live:
            from visualization.components.py_simulator import simulate_heston
            return simulate_heston(S0, mu, v0, kappa, theta, sigma_v, rho, T, min(n_paths, 50_000), n_steps)
        config = self.qe.SimConfig()
        config.n_paths = n_paths
        config.n_steps = n_steps
        config.seed = 42
        config.antithetic = True
        p = self.qe.HestonParams()
        p.S0, p.mu, p.v0 = S0, mu, v0
        p.kappa, p.theta, p.sigma_v, p.rho = kappa, theta, sigma_v, rho
        p.T, p.n_steps = T, n_steps
        prices, variances = self.qe.simulate_heston_full(p, config, True)
        prices = np.array(prices)
        variances = np.array(variances)
        returns = np.log(prices[:, -1] / prices[:, 0])
        return {'prices': prices, 'variances': variances,
                'terminal': prices[:, -1], 'returns': returns}

    def simulate_rough_heston(self, S0, mu, v0, kappa, theta, sigma_v, rho, H, T, n_paths, n_steps):
        if not self.live:
            from visualization.components.py_simulator import simulate_rough_heston
            return simulate_rough_heston(S0, mu, v0, kappa, theta, sigma_v, rho, H, T, min(n_paths, 50_000), n_steps)
        config = self.qe.SimConfig()
        config.n_paths = n_paths
        config.n_steps = n_steps
        config.seed = 42
        config.antithetic = True
        p = self.qe.RoughHestonParams()
        p.S0, p.mu, p.v0 = S0, mu, v0
        p.kappa, p.theta, p.sigma_v, p.rho = kappa, theta, sigma_v, rho
        p.H, p.T, p.n_steps = H, T, n_steps
        prices, variances = self.qe.simulate_rough_heston_full(p, config, 8)
        prices = np.array(prices)
        variances = np.array(variances)
        returns = np.log(prices[:, -1] / prices[:, 0])
        return {'prices': prices, 'variances': variances,
                'terminal': prices[:, -1], 'returns': returns}


def compute_risk_metrics(returns, confidence=0.99):
    """Compute VaR and ES from return array."""
    sorted_r = np.sort(returns)
    n = len(sorted_r)
    idx = int(n * (1 - confidence))
    var = -sorted_r[idx]
    tail = sorted_r[:idx]
    es = -np.mean(tail) if len(tail) > 0 else var
    return {
        'var': var, 'es': es,
        'mean': np.mean(returns), 'std': np.std(returns, ddof=1),
        'skewness': float(_skew(returns)), 'kurtosis': float(_kurt(returns)),
    }


def _skew(x):
    m, s = np.mean(x), np.std(x, ddof=1)
    return np.mean(((x - m) / s) ** 3) if s > 1e-15 else 0.0


def _kurt(x):
    m, s = np.mean(x), np.std(x, ddof=1)
    return np.mean(((x - m) / s) ** 4) - 3.0 if s > 1e-15 else 0.0
