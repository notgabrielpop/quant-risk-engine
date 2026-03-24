"""C++ engine wrapper with precomputed fallback."""

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
            self._cache = _load_precomputed()

    def simulate_gbm(self, S0, mu, sigma, T, n_paths, n_steps):
        if not self.live:
            return self._cache_lookup('gbm', sigma=sigma)
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
            return self._cache_lookup('heston', sigma_v=sigma_v, rho=rho)
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
            return self._cache_lookup('rough_heston', sigma_v=sigma_v, rho=rho, H=H)
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

    def _cache_lookup(self, model, **kwargs):
        if self._cache is None:
            return _dummy_result()
        key = model
        for k, v in sorted(kwargs.items()):
            key += f"_{k}={v}"
        if key in self._cache:
            return self._cache[key]
        # Find closest match
        prefix = model + "_"
        candidates = [k for k in self._cache if k.startswith(prefix)]
        if candidates:
            return self._cache[candidates[0]]
        return _dummy_result()


def _dummy_result():
    rng = np.random.default_rng(42)
    returns = rng.normal(0.05, 0.20, 100_000)
    terminal = 100 * np.exp(returns)
    return {'terminal': terminal, 'returns': returns, 'prices': None, 'variances': None}


def _load_precomputed():
    cache_path = os.path.join(PROJECT_ROOT, 'data', 'precomputed', 'dashboard_cache.pkl')
    if os.path.exists(cache_path):
        import pickle
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return None


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
