"""Pre-compute simulation results for dashboard fallback mode.

Run this before deploying where C++ engine is unavailable.
"""

import sys
import os
import pickle
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'build', 'src', 'cpp'))

import quant_engine_py as qe


def simulate(model, params, n_paths=100_000, n_steps=252, T=1.0):
    config = qe.SimConfig()
    config.n_paths = n_paths
    config.n_steps = n_steps
    config.seed = 42
    config.antithetic = True

    if model == 'gbm':
        p = qe.GBMParams()
        p.S0, p.mu, p.sigma, p.T, p.n_steps = 100.0, 0.05, params['sigma'], T, n_steps
        prices = np.array(qe.simulate_gbm_paths(p, config))
        returns = np.log(prices[:, -1] / prices[:, 0])
        return {'prices': prices, 'terminal': prices[:, -1], 'returns': returns, 'variances': None}

    elif model == 'heston':
        p = qe.HestonParams()
        p.S0, p.mu = 100.0, 0.05
        p.v0, p.kappa, p.theta = params['v0'], params['kappa'], params['theta']
        p.sigma_v, p.rho = params['sigma_v'], params['rho']
        p.T, p.n_steps = T, n_steps
        prices, variances = qe.simulate_heston_full(p, config, True)
        prices, variances = np.array(prices), np.array(variances)
        returns = np.log(prices[:, -1] / prices[:, 0])
        return {'prices': prices, 'terminal': prices[:, -1], 'returns': returns, 'variances': variances}

    elif model == 'rough_heston':
        p = qe.RoughHestonParams()
        p.S0, p.mu = 100.0, 0.05
        p.v0, p.kappa, p.theta = params['v0'], params['kappa'], params['theta']
        p.sigma_v, p.rho, p.H = params['sigma_v'], params['rho'], params['H']
        p.T, p.n_steps = T, n_steps
        prices, variances = qe.simulate_rough_heston_full(p, config, 8)
        prices, variances = np.array(prices), np.array(variances)
        returns = np.log(prices[:, -1] / prices[:, 0])
        return {'prices': prices, 'terminal': prices[:, -1], 'returns': returns, 'variances': variances}


def make_key(model, **kwargs):
    key = model
    for k, v in sorted(kwargs.items()):
        key += f"_{k}={v}"
    return key


def summarize(result):
    """Keep only what's needed for dashboard — minimal data."""
    return {
        'returns': result['returns'].astype(np.float32),
        'terminal': result['terminal'].astype(np.float32),
        'prices': result['prices'][:20].astype(np.float32) if result['prices'] is not None else None,
        'variances': None,
    }


def main():
    cache = {}
    print("Pre-computing dashboard cache...", flush=True)

    # GBM grid
    for sigma in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.60]:
        key = make_key('gbm', sigma=sigma)
        print(f"  {key}", flush=True)
        cache[key] = summarize(simulate('gbm', {'sigma': sigma}))

    # Heston grid
    base = dict(v0=0.04, kappa=2.0, theta=0.04)
    for sigma_v in [0.1, 0.3, 0.5, 0.8]:
        for rho in [-0.9, -0.7, -0.5, -0.3]:
            params = {**base, 'sigma_v': sigma_v, 'rho': rho}
            key = make_key('heston', sigma_v=sigma_v, rho=rho)
            print(f"  {key}", flush=True)
            cache[key] = summarize(simulate('heston', params))

    # Rough Heston grid (subset)
    for sigma_v in [0.3, 0.5]:
        for rho in [-0.7, -0.5]:
            for H in [0.05, 0.10, 0.20]:
                params = {**base, 'sigma_v': sigma_v, 'rho': rho, 'H': H}
                key = make_key('rough_heston', sigma_v=sigma_v, rho=rho, H=H)
                print(f"  {key}", flush=True)
                cache[key] = summarize(simulate('rough_heston', params))

    out_dir = os.path.join(PROJECT_ROOT, 'data', 'precomputed')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'dashboard_cache.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nSaved: {out_path} ({size_mb:.1f} MB, {len(cache)} entries)", flush=True)


if __name__ == '__main__':
    main()
