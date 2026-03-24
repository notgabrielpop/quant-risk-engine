"""Dynamic delta-hedging simulation engine.

Simulates a trader who sells an ATM European call and delta-hedges using
Black-Scholes delta under different true market dynamics (GBM, Heston,
Rough Heston). Quantifies hidden P&L risk from model misspecification.
"""

import sys
import os
import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'build', 'src', 'cpp'))


# ── Black-Scholes formulas ──

def bs_price(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T < 1e-10:
        return np.maximum(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_delta(S, K, T, r, sigma):
    """Black-Scholes call delta."""
    if T < 1e-10:
        return np.where(S > K, 1.0, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


# ── Hedging simulation ──

def simulate_hedging(model_type, model_params_dict, bs_sigma,
                     n_paths=100_000, S0=100.0, K=100.0, T=0.25,
                     r=0.05, n_steps=63, strategy='constant'):
    """Simulate delta-hedging under specified true dynamics.

    Args:
        model_type: 'gbm', 'heston', or 'rough_heston'
        model_params_dict: dict of model parameters
        bs_sigma: volatility the trader uses for BS delta
        strategy: 'constant' (use bs_sigma) or 'local' (use sqrt(v_t))
        n_paths: number of simulation paths

    Returns dict with:
        total_pnl, option_premium, hedge_pnl, option_payoff, paths, variances
    """
    import quant_engine_py as qe

    config = qe.SimConfig()
    config.n_paths = n_paths
    config.n_steps = n_steps
    config.seed = 42
    config.antithetic = True

    dt = T / n_steps
    variances = None

    if model_type == 'gbm':
        p = qe.GBMParams()
        p.S0 = S0
        p.mu = model_params_dict.get('mu', r)
        p.sigma = model_params_dict.get('sigma', bs_sigma)
        p.T = T
        p.n_steps = n_steps
        paths = np.array(qe.simulate_gbm_paths(p, config))

    elif model_type == 'heston':
        p = qe.HestonParams()
        p.S0 = S0
        p.mu = model_params_dict.get('mu', r)
        p.v0 = model_params_dict['v0']
        p.kappa = model_params_dict['kappa']
        p.theta = model_params_dict['theta']
        p.sigma_v = model_params_dict['sigma_v']
        p.rho = model_params_dict['rho']
        p.T = T
        p.n_steps = n_steps
        paths, variances = qe.simulate_heston_full(p, config, True)
        paths = np.array(paths)
        variances = np.array(variances)

    elif model_type == 'rough_heston':
        p = qe.RoughHestonParams()
        p.S0 = S0
        p.mu = model_params_dict.get('mu', r)
        p.v0 = model_params_dict['v0']
        p.kappa = model_params_dict['kappa']
        p.theta = model_params_dict['theta']
        p.sigma_v = model_params_dict['sigma_v']
        p.rho = model_params_dict['rho']
        p.H = model_params_dict.get('H', 0.1)
        p.T = T
        p.n_steps = n_steps
        paths, variances = qe.simulate_rough_heston_full(p, config, 8)
        paths = np.array(paths)
        variances = np.array(variances)

    # Option premium received at t=0
    option_premium = bs_price(S0, K, T, r, bs_sigma)

    # Delta hedge simulation
    hedge_pnl = np.zeros(n_paths)
    cum_hedge_pnl = np.zeros((n_paths, n_steps))

    for t in range(n_steps):
        S_t = paths[:, t]
        S_t1 = paths[:, t + 1]
        tau = T - t * dt

        if tau > 1e-10:
            if strategy == 'local' and variances is not None:
                # Use instantaneous vol for delta
                sigma_t = np.sqrt(np.maximum(variances[:, t], 1e-8))
                delta = bs_delta(S_t, K, tau, r, sigma_t)
            else:
                # Constant BS vol
                delta = bs_delta(S_t, K, tau, r, bs_sigma)
        else:
            delta = np.where(S_t > K, 1.0, 0.0)

        step_pnl = delta * (S_t1 - S_t)
        hedge_pnl += step_pnl
        cum_hedge_pnl[:, t] = hedge_pnl.copy()

    # Option payoff
    S_T = paths[:, -1]
    option_payoff = np.maximum(S_T - K, 0.0)

    # Total P&L = premium + hedge gains - option payoff
    total_pnl = option_premium + hedge_pnl - option_payoff

    return {
        'total_pnl': total_pnl,
        'option_premium': option_premium,
        'hedge_pnl': hedge_pnl,
        'option_payoff': option_payoff,
        'paths': paths,
        'variances': variances,
        'cum_hedge_pnl': cum_hedge_pnl,
    }
