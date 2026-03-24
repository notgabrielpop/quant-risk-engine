"""Pure Python simulation fallback for when C++ engine is unavailable.

Implements GBM, Heston, and Rough Heston using numpy.
Slower than C++ but fully functional for interactive dashboard use.
"""

import numpy as np


def simulate_gbm(S0, mu, sigma, T, n_paths, n_steps, seed=42):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    Z = rng.standard_normal((n_paths, n_steps))
    log_S = np.zeros((n_paths, n_steps + 1))
    log_S[:, 0] = np.log(S0)
    drift = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)
    for t in range(n_steps):
        log_S[:, t + 1] = log_S[:, t] + drift + diffusion * Z[:, t]
    prices = np.exp(log_S)
    returns = log_S[:, -1] - log_S[:, 0]
    return {'prices': prices, 'terminal': prices[:, -1], 'returns': returns}


def simulate_heston(S0, mu, v0, kappa, theta, sigma_v, rho, T, n_paths, n_steps, seed=42):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    Z1 = rng.standard_normal((n_paths, n_steps))
    Z2 = rng.standard_normal((n_paths, n_steps))
    W_S = Z1
    W_v = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2

    log_S = np.zeros((n_paths, n_steps + 1))
    v = np.zeros((n_paths, n_steps + 1))
    log_S[:, 0] = np.log(S0)
    v[:, 0] = v0

    for t in range(n_steps):
        v_pos = np.maximum(v[:, t], 0.0)
        sqrt_v = np.sqrt(v_pos)
        log_S[:, t + 1] = log_S[:, t] + (mu - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * W_S[:, t]
        v[:, t + 1] = v[:, t] + kappa * (theta - v_pos) * dt + sigma_v * sqrt_v * sqrt_dt * W_v[:, t]

    prices = np.exp(log_S)
    returns = log_S[:, -1] - log_S[:, 0]
    return {
        'prices': prices, 'terminal': prices[:, -1],
        'returns': returns, 'variances': v,
    }


def simulate_rough_heston(S0, mu, v0, kappa, theta, sigma_v, rho, H, T, n_paths, n_steps, seed=42):
    """Rough Heston via hybrid scheme: Volterra kernel for variance."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    alpha = H - 0.5  # fractional exponent

    Z1 = rng.standard_normal((n_paths, n_steps))
    Z2 = rng.standard_normal((n_paths, n_steps))
    W_S = Z1
    W_v = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2

    # Precompute kernel weights: (t_j)^alpha / Gamma(alpha+1)
    from math import gamma as gamma_func
    max_memory = min(n_steps, 50)  # truncate kernel for speed
    kernel = np.zeros(max_memory)
    for j in range(max_memory):
        kernel[j] = ((j + 1) * dt) ** alpha / gamma_func(alpha + 1)
    kernel /= kernel.sum() if kernel.sum() > 0 else 1.0

    log_S = np.zeros((n_paths, n_steps + 1))
    v = np.zeros((n_paths, n_steps + 1))
    log_S[:, 0] = np.log(S0)
    v[:, 0] = v0

    # Store variance increments for convolution
    dv_increments = np.zeros((n_paths, n_steps))

    for t in range(n_steps):
        v_pos = np.maximum(v[:, t], 0.0)
        sqrt_v = np.sqrt(v_pos)
        log_S[:, t + 1] = log_S[:, t] + (mu - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * W_S[:, t]

        # Variance increment (before kernel)
        dv_increments[:, t] = kappa * (theta - v_pos) * dt + sigma_v * sqrt_v * sqrt_dt * W_v[:, t]

        # Apply fractional kernel via vectorized dot product
        lookback = min(t + 1, max_memory)
        start = t + 1 - lookback
        v[:, t + 1] = v0 + dv_increments[:, start:t + 1] @ kernel[:lookback][::-1]

    prices = np.exp(log_S)
    returns = log_S[:, -1] - log_S[:, 0]
    return {
        'prices': prices, 'terminal': prices[:, -1],
        'returns': returns, 'variances': v,
    }
