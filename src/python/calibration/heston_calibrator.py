"""Heston model calibration via differential evolution on moment-matching.

Finds Heston parameters (kappa, theta, sigma_v, rho, v0) that best reproduce
the empirical return distribution by minimizing the distance between model-implied
and empirical moments (variance, skewness, kurtosis, ACF of |returns|, leverage corr).
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'build', 'src', 'cpp'))


@dataclass
class CalibrationResult:
    asset: str
    model: str
    kappa: float
    theta: float
    sigma_v: float
    rho: float
    v0: float
    H: float  # only for rough heston
    loss: float
    n_evals: int
    empirical_moments: dict
    model_moments: dict


def _load_engine():
    import quant_engine_py as qe
    return qe


def compute_empirical_targets(returns: np.ndarray) -> dict:
    """Compute target moments from historical returns."""
    variance = np.var(returns, ddof=1) * 252
    skewness = _skewness(returns)
    kurtosis = _kurtosis(returns)
    acf_abs = [_acf_1d(np.abs(returns), lag) for lag in [1, 5, 10, 20]]
    acf_abs_long = [_acf_1d(np.abs(returns), lag) for lag in [30, 50]]

    # Leverage correlation
    n = len(returns)
    if n > 40:
        rv = np.array([np.var(returns[max(0, i-20):i], ddof=1) * 252
                       for i in range(20, n)])
        ret = returns[20:n]
        if len(ret) > 1 and len(rv) > 1:
            min_len = min(len(ret) - 1, len(rv) - 1)
            leverage = np.corrcoef(ret[:min_len], rv[1:min_len+1] - rv[:min_len])[0, 1]
            if np.isnan(leverage):
                leverage = -0.5
        else:
            leverage = -0.5
    else:
        leverage = -0.5

    return {
        'variance': variance,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'acf_abs': acf_abs,
        'acf_abs_long': acf_abs_long,
        'leverage': leverage,
    }


def _skewness(x):
    m = np.mean(x)
    s = np.std(x, ddof=1)
    return np.mean(((x - m) / s) ** 3) if s > 1e-15 else 0.0


def _kurtosis(x):
    m = np.mean(x)
    s = np.std(x, ddof=1)
    return np.mean(((x - m) / s) ** 4) - 3.0 if s > 1e-15 else 0.0


def _acf_1d(x, lag):
    """ACF for a single 1D series."""
    if lag >= len(x):
        return 0.0
    x_centered = x - np.mean(x)
    c0 = np.mean(x_centered ** 2)
    if c0 < 1e-15:
        return 0.0
    return np.mean(x_centered[lag:] * x_centered[:-lag]) / c0


def _acf_paths(abs_returns, lag):
    """Vectorized ACF averaged over paths. abs_returns: (n_paths, n_steps)."""
    if lag >= abs_returns.shape[1]:
        return 0.0
    x = abs_returns - abs_returns.mean(axis=1, keepdims=True)
    c0 = (x ** 2).mean(axis=1)
    c_lag = (x[:, lag:] * x[:, :-lag]).mean(axis=1)
    valid = c0 > 1e-15
    if valid.sum() == 0:
        return 0.0
    return float((c_lag[valid] / c0[valid]).mean())


def _simulate_paths(qe, params_dict, model, n_paths, n_steps=252):
    """Simulate price paths using C++ engine. Returns (n_paths, n_steps+1) array."""
    config = qe.SimConfig()
    config.n_paths = n_paths
    config.n_steps = n_steps
    config.seed = 42
    config.antithetic = True

    if model == 'heston':
        p = qe.HestonParams()
        p.S0 = 100.0
        p.mu = params_dict.get('mu', 0.08)
        p.v0 = params_dict['v0']
        p.kappa = params_dict['kappa']
        p.theta = params_dict['theta']
        p.sigma_v = params_dict['sigma_v']
        p.rho = params_dict['rho']
        p.T = 1.0
        p.n_steps = n_steps
        prices, _ = qe.simulate_heston_full(p, config, True)
    elif model == 'rough_heston':
        p = qe.RoughHestonParams()
        p.S0 = 100.0
        p.mu = params_dict.get('mu', 0.08)
        p.v0 = params_dict['v0']
        p.kappa = params_dict['kappa']
        p.theta = params_dict['theta']
        p.sigma_v = params_dict['sigma_v']
        p.rho = params_dict['rho']
        p.H = params_dict.get('H', 0.1)
        p.T = 1.0
        p.n_steps = n_steps
        prices, _ = qe.simulate_rough_heston_full(p, config, 8)
    else:
        raise ValueError(f"Unknown model: {model}")

    return np.array(prices)


def _compute_moments(log_returns):
    """Compute moments from log returns array (n_paths, n_steps)."""
    flat = log_returns.flatten()
    var_ann = np.var(flat, ddof=1) * 252
    skew = _skewness(flat)
    kurt = _kurtosis(flat)

    # Per-path ACF (vectorized)
    abs_r = np.abs(log_returns)
    acf_short = [_acf_paths(abs_r, lag) for lag in [1, 5, 10, 20]]
    acf_long = [_acf_paths(abs_r, lag) for lag in [30, 50]]

    return {
        'variance': var_ann,
        'skewness': skew,
        'kurtosis': kurt,
        'acf_abs': acf_short,
        'acf_abs_long': acf_long,
    }


def _compute_loss(params_vec, targets, qe, model, mu, n_paths, tracker):
    """Loss function for differential evolution."""
    if model == 'heston':
        kappa, theta, sigma_v, rho, v0 = params_vec
        params_dict = dict(kappa=kappa, theta=theta, sigma_v=sigma_v,
                           rho=rho, v0=v0, mu=mu)
    else:  # rough_heston
        kappa, theta, sigma_v, rho, v0, H = params_vec
        params_dict = dict(kappa=kappa, theta=theta, sigma_v=sigma_v,
                           rho=rho, v0=v0, mu=mu, H=H)

    try:
        prices = _simulate_paths(qe, params_dict, model, n_paths, 252)
        log_returns = np.log(prices[:, 1:] / prices[:, :-1])
        moments = _compute_moments(log_returns)
    except Exception:
        return 1e6

    # Loss components
    loss = 0.0
    loss += 1.0 * ((targets['variance'] - moments['variance']) / max(targets['variance'], 1e-6)) ** 2
    loss += 2.0 * (targets['skewness'] - moments['skewness']) ** 2
    loss += 2.0 * ((targets['kurtosis'] - moments['kurtosis']) / max(abs(targets['kurtosis']), 1.0)) ** 2

    for emp, sim in zip(targets['acf_abs'], moments['acf_abs']):
        loss += 0.25 * (emp - sim) ** 2

    loss += 1.0 * (targets['leverage'] - rho) ** 2

    if model == 'rough_heston':
        for emp, sim in zip(targets['acf_abs_long'], moments['acf_abs_long']):
            loss += 0.5 * (emp - sim) ** 2

    # Regularization penalties
    loss += 10.0 * max(0, sigma_v - 0.8) ** 2      # penalize extreme vol-of-vol
    loss += 5.0 * (rho + 0.6) ** 2 * 0.1            # soft prior: rho near -0.6
    loss += 2.0 * max(0, kappa - 8.0) ** 2           # penalize extreme mean reversion

    if model == 'rough_heston':
        loss += 5.0 * (H - 0.1) ** 2 * 0.5          # soft prior: H near 0.1

    # Track best (no extra simulation needed!)
    tracker['n_evals'] += 1
    if loss < tracker['best_loss']:
        tracker['best_loss'] = loss

    return loss


class HestonCalibrator:
    """Calibrate Heston/Rough Heston via differential evolution on moments."""

    def __init__(self, n_paths: int = 30_000, maxiter: int = 20,
                 popsize: int = 10):
        self.n_paths = n_paths
        self.maxiter = maxiter
        self.popsize = popsize
        self.qe = _load_engine()
        self.convergence_history = []

    def calibrate(self, returns: np.ndarray, asset: str,
                  model: str = 'heston',
                  mu: float = 0.08) -> CalibrationResult:
        """Run DE calibration."""
        targets = compute_empirical_targets(returns)
        print(f"  Empirical targets: var={targets['variance']:.4f}, "
              f"skew={targets['skewness']:.3f}, kurt={targets['kurtosis']:.3f}, "
              f"leverage={targets['leverage']:.3f}", flush=True)

        # Bounds
        if model == 'heston':
            bounds = [
                (0.1, 10.0),    # kappa
                (0.001, 0.25),  # theta
                (0.05, 1.0),    # sigma_v
                (-0.95, -0.15), # rho (must be negative for equities)
                (0.005, 0.20),  # v0
            ]
        else:  # rough_heston
            bounds = [
                (0.1, 10.0),    # kappa
                (0.001, 0.25),  # theta
                (0.05, 1.0),    # sigma_v
                (-0.95, -0.15), # rho (must be negative for equities)
                (0.005, 0.20),  # v0
                (0.05, 0.25),   # H (literature ~0.1)
            ]

        self.convergence_history = []
        tracker = {'n_evals': 0, 'best_loss': 1e6}
        gen_count = [0]
        t0 = time.time()

        def callback(xk, convergence):
            gen_count[0] += 1
            self.convergence_history.append(tracker['best_loss'])
            elapsed = time.time() - t0
            print(f"    Gen {gen_count[0]:3d}: best_loss={tracker['best_loss']:.6f}, "
                  f"evals={tracker['n_evals']}, elapsed={elapsed:.0f}s", flush=True)

        result = differential_evolution(
            _compute_loss, bounds,
            args=(targets, self.qe, model, mu, self.n_paths, tracker),
            maxiter=self.maxiter,
            popsize=self.popsize,
            tol=1e-4,
            seed=42,
            callback=callback,
            disp=False,
        )

        elapsed = time.time() - t0
        print(f"  DE finished: {result.nfev} evaluations in {elapsed:.0f}s", flush=True)

        # Extract params
        if model == 'heston':
            kappa, theta, sigma_v, rho, v0 = result.x
            H = np.nan
        else:
            kappa, theta, sigma_v, rho, v0, H = result.x

        # Compute final model moments with more paths for accuracy
        params_dict = dict(kappa=kappa, theta=theta, sigma_v=sigma_v,
                           rho=rho, v0=v0, mu=mu)
        if model == 'rough_heston':
            params_dict['H'] = H

        prices = _simulate_paths(self.qe, params_dict, model, 200_000, 252)
        log_returns = np.log(prices[:, 1:] / prices[:, :-1])
        model_moments = _compute_moments(log_returns)

        print(f"  Calibrated: kappa={kappa:.3f}, theta={theta:.5f}, "
              f"sigma_v={sigma_v:.3f}, rho={rho:.3f}, v0={v0:.5f}"
              + (f", H={H:.3f}" if not np.isnan(H) else ""), flush=True)
        print(f"  Model moments: var={model_moments['variance']:.4f}, "
              f"skew={model_moments['skewness']:.3f}, "
              f"kurt={model_moments['kurtosis']:.3f}", flush=True)
        print(f"  Feller condition: 2*kappa*theta={2*kappa*theta:.4f} vs "
              f"sigma_v^2={sigma_v**2:.4f} "
              f"({'satisfied' if 2*kappa*theta >= sigma_v**2 else 'VIOLATED'})", flush=True)

        return CalibrationResult(
            asset=asset, model=model,
            kappa=kappa, theta=theta, sigma_v=sigma_v, rho=rho,
            v0=v0, H=H, loss=result.fun,
            n_evals=result.nfev,
            empirical_moments=targets,
            model_moments=model_moments,
        )


def run_calibration():
    """Run full calibration pipeline."""
    import warnings
    warnings.filterwarnings('ignore')

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    data_dir = os.path.join(project_root, 'data', 'processed')
    table_dir = os.path.join(project_root, 'outputs', 'tables')
    fig_dir = os.path.join(project_root, 'outputs', 'figures', 'calibration')
    os.makedirs(fig_dir, exist_ok=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    returns_df = pd.read_csv(
        os.path.join(data_dir, 'portfolio_returns.csv'),
        parse_dates=['Date'], index_col='Date'
    )
    # Training set: pre-2022
    train = returns_df[returns_df.index < '2022-01-01']

    assets = ['SPY']
    calibrator = HestonCalibrator(n_paths=100_000, maxiter=30, popsize=15)

    # Load old params for comparison
    old_mle_path = os.path.join(table_dir, 'calibrated_params_mle.csv')
    old_params = pd.read_csv(old_mle_path) if os.path.exists(old_mle_path) else None

    results = []
    convergence_data = {}

    total_t0 = time.time()

    for asset in assets:
        ret = train[asset].dropna().values
        mu = np.mean(ret) * 252

        print(f"\n{'='*60}", flush=True)
        print(f"Calibrating Heston for {asset} ({len(ret)} returns)", flush=True)
        print(f"{'='*60}", flush=True)
        h_result = calibrator.calibrate(ret, asset, model='heston', mu=mu)
        results.append(h_result)
        convergence_data[f'{asset}_heston'] = calibrator.convergence_history.copy()

        print(f"\n{'='*60}", flush=True)
        print(f"Calibrating Rough Heston for {asset}", flush=True)
        print(f"{'='*60}", flush=True)
        rh_result = calibrator.calibrate(ret, asset, model='rough_heston', mu=mu)
        results.append(rh_result)
        convergence_data[f'{asset}_rough_heston'] = calibrator.convergence_history.copy()

    total_elapsed = time.time() - total_t0
    print(f"\nTotal calibration time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)", flush=True)

    # ── Comparison: old vs new vs empirical ──
    if old_params is not None:
        print(f"\n{'='*60}", flush=True)
        print("COMPARISON: Old Params vs New Params vs Empirical", flush=True)
        print(f"{'='*60}", flush=True)
        param_names = ['kappa', 'theta', 'sigma_v', 'rho', 'v0']
        for r in results:
            old_row = old_params[(old_params['Asset'] == r.asset) & (old_params['Model'] == r.model)]
            if len(old_row) == 0:
                continue
            old_row = old_row.iloc[0]
            print(f"\n  {r.asset} — {r.model}:", flush=True)
            print(f"  {'Param':<10} {'Old':>10} {'New':>10} {'Change':>10}", flush=True)
            print(f"  {'-'*42}", flush=True)
            for p in param_names:
                old_v = old_row[p]
                new_v = getattr(r, p)
                print(f"  {p:<10} {old_v:>10.4f} {new_v:>10.4f} {new_v - old_v:>+10.4f}", flush=True)
            if r.model == 'rough_heston':
                print(f"  {'H':<10} {old_row['H']:>10.4f} {r.H:>10.4f} {r.H - old_row['H']:>+10.4f}", flush=True)
            print(f"  {'loss':<10} {old_row['loss']:>10.4f} {r.loss:>10.4f}", flush=True)
            print(f"\n  Empirical: var={r.empirical_moments['variance']:.4f}, "
                  f"skew={r.empirical_moments['skewness']:.3f}, "
                  f"kurt={r.empirical_moments['kurtosis']:.3f}", flush=True)
            print(f"  Model:     var={r.model_moments['variance']:.4f}, "
                  f"skew={r.model_moments['skewness']:.3f}, "
                  f"kurt={r.model_moments['kurtosis']:.3f}", flush=True)

    # Save calibrated params
    rows = []
    for r in results:
        rows.append({
            'Asset': r.asset, 'Model': r.model,
            'kappa': r.kappa, 'theta': r.theta, 'sigma_v': r.sigma_v,
            'rho': r.rho, 'v0': r.v0, 'H': r.H, 'loss': r.loss,
            'Feller': 'Yes' if 2*r.kappa*r.theta >= r.sigma_v**2 else 'No',
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(table_dir, 'calibrated_params_mle.csv'), index=False
    )
    print("Saved: calibrated_params_mle.csv", flush=True)

    # Save fit comparison
    fit_rows = []
    for r in results:
        fit_rows.append({
            'Asset': r.asset, 'Model': r.model,
            'Emp_Var': r.empirical_moments['variance'],
            'Mod_Var': r.model_moments['variance'],
            'Emp_Skew': r.empirical_moments['skewness'],
            'Mod_Skew': r.model_moments['skewness'],
            'Emp_Kurt': r.empirical_moments['kurtosis'],
            'Mod_Kurt': r.model_moments['kurtosis'],
        })
    pd.DataFrame(fit_rows).to_csv(
        os.path.join(table_dir, 'calibration_fit_comparison.csv'), index=False
    )
    print("Saved: calibration_fit_comparison.csv", flush=True)

    # ── Figure 1: Calibration fit for SPY ──
    spy_heston = [r for r in results if r.asset == 'SPY' and r.model == 'heston'][0]
    spy_rh = [r for r in results if r.asset == 'SPY' and r.model == 'rough_heston'][0]

    qe = _load_engine()
    spy_ret = train['SPY'].dropna().values
    mu_spy = np.mean(spy_ret) * 252

    h_params = dict(kappa=spy_heston.kappa, theta=spy_heston.theta,
                    sigma_v=spy_heston.sigma_v, rho=spy_heston.rho,
                    v0=spy_heston.v0, mu=mu_spy)
    rh_params = dict(kappa=spy_rh.kappa, theta=spy_rh.theta,
                     sigma_v=spy_rh.sigma_v, rho=spy_rh.rho,
                     v0=spy_rh.v0, mu=mu_spy, H=spy_rh.H)

    # Simulate for plotting
    h_prices = _simulate_paths(qe, h_params, 'heston', 200_000, 252)
    h_logr = np.log(h_prices[:, 1:] / h_prices[:, :-1])
    rh_prices = _simulate_paths(qe, rh_params, 'rough_heston', 200_000, 252)
    rh_logr = np.log(rh_prices[:, 1:] / rh_prices[:, :-1])

    # Subsample flattened returns for histogram (5M points is plenty)
    h_flat = h_logr.flatten()
    rh_flat = rh_logr.flatten()
    rng = np.random.default_rng(42)
    n_hist = min(5_000_000, len(h_flat))
    h_hist = rng.choice(h_flat, n_hist, replace=False)
    rh_hist = rng.choice(rh_flat, n_hist, replace=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Return distribution
    ax = axes[0, 0]
    bins = np.linspace(-0.08, 0.08, 200)
    ax.hist(spy_ret, bins=bins, density=True, alpha=0.5, color='black', label='Empirical')
    ax.hist(h_hist, bins=bins, density=True, alpha=0.4, color='#FF5722', label='Heston')
    ax.hist(rh_hist, bins=bins, density=True, alpha=0.3, color='#4CAF50', label='Rough Heston')
    ax.set_title('Return Distribution: SPY')
    ax.legend()
    ax.set_xlim(-0.06, 0.06)

    # QQ plot
    ax = axes[0, 1]
    q_points = np.linspace(0.001, 0.999, 500)
    emp_q = np.quantile(spy_ret, q_points)
    h_q = np.quantile(h_hist, q_points)
    rh_q = np.quantile(rh_hist, q_points)
    ax.scatter(emp_q, h_q, s=3, alpha=0.5, color='#FF5722', label='Heston')
    ax.scatter(emp_q, rh_q, s=3, alpha=0.5, color='#4CAF50', label='Rough Heston')
    lims = [min(emp_q.min(), h_q.min()), max(emp_q.max(), h_q.max())]
    ax.plot(lims, lims, 'k--', lw=0.8)
    ax.set_title('QQ Plot vs Empirical')
    ax.set_xlabel('Empirical')
    ax.set_ylabel('Model')
    ax.legend()

    # ACF of |returns| (per-path averaged)
    ax = axes[1, 0]
    lags = list(range(1, 60))
    emp_acf = [_acf_1d(np.abs(spy_ret), lag) for lag in lags]
    h_abs = np.abs(h_logr)
    rh_abs = np.abs(rh_logr)
    h_acf = [_acf_paths(h_abs, lag) for lag in lags]
    rh_acf = [_acf_paths(rh_abs, lag) for lag in lags]
    ax.plot(lags, emp_acf, 'k-', lw=2, label='Empirical')
    ax.plot(lags, h_acf, '--', color='#FF5722', lw=1.5, label='Heston')
    ax.plot(lags, rh_acf, '--', color='#4CAF50', lw=1.5, label='Rough Heston')
    ax.set_title('ACF of |Returns|')
    ax.set_xlabel('Lag (days)')
    ax.legend()

    # Parameter table
    ax = axes[1, 1]
    ax.axis('off')
    cell_text = [
        ['kappa', f'{spy_heston.kappa:.3f}', f'{spy_rh.kappa:.3f}'],
        ['theta', f'{spy_heston.theta:.5f}', f'{spy_rh.theta:.5f}'],
        ['sigma_v', f'{spy_heston.sigma_v:.3f}', f'{spy_rh.sigma_v:.3f}'],
        ['rho', f'{spy_heston.rho:.3f}', f'{spy_rh.rho:.3f}'],
        ['v0', f'{spy_heston.v0:.5f}', f'{spy_rh.v0:.5f}'],
        ['H', 'N/A', f'{spy_rh.H:.3f}'],
        ['Loss', f'{spy_heston.loss:.4f}', f'{spy_rh.loss:.4f}'],
    ]
    table = ax.table(cellText=cell_text, colLabels=['Param', 'Heston', 'Rough Heston'],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax.set_title('Calibrated Parameters: SPY', fontsize=11)

    fig.suptitle('Calibration Fit: SPY', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'calibration_fit_spy.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Saved: calibration_fit_spy.png", flush=True)

    # ── Figure 2: Parameter comparison ──
    quick_path = os.path.join(table_dir, 'calibrated_parameters.csv')
    if os.path.exists(quick_path):
        quick_params = pd.read_csv(quick_path)
        n_assets = len(assets)
        fig, axes = plt.subplots(1, max(n_assets, 1), figsize=(6 * max(n_assets, 1), 5),
                                 squeeze=False)
        axes = axes.flatten()
        param_names = ['kappa', 'theta', 'sigma_v', 'rho', 'v0']
        for i, asset in enumerate(assets):
            ax = axes[i]
            quick_h = quick_params[(quick_params['Asset'] == asset) & (quick_params['Model'] == 'heston')]
            mle_h = [r for r in results if r.asset == asset and r.model == 'heston']
            if len(quick_h) == 0 or len(mle_h) == 0:
                continue
            mle_h = mle_h[0]
            quick_vals = [quick_h[p].values[0] for p in param_names]
            mle_vals = [getattr(mle_h, p) for p in param_names]
            x = np.arange(len(param_names))
            ax.bar(x - 0.2, quick_vals, 0.4, label='Quick (Day 9)', color='#2196F3')
            ax.bar(x + 0.2, mle_vals, 0.4, label='MLE (Day 10)', color='#FF5722')
            ax.set_xticks(x)
            ax.set_xticklabels(param_names, rotation=30)
            ax.set_title(asset, fontweight='bold')
            ax.legend(fontsize=8)
        fig.suptitle('Parameter Comparison: Quick vs MLE Calibration',
                     fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, 'parameter_comparison.png'),
                    dpi=200, bbox_inches='tight')
        plt.close(fig)
        print("Saved: parameter_comparison.png", flush=True)
    else:
        print("Skipped parameter_comparison.png (no quick calibration file)", flush=True)

    # ── Figure 3: DE convergence ──
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, history in convergence_data.items():
        if history:
            ax.plot(range(1, len(history) + 1), history, 'o-', lw=1.5,
                    ms=3, label=key)
    ax.set_xlabel('DE Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Differential Evolution Convergence', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'de_convergence.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Saved: de_convergence.png", flush=True)

    print("\n=== Calibration complete ===", flush=True)
    return results


if __name__ == '__main__':
    run_calibration()
