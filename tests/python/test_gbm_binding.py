"""
Day 4 — Python validation of the C++ GBM engine via pybind11 bindings.

Generates:
  outputs/figures/engine/gbm_validation.png
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'src', 'cpp'))

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import quant_engine_py as qe

# ── Parameters ──────────────────────────────────────────────────────
S0, mu, sigma, T, n_steps = 100.0, 0.05, 0.2, 1.0, 252
N = 1_000_000

params = qe.GBMParams()
params.S0 = S0; params.mu = mu; params.sigma = sigma
params.T = T; params.n_steps = n_steps

cfg = qe.SimConfig()
cfg.n_paths = N; cfg.n_steps = n_steps; cfg.seed = 42

# ── Simulate ────────────────────────────────────────────────────────
print("Running GBM simulation (1M paths, 252 steps)…")
terminal = qe.simulate_gbm(params, cfg)
print(f"  Returned numpy array: shape={terminal.shape}, dtype={terminal.dtype}")

# ── Risk metrics ────────────────────────────────────────────────────
rm = qe.compute_risk(terminal, S0)
print("\n── Risk Metrics ──")
for attr in ['mean_price','std_price','skewness','kurtosis',
             'var_95','var_99','es_95','es_99',
             'min_price','max_price','median_price']:
    print(f"  {attr:15s} = {getattr(rm, attr):.4f}")

# ── Analytical comparison ───────────────────────────────────────────
analytical_mean = S0 * np.exp(mu * T)
sim_mean = terminal.mean()
rel_err = abs(sim_mean - analytical_mean) / analytical_mean
print(f"\nE[S_T] analytical = {analytical_mean:.4f}, simulated = {sim_mean:.4f}, rel err = {rel_err:.4%}")
assert rel_err < 0.002, f"Mean relative error too large: {rel_err:.4%}"

log_ret = np.log(terminal / S0)
analytical_var_lr = sigma**2 * T
sim_var_lr = log_ret.var(ddof=1)
rel_err_v = abs(sim_var_lr - analytical_var_lr) / analytical_var_lr
print(f"Var[ln R] analytical = {analytical_var_lr:.6f}, simulated = {sim_var_lr:.6f}, rel err = {rel_err_v:.4%}")
assert rel_err_v < 0.01, f"Variance relative error too large: {rel_err_v:.4%}"

# ── Convergence study ───────────────────────────────────────────────
sizes = [1_000, 10_000, 100_000, 1_000_000]
errors = []
for n in sizes:
    c = qe.SimConfig()
    c.n_paths = n; c.n_steps = n_steps; c.seed = 42
    t = qe.simulate_gbm(params, c)
    errors.append(abs(t.mean() - analytical_mean))
print("\n── Convergence ──")
print(f"  {'N':>10s}  {'Error':>12s}  {'Ratio':>8s}")
for i, (n, e) in enumerate(zip(sizes, errors)):
    ratio = errors[i]/errors[i-1] if i > 0 else 0.0
    print(f"  {n:>10d}  {e:>12.6f}  {ratio:>8.3f}")

# ── Full paths (small sample for plot) ──────────────────────────────
cfg_small = qe.SimConfig()
cfg_small.n_paths = 200; cfg_small.n_steps = n_steps; cfg_small.seed = 42
paths = qe.simulate_gbm_paths(params, cfg_small)
print(f"\nFull paths array: shape={paths.shape}")

# ── Figure ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Day 4 — GBM Engine Validation', fontsize=14, fontweight='bold')

# Panel 1: Terminal price histogram + log-normal PDF
ax = axes[0, 0]
ax.hist(terminal, bins=200, density=True, alpha=0.7, color='steelblue', label='Simulated')
m_ln = (mu - 0.5*sigma**2)*T
s_ln = sigma*np.sqrt(T)
x = np.linspace(terminal.min(), np.percentile(terminal, 99.5), 500)
pdf = stats.lognorm.pdf(x, s=s_ln, scale=S0*np.exp(m_ln))
ax.plot(x, pdf, 'r-', lw=2, label='Analytical log-normal')
ax.axvline(rm.mean_price, color='k', ls='--', lw=1, label=f'Mean = {rm.mean_price:.2f}')
ax.axvline(S0 - rm.var_99, color='orange', ls='--', lw=1, label=f'VaR₉₉ boundary = {S0-rm.var_99:.2f}')
ax.set_xlabel('Terminal Price S_T')
ax.set_ylabel('Density')
ax.set_title('Terminal Price Distribution (1M paths)')
ax.legend(fontsize=8)

# Panel 2: Sample paths
ax = axes[0, 1]
t_grid = np.linspace(0, T, n_steps + 1)
for i in range(50):
    ax.plot(t_grid, paths[i], alpha=0.3, lw=0.5, color='steelblue')
ax.plot(t_grid, paths[:50].mean(axis=0), 'r-', lw=2, label='Sample mean (50 paths)')
ax.axhline(analytical_mean, color='k', ls='--', lw=1, label=f'E[S_T] = {analytical_mean:.2f}')
ax.set_xlabel('Time (years)')
ax.set_ylabel('Price')
ax.set_title('Sample GBM Paths')
ax.legend(fontsize=8)

# Panel 3: Convergence plot
ax = axes[1, 0]
ax.loglog(sizes, errors, 'bo-', lw=2, label='Simulated error')
# Reference 1/sqrt(N) line
ref = errors[0] * np.sqrt(sizes[0]) / np.sqrt(sizes)
ax.loglog(sizes, ref, 'r--', lw=1, label=r'$O(1/\sqrt{N})$ reference')
ax.set_xlabel('Number of Paths (N)')
ax.set_ylabel('|Mean Error|')
ax.set_title('Monte Carlo Convergence Rate')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 4: QQ plot of log-returns
ax = axes[1, 1]
sample_lr = np.sort(log_ret[::100])  # subsample for QQ
theoretical_q = stats.norm.ppf(np.linspace(0.001, 0.999, len(sample_lr)),
                                loc=m_ln, scale=s_ln)
ax.scatter(theoretical_q, sample_lr, s=1, alpha=0.5, color='steelblue')
lims = [min(theoretical_q.min(), sample_lr.min()), max(theoretical_q.max(), sample_lr.max())]
ax.plot(lims, lims, 'r-', lw=1)
ax.set_xlabel('Theoretical Quantiles (Normal)')
ax.set_ylabel('Sample Quantiles')
ax.set_title('QQ Plot of Log-Returns')
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', 'figures', 'engine', 'gbm_validation.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved: {out_path}")
print("\n✓ All Python validation checks passed.")
