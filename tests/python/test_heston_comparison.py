"""
Day 5 — Heston vs GBM comparison: stylized facts, risk metrics, parameter sensitivity.

Generates 8 figures in outputs/figures/engine/.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'src', 'cpp'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import acf

import quant_engine_py as qe

OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', 'figures', 'engine')
os.makedirs(OUT, exist_ok=True)
DATA = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')

# ── Parameters ──────────────────────────────────────────────────────
S0, mu, T, n_steps = 100.0, 0.05, 1.0, 252
N = 1_000_000
N_paths_vis = 200  # for path visualisation

gbm_p = qe.GBMParams()
gbm_p.S0 = S0; gbm_p.mu = mu; gbm_p.sigma = 0.2; gbm_p.T = T; gbm_p.n_steps = n_steps

hp = qe.HestonParams()
hp.S0 = S0; hp.mu = mu; hp.T = T; hp.n_steps = n_steps
hp.v0 = 0.04; hp.kappa = 2.0; hp.theta = 0.04; hp.sigma_v = 0.3; hp.rho = -0.7

cfg = qe.SimConfig(); cfg.n_paths = N; cfg.n_steps = n_steps; cfg.seed = 42

# ── Simulate ────────────────────────────────────────────────────────
print("Simulating GBM (1M paths)...")
gbm_term = qe.simulate_gbm(gbm_p, cfg)
print("Simulating Heston QE (1M paths)...")
heston_term = qe.simulate_heston(hp, cfg, True)

cfg_vis = qe.SimConfig(); cfg_vis.n_paths = N_paths_vis; cfg_vis.n_steps = n_steps; cfg_vis.seed = 42
print("Generating full paths for visualization...")
gbm_paths = qe.simulate_gbm_paths(gbm_p, cfg_vis)
heston_price_paths, heston_var_paths = qe.simulate_heston_full(hp, cfg_vis, True)

# Log-returns
gbm_lr = np.log(gbm_term / S0)
heston_lr = np.log(heston_term / S0)

# Real data
real_ret = pd.read_csv(os.path.join(DATA, 'portfolio_returns.csv'), index_col=0)['SPY'].dropna().values

# Risk metrics
rm_gbm = qe.compute_risk(gbm_term, S0)
rm_hes = qe.compute_risk(heston_term, S0)

print(f"\n{'Metric':>15s}  {'GBM':>10s}  {'Heston':>10s}  {'Diff%':>8s}")
for attr in ['mean_price','std_price','var_95','var_99','es_95','es_99']:
    g = getattr(rm_gbm, attr); h = getattr(rm_hes, attr)
    d = (h - g) / g * 100 if g != 0 else 0
    print(f"  {attr:>15s}  {g:>10.2f}  {h:>10.2f}  {d:>+7.1f}%")

# ──────────────────────────────────────────────────────────────────
# Figure 1: Return Distribution Comparison (GBM vs Heston vs Real)
# ──────────────────────────────────────────────────────────────────
print("\nFigure 1: Return distributions...")
fig, ax = plt.subplots(figsize=(10, 6))

# Scale real returns to annual for comparability
real_annual = real_ret * np.sqrt(252)  # rough scaling for overlay
# Use standardised log-returns for all three
gbm_std = (gbm_lr - gbm_lr.mean()) / gbm_lr.std()
hes_std = (heston_lr - heston_lr.mean()) / heston_lr.std()
real_std = (real_ret - real_ret.mean()) / real_ret.std()

bins = np.linspace(-5, 5, 150)
ax.hist(gbm_std, bins=bins, density=True, alpha=0.4, color='steelblue', label=f'GBM (skew={stats.skew(gbm_lr):.3f}, kurt={stats.kurtosis(gbm_lr):.3f})')
ax.hist(hes_std, bins=bins, density=True, alpha=0.4, color='crimson', label=f'Heston (skew={stats.skew(heston_lr):.3f}, kurt={stats.kurtosis(heston_lr):.3f})')
ax.hist(real_std, bins=bins, density=True, alpha=0.35, color='forestgreen', label=f'Real SPY (skew={stats.skew(real_ret):.3f}, kurt={stats.kurtosis(real_ret):.3f})')

x = np.linspace(-5, 5, 300)
ax.plot(x, stats.norm.pdf(x), 'k--', lw=2, label='Standard Normal')
ax.set_xlabel('Standardised Log-Returns')
ax.set_ylabel('Density')
ax.set_title('Return Distribution: GBM vs Heston vs Real Data')
ax.legend(fontsize=9)
ax.set_xlim(-5, 5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'heston_vs_gbm_distributions.png'), dpi=150)
plt.close(fig)

# ──────────────────────────────────────────────────────────────────
# Figure 2: QQ-Plot Comparison
# ──────────────────────────────────────────────────────────────────
print("Figure 2: QQ plots...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for ax, data, title in [(ax1, gbm_lr, 'GBM Log-Returns'), (ax2, heston_lr, 'Heston Log-Returns')]:
    std_data = (data - data.mean()) / data.std()
    sample = np.sort(std_data[::max(1, len(std_data)//2000)])
    theoretical = stats.norm.ppf(np.linspace(0.001, 0.999, len(sample)))
    ax.scatter(theoretical, sample, s=2, alpha=0.5, color='steelblue' if 'GBM' in title else 'crimson')
    lims = [min(theoretical.min(), sample.min()), max(theoretical.max(), sample.max())]
    ax.plot(lims, lims, 'k-', lw=1)
    ax.set_xlabel('Theoretical Quantiles (Normal)')
    ax.set_ylabel('Sample Quantiles')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

fig.suptitle('QQ-Plot: GBM (normal) vs Heston (fat tails)', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'heston_vs_gbm_qq.png'), dpi=150)
plt.close(fig)

# ──────────────────────────────────────────────────────────────────
# Figure 3: Sample Paths
# ──────────────────────────────────────────────────────────────────
print("Figure 3: Sample paths...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
t_grid = np.linspace(0, T, n_steps + 1)
n_show = 20

for i in range(n_show):
    ax1.plot(t_grid, gbm_paths[i], alpha=0.4, lw=0.6, color='steelblue')
    ax2.plot(t_grid, heston_price_paths[i], alpha=0.4, lw=0.6, color='crimson')

ax1.set_ylabel('Price'); ax1.set_title('GBM Sample Paths (constant volatility)')
ax2.set_ylabel('Price'); ax2.set_title('Heston Sample Paths (stochastic volatility)')
ax2.set_xlabel('Time (years)')
for ax in [ax1, ax2]:
    ax.axhline(S0, color='k', ls=':', lw=0.8)
    ax.grid(True, alpha=0.2)

fig.tight_layout()
fig.savefig(os.path.join(OUT, 'heston_vs_gbm_paths.png'), dpi=150)
plt.close(fig)

# ──────────────────────────────────────────────────────────────────
# Figure 4: Variance Paths
# ──────────────────────────────────────────────────────────────────
print("Figure 4: Variance paths...")
fig, ax = plt.subplots(figsize=(10, 5))
for i in range(n_show):
    ax.plot(t_grid, heston_var_paths[i], alpha=0.35, lw=0.6, color='crimson')

# Analytical mean E[v_t]
ev = hp.theta + (hp.v0 - hp.theta) * np.exp(-hp.kappa * t_grid)
ax.plot(t_grid, ev, 'k--', lw=2.5, label=r'$E[v_t] = \theta + (v_0 - \theta)e^{-\kappa t}$')
ax.axhline(hp.theta, color='gray', ls=':', lw=1, label=r'$\theta$ (long-run variance)')
ax.set_xlabel('Time (years)'); ax.set_ylabel('Variance $v_t$')
ax.set_title('Heston Variance Paths with Mean Reversion')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'heston_variance_paths.png'), dpi=150)
plt.close(fig)

# ──────────────────────────────────────────────────────────────────
# Figure 5: ACF Comparison
# ──────────────────────────────────────────────────────────────────
print("Figure 5: ACF comparison...")

def compute_daily_returns(price_paths):
    """Compute daily log-returns from price paths array (n_paths, n_steps+1)."""
    return np.log(price_paths[:, 1:] / price_paths[:, :-1])

gbm_daily = compute_daily_returns(gbm_paths)
hes_daily = compute_daily_returns(heston_price_paths)

n_lags = 40

def avg_acf(daily_rets, transform, nlags):
    """Average ACF across paths."""
    acf_sum = np.zeros(nlags + 1)
    count = 0
    for i in range(daily_rets.shape[0]):
        series = transform(daily_rets[i])
        if np.std(series) < 1e-15:
            continue
        a = acf(series, nlags=nlags, fft=True)
        acf_sum += a
        count += 1
    return acf_sum / count if count > 0 else acf_sum

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
configs = [
    (axes[0,0], gbm_daily, np.abs, 'GBM |returns| ACF', 'steelblue'),
    (axes[0,1], hes_daily, np.abs, 'Heston |returns| ACF', 'crimson'),
    (axes[1,0], gbm_daily, lambda x: x**2, 'GBM squared returns ACF', 'steelblue'),
    (axes[1,1], hes_daily, lambda x: x**2, 'Heston squared returns ACF', 'crimson'),
]

for ax, data, transform, title, color in configs:
    a = avg_acf(data, transform, n_lags)
    ax.bar(range(len(a)), a, color=color, alpha=0.7, width=0.8)
    ax.axhline(0, color='k', lw=0.5)
    n_obs = data.shape[1]
    ax.axhline(1.96/np.sqrt(n_obs), color='gray', ls='--', lw=0.8)
    ax.axhline(-1.96/np.sqrt(n_obs), color='gray', ls='--', lw=0.8)
    ax.set_title(title)
    ax.set_xlabel('Lag'); ax.set_ylabel('ACF')
    ax.set_ylim(-0.1, max(0.5, a[1:].max() * 1.3) if a[1:].max() > 0.05 else 0.15)

fig.suptitle('Volatility Clustering: GBM (none) vs Heston (strong)', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'heston_vs_gbm_acf.png'), dpi=150)
plt.close(fig)

# ──────────────────────────────────────────────────────────────────
# Figure 6: VaR/ES Comparison
# ──────────────────────────────────────────────────────────────────
print("Figure 6: VaR/ES comparison...")
fig, ax = plt.subplots(figsize=(10, 6))

metrics = ['var_95', 'var_99', 'es_95', 'es_99']
labels = ['VaR 95%', 'VaR 99%', 'ES 95%', 'ES 99%']
gbm_vals = [getattr(rm_gbm, m) for m in metrics]
hes_vals = [getattr(rm_hes, m) for m in metrics]

x = np.arange(len(metrics))
width = 0.35
bars1 = ax.bar(x - width/2, gbm_vals, width, label='GBM', color='steelblue', alpha=0.8)
bars2 = ax.bar(x + width/2, hes_vals, width, label='Heston', color='crimson', alpha=0.8)

# Annotate differences
for i in range(len(metrics)):
    diff_pct = (hes_vals[i] - gbm_vals[i]) / gbm_vals[i] * 100
    y_pos = max(gbm_vals[i], hes_vals[i]) + 1
    ax.text(x[i], y_pos, f'+{diff_pct:.1f}%', ha='center', fontsize=10, fontweight='bold', color='darkred')

ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel('Loss Amount ($)')
ax.set_title('Risk Metrics: GBM vs Heston — Model Risk Quantified')
ax.legend(fontsize=11)
ax.grid(True, axis='y', alpha=0.3)

# Add text box
diff_es99 = (rm_hes.es_99 - rm_gbm.es_99) / rm_gbm.es_99 * 100
textstr = f'Heston ES$_{{99}}$ is {diff_es99:.1f}% higher than GBM ES$_{{99}}$\nThis difference IS model risk'
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)

fig.tight_layout()
fig.savefig(os.path.join(OUT, 'heston_vs_gbm_risk.png'), dpi=150)
plt.close(fig)

# ──────────────────────────────────────────────────────────────────
# Figure 7: Sensitivity Analysis (2x2)
# ──────────────────────────────────────────────────────────────────
print("Figure 7: Sensitivity analysis...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

cfg_sens = qe.SimConfig(); cfg_sens.n_paths = 500_000; cfg_sens.n_steps = n_steps; cfg_sens.seed = 42

# Top-left: Effect of sigma_v on return distribution
ax = axes[0, 0]
sigma_vs = [0.1, 0.3, 0.5, 0.8]
colors_sv = ['#2196F3', '#FF9800', '#E91E63', '#9C27B0']
for sv, col in zip(sigma_vs, colors_sv):
    hp_s = qe.HestonParams()
    hp_s.S0=S0; hp_s.mu=mu; hp_s.T=T; hp_s.n_steps=n_steps
    hp_s.v0=0.04; hp_s.kappa=2.0; hp_s.theta=0.04; hp_s.sigma_v=sv; hp_s.rho=-0.7
    term = qe.simulate_heston(hp_s, cfg_sens, True)
    lr = np.log(term / S0)
    std_lr = (lr - lr.mean()) / lr.std()
    ax.hist(std_lr, bins=np.linspace(-5,5,120), density=True, alpha=0.35, color=col,
            label=f'$\\sigma_v$={sv} (kurt={stats.kurtosis(lr):.2f})')
ax.plot(np.linspace(-5,5,200), stats.norm.pdf(np.linspace(-5,5,200)), 'k--', lw=1.5)
ax.set_title(r'Effect of $\sigma_v$ (vol of vol)')
ax.set_xlabel('Standardised log-returns'); ax.legend(fontsize=8); ax.set_xlim(-5,5)

# Top-right: Effect of rho on return distribution
ax = axes[0, 1]
rhos = [-0.9, -0.5, 0.0, 0.5]
colors_rho = ['#E91E63', '#FF9800', '#4CAF50', '#2196F3']
for rho_val, col in zip(rhos, colors_rho):
    hp_r = qe.HestonParams()
    hp_r.S0=S0; hp_r.mu=mu; hp_r.T=T; hp_r.n_steps=n_steps
    hp_r.v0=0.04; hp_r.kappa=2.0; hp_r.theta=0.04; hp_r.sigma_v=0.3; hp_r.rho=rho_val
    term = qe.simulate_heston(hp_r, cfg_sens, True)
    lr = np.log(term / S0)
    std_lr = (lr - lr.mean()) / lr.std()
    ax.hist(std_lr, bins=np.linspace(-5,5,120), density=True, alpha=0.35, color=col,
            label=f'$\\rho$={rho_val} (skew={stats.skew(lr):.2f})')
ax.plot(np.linspace(-5,5,200), stats.norm.pdf(np.linspace(-5,5,200)), 'k--', lw=1.5)
ax.set_title(r'Effect of $\rho$ (leverage correlation)')
ax.set_xlabel('Standardised log-returns'); ax.legend(fontsize=8); ax.set_xlim(-5,5)

# Bottom-left: Effect of kappa on variance paths
ax = axes[1, 0]
kappas = [0.5, 2.0, 5.0, 10.0]
colors_k = ['#E91E63', '#FF9800', '#4CAF50', '#2196F3']
cfg_var = qe.SimConfig(); cfg_var.n_paths = 3; cfg_var.n_steps = n_steps; cfg_var.seed = 42
for k_val, col in zip(kappas, colors_k):
    hp_k = qe.HestonParams()
    hp_k.S0=S0; hp_k.mu=mu; hp_k.T=T; hp_k.n_steps=n_steps
    hp_k.v0=0.09; hp_k.kappa=k_val; hp_k.theta=0.04; hp_k.sigma_v=0.3; hp_k.rho=-0.7
    _, var_p = qe.simulate_heston_full(hp_k, cfg_var, True)
    ev = 0.04 + 0.05 * np.exp(-k_val * t_grid)
    ax.plot(t_grid, ev, '--', color=col, lw=2, label=f'$\\kappa$={k_val}')
    for j in range(2):
        ax.plot(t_grid, var_p[j], '-', color=col, alpha=0.3, lw=0.6)
ax.axhline(0.04, color='gray', ls=':', lw=0.8)
ax.set_title(r'Effect of $\kappa$ (mean reversion speed)')
ax.set_xlabel('Time (years)'); ax.set_ylabel('Variance $v_t$')
ax.legend(fontsize=8)

# Bottom-right: Effect of sigma_v on VaR/ES
ax = axes[1, 1]
sv_range = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8])
var99_vals = []; es99_vals = []
for sv in sv_range:
    hp_v = qe.HestonParams()
    hp_v.S0=S0; hp_v.mu=mu; hp_v.T=T; hp_v.n_steps=n_steps
    hp_v.v0=0.04; hp_v.kappa=2.0; hp_v.theta=0.04; hp_v.sigma_v=sv; hp_v.rho=-0.7
    cfg_r = qe.SimConfig(); cfg_r.n_paths = 200_000; cfg_r.n_steps = n_steps; cfg_r.seed = 42
    term = qe.simulate_heston(hp_v, cfg_r, True)
    rm = qe.compute_risk(term, S0)
    var99_vals.append(rm.var_99); es99_vals.append(rm.es_99)

ax.plot(sv_range, var99_vals, 'bo-', lw=2, label='VaR 99%')
ax.plot(sv_range, es99_vals, 'rs-', lw=2, label='ES 99%')
ax.axhline(rm_gbm.var_99, color='steelblue', ls='--', lw=1, alpha=0.7, label=f'GBM VaR 99% = {rm_gbm.var_99:.1f}')
ax.axhline(rm_gbm.es_99, color='salmon', ls='--', lw=1, alpha=0.7, label=f'GBM ES 99% = {rm_gbm.es_99:.1f}')
ax.set_xlabel(r'$\sigma_v$ (vol of vol)')
ax.set_ylabel('Risk Measure ($)')
ax.set_title(r'VaR$_{99}$ and ES$_{99}$ vs $\sigma_v$')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.suptitle('Heston Parameter Sensitivity Analysis', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'heston_sensitivity.png'), dpi=150)
plt.close(fig)

# ──────────────────────────────────────────────────────────────────
# Figure 8: Summary Dashboard (2x3)
# ──────────────────────────────────────────────────────────────────
print("Figure 8: Summary dashboard...")
fig = plt.figure(figsize=(18, 12))
fig.suptitle('GBM vs Heston: Why Stochastic Volatility Matters', fontsize=16, fontweight='bold')

# Panel 1: Return distributions
ax1 = fig.add_subplot(2, 3, 1)
bins = np.linspace(-5, 5, 100)
ax1.hist(gbm_std, bins=bins, density=True, alpha=0.4, color='steelblue', label='GBM')
ax1.hist(hes_std, bins=bins, density=True, alpha=0.4, color='crimson', label='Heston')
ax1.hist(real_std, bins=bins, density=True, alpha=0.3, color='forestgreen', label='Real SPY')
ax1.plot(np.linspace(-5,5,200), stats.norm.pdf(np.linspace(-5,5,200)), 'k--', lw=1)
ax1.set_title('Return Distributions')
ax1.legend(fontsize=7); ax1.set_xlim(-5,5)

# Panel 2: QQ plots (combined)
ax2 = fig.add_subplot(2, 3, 2)
for data, color, label in [(gbm_lr, 'steelblue', 'GBM'), (heston_lr, 'crimson', 'Heston')]:
    std_data = (data - data.mean()) / data.std()
    sample = np.sort(std_data[::max(1, len(std_data)//1000)])
    theoretical = stats.norm.ppf(np.linspace(0.001, 0.999, len(sample)))
    ax2.scatter(theoretical, sample, s=1, alpha=0.4, color=color, label=label)
lims = [-4, 4]
ax2.plot(lims, lims, 'k-', lw=1)
ax2.set_title('QQ-Plot vs Normal')
ax2.legend(fontsize=8)

# Panel 3: Sample paths
ax3 = fig.add_subplot(2, 3, 3)
for i in range(10):
    ax3.plot(t_grid, gbm_paths[i], alpha=0.3, lw=0.5, color='steelblue')
    ax3.plot(t_grid, heston_price_paths[i], alpha=0.3, lw=0.5, color='crimson')
ax3.plot([], [], color='steelblue', label='GBM')
ax3.plot([], [], color='crimson', label='Heston')
ax3.set_title('Sample Paths')
ax3.legend(fontsize=8)

# Panel 4: Variance paths
ax4 = fig.add_subplot(2, 3, 4)
for i in range(10):
    ax4.plot(t_grid, heston_var_paths[i], alpha=0.3, lw=0.6, color='crimson')
ev = hp.theta + (hp.v0 - hp.theta) * np.exp(-hp.kappa * t_grid)
ax4.plot(t_grid, ev, 'k--', lw=2, label=r'$E[v_t]$')
ax4.axhline(hp.theta, color='gray', ls=':', lw=0.8)
ax4.set_title('Variance Paths (mean reversion)')
ax4.legend(fontsize=8)

# Panel 5: ACF
ax5 = fig.add_subplot(2, 3, 5)
acf_gbm = avg_acf(gbm_daily, np.abs, n_lags)
acf_hes = avg_acf(hes_daily, np.abs, n_lags)
ax5.bar(np.arange(n_lags+1) - 0.2, acf_gbm, 0.4, color='steelblue', alpha=0.7, label='GBM')
ax5.bar(np.arange(n_lags+1) + 0.2, acf_hes, 0.4, color='crimson', alpha=0.7, label='Heston')
ax5.set_title('ACF of |returns|')
ax5.legend(fontsize=8); ax5.set_xlabel('Lag')

# Panel 6: VaR/ES
ax6 = fig.add_subplot(2, 3, 6)
x = np.arange(len(metrics)); width = 0.35
ax6.bar(x - width/2, gbm_vals, width, label='GBM', color='steelblue', alpha=0.8)
ax6.bar(x + width/2, hes_vals, width, label='Heston', color='crimson', alpha=0.8)
ax6.set_xticks(x); ax6.set_xticklabels(labels, fontsize=9)
ax6.set_title('Risk Metrics Comparison')
ax6.legend(fontsize=8)
for i in range(len(metrics)):
    diff_pct = (hes_vals[i] - gbm_vals[i]) / gbm_vals[i] * 100
    y_pos = max(gbm_vals[i], hes_vals[i]) + 0.5
    ax6.text(x[i], y_pos, f'+{diff_pct:.0f}%', ha='center', fontsize=8, fontweight='bold', color='darkred')

fig.tight_layout()
fig.savefig(os.path.join(OUT, 'heston_comparison_summary.png'), dpi=300)
plt.close(fig)

print(f"\nAll 8 figures saved to {OUT}/")
print("Done.")
