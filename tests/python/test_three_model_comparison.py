"""
Day 6 — Three-model comparison: GBM → Heston → Rough Heston.
Generates all comparison figures in outputs/figures/engine/.
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
N_vis = 200

gbm_p = qe.GBMParams()
gbm_p.S0 = S0; gbm_p.mu = mu; gbm_p.sigma = 0.2; gbm_p.T = T; gbm_p.n_steps = n_steps

hp = qe.HestonParams()
hp.S0 = S0; hp.mu = mu; hp.T = T; hp.n_steps = n_steps
hp.v0 = 0.04; hp.kappa = 2.0; hp.theta = 0.04; hp.sigma_v = 0.3; hp.rho = -0.7

rp = qe.RoughHestonParams()
rp.S0 = S0; rp.mu = mu; rp.T = T; rp.n_steps = n_steps
rp.v0 = 0.04; rp.kappa = 2.0; rp.theta = 0.04; rp.sigma_v = 0.3; rp.rho = -0.7
rp.H = 0.1

cfg = qe.SimConfig(); cfg.n_paths = N; cfg.n_steps = n_steps; cfg.seed = 42
cfg_vis = qe.SimConfig(); cfg_vis.n_paths = N_vis; cfg_vis.n_steps = n_steps; cfg_vis.seed = 42

# ── Simulate ────────────────────────────────────────────────────────
print("Simulating GBM...")
gbm_term = qe.simulate_gbm(gbm_p, cfg)
print("Simulating Heston QE...")
heston_term = qe.simulate_heston(hp, cfg, True)
print("Simulating Rough Heston (8 factors)...")
rh_term = qe.simulate_rough_heston(rp, cfg, 8)

print("Generating visualization paths...")
gbm_paths = qe.simulate_gbm_paths(gbm_p, cfg_vis)
heston_pp, heston_vp = qe.simulate_heston_full(hp, cfg_vis, True)
rh_pp, rh_vp = qe.simulate_rough_heston_full(rp, cfg_vis, 8)

# Log-returns
gbm_lr = np.log(gbm_term / S0)
hes_lr = np.log(heston_term / S0)
rh_lr  = np.log(rh_term / S0)

# Real data
real_ret = pd.read_csv(os.path.join(DATA, 'portfolio_returns.csv'), index_col=0)['SPY'].dropna().values

# Risk
rm_g = qe.compute_risk(gbm_term, S0)
rm_h = qe.compute_risk(heston_term, S0)
rm_r = qe.compute_risk(rh_term, S0)

print(f"\n{'Metric':>15s}  {'GBM':>8s}  {'Heston':>8s}  {'RoughH':>8s}")
for attr in ['mean_price','std_price','skewness','kurtosis','var_95','var_99','es_95','es_99']:
    g,h,r = getattr(rm_g,attr), getattr(rm_h,attr), getattr(rm_r,attr)
    print(f"  {attr:>15s}  {g:>8.2f}  {h:>8.2f}  {r:>8.2f}")

t_grid = np.linspace(0, T, n_steps + 1)

# ── Standardised returns ────────────────────────────────────────────
def standardise(x): return (x - x.mean()) / x.std()
gbm_std = standardise(gbm_lr)
hes_std = standardise(hes_lr)
rh_std  = standardise(rh_lr)
real_std = standardise(real_ret)

# ==================================================================
# Figure 1: Return Distribution — Three Models + Real Data (CENTRAL)
# ==================================================================
print("\nFigure 1: Return distributions...")
fig, ax = plt.subplots(figsize=(11, 6))
bins = np.linspace(-5, 5, 150)
ax.hist(gbm_std, bins=bins, density=True, alpha=0.3, color='steelblue',
        label=f'GBM (skew={stats.skew(gbm_lr):.2f}, kurt={stats.kurtosis(gbm_lr):.2f})')
ax.hist(hes_std, bins=bins, density=True, alpha=0.3, color='darkorange',
        label=f'Heston (skew={stats.skew(hes_lr):.2f}, kurt={stats.kurtosis(hes_lr):.2f})')
ax.hist(rh_std, bins=bins, density=True, alpha=0.3, color='crimson',
        label=f'Rough Heston (skew={stats.skew(rh_lr):.2f}, kurt={stats.kurtosis(rh_lr):.2f})')
ax.hist(real_std, bins=bins, density=True, alpha=0.25, color='gray',
        label=f'Real SPY (skew={stats.skew(real_ret):.2f}, kurt={stats.kurtosis(real_ret):.2f})')
x = np.linspace(-5, 5, 300)
ax.plot(x, stats.norm.pdf(x), 'k--', lw=2, label='Normal')
ax.set_xlabel('Standardised Log-Returns', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Return Distributions: GBM vs Heston vs Rough Heston vs Real Data', fontsize=13, fontweight='bold')
ax.legend(fontsize=9); ax.set_xlim(-5, 5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'three_model_distributions.png'), dpi=150)
plt.close(fig)

# ==================================================================
# Figure 2: QQ-Plot Comparison (1x3)
# ==================================================================
print("Figure 2: QQ plots...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, data, title, color in [
    (axes[0], gbm_lr, 'GBM', 'steelblue'),
    (axes[1], hes_lr, 'Heston', 'darkorange'),
    (axes[2], rh_lr, 'Rough Heston', 'crimson'),
]:
    std_d = standardise(data)
    sample = np.sort(std_d[::max(1, len(std_d)//2000)])
    theo = stats.norm.ppf(np.linspace(0.001, 0.999, len(sample)))
    ax.scatter(theo, sample, s=2, alpha=0.5, color=color)
    ax.plot([-4,4], [-4,4], 'k-', lw=1)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Theoretical (Normal)'); ax.set_ylabel('Sample')
    ax.set_xlim(-4,4); ax.set_ylim(-4,4); ax.grid(True, alpha=0.3)
fig.suptitle('QQ-Plot vs Normal: Increasing Tail Heaviness', fontsize=13, fontweight='bold')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'three_model_qq.png'), dpi=150); plt.close(fig)

# ==================================================================
# Figure 3: ACF Comparison — The Memory Test
# ==================================================================
print("Figure 3: ACF comparison...")
n_lags = 60

def daily_returns(price_paths):
    return np.log(price_paths[:, 1:] / price_paths[:, :-1])

def avg_acf(daily_rets, transform, nlags):
    acf_sum = np.zeros(nlags + 1); count = 0
    for i in range(daily_rets.shape[0]):
        series = transform(daily_rets[i])
        if np.std(series) < 1e-15: continue
        acf_sum += acf(series, nlags=nlags, fft=True); count += 1
    return acf_sum / count if count > 0 else acf_sum

gbm_d = daily_returns(gbm_paths)
hes_d = daily_returns(heston_pp)
rh_d  = daily_returns(rh_pp)

fig, ax = plt.subplots(figsize=(11, 6))
lags = np.arange(1, n_lags + 1)
acf_gbm = avg_acf(gbm_d, np.abs, n_lags)[1:]
acf_hes = avg_acf(hes_d, np.abs, n_lags)[1:]
acf_rh  = avg_acf(rh_d,  np.abs, n_lags)[1:]

# Real data ACF
real_abs = np.abs(real_ret)
acf_real = acf(real_abs, nlags=n_lags, fft=True)[1:]

ax.plot(lags, acf_gbm, 'o-', color='steelblue', ms=3, lw=1.5, label='GBM')
ax.plot(lags, acf_hes, 's-', color='darkorange', ms=3, lw=1.5, label='Heston')
ax.plot(lags, acf_rh,  '^-', color='crimson', ms=3, lw=1.5, label='Rough Heston')
ax.plot(lags, acf_real, 'D-', color='gray', ms=3, lw=1.5, alpha=0.7, label='Real SPY')
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('Lag (days)', fontsize=12); ax.set_ylabel('ACF of |returns|', fontsize=12)
ax.set_title('Autocorrelation of Absolute Returns: The Memory Test', fontsize=13, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'three_model_acf.png'), dpi=150); plt.close(fig)

# ==================================================================
# Figure 4: Model Risk Progression (grouped bar chart)
# ==================================================================
print("Figure 4: Risk metrics comparison...")
fig, ax = plt.subplots(figsize=(10, 6))
metrics = ['var_95', 'var_99', 'es_95', 'es_99']
labels = ['VaR 95%', 'VaR 99%', 'ES 95%', 'ES 99%']
g_vals = [getattr(rm_g, m) for m in metrics]
h_vals = [getattr(rm_h, m) for m in metrics]
r_vals = [getattr(rm_r, m) for m in metrics]

x = np.arange(len(metrics)); w = 0.25
ax.bar(x - w, g_vals, w, label='GBM', color='steelblue', alpha=0.8)
ax.bar(x,     h_vals, w, label='Heston', color='darkorange', alpha=0.8)
ax.bar(x + w, r_vals, w, label='Rough Heston', color='crimson', alpha=0.8)
for i in range(len(metrics)):
    d_hr = (r_vals[i] - g_vals[i]) / g_vals[i] * 100
    ax.text(x[i] + w, r_vals[i] + 0.5, f'+{d_hr:.0f}%', ha='center', fontsize=9, fontweight='bold', color='darkred')
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel('Loss ($)'); ax.set_title('Model Risk Progression: GBM < Heston < Rough Heston', fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, axis='y', alpha=0.3)
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
d_es99 = (rm_r.es_99 - rm_g.es_99) / rm_g.es_99 * 100
ax.text(0.02, 0.97, f'Rough Heston ES$_{{99}}$ is {d_es99:.1f}% higher than GBM\nThis gap IS model risk',
        transform=ax.transAxes, fontsize=10, va='top', bbox=props)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'three_model_risk.png'), dpi=150); plt.close(fig)

# ==================================================================
# Figure 5: Rough vs Classic Variance Paths
# ==================================================================
print("Figure 5: Variance paths...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
n_show = 10
for i in range(n_show):
    ax1.plot(t_grid, heston_vp[i], alpha=0.35, lw=0.6, color='darkorange')
    ax2.plot(t_grid, rh_vp[i], alpha=0.35, lw=0.6, color='crimson')
ev = hp.theta + (hp.v0 - hp.theta) * np.exp(-hp.kappa * t_grid)
for ax in [ax1, ax2]:
    ax.plot(t_grid, ev, 'k--', lw=2, label=r'$E[v_t]$')
    ax.axhline(hp.theta, color='gray', ls=':', lw=0.8)
    ax.set_ylabel('Variance $v_t$'); ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
ax1.set_title('Heston Variance Paths (smoother)', fontsize=12)
ax2.set_title('Rough Heston Variance Paths (H=0.1, rougher)', fontsize=12)
ax2.set_xlabel('Time (years)')
fig.suptitle('Variance Path Roughness: Classic vs Rough Heston', fontsize=13, fontweight='bold')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'rough_vs_classic_variance_paths.png'), dpi=150); plt.close(fig)

# ==================================================================
# Figure 6: H-Exponent Sensitivity
# ==================================================================
print("Figure 6: H sensitivity...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
cfg_s = qe.SimConfig(); cfg_s.n_paths = 300_000; cfg_s.n_steps = n_steps; cfg_s.seed = 42
cfg_v = qe.SimConfig(); cfg_v.n_paths = 5; cfg_v.n_steps = n_steps; cfg_v.seed = 42

H_vals = [0.05, 0.1, 0.2, 0.49]
H_colors = ['#E91E63', '#FF5722', '#FF9800', '#4CAF50']

# Top-left: Return distributions
ax = axes[0, 0]
for Hv, col in zip(H_vals, H_colors):
    rp_h = qe.RoughHestonParams()
    rp_h.S0=S0; rp_h.mu=mu; rp_h.T=T; rp_h.n_steps=n_steps
    rp_h.v0=0.04; rp_h.kappa=2.0; rp_h.theta=0.04; rp_h.sigma_v=0.3; rp_h.rho=-0.7; rp_h.H=Hv
    nf = 16 if Hv > 0.4 else 8
    term = qe.simulate_rough_heston(rp_h, cfg_s, nf)
    lr = np.log(term / S0); std_lr = standardise(lr)
    ax.hist(std_lr, bins=np.linspace(-5,5,120), density=True, alpha=0.3, color=col,
            label=f'H={Hv} (kurt={stats.kurtosis(lr):.2f})')
ax.plot(np.linspace(-5,5,200), stats.norm.pdf(np.linspace(-5,5,200)), 'k--', lw=1.5)
ax.set_title('Return Distribution vs H'); ax.legend(fontsize=8); ax.set_xlim(-5,5)

# Top-right: ACF of |returns|
ax = axes[0, 1]
for Hv, col in zip(H_vals, H_colors):
    rp_h = qe.RoughHestonParams()
    rp_h.S0=S0; rp_h.mu=mu; rp_h.T=T; rp_h.n_steps=n_steps
    rp_h.v0=0.04; rp_h.kappa=2.0; rp_h.theta=0.04; rp_h.sigma_v=0.3; rp_h.rho=-0.7; rp_h.H=Hv
    nf = 16 if Hv > 0.4 else 8
    cfg_acf = qe.SimConfig(); cfg_acf.n_paths = 200; cfg_acf.n_steps = n_steps; cfg_acf.seed = 42
    pp, _ = qe.simulate_rough_heston_full(rp_h, cfg_acf, nf)
    dr = daily_returns(pp)
    a = avg_acf(dr, np.abs, 40)[1:]
    ax.plot(range(1, 41), a, '-', color=col, lw=2, label=f'H={Hv}')
ax.set_title('ACF of |returns| vs H'); ax.set_xlabel('Lag'); ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Bottom-left: ES99 vs H
ax = axes[1, 0]
H_fine = [0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.49]
es99_h = []; kurt_h = []
cfg_es = qe.SimConfig(); cfg_es.n_paths = 200_000; cfg_es.n_steps = n_steps; cfg_es.seed = 42
for Hv in H_fine:
    rp_h = qe.RoughHestonParams()
    rp_h.S0=S0; rp_h.mu=mu; rp_h.T=T; rp_h.n_steps=n_steps
    rp_h.v0=0.04; rp_h.kappa=2.0; rp_h.theta=0.04; rp_h.sigma_v=0.3; rp_h.rho=-0.7; rp_h.H=Hv
    nf = 16 if Hv > 0.35 else 8
    term = qe.simulate_rough_heston(rp_h, cfg_es, nf)
    rm = qe.compute_risk(term, S0)
    es99_h.append(rm.es_99)
    lr = np.log(term / S0)
    kurt_h.append(stats.kurtosis(lr))
ax.plot(H_fine, es99_h, 'rs-', lw=2, ms=6)
ax.axhline(rm_g.es_99, color='steelblue', ls='--', lw=1, label=f'GBM ES99={rm_g.es_99:.1f}')
ax.axhline(rm_h.es_99, color='darkorange', ls='--', lw=1, label=f'Heston ES99={rm_h.es_99:.1f}')
ax.set_xlabel('Hurst exponent H'); ax.set_ylabel('ES 99% ($)')
ax.set_title(r'ES$_{99}$ vs H'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Bottom-right: Kurtosis vs H
ax = axes[1, 1]
ax.plot(H_fine, kurt_h, 'ro-', lw=2, ms=6)
ax.axhline(stats.kurtosis(gbm_lr), color='steelblue', ls='--', lw=1, label=f'GBM kurt={stats.kurtosis(gbm_lr):.2f}')
ax.axhline(stats.kurtosis(hes_lr), color='darkorange', ls='--', lw=1, label=f'Heston kurt={stats.kurtosis(hes_lr):.2f}')
ax.set_xlabel('Hurst exponent H'); ax.set_ylabel('Excess Kurtosis')
ax.set_title('Excess Kurtosis vs H'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

fig.suptitle('Rough Heston: Effect of Hurst Exponent H', fontsize=14, fontweight='bold')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'rough_heston_h_sensitivity.png'), dpi=150); plt.close(fig)

# ==================================================================
# Figure 7: Variogram Comparison
# ==================================================================
print("Figure 7: Variogram...")
fig, ax = plt.subplots(figsize=(10, 6))
cfg_vg = qe.SimConfig(); cfg_vg.n_paths = 500; cfg_vg.n_steps = n_steps; cfg_vg.seed = 42

for label, pp, vp, color in [
    ('Heston', *qe.simulate_heston_full(hp, cfg_vg, True), 'darkorange'),
    ('Rough Heston', *qe.simulate_rough_heston_full(rp, cfg_vg, 8), 'crimson'),
]:
    taus = [1, 2, 3, 5, 8, 13, 21, 34, 55]
    variogram = []
    for tau in taus:
        diffs = []
        for i in range(vp.shape[0]):
            log_v = np.log(np.maximum(vp[i], 1e-10))
            d = log_v[tau:] - log_v[:-tau]
            diffs.extend(d.tolist())
        variogram.append(np.mean(np.array(diffs)**2))
    tau_years = np.array(taus) / 252.0
    ax.loglog(tau_years, variogram, 'o-', color=color, lw=2, ms=5, label=label)

# Reference slopes
tau_ref = np.array([1, 55]) / 252.0
v0_h = variogram[0]
ax.loglog(tau_ref, v0_h * (tau_ref/tau_ref[0])**1.0, '--', color='darkorange', alpha=0.5, label='slope=1 (H=0.5)')
ax.loglog(tau_ref, v0_h * (tau_ref/tau_ref[0])**0.2, '--', color='crimson', alpha=0.5, label='slope=0.2 (H=0.1)')
ax.set_xlabel(r'Lag $\tau$ (years)', fontsize=12); ax.set_ylabel(r'$E[(\log v_{t+\tau} - \log v_t)^2]$', fontsize=12)
ax.set_title('Variogram of Log-Volatility: Roughness Verification', fontsize=13, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3, which='both')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'variogram_comparison.png'), dpi=150); plt.close(fig)

# ==================================================================
# Figure 8: Summary Dashboard (2x3, 300 DPI)
# ==================================================================
print("Figure 8: Summary dashboard...")
fig = plt.figure(figsize=(18, 12))
fig.suptitle('Three-Model Hierarchy: From Constant to Rough Volatility', fontsize=16, fontweight='bold')

# 1: Return distributions
ax = fig.add_subplot(2, 3, 1)
bins = np.linspace(-5, 5, 100)
ax.hist(gbm_std, bins=bins, density=True, alpha=0.3, color='steelblue', label='GBM')
ax.hist(hes_std, bins=bins, density=True, alpha=0.3, color='darkorange', label='Heston')
ax.hist(rh_std, bins=bins, density=True, alpha=0.3, color='crimson', label='Rough Heston')
ax.hist(real_std, bins=bins, density=True, alpha=0.2, color='gray', label='Real SPY')
ax.plot(np.linspace(-5,5,200), stats.norm.pdf(np.linspace(-5,5,200)), 'k--', lw=1)
ax.set_title('Return Distributions'); ax.legend(fontsize=6); ax.set_xlim(-5,5)

# 2: QQ combined
ax = fig.add_subplot(2, 3, 2)
for data, color, lbl in [(gbm_lr,'steelblue','GBM'),(hes_lr,'darkorange','Heston'),(rh_lr,'crimson','RoughH')]:
    s = np.sort(standardise(data)[::max(1,len(data)//800)])
    t = stats.norm.ppf(np.linspace(0.001,0.999,len(s)))
    ax.scatter(t, s, s=1, alpha=0.3, color=color, label=lbl)
ax.plot([-4,4],[-4,4],'k-',lw=1); ax.set_title('QQ-Plot vs Normal'); ax.legend(fontsize=7)

# 3: ACF comparison
ax = fig.add_subplot(2, 3, 3)
for a, col, lbl in [(acf_gbm,'steelblue','GBM'),(acf_hes,'darkorange','Heston'),(acf_rh,'crimson','RoughH'),(acf_real,'gray','Real SPY')]:
    ax.plot(lags[:40], a[:40], '-', color=col, lw=1.5, label=lbl)
ax.set_title('ACF of |returns|'); ax.legend(fontsize=7); ax.set_xlabel('Lag')

# 4: Sample paths
ax = fig.add_subplot(2, 3, 4)
for i in range(5):
    ax.plot(t_grid, heston_pp[i], alpha=0.3, lw=0.5, color='darkorange')
    ax.plot(t_grid, rh_pp[i], alpha=0.3, lw=0.5, color='crimson')
ax.plot([],[],color='darkorange',label='Heston'); ax.plot([],[],color='crimson',label='RoughH')
ax.set_title('Sample Price Paths'); ax.legend(fontsize=7)

# 5: Kurtosis vs H
ax = fig.add_subplot(2, 3, 5)
ax.plot(H_fine, kurt_h, 'ro-', lw=2, ms=5)
ax.axhline(stats.kurtosis(gbm_lr), color='steelblue', ls='--', lw=1, label='GBM')
ax.axhline(stats.kurtosis(hes_lr), color='darkorange', ls='--', lw=1, label='Heston')
ax.set_title('Excess Kurtosis vs H'); ax.set_xlabel('H'); ax.legend(fontsize=7)

# 6: Risk comparison
ax = fig.add_subplot(2, 3, 6)
x = np.arange(4); w = 0.25
ax.bar(x-w, g_vals, w, color='steelblue', alpha=0.8, label='GBM')
ax.bar(x, h_vals, w, color='darkorange', alpha=0.8, label='Heston')
ax.bar(x+w, r_vals, w, color='crimson', alpha=0.8, label='Rough Heston')
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_title('Risk Metrics'); ax.legend(fontsize=7)

fig.tight_layout()
fig.savefig(os.path.join(OUT, 'three_model_summary.png'), dpi=300)
plt.close(fig)

# ==================================================================
# Figure 9: Kernel approximation quality
# ==================================================================
print("Figure 9: Kernel approximation...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
t_arr = np.logspace(np.log10(1/252), np.log10(2), 500)

from scipy.special import gamma as scipy_gamma

def K_exact(t, H=0.1):
    return t**(H - 0.5) / scipy_gamma(H + 0.5)

for nf, ls in [(4, ':'), (6, '--'), (8, '-'), (10, '-.')]:
    # Compute spectral weights
    alpha = 0.5 - 0.1  # H=0.1
    lam_min, lam_max = 0.1, 200.0
    delta_u = (np.log(lam_max) - np.log(lam_min)) / nf
    gamma_prod = np.pi / np.sin(np.pi * alpha)
    lams = np.exp(np.log(lam_min) + (np.arange(nf) + 0.5) * delta_u)
    ws = lams**alpha * delta_u / gamma_prod
    K_approx = np.array([np.sum(ws * np.exp(-lams * t)) for t in t_arr])
    ax1.semilogx(t_arr, K_approx, ls, lw=1.5, label=f'{nf} factors')

K_ex = np.array([K_exact(t) for t in t_arr])
ax1.semilogx(t_arr, K_ex, 'k-', lw=2.5, label='Exact')
ax1.set_xlabel('t'); ax1.set_ylabel('K(t)'); ax1.set_title('Kernel Approximation (H=0.1)')
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

# Relative error
for nf, col in [(4, '#2196F3'), (6, '#FF9800'), (8, '#E91E63'), (10, '#9C27B0')]:
    alpha = 0.4; delta_u = (np.log(200) - np.log(0.1)) / nf
    gamma_prod = np.pi / np.sin(np.pi * alpha)
    lams = np.exp(np.log(0.1) + (np.arange(nf) + 0.5) * delta_u)
    ws = lams**alpha * delta_u / gamma_prod
    K_approx = np.array([np.sum(ws * np.exp(-lams * t)) for t in t_arr])
    rel_err = np.abs(K_ex - K_approx) / K_ex * 100
    ax2.semilogx(t_arr, rel_err, '-', color=col, lw=1.5, label=f'{nf} factors')
ax2.set_xlabel('t'); ax2.set_ylabel('Relative Error (%)')
ax2.set_title('Kernel Approximation Error'); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 50)

fig.tight_layout(); fig.savefig(os.path.join(OUT, 'kernel_approximation.png'), dpi=150); plt.close(fig)

print(f"\nAll figures saved to {OUT}/")
print("Done.")
