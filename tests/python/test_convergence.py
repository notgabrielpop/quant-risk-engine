#!/usr/bin/env python3
"""
Day 7: Convergence analysis — Sobol QMC, Control Variates, Importance Sampling.

Generates:
  1. convergence_comparison.png      — MC vs QMC mean convergence (KEY FIGURE)
  2. es_convergence_comparison.png   — ES convergence
  3. variance_reduction_summary.png  — VR factor bar chart
  4. importance_sampling_diagnostics.png — IS weight/ESS diagnostics
  5. sobol_vs_mc_scatter.png         — 2D point distribution
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'src', 'cpp'))
import quant_engine_py as qe
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', 'figures', 'engine')
os.makedirs(OUT, exist_ok=True)

# Reference params
def make_gbm():
    p = qe.GBMParams(); p.S0 = 100; p.mu = 0.05; p.sigma = 0.2; p.T = 1.0; p.n_steps = 252
    return p

def make_heston():
    p = qe.HestonParams(); p.S0 = 100; p.mu = 0.05; p.T = 1.0; p.n_steps = 252
    p.v0 = 0.04; p.kappa = 2.0; p.theta = 0.04; p.sigma_v = 0.3; p.rho = -0.7
    return p

def make_config(n, seed=42):
    c = qe.SimConfig(); c.n_paths = n; c.n_steps = 252; c.seed = seed; c.n_threads = 0
    return c

gp = make_gbm()
hp = make_heston()
analytical_mean = gp.S0 * np.exp(gp.mu * gp.T)

print("Analytical GBM mean:", analytical_mean)

# ==================================================================
# Figure 1: THE CONVERGENCE PLOT (KEY FIGURE)
# ==================================================================
print("Figure 1: Convergence comparison...")
Ns = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]
n_reps = 32

mc_errors = []
mc_anti_errors = []
qmc_errors = []

for N in Ns:
    mc_errs = []
    anti_errs = []
    qmc_errs = []

    for r in range(n_reps):
        cfg = make_config(N, seed=1000 + r)

        # MC
        prices = qe.simulate_gbm(gp, cfg)
        mc_errs.append(abs(np.mean(prices) - analytical_mean))

        # MC + Antithetic
        cfg.antithetic = True
        # Use the standard simulate (antithetic would need separate binding)
        # Instead, use the standard MC with half paths paired
        cfg.antithetic = False  # We don't have antithetic binding yet — skip

        # QMC (Sobol with scrambling)
        prices_qmc = qe.simulate_gbm_sobol(gp, cfg, scramble_seed=r + 1)
        qmc_errs.append(abs(np.mean(prices_qmc) - analytical_mean))

    mc_errors.append(mc_errs)
    qmc_errors.append(qmc_errs)
    print(f"  N={N:>7d}: MC={np.mean(mc_errs):.6f}  QMC={np.mean(qmc_errs):.6f}")

mc_means = [np.mean(e) for e in mc_errors]
mc_q25   = [np.percentile(e, 25) for e in mc_errors]
mc_q75   = [np.percentile(e, 75) for e in mc_errors]
qmc_means = [np.mean(e) for e in qmc_errors]
qmc_q25   = [np.percentile(e, 25) for e in qmc_errors]
qmc_q75   = [np.percentile(e, 75) for e in qmc_errors]

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
ax.fill_between(Ns, mc_q25, mc_q75, alpha=0.15, color='#2196F3')
ax.loglog(Ns, mc_means, 'o-', color='#2196F3', lw=2, ms=6, label='MC (Mersenne Twister)')
ax.fill_between(Ns, qmc_q25, qmc_q75, alpha=0.15, color='#E91E63')
ax.loglog(Ns, qmc_means, 's-', color='#E91E63', lw=2, ms=6, label='QMC (Sobol)')

# Reference slopes
N_ref = np.array([Ns[0], Ns[-1]])
mc_ref = mc_means[0] * (N_ref / N_ref[0])**(-0.5)
qmc_ref = qmc_means[0] * (N_ref / N_ref[0])**(-1.0)
ax.loglog(N_ref, mc_ref, '--', color='gray', lw=1, alpha=0.7, label=r'$O(1/\sqrt{N})$ reference')
ax.loglog(N_ref, qmc_ref, ':', color='gray', lw=1, alpha=0.7, label=r'$O(1/N)$ reference')

ax.set_xlabel('Number of Paths (N)', fontsize=13)
ax.set_ylabel('Absolute Error in Mean Price', fontsize=13)
ax.set_title('Monte Carlo vs Quasi-Monte Carlo Convergence (GBM)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, which='both')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'convergence_comparison.png'), dpi=150)
plt.close(fig)

# ==================================================================
# Figure 2: ES Convergence
# ==================================================================
print("Figure 2: ES convergence...")
# Reference ES from high-N run
cfg_ref = make_config(2000000, seed=9999)
prices_ref = qe.simulate_gbm(gp, cfg_ref)
rm_ref = qe.compute_risk(prices_ref, gp.S0)
es99_ref = rm_ref.es_99
print(f"  Reference ES99 = {es99_ref:.4f}")

Ns_es = [5000, 10000, 20000, 50000, 100000, 200000, 500000]
n_reps_es = 16

mc_es_errors = []
qmc_es_errors = []

for N in Ns_es:
    mc_errs = []
    qmc_errs = []
    for r in range(n_reps_es):
        cfg = make_config(N, seed=2000 + r)

        prices_mc = qe.simulate_gbm(gp, cfg)
        rm_mc = qe.compute_risk(prices_mc, gp.S0)
        mc_errs.append(abs(rm_mc.es_99 - es99_ref))

        prices_qmc = qe.simulate_gbm_sobol(gp, cfg, scramble_seed=r + 1)
        rm_qmc = qe.compute_risk(prices_qmc, gp.S0)
        qmc_errs.append(abs(rm_qmc.es_99 - es99_ref))

    mc_es_errors.append(mc_errs)
    qmc_es_errors.append(qmc_errs)
    print(f"  N={N:>7d}: MC={np.mean(mc_errs):.4f}  QMC={np.mean(qmc_errs):.4f}")

mc_es_means = [np.mean(e) for e in mc_es_errors]
qmc_es_means = [np.mean(e) for e in qmc_es_errors]

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
ax.loglog(Ns_es, mc_es_means, 'o-', color='#2196F3', lw=2, ms=6, label='MC')
ax.loglog(Ns_es, qmc_es_means, 's-', color='#E91E63', lw=2, ms=6, label='QMC (Sobol)')

N_ref = np.array([Ns_es[0], Ns_es[-1]])
mc_slope = mc_es_means[0] * (N_ref / N_ref[0])**(-0.5)
ax.loglog(N_ref, mc_slope, '--', color='gray', lw=1, alpha=0.7, label=r'$O(1/\sqrt{N})$')

ax.set_xlabel('Number of Paths (N)', fontsize=13)
ax.set_ylabel('Absolute Error in ES₉₉', fontsize=13)
ax.set_title('ES₉₉ Convergence: MC vs QMC', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, which='both')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'es_convergence_comparison.png'), dpi=150)
plt.close(fig)

# ==================================================================
# Figure 3: Variance Reduction Summary
# ==================================================================
print("Figure 3: Variance reduction summary...")
N_vr = 100000
n_reps_vr = 16

# Collect mean-price standard errors for each method
methods = {}

# MC
ses = []
for r in range(n_reps_vr):
    cfg = make_config(N_vr, seed=3000 + r)
    p = qe.simulate_gbm(gp, cfg)
    ses.append(np.std(p) / np.sqrt(len(p)))
methods['MC'] = np.mean(ses)

# QMC
means = []
for r in range(n_reps_vr):
    cfg = make_config(N_vr, seed=3000 + r)
    p = qe.simulate_gbm_sobol(gp, cfg, scramble_seed=r + 1)
    means.append(np.mean(p))
methods['QMC'] = np.std(means) / np.sqrt(n_reps_vr) if n_reps_vr > 1 else methods['MC']

# CV (Heston)
gp_cv = make_gbm()
gp_cv.sigma = np.sqrt(hp.v0)
ses_std = []
ses_cv = []
for r in range(n_reps_vr):
    cfg = make_config(N_vr, seed=3000 + r)
    # Standard Heston
    p = qe.simulate_heston(hp, cfg)
    ses_std.append(np.std(p) / np.sqrt(len(p)))
    # CV Heston
    cv = qe.simulate_heston_with_cv(hp, gp_cv, cfg)
    cp = np.array(cv['controlled_prices'])
    ses_cv.append(np.std(cp) / np.sqrt(len(cp)))

methods['Heston MC'] = np.mean(ses_std)
methods['Heston+CV'] = np.mean(ses_cv)

# VR factors relative to MC
mc_baseline = methods['MC']
heston_baseline = methods['Heston MC']

labels = ['MC', 'QMC\n(Sobol)', 'Heston\nMC', 'Heston\n+CV']
values = [1.0,
          mc_baseline / methods['QMC'] if methods['QMC'] > 0 else 1,
          1.0,
          heston_baseline / methods['Heston+CV'] if methods['Heston+CV'] > 0 else 1]
colors = ['#90CAF9', '#E91E63', '#90CAF9', '#4CAF50']

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, values, color=colors, edgecolor='white', lw=1.5)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{val:.1f}x', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('Variance Reduction Factor (vs baseline)', fontsize=12)
ax.set_title('Variance Reduction: Standard Error Improvement', fontsize=14, fontweight='bold')
ax.axhline(y=1.0, color='black', lw=0.8, ls='--', alpha=0.3)
ax.set_ylim(0, max(values) * 1.3)
ax.grid(True, alpha=0.2, axis='y')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'variance_reduction_summary.png'), dpi=150)
plt.close(fig)

# ==================================================================
# Figure 4: Importance Sampling Diagnostics
# ==================================================================
print("Figure 4: IS diagnostics...")
N_is = 200000
cfg_is = make_config(N_is, seed=5000)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Top-left: IS weight histogram
is_result = qe.simulate_gbm_is(gp, cfg_is, mu_is=-2.0)
weights = np.array(is_result['weights'])
axes[0,0].hist(weights, bins=100, color='#2196F3', alpha=0.7, edgecolor='white')
axes[0,0].set_xlabel('Likelihood Ratio Weight')
axes[0,0].set_ylabel('Count')
axes[0,0].set_title(f'IS Weight Distribution (μ_IS=-2.0, ESS={is_result["ess_ratio"]*100:.1f}%)')
axes[0,0].set_xlim(0, np.percentile(weights, 99.5))

# Top-right: ESS ratio vs mu_IS
shifts = np.arange(-4.0, 0.1, 0.25)
ess_ratios = []
for mu in shifts:
    r = qe.simulate_gbm_is(gp, make_config(50000, seed=5001), mu_is=float(mu))
    ess_ratios.append(r['ess_ratio'])
axes[0,1].plot(shifts, ess_ratios, 'o-', color='#E91E63', lw=2)
axes[0,1].set_xlabel('μ_IS (shift)')
axes[0,1].set_ylabel('ESS / N')
axes[0,1].set_title('Effective Sample Size vs Shift')
axes[0,1].axhline(y=0.05, color='red', ls='--', lw=1, alpha=0.5, label='ESS=5% threshold')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Bottom-left: ES estimate vs mu_IS (unbiasedness)
shifts2 = [-0.5, -1.0, -1.5, -2.0, -2.5, -3.0]
es_estimates = []
for mu in shifts2:
    r = qe.simulate_gbm_is(gp, make_config(200000, seed=5002), mu_is=mu)
    prices = np.array(r['terminal_prices'])
    w = np.array(r['weights'])
    # Weighted mean
    wmean = np.sum(w * prices) / np.sum(w)
    es_estimates.append(wmean)

# Standard MC reference
ref_prices = qe.simulate_gbm(gp, make_config(200000, seed=5002))
ref_mean = np.mean(ref_prices)

axes[1,0].plot(shifts2, es_estimates, 'o-', color='#4CAF50', lw=2, ms=8, label='IS estimate')
axes[1,0].axhline(y=ref_mean, color='#2196F3', ls='--', lw=2, label=f'MC reference ({ref_mean:.2f})')
axes[1,0].set_xlabel('μ_IS (shift)')
axes[1,0].set_ylabel('Weighted Mean Price')
axes[1,0].set_title('Unbiasedness Check: IS Mean vs Shift')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Bottom-right: standard error of weighted mean vs mu_IS
n_reps_is = 16
shifts3 = [-0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5]
se_by_shift = []
for mu in shifts3:
    means = []
    for r in range(n_reps_is):
        res = qe.simulate_gbm_is(gp, make_config(50000, seed=6000 + r), mu_is=mu)
        w = np.array(res['weights'])
        p = np.array(res['terminal_prices'])
        means.append(np.sum(w * p) / np.sum(w))
    se_by_shift.append(np.std(means))

axes[1,1].plot(shifts3, se_by_shift, 'o-', color='#FF9800', lw=2, ms=8)
axes[1,1].set_xlabel('μ_IS (shift)')
axes[1,1].set_ylabel('Std Dev of Weighted Mean (across reps)')
axes[1,1].set_title('IS Efficiency: Variance vs Shift')
axes[1,1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUT, 'importance_sampling_diagnostics.png'), dpi=150)
plt.close(fig)

# ==================================================================
# Figure 5: Sobol vs MC scatter
# ==================================================================
print("Figure 5: Sobol vs MC scatter...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# MC: 1000 uniform points in 2D
rng = np.random.RandomState(42)
mc_pts = rng.rand(1000, 2)
ax1.scatter(mc_pts[:, 0], mc_pts[:, 1], s=3, alpha=0.5, color='#2196F3')
ax1.set_title('Pseudo-Random (Mersenne Twister)', fontsize=13, fontweight='bold')
ax1.set_xlabel('Dimension 1'); ax1.set_ylabel('Dimension 2')
ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
ax1.set_aspect('equal')

# Sobol: generate 1000 points in 2D via Python
# Use a simple van der Corput + Sobol-like sequence
# Actually, let's use the C++ Sobol via simulation trick:
# Generate 1000 GBM paths with 2 steps — use Sobol uniform
# For simplicity, generate 2D Sobol manually
def van_der_corput(n, base=2):
    seq = np.zeros(n)
    for i in range(1, n + 1):
        f, r = 1.0, 0.0
        num = i
        while num > 0:
            f /= base
            r += f * (num % base)
            num //= base
        seq[i-1] = r
    return seq

sobol_x = van_der_corput(1000, base=2)
sobol_y = van_der_corput(1000, base=3)
ax2.scatter(sobol_x, sobol_y, s=3, alpha=0.5, color='#E91E63')
ax2.set_title('Low-Discrepancy (Sobol/Halton)', fontsize=13, fontweight='bold')
ax2.set_xlabel('Dimension 1'); ax2.set_ylabel('Dimension 2')
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
ax2.set_aspect('equal')

fig.suptitle('Point Distribution: Random vs Low-Discrepancy', fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'sobol_vs_mc_scatter.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"\nAll figures saved to {OUT}")
print("Done.")
