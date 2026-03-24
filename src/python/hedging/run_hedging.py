#!/usr/bin/env python3
"""Day 10: Dynamic hedging simulation and model risk quantification.

Uses calibrated Heston parameters (MLE from Part 1 if available, else Day 9 quick).
Generates figures showing hidden P&L risk from model misspecification.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(project_root, 'src', 'python'))
sys.path.insert(0, os.path.join(project_root, 'build', 'src', 'cpp'))

from hedging.dynamic_hedging import simulate_hedging, bs_price, bs_delta
from hedging.hedging_analysis import compute_hedging_metrics

TABLE_DIR = os.path.join(project_root, 'outputs', 'tables')
FIG_DIR = os.path.join(project_root, 'outputs', 'figures', 'hedging')
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': '#fafafa',
    'axes.grid': True, 'grid.alpha': 0.3, 'font.size': 10,
})

N_PATHS = 100_000
S0, K, T, R = 100.0, 100.0, 0.25, 0.05
N_STEPS = 63
BS_SIGMA = 0.20


def load_params():
    """Load calibrated params (MLE preferred, fallback to Day 9 quick)."""
    mle_path = os.path.join(TABLE_DIR, 'calibrated_params_mle.csv')
    quick_path = os.path.join(TABLE_DIR, 'calibrated_parameters.csv')

    heston_params = None
    rh_params = None

    if os.path.exists(mle_path):
        df = pd.read_csv(mle_path)
        h = df[(df['Asset'] == 'SPY') & (df['Model'] == 'heston')]
        if len(h) > 0:
            h = h.iloc[0]
            heston_params = dict(kappa=h['kappa'], theta=h['theta'],
                                 sigma_v=h['sigma_v'], rho=h['rho'],
                                 v0=h['v0'], mu=R)
        rh = df[(df['Asset'] == 'SPY') & (df['Model'] == 'rough_heston')]
        if len(rh) > 0:
            rh = rh.iloc[0]
            rh_params = dict(kappa=rh['kappa'], theta=rh['theta'],
                             sigma_v=rh['sigma_v'], rho=rh['rho'],
                             v0=rh['v0'], H=rh['H'], mu=R)

    if heston_params is None and os.path.exists(quick_path):
        df = pd.read_csv(quick_path)
        h = df[(df['Asset'] == 'SPY') & (df['Model'] == 'heston')]
        if len(h) > 0:
            h = h.iloc[0]
            heston_params = dict(kappa=h['kappa'], theta=h['theta'],
                                 sigma_v=h['sigma_v'], rho=h['rho'],
                                 v0=h['v0'], mu=R)
        rh = df[(df['Asset'] == 'SPY') & (df['Model'] == 'rough_heston')]
        if len(rh) > 0:
            rh = rh.iloc[0]
            rh_params = dict(kappa=rh['kappa'], theta=rh['theta'],
                             sigma_v=rh['sigma_v'], rho=rh['rho'],
                             v0=rh['v0'], H=rh['H'], mu=R)

    if heston_params is None:
        heston_params = dict(kappa=2.0, theta=0.04, sigma_v=0.3,
                             rho=-0.7, v0=0.04, mu=R)
    if rh_params is None:
        rh_params = dict(kappa=2.0, theta=0.04, sigma_v=0.3,
                         rho=-0.7, v0=0.04, H=0.1, mu=R)

    return heston_params, rh_params


def main():
    print("=" * 70)
    print("Day 10: Dynamic Hedging Simulation")
    print("=" * 70)

    heston_params, rh_params = load_params()
    gbm_params = dict(mu=R, sigma=BS_SIGMA)

    print(f"\nOption: S0={S0}, K={K}, T={T}, r={R}")
    premium = bs_price(S0, K, T, R, BS_SIGMA)
    print(f"BS price (sigma={BS_SIGMA}): ${premium:.2f}")
    print(f"Heston params: {heston_params}")
    print(f"Rough Heston params: {rh_params}")

    # ── Run hedging simulations ──
    print(f"\nSimulating {N_PATHS} paths per scenario...")

    # Scenario 1: GBM true, BS constant hedge
    print("  [1/5] GBM true + BS constant vol hedge...")
    gbm_const = simulate_hedging('gbm', gbm_params, BS_SIGMA,
                                 N_PATHS, S0, K, T, R, N_STEPS, 'constant')

    # Scenario 2: Heston true, BS constant hedge
    print("  [2/5] Heston true + BS constant vol hedge...")
    hes_const = simulate_hedging('heston', heston_params, BS_SIGMA,
                                 N_PATHS, S0, K, T, R, N_STEPS, 'constant')

    # Scenario 3: Heston true, BS local vol hedge
    print("  [3/5] Heston true + BS local vol hedge...")
    hes_local = simulate_hedging('heston', heston_params, BS_SIGMA,
                                 N_PATHS, S0, K, T, R, N_STEPS, 'local')

    # Scenario 4: Rough Heston true, BS constant hedge
    print("  [4/5] Rough Heston true + BS constant vol hedge...")
    rh_const = simulate_hedging('rough_heston', rh_params, BS_SIGMA,
                                N_PATHS, S0, K, T, R, N_STEPS, 'constant')

    # Scenario 5: Rough Heston true, BS local vol hedge
    print("  [5/5] Rough Heston true + BS local vol hedge...")
    rh_local = simulate_hedging('rough_heston', rh_params, BS_SIGMA,
                                N_PATHS, S0, K, T, R, N_STEPS, 'local')

    # ── Compute metrics ──
    scenarios = [
        (gbm_const, 'GBM true', 'constant'),
        (hes_const, 'Heston true', 'constant'),
        (hes_local, 'Heston true', 'local vol'),
        (rh_const, 'Rough Heston true', 'constant'),
        (rh_local, 'Rough Heston true', 'local vol'),
    ]

    metrics = []
    for result, label, strat in scenarios:
        m = compute_hedging_metrics(result['total_pnl'], premium, label, strat)
        metrics.append(m)
        print(f"\n  {label} ({strat}):")
        print(f"    Mean P&L: ${m.mean_pnl:.3f}, Std: ${m.std_pnl:.3f}")
        print(f"    Skew: {m.skewness:.3f}, Kurt: {m.kurtosis:.3f}")
        print(f"    VaR99: ${m.var_99:.3f}, ES99: ${m.es_99:.3f}")
        print(f"    P(loss > 2x premium): {m.prob_loss_2x_premium:.2%}")

    # ── Save table ──
    rows = []
    for m in metrics:
        rows.append({
            'Scenario': m.label, 'Strategy': m.strategy,
            'Premium': m.option_premium,
            'PnL_Mean': m.mean_pnl, 'PnL_Std': m.std_pnl,
            'Skewness': m.skewness, 'Kurtosis': m.kurtosis,
            'VaR_99': m.var_99, 'ES_99': m.es_99,
            'P_loss_2x': m.prob_loss_2x_premium,
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(TABLE_DIR, 'hedging_results.csv'), index=False
    )

    # ── Figures ──
    print(f"\n{'='*60}")
    print("Generating Figures")
    print(f"{'='*60}")

    plot_pnl_distributions(gbm_const, hes_const, rh_const, metrics)
    plot_pnl_paths(gbm_const, hes_const)
    plot_hedging_model_risk(metrics)
    plot_strategy_comparison(hes_const, hes_local, rh_const, rh_local, metrics)
    plot_hidden_risk(gbm_const, hes_const, rh_const)
    plot_summary_dashboard(gbm_const, hes_const, rh_const, hes_local,
                           metrics, heston_params, rh_params)

    print(f"\nAll figures saved to {FIG_DIR}")
    print(f"Table saved to {TABLE_DIR}/hedging_results.csv")
    print("=" * 70)


# ──────────────────────────────────────────────────────────────────
# FIGURE FUNCTIONS
# ──────────────────────────────────────────────────────────────────

def plot_pnl_distributions(gbm, heston, rough, metrics):
    """Figure 1: Hedge P&L distributions overlaid."""
    fig, ax = plt.subplots(figsize=(12, 7))

    bins = np.linspace(-15, 10, 300)
    ax.hist(gbm['total_pnl'], bins=bins, density=True, alpha=0.5,
            color='#2196F3', label=f'GBM true (std=${metrics[0].std_pnl:.2f})')
    ax.hist(heston['total_pnl'], bins=bins, density=True, alpha=0.5,
            color='#FF5722', label=f'Heston true (std=${metrics[1].std_pnl:.2f})')
    ax.hist(rough['total_pnl'], bins=bins, density=True, alpha=0.4,
            color='#4CAF50', label=f'Rough Heston true (std=${metrics[3].std_pnl:.2f})')

    ax.axvline(0, color='black', ls='--', lw=1.5, label='Break-even')
    ax.set_xlabel('Hedge P&L ($)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Delta-Hedge P&L Distribution Under Different True Dynamics',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)

    ax.text(0.02, 0.95,
            f'Option premium: ${metrics[0].option_premium:.2f}\n'
            f'Hidden risk (Heston ES99): ${metrics[1].es_99:.2f}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow'))

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'hedge_pnl_distributions.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: hedge_pnl_distributions.png")


def plot_pnl_paths(gbm, heston):
    """Figure 2: Cumulative P&L paths."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    n_show = 50
    t = np.arange(N_STEPS)

    ax = axes[0]
    for i in range(n_show):
        ax.plot(t, gbm['cum_hedge_pnl'][i] - gbm['option_payoff'][i] * t / N_STEPS + gbm['option_premium'],
                color='#2196F3', alpha=0.2, lw=0.5)
    ax.axhline(0, color='black', ls='--', lw=0.8)
    ax.set_title('GBM True: Hedge P&L Paths (tight around zero)', fontsize=11)
    ax.set_ylabel('Cumulative P&L ($)')

    ax = axes[1]
    for i in range(n_show):
        ax.plot(t, heston['cum_hedge_pnl'][i] - heston['option_payoff'][i] * t / N_STEPS + heston['option_premium'],
                color='#FF5722', alpha=0.2, lw=0.5)
    ax.axhline(0, color='black', ls='--', lw=0.8)
    ax.set_title('Heston True: Hedge P&L Paths (wider dispersion)', fontsize=11)
    ax.set_ylabel('Cumulative P&L ($)')
    ax.set_xlabel('Trading Day')

    fig.suptitle('Hedge P&L Path Comparison', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'hedge_pnl_paths.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: hedge_pnl_paths.png")


def plot_hedging_model_risk(metrics):
    """Figure 3: Model risk in hedging — bar chart."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    labels = [f'{m.label}\n({m.strategy})' for m in metrics]
    colors = ['#2196F3', '#FF5722', '#FF9800', '#4CAF50', '#8BC34A']

    for ax, attr, title in zip(axes,
                                ['std_pnl', 'skewness', 'var_99', 'es_99'],
                                ['P&L Std ($)', 'P&L Skewness', 'VaR 99% ($)', 'ES 99% ($)']):
        vals = [getattr(m, attr) for m in metrics]
        ax.bar(range(len(vals)), vals, color=colors)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=7)
        ax.set_title(title, fontsize=11, fontweight='bold')

    fig.suptitle('Model Risk in Delta Hedging', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'hedging_model_risk.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: hedging_model_risk.png")


def plot_strategy_comparison(hes_const, hes_local, rh_const, rh_local, metrics):
    """Figure 4: Hedging strategy comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    bins = np.linspace(-15, 10, 200)

    ax = axes[0, 0]
    ax.hist(hes_const['total_pnl'], bins=bins, density=True, alpha=0.6,
            color='#FF5722', label='Constant vol')
    ax.axvline(0, color='black', ls='--', lw=0.8)
    ax.set_title('Heston: Constant Vol Hedge', fontsize=11)
    ax.legend()

    ax = axes[0, 1]
    ax.hist(hes_local['total_pnl'], bins=bins, density=True, alpha=0.6,
            color='#FF9800', label='Local vol')
    ax.axvline(0, color='black', ls='--', lw=0.8)
    ax.set_title('Heston: Local Vol Hedge', fontsize=11)
    ax.legend()

    # P&L std vs rebalancing frequency (use Heston constant as base)
    ax = axes[1, 0]
    # Approximate: subsample the hedging by skipping steps
    freqs = [1, 2, 5, 10, 21]
    freq_labels = ['Daily', '2-day', 'Weekly', 'Bi-weekly', 'Monthly']
    stds = []
    for freq in freqs:
        # Re-compute hedge with fewer rebalancing steps
        pnl = _rebalanced_pnl(hes_const, freq)
        stds.append(np.std(pnl, ddof=1))
    ax.bar(range(len(freqs)), stds, color='#FF5722', alpha=0.8)
    ax.set_xticks(range(len(freqs)))
    ax.set_xticklabels(freq_labels, rotation=20)
    ax.set_ylabel('P&L Std ($)')
    ax.set_title('P&L Std vs Rebalancing Frequency', fontsize=11)

    # P&L std vs sigma_v
    ax = axes[1, 1]
    m_const = [m for m in metrics if m.strategy == 'constant']
    m_local = [m for m in metrics if m.strategy == 'local vol']
    labels_c = [m.label for m in m_const]
    ax.bar(np.arange(len(m_const)) - 0.2,
           [m.std_pnl for m in m_const], 0.4,
           label='Constant vol', color='#FF5722')
    ax.bar(np.arange(len(m_local)) + 0.2,
           [m.std_pnl for m in m_local], 0.4,
           label='Local vol', color='#FF9800')
    ax.set_xticks(range(len(m_const)))
    ax.set_xticklabels(labels_c, fontsize=8)
    ax.set_ylabel('P&L Std ($)')
    ax.set_title('Constant vs Local Vol Hedge', fontsize=11)
    ax.legend(fontsize=8)

    fig.suptitle('Hedging Strategy Comparison', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'hedging_strategy_comparison.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: hedging_strategy_comparison.png")


def plot_hidden_risk(gbm, heston, rough):
    """Figure 5: THE KEY FIGURE — Hidden risk percentile plot."""
    fig, ax = plt.subplots(figsize=(12, 7))

    percentiles = np.arange(0.1, 50.1, 0.5)

    gbm_q = np.percentile(gbm['total_pnl'], percentiles)
    hes_q = np.percentile(heston['total_pnl'], percentiles)
    rh_q = np.percentile(rough['total_pnl'], percentiles)

    ax.plot(percentiles, gbm_q, color='#2196F3', lw=2.5, label='GBM true')
    ax.plot(percentiles, hes_q, color='#FF5722', lw=2.5, label='Heston true')
    ax.plot(percentiles, rh_q, color='#4CAF50', lw=2.5, label='Rough Heston true')

    ax.fill_between(percentiles, gbm_q, hes_q, alpha=0.15, color='#FF5722',
                    label='Hidden risk (Heston)')

    ax.axhline(0, color='black', ls='--', lw=0.8)
    ax.set_xlabel('Percentile of P&L Distribution (%)', fontsize=12)
    ax.set_ylabel('P&L ($)', fontsize=12)
    ax.set_title('Hidden Risk: The Cost of Assuming Constant Volatility',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)

    # Annotate the gap at 1st percentile
    gap_1 = gbm_q[2] - hes_q[2]  # percentile=1.0
    ax.annotate(
        f'Hidden loss at 1st pctile: ${abs(gap_1):.2f}\n'
        f'on a ${S0} notional position',
        xy=(1.0, hes_q[2]), xytext=(10, hes_q[2] - 2),
        fontsize=10,
        arrowprops=dict(arrowstyle='->', color='red'),
        bbox=dict(boxstyle='round', facecolor='lightyellow'),
    )

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'hidden_risk.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: hidden_risk.png (KEY FIGURE)")


def plot_summary_dashboard(gbm, heston, rough, hes_local,
                           metrics, heston_params, rh_params):
    """Figure 6: 2x3 summary dashboard."""
    fig = plt.figure(figsize=(20, 13))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 1. P&L distributions
    ax = fig.add_subplot(gs[0, 0])
    bins = np.linspace(-15, 10, 150)
    ax.hist(gbm['total_pnl'], bins=bins, density=True, alpha=0.5,
            color='#2196F3', label='GBM')
    ax.hist(heston['total_pnl'], bins=bins, density=True, alpha=0.5,
            color='#FF5722', label='Heston')
    ax.axvline(0, color='black', ls='--', lw=0.8)
    ax.legend(fontsize=8)
    ax.set_title('Hedge P&L Distributions', fontsize=10, fontweight='bold')

    # 2. Hidden risk
    ax = fig.add_subplot(gs[0, 1])
    pctiles = np.arange(0.5, 30, 1)
    ax.plot(pctiles, np.percentile(gbm['total_pnl'], pctiles),
            color='#2196F3', lw=2, label='GBM')
    ax.plot(pctiles, np.percentile(heston['total_pnl'], pctiles),
            color='#FF5722', lw=2, label='Heston')
    ax.fill_between(pctiles,
                    np.percentile(gbm['total_pnl'], pctiles),
                    np.percentile(heston['total_pnl'], pctiles),
                    alpha=0.2, color='#FF5722')
    ax.axhline(0, color='black', ls='--', lw=0.5)
    ax.legend(fontsize=8)
    ax.set_title('Hidden Risk (shaded)', fontsize=10, fontweight='bold')
    ax.set_xlabel('Percentile')

    # 3. Strategy comparison
    ax = fig.add_subplot(gs[0, 2])
    m_const = [m for m in metrics if m.label == 'Heston true' and m.strategy == 'constant']
    m_local = [m for m in metrics if m.label == 'Heston true' and m.strategy == 'local vol']
    if m_const and m_local:
        vals = [m_const[0].std_pnl, m_local[0].std_pnl]
        ax.bar(['Constant vol', 'Local vol'], vals, color=['#FF5722', '#FF9800'])
        ax.set_ylabel('P&L Std ($)')
    ax.set_title('Hedge Strategy Comparison', fontsize=10, fontweight='bold')

    # 4. P&L paths
    ax = fig.add_subplot(gs[1, 0])
    for i in range(30):
        ax.plot(range(N_STEPS), gbm['cum_hedge_pnl'][i],
                color='#2196F3', alpha=0.15, lw=0.5)
        ax.plot(range(N_STEPS), heston['cum_hedge_pnl'][i],
                color='#FF5722', alpha=0.15, lw=0.5)
    ax.axhline(0, color='black', ls='--', lw=0.5)
    ax.set_title('Cumulative Hedge P&L Paths', fontsize=10, fontweight='bold')
    ax.set_xlabel('Trading Day')

    # 5. Risk metrics table
    ax = fig.add_subplot(gs[1, 1])
    ax.axis('off')
    cell_text = []
    for m in metrics[:3]:  # GBM, Heston const, RH const
        cell_text.append([
            f'{m.label}',
            f'${m.std_pnl:.2f}',
            f'{m.skewness:.2f}',
            f'${m.es_99:.2f}',
        ])
    table = ax.table(cellText=cell_text,
                     colLabels=['Scenario', 'Std', 'Skew', 'ES99'],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    ax.set_title('Risk Metrics Summary', fontsize=10, fontweight='bold')

    # 6. Parameters
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    param_text = (
        f"Heston (SPY calibrated):\n"
        f"  kappa={heston_params['kappa']:.3f}, theta={heston_params['theta']:.5f}\n"
        f"  sigma_v={heston_params['sigma_v']:.3f}, rho={heston_params['rho']:.3f}\n"
        f"  v0={heston_params['v0']:.5f}\n\n"
        f"Rough Heston:\n"
        f"  H={rh_params.get('H', 0.1):.3f}\n"
        f"  (other params same structure)\n\n"
        f"BS hedge sigma: {BS_SIGMA}"
    )
    ax.text(0.1, 0.9, param_text, transform=ax.transAxes, fontsize=9,
            va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax.set_title('Calibrated Parameters', fontsize=10, fontweight='bold')

    fig.suptitle('Calibration and Dynamic Hedging: Model Risk Has Real Consequences',
                 fontsize=16, fontweight='bold', y=0.98)
    fig.savefig(os.path.join(FIG_DIR, 'calibration_hedging_summary.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: calibration_hedging_summary.png (300 DPI)")


def _rebalanced_pnl(result, freq):
    """Approximate P&L with less frequent rebalancing."""
    paths = result['paths']
    n_paths, n_total = paths.shape[0], paths.shape[1] - 1
    pnl = np.zeros(n_paths)
    dt = T / n_total

    for t in range(0, n_total, freq):
        S_t = paths[:, t]
        end = min(t + freq, n_total)
        S_end = paths[:, end]
        tau = T - t * dt
        if tau > 1e-10:
            delta = bs_delta(S_t, K, tau, R, BS_SIGMA)
        else:
            delta = np.where(S_t > K, 1.0, 0.0)
        pnl += delta * (S_end - S_t)

    S_T = paths[:, -1]
    payoff = np.maximum(S_T - K, 0)
    premium = bs_price(S0, K, T, R, BS_SIGMA)
    return premium + pnl - payoff


if __name__ == '__main__':
    main()
