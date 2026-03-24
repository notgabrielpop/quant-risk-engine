#!/usr/bin/env python3
"""Day 8: Multi-asset portfolio risk with vine copulas.

Runs the full analysis pipeline:
1. Marginal fitting and validation
2. Vine copula estimation
3. Multi-asset simulation under 4 dependence assumptions
4. Portfolio risk metrics and Euler allocation
5. Regime-conditional analysis
6. Generates 10 figures and 5 CSV tables
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
from scipy import stats

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(project_root, 'src', 'python'))

from copula.marginals import fit_marginals, get_uniform_data, marginal_fits_table
from copula.vine_copula import fit_vine_copula, simulate_vine, vine_structure_table, get_pair_copula_matrix
from copula.multi_asset_simulator import MultiAssetSimulator
from copula.copula_families import family_name, tail_dependence_bicop, FAMILY_NAMES
from copula.regime_copula import (
    load_vix, fit_regime_copulas, regime_tail_dependence,
    rolling_copula_analysis, classify_regimes
)
from risk.portfolio_risk import (
    portfolio_returns, compute_risk_metrics, compute_es_at_quantiles,
    individual_es, risk_comparison_table
)
from risk.euler_allocation import euler_es_allocation, allocation_table

import pyvinecopulib as pv

# Paths
DATA_DIR = os.path.join(project_root, 'data', 'processed')
FIG_DIR = os.path.join(project_root, 'outputs', 'figures', 'copula')
TABLE_DIR = os.path.join(project_root, 'outputs', 'tables')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

# Portfolio definitions
EQUAL_WEIGHTS = None  # set after loading data
MKTCAP_WEIGHTS = np.array([0.05, 0.30, 0.15, 0.15, 0.10, 0.25])

N_SCENARIOS = 500_000
SEED = 42

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 10,
})

COLORS = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800', '#607D8B']


def main():
    print("=" * 70)
    print("Day 8: Multi-Asset Portfolio Risk with Vine Copulas")
    print("=" * 70)

    # ── Load data ──
    print("\n[1/8] Loading portfolio returns...")
    returns = pd.read_csv(
        os.path.join(DATA_DIR, 'portfolio_returns.csv'),
        parse_dates=['Date'], index_col='Date'
    )
    asset_names = list(returns.columns)
    d = len(asset_names)
    global EQUAL_WEIGHTS
    EQUAL_WEIGHTS = np.ones(d) / d
    print(f"  Assets: {asset_names}")
    print(f"  Observations: {len(returns)}")
    print(f"  Date range: {returns.index[0].date()} to {returns.index[-1].date()}")

    # ── Part 1: Marginal fitting ──
    print("\n[2/8] Fitting marginal distributions...")
    marginal_results = fit_marginals(returns)
    marg_table = marginal_fits_table(marginal_results)
    marg_table.to_csv(os.path.join(TABLE_DIR, 'marginal_fits.csv'), index=False)
    print(marg_table[['Asset', 't_df', 't_AIC', 'norm_AIC', 'Best',
                       't_KS_pvalue']].to_string(index=False))

    u_data = get_uniform_data(marginal_results, asset_names)
    plot_marginal_diagnostics(returns, marginal_results, asset_names)

    # ── Part 2: Vine copula estimation ──
    print("\n[3/8] Fitting vine copula...")
    vine = fit_vine_copula(u_data, trunc_lvl=5)
    print(f"  Vine structure:\n{vine.format()}")

    vtable = vine_structure_table(vine, asset_names)
    vtable.to_csv(os.path.join(TABLE_DIR, 'vine_copula_structure.csv'), index=False)
    print(f"  Vine edges saved ({len(vtable)} pairs)")

    plot_copula_families_heatmap(vine, asset_names, d)
    plot_vine_structure(vine, asset_names, d)

    # ── Part 3: Multi-asset simulation ──
    print("\n[4/8] Simulating under 4 dependence assumptions...")
    sim = MultiAssetSimulator(returns)
    sim.marginal_results = marginal_results
    sim.u_data = u_data
    sim.vine_copula = vine

    r_vine = sim.simulate(N_SCENARIOS, seed=SEED)
    r_gauss = sim.simulate_gaussian(N_SCENARIOS, seed=SEED)
    r_indep = sim.simulate_independent(N_SCENARIOS, seed=SEED)
    r_como = sim.simulate_comonotonic(N_SCENARIOS, seed=SEED)

    scenarios = {
        'Independent': r_indep,
        'Gaussian': r_gauss,
        'Vine Copula': r_vine,
        'Comonotonic': r_como,
    }

    # ── Part 4: Portfolio risk metrics ──
    print("\n[5/8] Computing portfolio risk metrics...")
    risk_results_eq = {}
    risk_results_mc = {}
    for label, r_sim in scenarios.items():
        rp_eq = portfolio_returns(r_sim, EQUAL_WEIGHTS)
        rp_mc = portfolio_returns(r_sim, MKTCAP_WEIGHTS)
        risk_results_eq[label] = compute_risk_metrics(rp_eq)
        risk_results_mc[label] = compute_risk_metrics(rp_mc)

    # Print comparison
    eq_table = risk_comparison_table(risk_results_eq)
    mc_table = risk_comparison_table(risk_results_mc)
    print("\n  Equal-weight portfolio:")
    print(eq_table.to_string(index=False))
    print("\n  Market-cap-weight portfolio:")
    print(mc_table.to_string(index=False))

    # Save combined
    eq_table['Weighting'] = 'Equal'
    mc_table['Weighting'] = 'MarketCap'
    pd.concat([eq_table, mc_table]).to_csv(
        os.path.join(TABLE_DIR, 'portfolio_risk_comparison.csv'), index=False
    )

    plot_portfolio_es_comparison(risk_results_eq, risk_results_mc)

    # ── Part 5: Euler allocation ──
    print("\n[6/8] Computing Euler allocation...")
    alloc_eq = euler_es_allocation(r_vine, EQUAL_WEIGHTS, quantile=0.01)
    alloc_mc = euler_es_allocation(r_vine, MKTCAP_WEIGHTS, quantile=0.01)

    alloc_eq_df = allocation_table(alloc_eq, asset_names)
    alloc_mc_df = allocation_table(alloc_mc, asset_names)
    alloc_eq_df['Weighting'] = 'Equal'
    alloc_mc_df['Weighting'] = 'MarketCap'
    pd.concat([alloc_eq_df, alloc_mc_df]).to_csv(
        os.path.join(TABLE_DIR, 'euler_allocation.csv'), index=False
    )

    # Verify sum
    sum_check_eq = np.sum(alloc_eq['contributions'])
    print(f"  Equal-weight: sum(ES_i)={sum_check_eq:.6f}, ES_portfolio={alloc_eq['total_es']:.6f}")
    print(f"  Relative error: {abs(sum_check_eq - alloc_eq['total_es']) / alloc_eq['total_es']:.4%}")
    print(f"\n  Equal-weight Euler allocation:")
    print(alloc_eq_df.to_string(index=False))

    plot_euler_allocation(alloc_eq, alloc_mc, asset_names)

    # ── Part 6: Tail dependence and diversification failure ──
    print("\n[7/8] Computing diversification failure curve...")
    plot_tail_dependence(vine, asset_names, d)
    plot_diversification_failure(scenarios, EQUAL_WEIGHTS)

    # ── Part 7: Regime analysis ──
    print("\n[8/8] Regime-conditional analysis...")
    vix_path = os.path.join(DATA_DIR, 'vix_daily.csv')
    vix = load_vix(vix_path)

    regime_results = fit_regime_copulas(returns, vix)

    if regime_results.get('calm') and regime_results.get('crisis'):
        regime_td = regime_tail_dependence(regime_results, d)
        regime_td.to_csv(os.path.join(TABLE_DIR, 'regime_copula_comparison.csv'), index=False)
        print(f"  Calm periods: {regime_results['calm']['n_obs']} days")
        print(f"  Crisis periods: {regime_results['crisis']['n_obs']} days")
        print(regime_td.to_string(index=False))

        plot_regime_dependence(returns, vix, regime_results, asset_names)
    else:
        print("  Warning: insufficient data for one regime. Skipping regime plots.")
        # Create a minimal regime comparison CSV
        pd.DataFrame({'Note': ['Insufficient crisis data']}).to_csv(
            os.path.join(TABLE_DIR, 'regime_copula_comparison.csv'), index=False
        )

    # Rolling analysis (subsample for speed)
    print("\n  Running rolling copula analysis (quarterly steps)...")
    rolling_df = rolling_copula_analysis(returns, window=252, step=63)
    print(f"  Rolling windows computed: {len(rolling_df)}")

    # ── Summary dashboard ──
    print("\n  Generating summary dashboard...")
    plot_summary_dashboard(
        vine, asset_names, d,
        risk_results_eq, risk_results_mc,
        alloc_eq, alloc_mc,
        scenarios, EQUAL_WEIGHTS,
        returns, vix, regime_results, rolling_df
    )

    print("\n" + "=" * 70)
    print("Day 8 complete!")
    print(f"  Figures: {FIG_DIR}")
    print(f"  Tables:  {TABLE_DIR}")
    print("=" * 70)


# ──────────────────────────────────────────────────────────────────────
# FIGURE FUNCTIONS
# ──────────────────────────────────────────────────────────────────────

def plot_marginal_diagnostics(returns, marginal_results, asset_names):
    """Figure 1: QQ plots + Figure 2: Uniform histograms."""
    # QQ plots
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for idx, (ax, name) in enumerate(zip(axes.flat, asset_names)):
        r = returns[name].dropna().values
        fit = marginal_results[name]['best']
        if fit.dist_name == 't':
            theoretical = stats.t.ppf(
                np.linspace(0.001, 0.999, len(r)), *fit.params
            )
        else:
            theoretical = stats.norm.ppf(
                np.linspace(0.001, 0.999, len(r)), *fit.params
            )
        empirical = np.sort(r)
        # Subsample if needed
        step = max(1, len(r) // 500)
        ax.scatter(theoretical[::step], empirical[::step], s=4, alpha=0.5,
                   color=COLORS[idx])
        lims = [min(theoretical.min(), empirical.min()),
                max(theoretical.max(), empirical.max())]
        ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.5)
        ax.set_title(f'{name} (Student-t, df={fit.params[0]:.1f})'
                     if fit.dist_name == 't' else f'{name} (Normal)')
        ax.set_xlabel('Theoretical')
        ax.set_ylabel('Empirical')
    fig.suptitle('QQ Plots: Fitted Marginals vs Empirical', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'marginal_qq.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: marginal_qq.png")

    # Uniform histograms
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for idx, (ax, name) in enumerate(zip(axes.flat, asset_names)):
        u = marginal_results[name]['uniform']
        ax.hist(u, bins=50, density=True, color=COLORS[idx], alpha=0.7,
                edgecolor='white', linewidth=0.3)
        ax.axhline(1.0, color='black', ls='--', lw=1, label='Uniform(0,1)')
        ks = stats.kstest(u, 'uniform')
        ax.set_title(f'{name} (KS p={ks.pvalue:.3f})')
        ax.set_xlabel('u')
        ax.set_ylabel('Density')
        ax.set_xlim(0, 1)
    fig.suptitle('Probability Integral Transform: Uniformity Check', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'uniform_histograms.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: uniform_histograms.png")


def plot_vine_structure(vine, asset_names, d):
    """Figure 3: Vine structure visualization."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Draw tree structure
    n_trees = min(d - 1, vine.trunc_lvl)
    y_positions = np.linspace(0.9, 0.1, n_trees + 1)

    # Draw nodes for Tree 1 (all variables)
    x_positions = np.linspace(0.05, 0.95, d)
    for i, name in enumerate(asset_names):
        ax.text(x_positions[i], y_positions[0], name,
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS[i], alpha=0.7),
                fontsize=11, fontweight='bold', color='white')

    # Draw edges for each tree level
    struct_matrix = np.array(vine.structure.matrix)
    for tree in range(n_trees):
        n_edges = d - 1 - tree
        for edge in range(n_edges):
            pc = vine.get_pair_copula(tree, edge)
            fam = family_name(pc.family)
            lam_L, lam_U = tail_dependence_bicop(pc)

            # Position edge at midpoint
            if tree == 0:
                x1 = x_positions[edge]
                x2 = x_positions[edge + 1]
            else:
                x1 = x_positions[edge] + tree * 0.02
                x2 = x_positions[edge + 1] - tree * 0.02

            y = (y_positions[tree] + y_positions[tree + 1]) / 2
            xm = (x1 + x2) / 2

            # Draw edge line
            ax.plot([x1, x2], [y, y], 'k-', lw=1.5, alpha=0.5)

            # Label
            label = f'{fam}'
            if lam_L > 0.01:
                label += f'\n$\\lambda_L$={lam_L:.3f}'
            ax.text(xm, y + 0.02, label, ha='center', va='bottom',
                    fontsize=7, style='italic',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        # Tree level label
        ax.text(-0.02, (y_positions[tree] + y_positions[tree + 1]) / 2,
                f'T{tree + 1}', ha='right', va='center', fontsize=10,
                fontweight='bold', color='gray')

    ax.set_xlim(-0.08, 1.02)
    ax.set_ylim(0, 1)
    ax.set_title('Vine Copula Structure', fontsize=14, fontweight='bold')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'vine_structure.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: vine_structure.png")


def plot_copula_families_heatmap(vine, asset_names, d):
    """Figure 4: Copula family selection heatmap."""
    # Build a matrix of copula families for Tree 1
    # For visualization: show all bivariate copula families selected
    pair_info = get_pair_copula_matrix(vine, d)

    # Create a family matrix for Tree 1 pairs
    fig, ax = plt.subplots(figsize=(10, 8))

    family_colors = {
        'Gaussian': '#2196F3',
        'Student-t': '#FF5722',
        'Clayton': '#E91E63',
        'Gumbel': '#4CAF50',
        'Frank': '#9E9E9E',
        'Joe': '#FF9800',
        'Independence': '#EEEEEE',
    }

    # Create a d x d matrix showing pair copula families
    matrix = np.zeros((d, d))
    labels = [[''] * d for _ in range(d)]
    color_map = {}

    fam_to_idx = {name: i for i, name in enumerate(family_colors.keys())}

    for (tree, edge), info in pair_info.items():
        if tree == 0:
            # Tree 1: direct pairs
            fam = info['family']
            lam_L = info['lambda_L']
            idx = fam_to_idx.get(fam, 6)
            matrix[edge][edge + 1] = idx + 1
            matrix[edge + 1][edge] = idx + 1
            labels[edge][edge + 1] = f'{fam}\n$\\lambda_L$={lam_L:.3f}'
            labels[edge + 1][edge] = f'{fam}\n$\\lambda_L$={lam_L:.3f}'

    # Create custom colormap
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap_colors = ['white'] + list(family_colors.values())
    cmap = ListedColormap(cmap_colors)
    bounds = np.arange(-0.5, len(cmap_colors) + 0.5)
    norm = BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='equal')

    # Labels
    ax.set_xticks(range(d))
    ax.set_yticks(range(d))
    ax.set_xticklabels(asset_names, fontsize=11)
    ax.set_yticklabels(asset_names, fontsize=11)

    # Annotate cells
    for i in range(d):
        for j in range(d):
            if labels[i][j]:
                ax.text(j, i, labels[i][j], ha='center', va='center',
                        fontsize=7, fontweight='bold')

    # Legend
    legend_elements = []
    from matplotlib.patches import Patch
    for name, color in family_colors.items():
        legend_elements.append(Patch(facecolor=color, label=name))
    ax.legend(handles=legend_elements, loc='upper left',
              bbox_to_anchor=(1.02, 1), fontsize=9)

    ax.set_title('Copula Families Selected (Tree 1)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'copula_families_heatmap.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: copula_families_heatmap.png")


def plot_tail_dependence(vine, asset_names, d):
    """Figure 5: Tail dependence comparison (model vs empirical)."""
    # Empirical tail dependence (chi statistic at 5th percentile)
    fig, ax = plt.subplots(figsize=(12, 6))

    pair_labels = []
    model_lam_L = []
    empirical_lam_L = []

    pair_info = get_pair_copula_matrix(vine, d)
    for (tree, edge), info in pair_info.items():
        if tree == 0:
            pair_labels.append(f'Edge {edge + 1}')
            model_lam_L.append(info['lambda_L'])

    # Empirical tail dependence: fraction of joint exceedances
    # Load uniforms
    returns = pd.read_csv(
        os.path.join(DATA_DIR, 'portfolio_returns.csv'),
        parse_dates=['Date'], index_col='Date'
    )
    marginal_results = fit_marginals(returns)
    u_data = get_uniform_data(marginal_results, asset_names)
    for edge in range(d - 1):
        u1 = u_data[:, edge]
        u2 = u_data[:, edge + 1]
        q = 0.05  # 5th percentile
        joint_exc = np.mean((u1 < q) & (u2 < q))
        empirical_lam_L.append(joint_exc / q)

    x = np.arange(len(pair_labels))
    width = 0.35
    ax.bar(x - width / 2, model_lam_L, width, label='Model (Copula)', color='#2196F3')
    ax.bar(x + width / 2, empirical_lam_L, width, label='Empirical (Chi)', color='#FF5722')
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, rotation=30)
    ax.set_ylabel('Lower Tail Dependence')
    ax.set_title('Tail Dependence: Model vs Empirical', fontsize=14, fontweight='bold')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'tail_dependence_copula.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: tail_dependence_copula.png")


def plot_portfolio_es_comparison(risk_eq, risk_mc):
    """Figure 6: Portfolio ES comparison (4 dependence assumptions, 2 weightings)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    models = list(risk_eq.keys())
    bar_colors = ['#4CAF50', '#2196F3', '#FF5722', '#9C27B0']

    for ax, (title, risk_dict) in zip(axes, [
        ('Equal Weight', risk_eq), ('Market-Cap Weight', risk_mc)
    ]):
        es99 = [risk_dict[m].es_99 for m in models]
        es95 = [risk_dict[m].es_95 for m in models]

        x = np.arange(len(models))
        width = 0.35
        ax.bar(x - width / 2, [e * 100 for e in es99], width,
               label='ES 99%', color=bar_colors, alpha=0.9)
        ax.bar(x + width / 2, [e * 100 for e in es95], width,
               label='ES 95%', color=bar_colors, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha='right')
        ax.set_ylabel('Expected Shortfall (%)')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend()

        # Annotate diversification benefit
        if len(es99) >= 4:
            vine_es = es99[2]
            como_es = es99[3]
            div_benefit = (como_es - vine_es) / como_es * 100
            ax.text(0.98, 0.95, f'Diversification\nbenefit: {div_benefit:.1f}%',
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow'))

    fig.suptitle('Portfolio ES Under Different Dependence Assumptions',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'portfolio_es_comparison.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: portfolio_es_comparison.png")


def plot_euler_allocation(alloc_eq, alloc_mc, asset_names):
    """Figure 7: Euler allocation — weights vs risk contributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (title, alloc) in zip(axes, [
        ('Equal Weight', alloc_eq), ('Market-Cap Weight', alloc_mc)
    ]):
        x = np.arange(len(asset_names))
        width = 0.35
        ax.bar(x - width / 2, alloc['weights'] * 100, width,
               label='Portfolio Weight', color='#2196F3', alpha=0.8)
        ax.bar(x + width / 2, alloc['pct_contributions'] * 100, width,
               label='Risk Contribution', color='#FF5722', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(asset_names)
        ax.set_ylabel('Percentage (%)')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend()

        # Highlight over/under-contributors
        for i in range(len(asset_names)):
            ratio = alloc['pct_contributions'][i] / alloc['weights'][i]
            if ratio > 1.2:
                ax.annotate(f'{ratio:.1f}x', (x[i] + width / 2, alloc['pct_contributions'][i] * 100),
                            textcoords='offset points', xytext=(0, 5),
                            ha='center', fontsize=8, color='red', fontweight='bold')

    fig.suptitle('Euler ES Allocation: Weight vs Risk Contribution',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'euler_allocation.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: euler_allocation.png")


def plot_diversification_failure(scenarios, weights):
    """Figure 8: THE KEY FIGURE — Diversification failure curve.

    Shows that diversification benefit decreases as you go deeper into the tail,
    and vine copula captures this while Gaussian copula does not.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    quantiles = np.array([0.50, 0.25, 0.10, 0.05, 0.025, 0.01, 0.005, 0.002, 0.001])

    for label, color, ls in [
        ('Independent', '#4CAF50', '--'),
        ('Gaussian', '#2196F3', '-.'),
        ('Vine Copula', '#FF5722', '-'),
        ('Comonotonic', '#9C27B0', ':'),
    ]:
        r_sim = scenarios[label]
        d = r_sim.shape[1]
        r_portfolio = portfolio_returns(r_sim, weights)

        # Portfolio ES at each quantile
        es_portfolio = compute_es_at_quantiles(r_portfolio, quantiles)

        # Sum of individual ES (no diversification)
        es_individual_sum = np.zeros(len(quantiles))
        for j in range(d):
            w_r = weights[j] * r_sim[:, j]
            es_j = compute_es_at_quantiles(w_r, quantiles)
            es_individual_sum += es_j

        # Diversification ratio: 1 = no diversification, < 1 = diversification benefit
        div_ratio = es_portfolio / es_individual_sum
        div_ratio = np.clip(div_ratio, 0, 2)  # safety clip

        ax.plot(quantiles * 100, div_ratio, color=color, ls=ls, lw=2.5,
                marker='o', markersize=6, label=label)

    ax.set_xscale('log')
    ax.set_xlabel('Quantile Level (%)', fontsize=12)
    ax.set_ylabel('Diversification Ratio (ES_portfolio / sum ES_individual)', fontsize=12)
    ax.set_title('Diversification Failure in the Tail', fontsize=14, fontweight='bold')
    ax.axhline(1.0, color='gray', ls='--', lw=1, alpha=0.5, label='No diversification')
    ax.legend(fontsize=10)

    ax.text(0.02, 0.02,
            'Vine copula shows diversification erodes in extreme tails\n'
            'Gaussian copula incorrectly maintains diversification benefit',
            transform=ax.transAxes, fontsize=9, va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.invert_xaxis()
    ax.set_xticks([50, 25, 10, 5, 2.5, 1, 0.5, 0.2, 0.1])
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:g}'))

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'diversification_failure.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: diversification_failure.png (KEY FIGURE)")


def plot_regime_dependence(returns, vix, regime_results, asset_names):
    """Figure 9: Calm vs crisis dependence."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    calm_ret, crisis_ret = classify_regimes(returns, vix)

    # Top row: scatter plots of two equity pair returns
    pair = ('IBM', 'JPM')
    if pair[0] in calm_ret.columns and pair[1] in calm_ret.columns:
        ax = axes[0, 0]
        ax.scatter(calm_ret[pair[0]], calm_ret[pair[1]], s=3, alpha=0.3, color='#2196F3')
        ax.set_title(f'{pair[0]} vs {pair[1]} — Calm (VIX<25)', fontsize=11)
        ax.set_xlabel(pair[0])
        ax.set_ylabel(pair[1])
        rho_calm = calm_ret[pair[0]].corr(calm_ret[pair[1]])
        ax.text(0.05, 0.95, f'$\\rho$={rho_calm:.3f}', transform=ax.transAxes,
                va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='white'))

        ax = axes[0, 1]
        ax.scatter(crisis_ret[pair[0]], crisis_ret[pair[1]], s=3, alpha=0.3, color='#FF5722')
        ax.set_title(f'{pair[0]} vs {pair[1]} — Crisis (VIX>25)', fontsize=11)
        ax.set_xlabel(pair[0])
        ax.set_ylabel(pair[1])
        rho_crisis = crisis_ret[pair[0]].corr(crisis_ret[pair[1]])
        ax.text(0.05, 0.95, f'$\\rho$={rho_crisis:.3f}', transform=ax.transAxes,
                va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='white'))

    # Bottom row: portfolio return distributions from regime copulas
    w = np.ones(len(asset_names)) / len(asset_names)
    for idx, (regime, color, ax) in enumerate(zip(
        ['calm', 'crisis'],
        ['#2196F3', '#FF5722'],
        [axes[1, 0], axes[1, 1]]
    )):
        res = regime_results.get(regime)
        if res is None:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            continue

        u_sim = simulate_vine(res['vine'], 200_000, seed=42)
        from copula.marginals import inverse_marginal_cdf
        r_sim = inverse_marginal_cdf(u_sim, res['marginals'], asset_names)
        r_p = r_sim @ w

        ax.hist(r_p, bins=200, density=True, color=color, alpha=0.7, edgecolor='none')
        es99 = -np.mean(r_p[r_p <= np.percentile(r_p, 1)])
        ax.axvline(-es99, color='black', ls='--', lw=1.5, label=f'ES 99%={es99:.4f}')
        ax.set_title(f'Portfolio Returns — {regime.title()} Regime', fontsize=11)
        ax.set_xlabel('Portfolio Return')
        ax.legend()

    fig.suptitle('Regime-Conditional Dependence', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'regime_dependence.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: regime_dependence.png")


def plot_summary_dashboard(vine, asset_names, d,
                           risk_eq, risk_mc, alloc_eq, alloc_mc,
                           scenarios, weights,
                           returns, vix, regime_results, rolling_df):
    """Figure 10: 2x3 summary dashboard."""
    fig = plt.figure(figsize=(20, 13))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel 1: Copula families
    ax1 = fig.add_subplot(gs[0, 0])
    pair_info = get_pair_copula_matrix(vine, d)
    families_t1 = []
    lam_L_t1 = []
    for (tree, edge), info in pair_info.items():
        if tree == 0:
            families_t1.append(info['family'])
            lam_L_t1.append(info['lambda_L'])
    fam_counts = pd.Series(families_t1).value_counts()
    ax1.bar(range(len(fam_counts)), fam_counts.values,
            color=['#FF5722', '#2196F3', '#4CAF50', '#9E9E9E', '#FF9800'][:len(fam_counts)])
    ax1.set_xticks(range(len(fam_counts)))
    ax1.set_xticklabels(fam_counts.index, rotation=30, ha='right')
    ax1.set_title('Copula Families (Tree 1)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Count')

    # Panel 2: ES comparison
    ax2 = fig.add_subplot(gs[0, 1])
    models = list(risk_eq.keys())
    es99_eq = [risk_eq[m].es_99 * 100 for m in models]
    bar_colors = ['#4CAF50', '#2196F3', '#FF5722', '#9C27B0']
    ax2.bar(range(len(models)), es99_eq, color=bar_colors)
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=20, ha='right', fontsize=8)
    ax2.set_ylabel('ES 99% (%)')
    ax2.set_title('Portfolio ES by Dependence', fontsize=11, fontweight='bold')

    # Panel 3: Euler allocation (equal weight)
    ax3 = fig.add_subplot(gs[0, 2])
    x = np.arange(len(asset_names))
    w = 0.35
    ax3.bar(x - w / 2, alloc_eq['weights'] * 100, w, label='Weight', color='#2196F3')
    ax3.bar(x + w / 2, alloc_eq['pct_contributions'] * 100, w, label='Risk', color='#FF5722')
    ax3.set_xticks(x)
    ax3.set_xticklabels(asset_names, fontsize=8)
    ax3.set_ylabel('%')
    ax3.set_title('Euler Allocation (EW)', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=8)

    # Panel 4: Calm vs crisis scatter
    ax4 = fig.add_subplot(gs[1, 0])
    calm_ret, crisis_ret = classify_regimes(returns, vix)
    pair = ('IBM', 'JPM')
    if pair[0] in calm_ret.columns:
        ax4.scatter(calm_ret[pair[0]], calm_ret[pair[1]], s=2, alpha=0.3,
                    color='#2196F3', label='Calm')
        ax4.scatter(crisis_ret[pair[0]], crisis_ret[pair[1]], s=2, alpha=0.3,
                    color='#FF5722', label='Crisis')
        ax4.legend(fontsize=8, markerscale=3)
    ax4.set_title(f'{pair[0]}-{pair[1]} Returns by Regime', fontsize=11, fontweight='bold')
    ax4.set_xlabel(pair[0])
    ax4.set_ylabel(pair[1])

    # Panel 5: Diversification failure (mini version)
    ax5 = fig.add_subplot(gs[1, 1])
    quantiles = np.array([0.50, 0.25, 0.10, 0.05, 0.025, 0.01, 0.005, 0.001])
    for label, color, ls in [('Gaussian', '#2196F3', '-.'), ('Vine Copula', '#FF5722', '-')]:
        r_sim = scenarios[label]
        r_p = portfolio_returns(r_sim, weights)
        es_p = compute_es_at_quantiles(r_p, quantiles)
        es_sum = np.zeros(len(quantiles))
        for j in range(r_sim.shape[1]):
            es_sum += compute_es_at_quantiles(weights[j] * r_sim[:, j], quantiles)
        div_ratio = np.clip(es_p / es_sum, 0, 2)
        ax5.plot(quantiles * 100, div_ratio, color=color, ls=ls, lw=2,
                 marker='o', ms=4, label=label)
    ax5.set_xscale('log')
    ax5.invert_xaxis()
    ax5.axhline(1.0, color='gray', ls='--', lw=0.8)
    ax5.set_xlabel('Quantile (%)')
    ax5.set_ylabel('Div. Ratio')
    ax5.set_title('Diversification Failure', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=8)

    # Panel 6: Rolling ES
    ax6 = fig.add_subplot(gs[1, 2])
    if len(rolling_df) > 0 and not rolling_df['ES_99_Vine'].isna().all():
        ax6.plot(rolling_df['Date'], rolling_df['ES_99_Vine'] * 100,
                 color='#FF5722', lw=1.5, label='Vine')
        ax6.plot(rolling_df['Date'], rolling_df['ES_99_Gaussian'] * 100,
                 color='#2196F3', lw=1.5, label='Gaussian')
        ax6.legend(fontsize=8)
        ax6.set_ylabel('ES 99% (%)')
        ax6.tick_params(axis='x', rotation=30)
    ax6.set_title('Rolling ES (1yr window)', fontsize=11, fontweight='bold')

    fig.suptitle('Multi-Asset Dependence: Vine Copulas and Portfolio Risk',
                 fontsize=16, fontweight='bold', y=0.98)
    fig.savefig(os.path.join(FIG_DIR, 'copula_summary.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: copula_summary.png (300 DPI)")


if __name__ == '__main__':
    main()
