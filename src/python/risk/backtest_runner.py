#!/usr/bin/env python3
"""Day 9: Walk-forward backtesting of VaR and ES models.

Runs Kupiec, Christoffersen, McNeil-Frey, and Basel tests on
ARIMA, GBM, Heston, and Rough Heston VaR/ES forecasts.
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

from risk.backtester import VaRBacktester, BacktestResults, add_arima_results
from risk.statistical_tests import (
    kupiec_test, christoffersen_test, mcneil_frey_test, basel_traffic_light
)

DATA_DIR = os.path.join(project_root, 'data', 'processed')
TABLE_DIR = os.path.join(project_root, 'outputs', 'tables')
FIG_DIR = os.path.join(project_root, 'outputs', 'figures', 'backtest')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

TEST_START = '2022-01-01'
N_PATHS = 50_000
RECALIB_EVERY = 60
ASSETS = ['SPY', 'IBM', 'JPM']
MODELS = ['gbm', 'heston', 'rough_heston']
ALPHAS = [0.95, 0.99]

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 10,
})
COLORS = {
    'arima': '#9E9E9E',
    'gbm': '#2196F3',
    'heston': '#FF5722',
    'rough_heston': '#4CAF50',
}
MODEL_LABELS = {
    'arima': 'ARIMA',
    'gbm': 'GBM',
    'heston': 'Heston',
    'rough_heston': 'Rough Heston',
}

# VIX crisis threshold
VIX_CRISIS = 25.0


def main():
    print("=" * 70)
    print("Day 9: Walk-Forward Backtesting")
    print("=" * 70)

    returns_df = pd.read_csv(
        os.path.join(DATA_DIR, 'portfolio_returns.csv'),
        parse_dates=['Date'], index_col='Date'
    )

    vix = pd.read_csv(
        os.path.join(DATA_DIR, 'vix_daily.csv'),
        parse_dates=['Date'], index_col='Date'
    )['VIX']

    all_results = {}
    all_models_used = set()

    # ── Run backtest per asset ──
    for asset in ASSETS:
        print(f"\n{'─'*60}")
        print(f"Asset: {asset}")
        print(f"{'─'*60}")

        ret = returns_df[asset].dropna()

        bt = VaRBacktester(
            returns=ret,
            test_start_date=TEST_START,
            models=MODELS,
            recalibrate_every=RECALIB_EVERY,
            n_paths=N_PATHS,
            confidence_levels=ALPHAS,
        )
        results = bt.run()

        # Add ARIMA
        arima_path = os.path.join(TABLE_DIR, f'arima_forecasts_{asset}.csv')
        if os.path.exists(arima_path):
            arima_df = pd.read_csv(arima_path)
            test_dates = ret.index[ret.index >= TEST_START]
            add_arima_results(results, arima_df, test_dates, ret)
            all_models_used.add('arima')

        all_results[asset] = results
        all_models_used.update(MODELS)

    all_models = sorted(all_models_used,
                        key=lambda m: ['arima', 'gbm', 'heston', 'rough_heston'].index(m)
                        if m in ['arima', 'gbm', 'heston', 'rough_heston'] else 99)

    # ── Statistical tests ──
    print(f"\n{'='*70}")
    print("Statistical Tests")
    print(f"{'='*70}")

    master_rows = []
    for asset in ASSETS:
        results = all_results[asset]
        for model in all_models:
            for alpha in ALPHAS:
                exc_series = results.exceedance_series(model, alpha)
                if len(exc_series) == 0:
                    continue

                n_exc = int(np.sum(exc_series))
                n_total = len(exc_series)
                expected_rate = 1 - alpha

                # Kupiec
                kup = kupiec_test(n_exc, n_total, expected_rate)

                # Christoffersen
                chris = christoffersen_test(exc_series, expected_rate)

                # McNeil-Frey
                var_arr = results.var_series(model, alpha)
                es_arr = results.es_series(model, alpha)
                act_arr = results.actual_series(model, alpha)
                mf = mcneil_frey_test(act_arr, var_arr, es_arr)

                # Basel (last 250 days)
                if n_total >= 250:
                    exc_last_250 = exc_series[-250:]
                    basel = basel_traffic_light(int(np.sum(exc_last_250)), 250)
                else:
                    basel = basel_traffic_light(n_exc, n_total)

                master_rows.append({
                    'Model': MODEL_LABELS.get(model, model),
                    'Asset': asset,
                    'Alpha': f'{alpha:.0%}',
                    'N_days': n_total,
                    'Exceedances': n_exc,
                    'Rate': f'{kup.actual_rate:.3%}',
                    'Kupiec_p': kup.p_value,
                    'Christ_p': chris.p_cc,
                    'Indep_p': chris.p_ind,
                    'Cluster': chris.clustering_ratio,
                    'Basel': basel.zone,
                    'MF_mean': mf.mean_residual,
                    'MF_p': mf.p_value,
                })

    master_df = pd.DataFrame(master_rows)
    master_df.to_csv(os.path.join(TABLE_DIR, 'backtest_master_comparison.csv'), index=False)
    print("\nMaster comparison table:")
    print(master_df.to_string(index=False))

    # ── Calibrated parameters snapshot ──
    save_calibration_snapshot(all_results)

    # ── Regime comparison ──
    print(f"\n{'='*70}")
    print("Regime Analysis")
    print(f"{'='*70}")
    regime_rows = []
    for asset in ASSETS:
        results = all_results[asset]
        ret = returns_df[asset].dropna()
        test_dates = [r.date for r in results.day_results
                      if r.model == (MODELS[0]) and r.alpha == 0.99]
        if not test_dates:
            continue

        vix_aligned = vix.reindex(test_dates)
        calm_mask = vix_aligned <= VIX_CRISIS
        crisis_mask = vix_aligned > VIX_CRISIS

        for model in all_models:
            exc = results.exceedance_series(model, 0.99)
            dates_m = results.dates(model, 0.99)
            if len(exc) == 0:
                continue

            # Align with VIX
            exc_df = pd.Series(exc, index=pd.DatetimeIndex(dates_m))
            vix_m = vix.reindex(exc_df.index)
            calm = vix_m <= VIX_CRISIS
            crisis = vix_m > VIX_CRISIS

            n_calm = calm.sum()
            n_crisis = crisis.sum()
            exc_calm = exc_df[calm].sum() if n_calm > 0 else 0
            exc_crisis = exc_df[crisis].sum() if n_crisis > 0 else 0

            regime_rows.append({
                'Model': MODEL_LABELS.get(model, model),
                'Asset': asset,
                'Calm_days': int(n_calm),
                'Calm_exc': int(exc_calm),
                'Calm_rate': f'{exc_calm/max(n_calm,1):.3%}',
                'Crisis_days': int(n_crisis),
                'Crisis_exc': int(exc_crisis),
                'Crisis_rate': f'{exc_crisis/max(n_crisis,1):.3%}',
                'Target': '1.0%',
            })

    regime_df = pd.DataFrame(regime_rows)
    regime_df.to_csv(os.path.join(TABLE_DIR, 'backtest_regime_comparison.csv'), index=False)
    print(regime_df.to_string(index=False))

    # ── Figures ──
    print(f"\n{'='*70}")
    print("Generating Figures")
    print(f"{'='*70}")

    plot_var_backtest_timeline(all_results, returns_df, vix, 'SPY', all_models)
    plot_exceedance_clustering(all_results, vix, 'SPY', all_models)
    plot_model_scorecard(master_df)
    plot_coverage_by_regime(regime_df)
    plot_var_adaptiveness(all_results, vix, 'SPY', all_models)
    plot_es_calibration(all_results, 'SPY', all_models)
    plot_portfolio_backtest_proxy(all_results, returns_df, vix, all_models)
    plot_summary_dashboard(all_results, returns_df, vix, master_df,
                           regime_df, 'SPY', all_models)

    # ── Summary ──
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for model in all_models:
        sub = master_df[master_df['Model'] == MODEL_LABELS.get(model, model)]
        sub_99 = sub[sub['Alpha'] == '99%']
        if len(sub_99) == 0:
            continue
        n_tests = len(sub_99) * 3  # Kupiec, Christ, Basel
        passes = 0
        for _, row in sub_99.iterrows():
            if row['Kupiec_p'] >= 0.05: passes += 1
            if row['Christ_p'] >= 0.05: passes += 1
            if row['Basel'] == 'green': passes += 1
        print(f"  {MODEL_LABELS.get(model, model):15s}: passes {passes}/{n_tests} tests (VaR99)")

    print(f"\n  Tables: {TABLE_DIR}")
    print(f"  Figures: {FIG_DIR}")
    print("=" * 70)


def save_calibration_snapshot(all_results):
    """Save calibrated parameters from the last calibration."""
    rows = []
    for asset, results in all_results.items():
        if results.calibration_log:
            last_cal = results.calibration_log[-1]
            for model_name, params in last_cal.items():
                row = {'Asset': asset, 'Model': model_name}
                for attr in ['S0', 'mu', 'sigma', 'v0', 'kappa', 'theta',
                             'sigma_v', 'rho', 'H']:
                    row[attr] = getattr(params, attr, np.nan)
                rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(TABLE_DIR, 'calibrated_parameters.csv'), index=False
        )


# ──────────────────────────────────────────────────────────────────
# FIGURE FUNCTIONS
# ──────────────────────────────────────────────────────────────────

def plot_var_backtest_timeline(all_results, returns_df, vix, asset, models):
    """Figure 1: VaR backtest timeline for one asset."""
    results = all_results[asset]
    n_models = len(models)
    fig, axes = plt.subplots(n_models, 1, figsize=(16, 3.5 * n_models), sharex=True)
    if n_models == 1:
        axes = [axes]

    ret = returns_df[asset]

    for ax, model in zip(axes, models):
        dates = results.dates(model, 0.99)
        if not dates:
            ax.set_title(f'{MODEL_LABELS.get(model, model)} — no data')
            continue

        var99 = results.var_series(model, 0.99)
        actual = results.actual_series(model, 0.99)
        exc = results.exceedance_series(model, 0.99)

        dates = pd.DatetimeIndex(dates)
        ax.plot(dates, actual, color='#333', lw=0.5, alpha=0.7, label='Actual return')
        ax.plot(dates, -var99, color=COLORS.get(model, 'blue'), lw=1.2,
                label=f'$-$VaR 99%')
        exc_dates = dates[exc]
        exc_vals = actual[exc]
        ax.scatter(exc_dates, exc_vals, color='red', s=15, zorder=5,
                   label=f'Exceedances ({int(np.sum(exc))})')

        # Shade crisis periods
        vix_aligned = vix.reindex(dates)
        _shade_crisis(ax, dates, vix_aligned)

        n_exc = int(np.sum(exc))
        ax.set_title(f'{MODEL_LABELS.get(model, model)} — {n_exc} exceedances '
                     f'({n_exc/len(dates)*100:.1f}%)', fontsize=11)
        ax.legend(loc='lower left', fontsize=8)
        ax.set_ylabel('Return')

    axes[-1].set_xlabel('Date')
    fig.suptitle(f'VaR 99% Backtest: {asset}', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'var_backtest_{asset.lower()}.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: var_backtest_{asset.lower()}.png")


def plot_exceedance_clustering(all_results, vix, asset, models):
    """Figure 2: Exceedance clustering comparison."""
    results = all_results[asset]
    n_models = len(models)
    fig, axes = plt.subplots(n_models, 1, figsize=(16, 2.5 * n_models), sharex=True)
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        exc = results.exceedance_series(model, 0.99)
        dates = results.dates(model, 0.99)
        if len(exc) == 0:
            continue
        dates = pd.DatetimeIndex(dates)

        ax.bar(dates, exc.astype(float), width=1, color=COLORS.get(model, 'blue'),
               alpha=0.8)

        vix_aligned = vix.reindex(dates)
        _shade_crisis(ax, dates, vix_aligned)

        chris = christoffersen_test(exc, 0.01)
        ax.set_title(f'{MODEL_LABELS.get(model, model)} — '
                     f'cluster ratio: {chris.clustering_ratio:.2f}, '
                     f'indep p={chris.p_ind:.3f}', fontsize=10)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['No', 'Yes'])

    axes[-1].set_xlabel('Date')
    fig.suptitle(f'VaR 99% Exceedance Clustering: {asset}', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'exceedance_clustering.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: exceedance_clustering.png")


def plot_model_scorecard(master_df):
    """Figure 3: Model comparison scorecard heatmap — KEY FIGURE."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Filter to VaR99 for scorecard
    df99 = master_df[master_df['Alpha'] == '99%'].copy()
    if len(df99) == 0:
        plt.close(fig)
        return

    # Build score matrix
    models = df99['Model'].unique()
    assets = df99['Asset'].unique()
    tests = ['Kupiec', 'Christoffersen', 'Independence', 'Basel', 'McNeil-Frey']

    n_rows = len(models) * len(assets)
    row_labels = []
    score_matrix = np.zeros((n_rows, len(tests)))

    for i, (_, row) in enumerate(df99.iterrows()):
        row_labels.append(f"{row['Model']} / {row['Asset']}")
        # Kupiec: p >= 0.05 = pass
        score_matrix[i, 0] = 2 if row['Kupiec_p'] >= 0.05 else (1 if row['Kupiec_p'] >= 0.01 else 0)
        # Christoffersen CC
        score_matrix[i, 1] = 2 if row['Christ_p'] >= 0.05 else (1 if row['Christ_p'] >= 0.01 else 0)
        # Independence
        score_matrix[i, 2] = 2 if row['Indep_p'] >= 0.05 else (1 if row['Indep_p'] >= 0.01 else 0)
        # Basel
        score_matrix[i, 3] = 2 if row['Basel'] == 'green' else (1 if row['Basel'] == 'yellow' else 0)
        # McNeil-Frey
        mf_p = row['MF_p']
        score_matrix[i, 4] = 2 if (not np.isnan(mf_p) and mf_p >= 0.05) else (
            1 if (not np.isnan(mf_p) and mf_p >= 0.01) else 0)

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#EF5350', '#FFC107', '#4CAF50'])

    im = ax.imshow(score_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=2)
    ax.set_xticks(range(len(tests)))
    ax.set_xticklabels(tests, fontsize=11)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9)

    # Annotate with actual values
    for i, (_, row) in enumerate(df99.iterrows()):
        vals = [
            f"p={row['Kupiec_p']:.3f}",
            f"p={row['Christ_p']:.3f}",
            f"p={row['Indep_p']:.3f}",
            row['Basel'].upper(),
            f"p={row['MF_p']:.3f}" if not np.isnan(row['MF_p']) else 'N/A',
        ]
        for j, v in enumerate(vals):
            color = 'white' if score_matrix[i, j] == 0 else 'black'
            ax.text(j, i, v, ha='center', va='center', fontsize=7,
                    fontweight='bold', color=color)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4CAF50', label='Pass (p>0.05)'),
        Patch(facecolor='#FFC107', label='Marginal (0.01<p<0.05)'),
        Patch(facecolor='#EF5350', label='Fail (p<0.01)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

    ax.set_title('Model Scorecard: VaR 99% Statistical Tests',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'model_scorecard.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: model_scorecard.png (KEY FIGURE)")


def plot_coverage_by_regime(regime_df):
    """Figure 4: Coverage by regime."""
    fig, axes = plt.subplots(1, len(ASSETS), figsize=(5 * len(ASSETS), 6), sharey=True)
    if len(ASSETS) == 1:
        axes = [axes]

    for ax, asset in zip(axes, ASSETS):
        sub = regime_df[regime_df['Asset'] == asset]
        if len(sub) == 0:
            continue

        models_in = sub['Model'].values
        calm_rates = [float(r.strip('%')) for r in sub['Calm_rate'].values]
        crisis_rates = [float(r.strip('%')) for r in sub['Crisis_rate'].values]

        x = np.arange(len(models_in))
        w = 0.35
        ax.bar(x - w/2, calm_rates, w, label='Calm', color='#2196F3', alpha=0.8)
        ax.bar(x + w/2, crisis_rates, w, label='Crisis', color='#FF5722', alpha=0.8)
        ax.axhline(1.0, color='black', ls='--', lw=1.5, label='Target (1%)')
        ax.set_xticks(x)
        ax.set_xticklabels(models_in, rotation=30, ha='right', fontsize=9)
        ax.set_title(f'{asset}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Exceedance Rate (%)')
        ax.legend(fontsize=8)

    fig.suptitle('VaR 99% Coverage by Market Regime', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'coverage_by_regime.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: coverage_by_regime.png")


def plot_var_adaptiveness(all_results, vix, asset, models):
    """Figure 5: VaR adaptiveness — how VaR tracks volatility."""
    results = all_results[asset]
    fig, ax1 = plt.subplots(figsize=(14, 6))

    for model in models:
        var99 = results.var_series(model, 0.99)
        dates = results.dates(model, 0.99)
        if len(var99) == 0:
            continue
        dates = pd.DatetimeIndex(dates)
        ax1.plot(dates, var99 * 100, color=COLORS.get(model, 'blue'),
                 lw=1.5, label=MODEL_LABELS.get(model, model), alpha=0.8)

    ax1.set_ylabel('VaR 99% (%)', fontsize=11)
    ax1.set_xlabel('Date')
    ax1.legend(loc='upper left', fontsize=9)

    # Overlay VIX
    ax2 = ax1.twinx()
    all_dates = []
    for model in models:
        d = results.dates(model, 0.99)
        if d:
            all_dates = d
            break
    if all_dates:
        vix_aligned = vix.reindex(pd.DatetimeIndex(all_dates))
        ax2.plot(pd.DatetimeIndex(all_dates), vix_aligned, color='gray',
                 lw=0.8, alpha=0.4, label='VIX')
        ax2.set_ylabel('VIX', fontsize=11, color='gray')
        ax2.legend(loc='upper right', fontsize=9)

    ax1.set_title(f'VaR 99% Adaptiveness: {asset}', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'var_adaptiveness.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: var_adaptiveness.png")


def plot_es_calibration(all_results, asset, models):
    """Figure 6: ES calibration scatter."""
    results = all_results[asset]
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5))
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        var99 = results.var_series(model, 0.99)
        es99 = results.es_series(model, 0.99)
        actual = results.actual_series(model, 0.99)
        exc = results.exceedance_series(model, 0.99)

        if len(exc) == 0 or np.sum(exc) < 2:
            ax.text(0.5, 0.5, 'Too few exceedances', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(MODEL_LABELS.get(model, model))
            continue

        # On exceedance days: plot (ES forecast, actual loss)
        exc_es = es99[exc]
        exc_loss = -actual[exc]

        ax.scatter(exc_es * 100, exc_loss * 100, color=COLORS.get(model, 'blue'),
                   s=30, alpha=0.7, zorder=5)

        lims = [0, max(np.max(exc_es), np.max(exc_loss)) * 100 * 1.2]
        ax.plot(lims, lims, 'k--', lw=1, alpha=0.5, label='Perfect calibration')
        ax.set_xlabel('ES Forecast (%)')
        ax.set_ylabel('Actual Loss (%)')
        ax.set_title(f'{MODEL_LABELS.get(model, model)} ({int(np.sum(exc))} exc.)',
                     fontsize=11)
        ax.legend(fontsize=8)

    fig.suptitle(f'ES Calibration on Exceedance Days: {asset}',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'es_calibration.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: es_calibration.png")


def plot_portfolio_backtest_proxy(all_results, returns_df, vix, models):
    """Figure 7: Portfolio backtest (equal-weight proxy from single-asset backtests)."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Compute equal-weight portfolio actual returns
    port_ret = returns_df[ASSETS].mean(axis=1)
    port_ret = port_ret[port_ret.index >= TEST_START]

    ax.plot(port_ret.index, port_ret.values, color='#333', lw=0.5, alpha=0.7,
            label='Portfolio return')

    # For each model, average VaR across assets as portfolio VaR proxy
    for model in models:
        var_lists = []
        date_list = None
        for asset in ASSETS:
            results = all_results[asset]
            v = results.var_series(model, 0.99)
            d = results.dates(model, 0.99)
            if len(v) > 0:
                var_lists.append(pd.Series(v, index=pd.DatetimeIndex(d)))
                if date_list is None:
                    date_list = d

        if not var_lists:
            continue

        # Average VaR (conservative diversified estimate)
        var_combined = pd.concat(var_lists, axis=1).mean(axis=1)
        # Scale by sqrt(n) diversification
        var_portfolio = var_combined / np.sqrt(len(ASSETS)) * np.sqrt(len(ASSETS) * 0.7)

        ax.plot(var_combined.index, -var_portfolio.values,
                color=COLORS.get(model, 'blue'), lw=1.2,
                label=f'{MODEL_LABELS.get(model, model)} VaR99')

    vix_aligned = vix.reindex(port_ret.index)
    _shade_crisis(ax, port_ret.index, vix_aligned)

    ax.set_ylabel('Return')
    ax.set_xlabel('Date')
    ax.set_title(f'Portfolio VaR 99% Backtest ({"+".join(ASSETS)}, equal weight)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'portfolio_backtest.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: portfolio_backtest.png")


def plot_summary_dashboard(all_results, returns_df, vix, master_df,
                           regime_df, asset, models):
    """Figure 8: 2x3 summary dashboard — 300 DPI."""
    results = all_results[asset]
    fig = plt.figure(figsize=(20, 13))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel 1: VaR timeline (best vs worst)
    ax1 = fig.add_subplot(gs[0, 0])
    for model in ['gbm', 'heston']:
        var99 = results.var_series(model, 0.99)
        dates = results.dates(model, 0.99)
        actual = results.actual_series(model, 0.99)
        if len(var99) == 0:
            continue
        dates = pd.DatetimeIndex(dates)
        ax1.plot(dates, actual, color='#333', lw=0.3, alpha=0.5)
        ax1.plot(dates, -var99, color=COLORS.get(model), lw=1,
                 label=MODEL_LABELS.get(model))
    ax1.legend(fontsize=7)
    ax1.set_title(f'VaR Timeline: {asset}', fontsize=10, fontweight='bold')
    ax1.tick_params(axis='x', rotation=30)

    # Panel 2: Scorecard mini
    ax2 = fig.add_subplot(gs[0, 1])
    df99 = master_df[(master_df['Alpha'] == '99%') & (master_df['Asset'] == asset)]
    if len(df99) > 0:
        models_sc = df99['Model'].values
        kupiec_pass = ['Pass' if p >= 0.05 else 'Fail' for p in df99['Kupiec_p']]
        chris_pass = ['Pass' if p >= 0.05 else 'Fail' for p in df99['Christ_p']]
        basel_vals = df99['Basel'].values

        cell_text = list(zip(kupiec_pass, chris_pass, [b.upper() for b in basel_vals]))
        table = ax2.table(cellText=cell_text, rowLabels=models_sc,
                          colLabels=['Kupiec', 'Christoffersen', 'Basel'],
                          loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)

        for i in range(len(models_sc)):
            for j in range(3):
                cell = table[i + 1, j]
                text = cell.get_text().get_text()
                if text in ['Pass', 'GREEN']:
                    cell.set_facecolor('#C8E6C9')
                elif text in ['YELLOW']:
                    cell.set_facecolor('#FFF9C4')
                else:
                    cell.set_facecolor('#FFCDD2')
    ax2.axis('off')
    ax2.set_title(f'Scorecard: {asset}', fontsize=10, fontweight='bold')

    # Panel 3: Coverage by regime
    ax3 = fig.add_subplot(gs[0, 2])
    sub = regime_df[regime_df['Asset'] == asset]
    if len(sub) > 0:
        models_r = sub['Model'].values
        calm = [float(r.strip('%')) for r in sub['Calm_rate'].values]
        crisis = [float(r.strip('%')) for r in sub['Crisis_rate'].values]
        x = np.arange(len(models_r))
        ax3.bar(x - 0.17, calm, 0.34, label='Calm', color='#2196F3')
        ax3.bar(x + 0.17, crisis, 0.34, label='Crisis', color='#FF5722')
        ax3.axhline(1.0, color='black', ls='--', lw=1)
        ax3.set_xticks(x)
        ax3.set_xticklabels(models_r, rotation=30, ha='right', fontsize=7)
        ax3.legend(fontsize=7)
    ax3.set_title('Coverage by Regime', fontsize=10, fontweight='bold')

    # Panel 4: Exceedance clustering
    ax4 = fig.add_subplot(gs[1, 0])
    for model in models[:3]:
        exc = results.exceedance_series(model, 0.99)
        dates = results.dates(model, 0.99)
        if len(exc) == 0:
            continue
        chris = christoffersen_test(exc, 0.01)
        ax4.bar([MODEL_LABELS.get(model, model)], [chris.clustering_ratio],
                color=COLORS.get(model, 'blue'))
    ax4.axhline(1.0, color='black', ls='--', lw=1)
    ax4.set_ylabel('Clustering Ratio')
    ax4.set_title('Exceedance Clustering', fontsize=10, fontweight='bold')

    # Panel 5: ES calibration mini
    ax5 = fig.add_subplot(gs[1, 1])
    for model in models[:3]:
        es99 = results.es_series(model, 0.99)
        actual = results.actual_series(model, 0.99)
        exc = results.exceedance_series(model, 0.99)
        if np.sum(exc) < 2:
            continue
        ax5.scatter(es99[exc] * 100, -actual[exc] * 100,
                    color=COLORS.get(model), s=20, alpha=0.7,
                    label=MODEL_LABELS.get(model))
    lims = ax5.get_xlim()
    ax5.plot([0, lims[1]], [0, lims[1]], 'k--', lw=0.8)
    ax5.set_xlabel('ES Forecast (%)')
    ax5.set_ylabel('Actual Loss (%)')
    ax5.legend(fontsize=7)
    ax5.set_title('ES Calibration', fontsize=10, fontweight='bold')

    # Panel 6: VaR adaptiveness
    ax6 = fig.add_subplot(gs[1, 2])
    for model in models[:3]:
        var99 = results.var_series(model, 0.99)
        dates = results.dates(model, 0.99)
        if len(var99) == 0:
            continue
        dates = pd.DatetimeIndex(dates)
        ax6.plot(dates, var99 * 100, color=COLORS.get(model), lw=1,
                 label=MODEL_LABELS.get(model))
    ax6.legend(fontsize=7)
    ax6.set_ylabel('VaR 99% (%)')
    ax6.tick_params(axis='x', rotation=30)
    ax6.set_title('VaR Adaptiveness', fontsize=10, fontweight='bold')

    fig.suptitle('Backtesting Results: Stochastic Volatility vs. Deterministic Models',
                 fontsize=16, fontweight='bold', y=0.98)
    fig.savefig(os.path.join(FIG_DIR, 'backtest_summary.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: backtest_summary.png (300 DPI)")


def _shade_crisis(ax, dates, vix_series):
    """Shade crisis periods (VIX > 25) on a plot."""
    if vix_series is None or vix_series.isna().all():
        return
    crisis = vix_series > VIX_CRISIS
    if crisis.sum() == 0:
        return
    in_crisis = False
    start = None
    for i, (d, c) in enumerate(zip(dates, crisis)):
        if c and not in_crisis:
            start = d
            in_crisis = True
        elif not c and in_crisis:
            ax.axvspan(start, d, color='red', alpha=0.05)
            in_crisis = False
    if in_crisis and start is not None:
        ax.axvspan(start, dates[-1], color='red', alpha=0.05)


if __name__ == '__main__':
    main()
