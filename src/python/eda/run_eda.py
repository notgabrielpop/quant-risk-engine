"""Comprehensive EDA: empirical evidence for stylized facts of financial returns.

Generates all figures (outputs/figures/eda/) and tables (outputs/tables/)
for the Day 2 analysis. Run end-to-end with:
    python src/python/eda/run_eda.py
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.nonparametric.smoothers_lowess import lowess

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from python.utils.config import DATA_PROCESSED, REGIMES, PORTFOLIO_TICKERS

FIG_DIR = PROJECT_ROOT / "outputs" / "figures" / "eda"
TBL_DIR = PROJECT_ROOT / "outputs" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

# Asset colour palette
COLORS = {
    "IBM": "#1f77b4",
    "AAPL": "#2ca02c",
    "JPM": "#d62728",
    "TLT": "#ff7f0e",
    "GLD": "#DAA520",
    "SPY": "#000000",
}
REGIME_COLORS = {
    "gfc_2008": "#e74c3c",
    "euro_debt_2011": "#9b59b6",
    "covid_crash": "#e67e22",
    "rate_hikes_2022": "#3498db",
    "calm_2013": "#2ecc71",
    "calm_2017": "#1abc9c",
}
STYLE = "seaborn-v0_8-whitegrid"
DPI = 200
DPI_HIGH = 300

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load portfolio returns, extended returns, BTC returns, VIX prices."""
    port = pd.read_csv(DATA_PROCESSED / "portfolio_returns.csv",
                       index_col="Date", parse_dates=True)
    ext = pd.read_csv(DATA_PROCESSED / "extended_returns.csv",
                      index_col="Date", parse_dates=True)
    btc = pd.read_csv(DATA_PROCESSED / "btc_returns.csv",
                      index_col="Date", parse_dates=True)
    vix = pd.read_csv(DATA_PROCESSED / "vix_daily.csv",
                      index_col="Date", parse_dates=True)
    return port, ext, btc, vix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def shade_regimes(ax: plt.Axes, regimes: dict | None = None,
                  label: bool = True) -> None:
    """Shade crisis/calm regime rectangles on a time-series axis."""
    if regimes is None:
        regimes = REGIMES
    ylim = ax.get_ylim()
    for name, (start, end) in regimes.items():
        color = REGIME_COLORS.get(name, "#cccccc")
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.18, color=color, zorder=0)
        if label:
            mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
            ax.text(mid, ylim[1] * 0.95, name.replace("_", " "),
                    ha="center", va="top", fontsize=7, rotation=90,
                    alpha=0.7)


# ===================================================================
# PART 1 — Univariate Return Statistics
# ===================================================================

def part1a_summary_statistics(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute and save summary statistics table."""
    print("\n" + "=" * 70)
    print("PART 1A: Summary Statistics")
    print("=" * 70)

    rows = []
    for col in returns.columns:
        r = returns[col].dropna()
        jb_stat, jb_p = stats.jarque_bera(r)
        rows.append({
            "Asset": col,
            "Count": len(r),
            "Ann. Mean (%)": r.mean() * 252 * 100,
            "Ann. Std (%)": r.std() * np.sqrt(252) * 100,
            "Min (%)": r.min() * 100,
            "Max (%)": r.max() * 100,
            "Skewness": stats.skew(r),
            "Excess Kurtosis": stats.kurtosis(r),
            "JB Statistic": jb_stat,
            "JB p-value": jb_p,
        })
    # Reference row
    rows.append({
        "Asset": "Normal Ref.",
        "Count": "-",
        "Ann. Mean (%)": "-",
        "Ann. Std (%)": "-",
        "Min (%)": "-",
        "Max (%)": "-",
        "Skewness": 0.0,
        "Excess Kurtosis": 0.0,
        "JB Statistic": 0.0,
        "JB p-value": 1.0,
    })

    df = pd.DataFrame(rows)
    df.to_csv(TBL_DIR / "summary_statistics.csv", index=False)
    print(df.to_string(index=False))
    return df


def part1b_return_histograms(returns: pd.DataFrame) -> None:
    """Histogram of daily log-returns with normal + t overlays."""
    print("\n" + "=" * 70)
    print("PART 1B: Return Distribution Histograms")
    print("=" * 70)

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(16, 11))
        axes = axes.ravel()

        for i, col in enumerate(returns.columns):
            ax = axes[i]
            r = returns[col].dropna().values
            mu, sigma = r.mean(), r.std()

            ax.hist(r, bins=100, density=True, alpha=0.55,
                    color=COLORS.get(col, "steelblue"), edgecolor="none",
                    label="Empirical")

            x = np.linspace(r.min() * 1.1, r.max() * 1.1, 500)
            ax.plot(x, stats.norm.pdf(x, mu, sigma),
                    "k-", lw=1.5, label="Normal")

            df_t, loc_t, scale_t = stats.t.fit(r)
            ax.plot(x, stats.t.pdf(x, df_t, loc_t, scale_t),
                    "r--", lw=1.5, label=f"Student-t (df={df_t:.1f})")

            for mult, ls in [(3, "--"), (5, ":")]:
                ax.axvline(mu + mult * sigma, color="grey", ls=ls, lw=0.8)
                ax.axvline(mu - mult * sigma, color="grey", ls=ls, lw=0.8)

            n_beyond_3s = np.sum(np.abs(r - mu) > 3 * sigma)
            pct = n_beyond_3s / len(r) * 100
            ax.text(0.97, 0.95, f">3\u03c3: {n_beyond_3s} ({pct:.2f}%)\nvs 0.27% expected",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

            ax.set_title(f"{col} Daily Log-Returns", fontsize=13, fontweight="bold")
            ax.set_xlabel("Log-return", fontsize=11)
            ax.set_ylabel("Density", fontsize=11)
            ax.legend(fontsize=8, loc="upper left")
            ax.tick_params(labelsize=9)

            # Individual figure
            fig_ind, ax_ind = plt.subplots(figsize=(10, 6))
            ax_ind.hist(r, bins=100, density=True, alpha=0.55,
                        color=COLORS.get(col, "steelblue"), edgecolor="none",
                        label="Empirical")
            ax_ind.plot(x, stats.norm.pdf(x, mu, sigma), "k-", lw=1.5, label="Normal")
            ax_ind.plot(x, stats.t.pdf(x, df_t, loc_t, scale_t),
                        "r--", lw=1.5, label=f"Student-t (df={df_t:.1f})")
            for mult, ls in [(3, "--"), (5, ":")]:
                ax_ind.axvline(mu + mult * sigma, color="grey", ls=ls, lw=0.8)
                ax_ind.axvline(mu - mult * sigma, color="grey", ls=ls, lw=0.8)
            ax_ind.text(0.97, 0.95, f">3\u03c3: {n_beyond_3s} ({pct:.2f}%)\nvs 0.27% expected",
                        transform=ax_ind.transAxes, ha="right", va="top", fontsize=10,
                        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
            ax_ind.set_title(f"{col} Daily Log-Returns Distribution", fontsize=14, fontweight="bold")
            ax_ind.set_xlabel("Log-return", fontsize=12)
            ax_ind.set_ylabel("Density", fontsize=12)
            ax_ind.legend(fontsize=10)
            fig_ind.tight_layout()
            fig_ind.savefig(FIG_DIR / f"hist_{col}.png", dpi=DPI)
            plt.close(fig_ind)

            print(f"  {col}: t-df={df_t:.1f}, >3\u03c3={n_beyond_3s} ({pct:.2f}%)")

        fig.suptitle("Daily Log-Return Distributions", fontsize=15, fontweight="bold", y=1.01)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "return_distributions_grid.png", dpi=DPI,
                    bbox_inches="tight")
        plt.close(fig)


def part1c_qq_plots(returns: pd.DataFrame) -> None:
    """QQ-plots against normal distribution."""
    print("\n" + "=" * 70)
    print("PART 1C: QQ Plots")
    print("=" * 70)

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(16, 11))
        axes = axes.ravel()

        for i, col in enumerate(returns.columns):
            ax = axes[i]
            r = returns[col].dropna().values
            (osm, osr), (slope, intercept, _) = stats.probplot(r, dist="norm")

            # Color points by tail severity
            colors = np.where(np.abs(osm) > 2, "#d62728", COLORS.get(col, "steelblue"))
            ax.scatter(osm, osr, c=colors, s=6, alpha=0.6, edgecolors="none")
            xline = np.array([osm.min(), osm.max()])
            ax.plot(xline, slope * xline + intercept, "k--", lw=1.2, label="45\u00b0 ref")
            ax.set_title(f"{col}", fontsize=13, fontweight="bold")
            ax.set_xlabel("Theoretical Quantiles", fontsize=11)
            ax.set_ylabel("Sample Quantiles", fontsize=11)
            ax.tick_params(labelsize=9)
            ax.legend(fontsize=8)

        fig.suptitle("QQ-Plots vs Normal Distribution (red = beyond \u00b12\u03c3 quantile)",
                     fontsize=14, fontweight="bold", y=1.01)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "qq_plots_grid.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)
    print("  Saved qq_plots_grid.png")


def part1d_extreme_events(returns: pd.DataFrame) -> pd.DataFrame:
    """Extreme events analysis: actual vs expected tail counts."""
    print("\n" + "=" * 70)
    print("PART 1D: Extreme Events Analysis")
    print("=" * 70)

    thresholds = [3, 4, 5]
    # Expected fractions under standard normal
    expected_frac = {s: 2 * (1 - stats.norm.cdf(s)) for s in thresholds}

    rows = []
    for col in returns.columns:
        r = returns[col].dropna().values
        mu, sigma = r.mean(), r.std()
        n = len(r)
        for s in thresholds:
            n_exceed = int(np.sum(np.abs(r - mu) > s * sigma))
            actual_pct = n_exceed / n * 100
            expected_pct = expected_frac[s] * 100
            ratio = actual_pct / expected_pct if expected_pct > 0 else np.inf
            rows.append({
                "Asset": col,
                "Threshold": f"{s}\u03c3",
                "Actual Count": n_exceed,
                "Actual %": round(actual_pct, 4),
                "Expected %": round(expected_pct, 4),
                "Ratio (Act/Exp)": round(ratio, 1),
            })

    df = pd.DataFrame(rows)
    df.to_csv(TBL_DIR / "extreme_events.csv", index=False)
    print(df.to_string(index=False))

    # Grouped bar chart
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
        assets = returns.columns.tolist()
        x = np.arange(len(assets))
        w = 0.35

        for k, s in enumerate(thresholds):
            ax = axes[k]
            sub = df[df["Threshold"] == f"{s}\u03c3"]
            actual = sub["Actual %"].values
            expected = np.full(len(assets), expected_frac[s] * 100)

            ax.bar(x - w / 2, actual, w, label="Actual", color="#d62728", alpha=0.8)
            ax.bar(x + w / 2, expected, w, label="Expected (Normal)", color="#1f77b4", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(assets, fontsize=9)
            ax.set_title(f"Beyond \u00b1{s}\u03c3", fontsize=13, fontweight="bold")
            ax.set_ylabel("% of observations", fontsize=11)
            ax.legend(fontsize=9)
            ax.tick_params(labelsize=9)

            for j, (a, e) in enumerate(zip(actual, expected)):
                if a > 0:
                    ratio = a / e if e > 0 else 0
                    ax.text(j - w / 2, a, f"{ratio:.0f}x", ha="center", va="bottom",
                            fontsize=7, fontweight="bold", color="#d62728")

        fig.suptitle("Extreme Events: Actual vs Normal-Expected",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "extreme_events_comparison.png", dpi=DPI,
                    bbox_inches="tight")
        plt.close(fig)

    return df


# ===================================================================
# PART 2 — Volatility Dynamics
# ===================================================================

def part2a_rolling_volatility(returns: pd.DataFrame, vix: pd.DataFrame) -> None:
    """Rolling volatility with crisis shading and VIX overlay."""
    print("\n" + "=" * 70)
    print("PART 2A: Rolling Volatility")
    print("=" * 70)

    spy = returns["SPY"]

    with plt.style.context(STYLE):
        # --- SPY detailed plot ---
        fig, ax1 = plt.subplots(figsize=(14, 6))

        vol_21 = spy.rolling(21).std() * np.sqrt(252) * 100
        vol_63 = spy.rolling(63).std() * np.sqrt(252) * 100

        ax1.plot(vol_21.index, vol_21, color="#d62728", lw=1.0, alpha=0.8,
                 label="21-day rolling vol")
        ax1.plot(vol_63.index, vol_63, color="#1f77b4", lw=1.5,
                 label="63-day rolling vol")
        ax1.set_ylabel("Annualised Volatility (%)", fontsize=12)
        ax1.set_xlabel("Date", fontsize=12)
        ax1.tick_params(labelsize=10)
        shade_regimes(ax1)

        # VIX overlay
        vix_aligned = vix["VIX"].reindex(spy.index).dropna()
        ax2 = ax1.twinx()
        ax2.plot(vix_aligned.index, vix_aligned, color="#DAA520", lw=0.8,
                 alpha=0.6, label="VIX (rhs)")
        ax2.set_ylabel("VIX Level", fontsize=12, color="#DAA520")
        ax2.tick_params(axis="y", labelcolor="#DAA520", labelsize=10)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")
        ax1.set_title("SPY Rolling Volatility with Crisis Regimes & VIX",
                      fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "rolling_volatility_spy.png", dpi=DPI)
        plt.close(fig)

        # --- All assets grid ---
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.ravel()
        for i, col in enumerate(returns.columns):
            ax = axes[i]
            vol = returns[col].rolling(21).std() * np.sqrt(252) * 100
            ax.plot(vol.index, vol, color=COLORS.get(col, "steelblue"), lw=0.9)
            shade_regimes(ax, label=False)
            ax.set_title(col, fontsize=13, fontweight="bold")
            ax.set_ylabel("Ann. Vol (%)", fontsize=10)
            ax.tick_params(labelsize=8)
            ax.xaxis.set_major_locator(mdates.YearLocator(4))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        fig.suptitle("21-Day Rolling Volatility (Annualised)",
                     fontsize=14, fontweight="bold", y=1.01)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "rolling_volatility_all.png", dpi=DPI,
                    bbox_inches="tight")
        plt.close(fig)

    print("  Saved rolling_volatility_spy.png, rolling_volatility_all.png")


def part2b_acf_comparison(returns: pd.DataFrame) -> None:
    """ACF of returns, |returns|, and returns² for SPY."""
    print("\n" + "=" * 70)
    print("PART 2B: Autocorrelation Analysis (Volatility Clustering)")
    print("=" * 70)

    spy = returns["SPY"].dropna()
    n_lags = 100

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        series_list = [
            (spy, "Raw Returns $r_t$", "#1f77b4"),
            (spy.abs(), "Absolute Returns $|r_t|$", "#d62728"),
            (spy ** 2, "Squared Returns $r_t^2$", "#2ca02c"),
        ]

        for ax, (s, title, color) in zip(axes, series_list):
            plot_acf(s, ax=ax, lags=n_lags, alpha=0.05,
                     color=color, vlines_kwargs={"colors": color},
                     title="")
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_ylabel("ACF", fontsize=11)
            ax.tick_params(labelsize=9)

        axes[2].set_xlabel("Lag (days)", fontsize=12)
        fig.suptitle("SPY — Autocorrelation: Returns vs Volatility Proxies",
                     fontsize=14, fontweight="bold", y=1.01)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "acf_comparison_spy.png", dpi=DPI,
                    bbox_inches="tight")
        plt.close(fig)

    print("  Key finding: raw returns show ~zero ACF; |r| and r² show strong,")
    print("  slowly decaying positive ACF → volatility clustering confirmed.")


def part2c_ljung_box(returns: pd.DataFrame) -> pd.DataFrame:
    """Ljung-Box test on raw and squared returns."""
    print("\n" + "=" * 70)
    print("PART 2C: Ljung-Box Tests")
    print("=" * 70)

    lags_test = [10, 20, 50]
    rows = []

    for col in returns.columns:
        r = returns[col].dropna()
        lb_raw = acorr_ljungbox(r, lags=lags_test, return_df=True)
        lb_sq = acorr_ljungbox(r ** 2, lags=lags_test, return_df=True)

        for lag in lags_test:
            rows.append({
                "Asset": col,
                "Lag": lag,
                "LB Stat (raw)": round(lb_raw.loc[lag, "lb_stat"], 2),
                "p-value (raw)": f"{lb_raw.loc[lag, 'lb_pvalue']:.4e}",
                "LB Stat (r²)": round(lb_sq.loc[lag, "lb_stat"], 2),
                "p-value (r²)": f"{lb_sq.loc[lag, 'lb_pvalue']:.4e}",
            })

    df = pd.DataFrame(rows)
    df.to_csv(TBL_DIR / "ljung_box_tests.csv", index=False)
    print(df.to_string(index=False))
    return df


def part2d_leverage_effect(returns: pd.DataFrame) -> None:
    """Scatter: return vs future realized vol, showing asymmetry."""
    print("\n" + "=" * 70)
    print("PART 2D: Leverage Effect (Return-Volatility Asymmetry)")
    print("=" * 70)

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax, col in zip(axes, ["SPY", "IBM"]):
            r = returns[col].dropna()
            fwd_vol = r.rolling(21).std().shift(-21) * np.sqrt(252) * 100
            mask = r.notna() & fwd_vol.notna()
            x_vals = r[mask].values
            y_vals = fwd_vol[mask].values

            ax.scatter(x_vals * 100, y_vals, s=3, alpha=0.15,
                       color=COLORS.get(col, "steelblue"))

            # LOWESS fit
            lw = lowess(y_vals, x_vals * 100, frac=0.3)
            ax.plot(lw[:, 0], lw[:, 1], "r-", lw=2.5, label="LOWESS fit")

            ax.set_title(f"{col} — Leverage Effect", fontsize=13, fontweight="bold")
            ax.set_xlabel("Daily Return (%)", fontsize=12)
            ax.set_ylabel("Forward 21-day Ann. Vol (%)", fontsize=12)
            ax.legend(fontsize=10)
            ax.tick_params(labelsize=9)

        fig.suptitle("Negative Returns → Higher Future Volatility (Leverage Effect)",
                     fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "leverage_effect.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)

    print("  Saved leverage_effect.png — asymmetry visible: left side (negative")
    print("  returns) maps to higher forward vol than symmetric positive returns.")


# ===================================================================
# PART 3 — Multi-Asset Dependence Structure
# ===================================================================

def part3a_correlation_matrices(returns: pd.DataFrame) -> None:
    """Pearson, Spearman, and difference correlation matrices."""
    print("\n" + "=" * 70)
    print("PART 3A: Correlation Matrices")
    print("=" * 70)

    pearson = returns.corr(method="pearson")
    spearman = returns.corr(method="spearman")
    diff = (pearson - spearman).abs()

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

        for ax, mat, title, cmap, vrange in [
            (axes[0], pearson,  "Pearson Correlation",  "RdBu_r", (-1, 1)),
            (axes[1], spearman, "Spearman Correlation", "RdBu_r", (-1, 1)),
            (axes[2], diff,     "|Pearson \u2212 Spearman|", "Oranges", (0, diff.max().max())),
        ]:
            sns.heatmap(mat, annot=True, fmt=".2f", cmap=cmap,
                        vmin=vrange[0], vmax=vrange[1],
                        ax=ax, square=True, linewidths=0.5,
                        annot_kws={"size": 10})
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.tick_params(labelsize=10)

        fig.suptitle("Return Correlation Structure",
                     fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "correlation_matrices.png", dpi=DPI,
                    bbox_inches="tight")
        plt.close(fig)

    print("  Pearson:\n", pearson.round(2).to_string())
    print("  Max |Pearson-Spearman| diff:", round(diff.max().max(), 4))


def part3b_rolling_correlations(returns: pd.DataFrame) -> None:
    """63-day rolling correlations for key pairs."""
    print("\n" + "=" * 70)
    print("PART 3B: Rolling Correlations")
    print("=" * 70)

    pairs = [("IBM", "SPY"), ("IBM", "TLT"), ("SPY", "GLD")]
    pair_colors = ["#1f77b4", "#d62728", "#DAA520"]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(14, 6))
        for (a, b), color in zip(pairs, pair_colors):
            rc = returns[a].rolling(63).corr(returns[b])
            ax.plot(rc.index, rc, lw=1.0, color=color, label=f"{a}\u2013{b}")

        shade_regimes(ax)
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.set_ylabel("63-Day Rolling Correlation", fontsize=12)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_title("Rolling Pair Correlations with Crisis Regimes",
                     fontsize=14, fontweight="bold")
        ax.legend(fontsize=10, loc="lower left")
        ax.tick_params(labelsize=10)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "rolling_correlations.png", dpi=DPI)
        plt.close(fig)

    print("  Saved rolling_correlations.png")


def part3c_tail_dependence(returns: pd.DataFrame) -> None:
    """Conditional correlation by market regime (tail vs normal)."""
    print("\n" + "=" * 70)
    print("PART 3C: Tail Dependence Analysis")
    print("=" * 70)

    spy = returns["SPY"]
    q05, q95 = spy.quantile(0.05), spy.quantile(0.95)
    q25, q75 = spy.quantile(0.25), spy.quantile(0.75)

    mask_lower = spy <= q05
    mask_upper = spy >= q95
    mask_normal = (spy >= q25) & (spy <= q75)

    pairs = [("IBM", "AAPL"), ("IBM", "JPM"), ("AAPL", "JPM"),
             ("SPY", "TLT"), ("SPY", "GLD")]

    rows = []
    for a, b in pairs:
        corr_lower = returns.loc[mask_lower, a].corr(returns.loc[mask_lower, b])
        corr_upper = returns.loc[mask_upper, a].corr(returns.loc[mask_upper, b])
        corr_normal = returns.loc[mask_normal, a].corr(returns.loc[mask_normal, b])
        rows.append({
            "Pair": f"{a}\u2013{b}",
            "Lower Tail": round(corr_lower, 3),
            "Normal Regime": round(corr_normal, 3),
            "Upper Tail": round(corr_upper, 3),
        })
        print(f"  {a}-{b}: lower={corr_lower:.3f}, normal={corr_normal:.3f}, upper={corr_upper:.3f}")

    td = pd.DataFrame(rows)

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(pairs))
        w = 0.25

        ax.bar(x - w, td["Lower Tail"], w, label="Lower Tail (SPY \u2264 5th pct)",
               color="#d62728", alpha=0.85)
        ax.bar(x,     td["Normal Regime"], w, label="Normal (25th\u201375th pct)",
               color="#1f77b4", alpha=0.85)
        ax.bar(x + w, td["Upper Tail"], w, label="Upper Tail (SPY \u2265 95th pct)",
               color="#2ca02c", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(td["Pair"], fontsize=11)
        ax.set_ylabel("Conditional Correlation", fontsize=12)
        ax.set_title("Tail Dependence: Correlations Conditional on SPY Regime",
                     fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.axhline(0, color="grey", lw=0.5)
        ax.tick_params(labelsize=10)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "tail_dependence.png", dpi=DPI)
        plt.close(fig)


def part3d_bivariate_scatter(returns: pd.DataFrame) -> None:
    """Bivariate scatter with color-coded regime points and marginals."""
    print("\n" + "=" * 70)
    print("PART 3D: Bivariate Scatter Plots with Marginals")
    print("=" * 70)

    # Build crisis mask from REGIMES
    crisis_mask = pd.Series(False, index=returns.index)
    for _, (start, end) in REGIMES.items():
        crisis_mask |= (returns.index >= start) & (returns.index <= end)

    pairs = [("IBM", "SPY"), ("IBM", "TLT"), ("SPY", "GLD"), ("AAPL", "JPM")]

    with plt.style.context(STYLE):
        fig = plt.figure(figsize=(16, 14))
        outer_gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.35)

        for idx, (a, b) in enumerate(pairs):
            inner = gridspec.GridSpecFromSubplotSpec(
                2, 2, subplot_spec=outer_gs[idx],
                height_ratios=[1, 4], width_ratios=[4, 1],
                hspace=0.05, wspace=0.05)

            ax_main = fig.add_subplot(inner[1, 0])
            ax_top  = fig.add_subplot(inner[0, 0], sharex=ax_main)
            ax_right = fig.add_subplot(inner[1, 1], sharey=ax_main)

            calm = ~crisis_mask
            ax_main.scatter(returns.loc[calm, a] * 100, returns.loc[calm, b] * 100,
                            s=4, alpha=0.20, color="#1f77b4", label="Calm", edgecolors="none")
            ax_main.scatter(returns.loc[crisis_mask, a] * 100,
                            returns.loc[crisis_mask, b] * 100,
                            s=8, alpha=0.55, color="#d62728", label="Crisis", edgecolors="none")
            ax_main.set_xlabel(f"{a} (%)", fontsize=10)
            ax_main.set_ylabel(f"{b} (%)", fontsize=10)
            ax_main.legend(fontsize=8, markerscale=2)
            ax_main.tick_params(labelsize=8)

            bins = 60
            ax_top.hist(returns.loc[calm, a] * 100, bins=bins, density=True,
                        alpha=0.5, color="#1f77b4", edgecolor="none")
            ax_top.hist(returns.loc[crisis_mask, a] * 100, bins=bins, density=True,
                        alpha=0.5, color="#d62728", edgecolor="none")
            ax_top.set_title(f"{a} vs {b}", fontsize=12, fontweight="bold")
            ax_top.tick_params(labelbottom=False, labelsize=7)

            ax_right.hist(returns.loc[calm, b] * 100, bins=bins, density=True,
                          orientation="horizontal", alpha=0.5,
                          color="#1f77b4", edgecolor="none")
            ax_right.hist(returns.loc[crisis_mask, b] * 100, bins=bins, density=True,
                          orientation="horizontal", alpha=0.5,
                          color="#d62728", edgecolor="none")
            ax_right.tick_params(labelleft=False, labelsize=7)

        fig.suptitle("Bivariate Returns: Calm (blue) vs Crisis (red) Regimes",
                     fontsize=14, fontweight="bold", y=1.01)
        fig.savefig(FIG_DIR / "bivariate_scatter_grid.png", dpi=DPI,
                    bbox_inches="tight")
        plt.close(fig)

    print("  Saved bivariate_scatter_grid.png")


# ===================================================================
# PART 4 — Extended Dataset: Long-History Extreme Events
# ===================================================================

KNOWN_EVENTS = {
    "1997-10-27": "Asian Crisis",
    "1998-08-31": "LTCM / Russia",
    "2000-04-14": "Dotcom sell-off",
    "2001-09-17": "Post-9/11 reopen",
    "2001-09-21": "9/11 week end",
    "2002-07-19": "WorldCom fraud",
    "2002-07-24": "Accounting scandals rally",
    "2008-09-29": "TARP vote fails",
    "2008-10-13": "Global bailout rally",
    "2008-10-15": "GFC sell-off",
    "2008-10-28": "GFC rally",
    "2008-11-13": "GFC sell-off",
    "2008-12-01": "Recession declared",
    "2009-03-10": "Bear-market bottom",
    "2009-03-23": "TALF / QE rally",
    "2010-05-06": "Flash Crash",
    "2010-05-10": "Flash Crash recovery",
    "2011-08-08": "US downgrade",
    "2015-08-24": "China deval. sell-off",
    "2018-02-05": "Volmageddon",
    "2020-03-09": "COVID oil crash",
    "2020-03-12": "COVID pandemic decl.",
    "2020-03-13": "COVID relief rally",
    "2020-03-16": "COVID circuit breaker",
    "2020-03-24": "COVID stimulus rally",
    "2022-06-13": "Rate-hike sell-off",
}


def part4a_long_history_extremes(ext_returns: pd.DataFrame) -> pd.DataFrame:
    """Top-20 extreme single-day returns for SPY in the extended dataset."""
    print("\n" + "=" * 70)
    print("PART 4A: Long-History Extreme Events (SPY, 1993\u20132026)")
    print("=" * 70)

    spy = ext_returns["SPY"].dropna()
    mu, sigma = spy.mean(), spy.std()

    # Top 20 by absolute return
    top20_idx = spy.abs().nlargest(20).index
    top20 = spy.loc[top20_idx].sort_values()

    rows = []
    for date, ret in top20.items():
        n_sigma = (ret - mu) / sigma
        normal_prob = 2 * (1 - stats.norm.cdf(abs(n_sigma)))
        date_str = date.strftime("%Y-%m-%d")
        event = KNOWN_EVENTS.get(date_str, "")
        rows.append({
            "Date": date_str,
            "Return (%)": round(ret * 100, 2),
            "# Sigma": round(n_sigma, 1),
            "Normal Probability": f"{normal_prob:.2e}",
            "Event": event,
        })

    df = pd.DataFrame(rows)
    df.to_csv(TBL_DIR / "top_20_extreme_days.csv", index=False)
    print(df.to_string(index=False))

    # Timeline figure
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(spy.index, spy * 100, color="grey", lw=0.3, alpha=0.5)

        for _, row in df.iterrows():
            date = pd.Timestamp(row["Date"])
            ret = row["Return (%)"]
            color = "#d62728" if ret < 0 else "#2ca02c"
            ax.scatter(date, ret, color=color, s=60, zorder=5, edgecolors="black", lw=0.5)
            label = row["Event"] if row["Event"] else row["Date"]
            ax.annotate(label, (date, ret), fontsize=6,
                        textcoords="offset points", xytext=(5, 8 if ret > 0 else -12),
                        ha="left", rotation=30)

        ax.axhline(0, color="grey", lw=0.5)
        ax.set_ylabel("Daily Return (%)", fontsize=12)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_title("SPY — 20 Largest Single-Day Moves (1993\u20132026)",
                     fontsize=14, fontweight="bold")
        ax.tick_params(labelsize=10)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "extreme_events_timeline.png", dpi=DPI)
        plt.close(fig)

    return df


# ===================================================================
# PART 5 — Summary Dashboard
# ===================================================================

def part5_summary_dashboard(returns: pd.DataFrame, vix: pd.DataFrame) -> None:
    """6-panel summary figure for the paper."""
    print("\n" + "=" * 70)
    print("PART 5: Summary Dashboard")
    print("=" * 70)

    spy = returns["SPY"].dropna()
    mu, sigma = spy.mean(), spy.std()

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))

        # 1 — SPY histogram with normal overlay
        ax = axes[0, 0]
        r = spy.values
        ax.hist(r, bins=100, density=True, alpha=0.55, color="#000000", edgecolor="none")
        x = np.linspace(r.min() * 1.1, r.max() * 1.1, 400)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), "r-", lw=1.5, label="Normal")
        df_t, loc_t, scale_t = stats.t.fit(r)
        ax.plot(x, stats.t.pdf(x, df_t, loc_t, scale_t), "b--", lw=1.5,
                label=f"Student-t (df={df_t:.1f})")
        ax.set_title("SPY Return Distribution", fontsize=13, fontweight="bold")
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=9)

        # 2 — SPY QQ plot
        ax = axes[0, 1]
        (osm, osr), (slope, intercept, _) = stats.probplot(r, dist="norm")
        colors = np.where(np.abs(osm) > 2, "#d62728", "#000000")
        ax.scatter(osm, osr, c=colors, s=5, alpha=0.6, edgecolors="none")
        xline = np.array([osm.min(), osm.max()])
        ax.plot(xline, slope * xline + intercept, "k--", lw=1)
        ax.set_title("QQ-Plot vs Normal", fontsize=13, fontweight="bold")
        ax.set_xlabel("Theoretical Quantiles", fontsize=10)
        ax.set_ylabel("Sample Quantiles", fontsize=10)
        ax.tick_params(labelsize=9)

        # 3 — ACF comparison (raw vs |r|)
        ax = axes[0, 2]
        from statsmodels.tsa.stattools import acf as compute_acf
        acf_raw = compute_acf(spy, nlags=80, fft=True)
        acf_abs = compute_acf(spy.abs(), nlags=80, fft=True)
        lags = np.arange(len(acf_raw))
        ax.bar(lags - 0.2, acf_raw, width=0.4, alpha=0.7, color="#1f77b4", label="$r_t$")
        ax.bar(lags + 0.2, acf_abs, width=0.4, alpha=0.7, color="#d62728", label="$|r_t|$")
        ci = 1.96 / np.sqrt(len(spy))
        ax.axhline(ci, ls="--", color="grey", lw=0.7)
        ax.axhline(-ci, ls="--", color="grey", lw=0.7)
        ax.set_title("ACF: Returns vs |Returns|", fontsize=13, fontweight="bold")
        ax.set_xlabel("Lag", fontsize=10)
        ax.legend(fontsize=9)
        ax.set_xlim(-1, 82)
        ax.tick_params(labelsize=9)

        # 4 — Rolling volatility with crisis shading
        ax = axes[1, 0]
        vol = spy.rolling(21).std() * np.sqrt(252) * 100
        ax.plot(vol.index, vol, color="#000000", lw=0.9)
        shade_regimes(ax, label=False)
        ax.set_title("SPY 21-Day Rolling Volatility", fontsize=13, fontweight="bold")
        ax.set_ylabel("Ann. Vol (%)", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.xaxis.set_major_locator(mdates.YearLocator(4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        # 5 — Tail dependence
        ax = axes[1, 1]
        q05, q95 = spy.quantile(0.05), spy.quantile(0.95)
        q25, q75 = spy.quantile(0.25), spy.quantile(0.75)
        mask_lo = spy <= q05
        mask_hi = spy >= q95
        mask_mid = (spy >= q25) & (spy <= q75)
        pairs = [("IBM", "AAPL"), ("IBM", "JPM"), ("AAPL", "JPM"), ("SPY", "TLT")]
        labels = [f"{a}\u2013{b}" for a, b in pairs]
        lo_vals, mid_vals, hi_vals = [], [], []
        for a, b in pairs:
            lo_vals.append(returns.loc[mask_lo, a].corr(returns.loc[mask_lo, b]))
            mid_vals.append(returns.loc[mask_mid, a].corr(returns.loc[mask_mid, b]))
            hi_vals.append(returns.loc[mask_hi, a].corr(returns.loc[mask_hi, b]))
        x = np.arange(len(pairs))
        w = 0.25
        ax.bar(x - w, lo_vals, w, color="#d62728", alpha=0.85, label="Lower tail")
        ax.bar(x,     mid_vals, w, color="#1f77b4", alpha=0.85, label="Normal")
        ax.bar(x + w, hi_vals, w, color="#2ca02c", alpha=0.85, label="Upper tail")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title("Tail Dependence", fontsize=13, fontweight="bold")
        ax.set_ylabel("Cond. Correlation", fontsize=10)
        ax.legend(fontsize=8)
        ax.axhline(0, color="grey", lw=0.5)
        ax.tick_params(labelsize=9)

        # 6 — Extreme events: actual vs expected
        ax = axes[1, 2]
        assets = returns.columns.tolist()
        s = 3  # 3-sigma threshold
        expected_pct = 2 * (1 - stats.norm.cdf(s)) * 100
        actual_pcts = []
        for col in assets:
            r_col = returns[col].dropna().values
            mu_c, sig_c = r_col.mean(), r_col.std()
            n_ex = np.sum(np.abs(r_col - mu_c) > s * sig_c)
            actual_pcts.append(n_ex / len(r_col) * 100)
        x = np.arange(len(assets))
        ax.bar(x - 0.2, actual_pcts, 0.4, color="#d62728", alpha=0.85, label="Actual")
        ax.bar(x + 0.2, [expected_pct] * len(assets), 0.4, color="#1f77b4",
               alpha=0.85, label="Expected (Normal)")
        ax.set_xticks(x)
        ax.set_xticklabels(assets, fontsize=9)
        ax.set_title("Events Beyond \u00b13\u03c3", fontsize=13, fontweight="bold")
        ax.set_ylabel("% of observations", fontsize=10)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=9)

        fig.suptitle("Stylized Facts of Financial Returns \u2014 Empirical Evidence",
                     fontsize=16, fontweight="bold", y=1.02)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "summary_dashboard.png", dpi=DPI_HIGH,
                    bbox_inches="tight")
        plt.close(fig)

    print("  Saved summary_dashboard.png (300 DPI)")


# ===================================================================
# MAIN
# ===================================================================

def main() -> None:
    """Run the complete EDA pipeline."""
    print("=" * 70)
    print("  QUANT RISK ENGINE — Exploratory Data Analysis")
    print("=" * 70)

    port, ext, btc, vix = load_data()
    assets = port.columns.tolist()
    print(f"\nPortfolio assets: {assets}")
    print(f"Portfolio shape:  {port.shape} ({port.index.min().date()} to {port.index.max().date()})")
    print(f"Extended shape:   {ext.shape}")

    # Part 1
    part1a_summary_statistics(port)
    part1b_return_histograms(port)
    part1c_qq_plots(port)
    part1d_extreme_events(port)

    # Part 2
    part2a_rolling_volatility(port, vix)
    part2b_acf_comparison(port)
    part2c_ljung_box(port)
    part2d_leverage_effect(port)

    # Part 3
    part3a_correlation_matrices(port)
    part3b_rolling_correlations(port)
    part3c_tail_dependence(port)
    part3d_bivariate_scatter(port)

    # Part 4
    part4a_long_history_extremes(ext)

    # Part 5
    part5_summary_dashboard(port, vix)

    # Verify outputs
    print("\n" + "=" * 70)
    print("  OUTPUT VERIFICATION")
    print("=" * 70)

    expected_figs = [
        "return_distributions_grid.png",
        "qq_plots_grid.png",
        "extreme_events_comparison.png",
        "rolling_volatility_spy.png",
        "rolling_volatility_all.png",
        "acf_comparison_spy.png",
        "leverage_effect.png",
        "correlation_matrices.png",
        "rolling_correlations.png",
        "tail_dependence.png",
        "bivariate_scatter_grid.png",
        "extreme_events_timeline.png",
        "summary_dashboard.png",
    ] + [f"hist_{t}.png" for t in assets]

    expected_tbls = [
        "summary_statistics.csv",
        "extreme_events.csv",
        "ljung_box_tests.csv",
        "top_20_extreme_days.csv",
    ]

    all_ok = True
    for f in expected_figs:
        path = FIG_DIR / f
        ok = path.exists()
        status = f"  [{('OK' if ok else 'MISSING'):>7s}] figures/eda/{f}"
        print(status)
        if not ok:
            all_ok = False

    for f in expected_tbls:
        path = TBL_DIR / f
        ok = path.exists()
        status = f"  [{('OK' if ok else 'MISSING'):>7s}] tables/{f}"
        print(status)
        if not ok:
            all_ok = False

    print("\n" + ("ALL OUTPUTS GENERATED SUCCESSFULLY" if all_ok
                  else "SOME OUTPUTS MISSING — CHECK ERRORS ABOVE"))
    print("=" * 70)


if __name__ == "__main__":
    main()
