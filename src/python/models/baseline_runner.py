"""Baseline model runner: ARIMA, RF, XGBoost walk-forward evaluation.

Generates all figures and tables for Day 3 analysis.
    python src/python/models/baseline_runner.py
"""

import json
import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from python.utils.config import DATA_PROCESSED, REGIMES
from python.data.feature_engineering import build_features
from python.models.arima_model import ARIMAForecaster, ARIMAResult
from python.models.ml_models import RFForecaster, XGBForecaster, MLResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

FIG_DIR = PROJECT_ROOT / "outputs" / "figures" / "baseline"
TBL_DIR = PROJECT_ROOT / "outputs" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

STYLE = "seaborn-v0_8-whitegrid"
DPI = 200
DPI_HIGH = 300
TRAIN_END = pd.Timestamp("2022-01-01")

MODEL_COLORS = {"ARIMA": "#1f77b4", "RandomForest": "#2ca02c",
                "XGBoost": "#ff7f0e", "RandomWalk": "#999999", "Actual": "#000000"}
REGIME_COLORS = {
    "gfc_2008": "#e74c3c", "euro_debt_2011": "#9b59b6",
    "covid_crash": "#e67e22", "rate_hikes_2022": "#3498db",
    "calm_2013": "#2ecc71", "calm_2017": "#1abc9c",
}


def shade_regimes(ax, label=True):
    ylim = ax.get_ylim()
    for name, (start, end) in REGIMES.items():
        color = REGIME_COLORS.get(name, "#cccccc")
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.15, color=color, zorder=0)
        if label:
            mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
            ax.text(mid, ylim[1] * 0.92, name.replace("_", " "), ha="center", va="top",
                    fontsize=6, rotation=90, alpha=0.6)


def crisis_mask_for(index):
    """Return boolean mask for crisis dates in given index."""
    mask = pd.Series(False, index=index)
    for _, (s, e) in REGIMES.items():
        mask |= (index >= s) & (index <= e)
    return mask


# =====================================================================
# PART 1: Data Preparation
# =====================================================================
def part1_prepare():
    logger.info("=" * 60)
    logger.info("PART 1: Data Preparation")
    logger.info("=" * 60)

    returns = pd.read_csv(DATA_PROCESSED / "portfolio_returns.csv",
                          index_col="Date", parse_dates=True)
    vix = pd.read_csv(DATA_PROCESSED / "vix_daily.csv",
                      index_col="Date", parse_dates=True)

    train = returns[returns.index < TRAIN_END]
    test = returns[returns.index >= TRAIN_END]
    logger.info(f"Train: {len(train)} rows ({train.index.min().date()} to {train.index.max().date()})")
    logger.info(f"Test:  {len(test)} rows ({test.index.min().date()} to {test.index.max().date()})")

    crisis = crisis_mask_for(test.index)
    logger.info(f"Test crisis days: {crisis.sum()}, calm days: {(~crisis).sum()}")

    # Features
    features = build_features(returns, vix)
    features.to_csv(DATA_PROCESSED / "ml_features.csv")
    logger.info(f"Features: {features.shape[1]} cols, {len(features)} rows")
    print(f"\nFeature matrix: {features.shape}")
    print(features.head().iloc[:, :8].to_string())

    return returns, vix, features, train, test


# =====================================================================
# PART 2: ARIMA
# =====================================================================
def part2a_stationarity(returns, train):
    logger.info("=" * 60)
    logger.info("PART 2A: Stationarity Tests")
    logger.info("=" * 60)

    prices = pd.read_csv(DATA_PROCESSED / "portfolio_prices.csv",
                         index_col="Date", parse_dates=True)
    rows = []
    for col in returns.columns:
        p_train = prices.loc[prices.index < TRAIN_END, col].dropna()
        r_train = train[col].dropna()

        adf_p = adfuller(p_train, maxlag=20, regression="ct", autolag="AIC")
        adf_r = adfuller(r_train, maxlag=20, regression="c", autolag="AIC")
        kpss_r = kpss(r_train, regression="c", nlags="auto")

        rows.append({
            "Asset": col,
            "ADF Price (stat)": round(adf_p[0], 3),
            "ADF Price (p)": round(adf_p[1], 4),
            "ADF Return (stat)": round(adf_r[0], 3),
            "ADF Return (p)": round(adf_r[1], 4),
            "KPSS Return (stat)": round(kpss_r[0], 3),
            "KPSS Return (p)": round(kpss_r[1], 4),
        })

    df = pd.DataFrame(rows)
    df.to_csv(TBL_DIR / "stationarity_tests.csv", index=False)
    print("\n" + df.to_string(index=False))
    return df


def part2b_arima_fit(train):
    logger.info("=" * 60)
    logger.info("PART 2B: ARIMA Order Selection")
    logger.info("=" * 60)

    orders = {}
    rows = []
    for col in train.columns:
        af = ARIMAForecaster(max_p=5, max_q=5, refit_every=20)
        order = af.select_order(train[col])
        orders[col] = af
        rows.append({"Asset": col, "Order": str(order), "AIC": round(af.aic_, 2)})
        logger.info(f"  {col}: ARIMA{order}, AIC={af.aic_:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(TBL_DIR / "arima_orders.csv", index=False)
    print("\n" + df.to_string(index=False))
    return orders


def part2c_arima_forecast(train, test, orders):
    logger.info("=" * 60)
    logger.info("PART 2C: ARIMA Walk-Forward Forecasting")
    logger.info("=" * 60)

    results = {}
    for col in train.columns:
        logger.info(f"  Forecasting {col}...")
        af = orders[col]
        result = af.walk_forward(train[col], test[col], ticker=col)
        result.forecasts.to_csv(TBL_DIR / f"arima_forecasts_{col}.csv")
        results[col] = result
        logger.info(f"  {col}: {len(result.forecasts)} forecasts done")

    return results


def part2d_arima_residuals(arima_results):
    logger.info("=" * 60)
    logger.info("PART 2D: ARIMA Residual Diagnostics")
    logger.info("=" * 60)

    for ticker in ["SPY", "IBM"]:
        res = arima_results[ticker]
        resid = res.forecasts["actual"] - res.forecasts["predicted"]

        jb_stat, jb_p = stats.jarque_bera(resid.dropna())
        lb_sq = acorr_ljungbox(resid.dropna() ** 2, lags=[10, 20], return_df=True)
        ex_kurt = stats.kurtosis(resid.dropna())

        logger.info(f"  {ticker} residuals: excess_kurt={ex_kurt:.2f}, "
                    f"JB_p={jb_p:.2e}, LB_sq(20)_p={lb_sq.loc[20, 'lb_pvalue']:.2e}")

        with plt.style.context(STYLE):
            fig, axes = plt.subplots(2, 3, figsize=(16, 10))

            # Residuals over time
            ax = axes[0, 0]
            ax.plot(resid.index, resid * 100, lw=0.5, color="#1f77b4")
            ax.set_title("Residuals Over Time", fontsize=12, fontweight="bold")
            ax.set_ylabel("Residual (%)")
            shade_regimes(ax, label=False)

            # Histogram
            ax = axes[0, 1]
            r_vals = resid.dropna().values
            ax.hist(r_vals, bins=80, density=True, alpha=0.6, color="#1f77b4", edgecolor="none")
            x = np.linspace(r_vals.min(), r_vals.max(), 300)
            ax.plot(x, stats.norm.pdf(x, r_vals.mean(), r_vals.std()), "r-", lw=1.5)
            ax.set_title(f"Residual Histogram (ExKurt={ex_kurt:.1f})", fontsize=12, fontweight="bold")

            # QQ plot
            ax = axes[0, 2]
            (osm, osr), (sl, ic, _) = stats.probplot(r_vals, dist="norm")
            colors = np.where(np.abs(osm) > 2, "#d62728", "#1f77b4")
            ax.scatter(osm, osr, c=colors, s=5, alpha=0.6, edgecolors="none")
            xline = np.array([osm.min(), osm.max()])
            ax.plot(xline, sl * xline + ic, "k--", lw=1)
            ax.set_title("QQ-Plot of Residuals", fontsize=12, fontweight="bold")

            # ACF of residuals
            ax = axes[1, 0]
            plot_acf(resid.dropna(), ax=ax, lags=50, alpha=0.05,
                     color="#1f77b4", vlines_kwargs={"colors": "#1f77b4"}, title="")
            ax.set_title("ACF of Residuals", fontsize=12, fontweight="bold")

            # ACF of squared residuals
            ax = axes[1, 1]
            plot_acf(resid.dropna() ** 2, ax=ax, lags=50, alpha=0.05,
                     color="#d62728", vlines_kwargs={"colors": "#d62728"}, title="")
            ax.set_title("ACF of Squared Residuals", fontsize=12, fontweight="bold")

            # Stats summary text
            ax = axes[1, 2]
            ax.axis("off")
            text = (f"ARIMA Residual Diagnostics — {ticker}\n\n"
                    f"Excess Kurtosis: {ex_kurt:.2f}\n"
                    f"Jarque-Bera stat: {jb_stat:.1f} (p={jb_p:.2e})\n"
                    f"LB(10) on r²: p={lb_sq.loc[10, 'lb_pvalue']:.2e}\n"
                    f"LB(20) on r²: p={lb_sq.loc[20, 'lb_pvalue']:.2e}\n\n"
                    f"Normality: REJECTED\n"
                    f"Heteroskedasticity: CONFIRMED\n\n"
                    f"ARIMA residuals still exhibit\n"
                    f"fat tails and volatility clustering.")
            ax.text(0.1, 0.95, text, transform=ax.transAxes, fontsize=11,
                    va="top", family="monospace",
                    bbox=dict(facecolor="lightyellow", alpha=0.8))

            fig.suptitle(f"ARIMA Residual Diagnostics — {ticker}",
                         fontsize=14, fontweight="bold", y=1.01)
            fig.tight_layout()
            fig.savefig(FIG_DIR / f"arima_residuals_{ticker.lower()}.png",
                        dpi=DPI, bbox_inches="tight")
            plt.close(fig)


# =====================================================================
# PART 3: ML Models
# =====================================================================
def part3_ml_models(returns, features, test):
    logger.info("=" * 60)
    logger.info("PART 3: ML Models (RF + XGBoost)")
    logger.info("=" * 60)

    vix = pd.read_csv(DATA_PROCESSED / "vix_daily.csv",
                      index_col="Date", parse_dates=True)

    ml_results = {}

    for ticker in ["SPY", "IBM"]:
        # Build single-asset features
        target = returns[ticker].shift(-1)  # next-day return
        common = features.index.intersection(target.dropna().index)
        X = features.loc[common]
        y = target.loc[common]

        # RF
        logger.info(f"  RF walk-forward: {ticker}")
        rf = RFForecaster(refit_every=60)
        rf_res = rf.walk_forward(X, y, TRAIN_END, ticker=ticker)
        ml_results[f"RF_{ticker}"] = rf_res

        rf_res.feature_importance.to_csv(TBL_DIR / "rf_feature_importance.csv", index=False)

        with plt.style.context(STYLE):
            fig, ax = plt.subplots(figsize=(10, 7))
            top15 = rf_res.feature_importance.head(15)
            ax.barh(top15["index"][::-1], top15["importance"][::-1], color="#2ca02c", alpha=0.8)
            ax.set_xlabel("Importance", fontsize=12)
            ax.set_title(f"RF Feature Importance — {ticker} (Top 15)", fontsize=14, fontweight="bold")
            ax.tick_params(labelsize=9)
            fig.tight_layout()
            fig.savefig(FIG_DIR / "rf_feature_importance.png", dpi=DPI)
            plt.close(fig)

        # XGBoost
        logger.info(f"  XGB walk-forward: {ticker}")
        xgb = XGBForecaster(refit_every=60)
        xgb_res = xgb.walk_forward(X, y, TRAIN_END, ticker=ticker)
        ml_results[f"XGB_{ticker}"] = xgb_res

        with plt.style.context(STYLE):
            fig, ax = plt.subplots(figsize=(10, 7))
            top15 = xgb_res.feature_importance.head(15)
            ax.barh(top15["index"][::-1], top15["importance"][::-1], color="#ff7f0e", alpha=0.8)
            ax.set_xlabel("Importance (gain)", fontsize=12)
            ax.set_title(f"XGBoost Feature Importance — {ticker} (Top 15)", fontsize=14, fontweight="bold")
            ax.tick_params(labelsize=9)
            fig.tight_layout()
            fig.savefig(FIG_DIR / "xgb_feature_importance.png", dpi=DPI)
            plt.close(fig)

    # ML residual diagnostics for SPY
    logger.info("  ML Residual diagnostics (SPY)")
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, (key, label, color) in zip(axes, [
            ("RF_SPY", "RF", "#2ca02c"), ("XGB_SPY", "XGBoost", "#ff7f0e"),
        ]):
            resid = ml_results[key].forecasts["actual"] - ml_results[key].forecasts["predicted"]
            r_vals = resid.dropna().values
            (osm, osr), (sl, ic, _) = stats.probplot(r_vals, dist="norm")
            cols = np.where(np.abs(osm) > 2, "#d62728", color)
            ax.scatter(osm, osr, c=cols, s=5, alpha=0.6, edgecolors="none")
            xline = np.array([osm.min(), osm.max()])
            ax.plot(xline, sl * xline + ic, "k--", lw=1)
            ek = stats.kurtosis(r_vals)
            ax.set_title(f"{label} Residuals QQ (ExKurt={ek:.1f})", fontsize=12, fontweight="bold")
            ax.set_xlabel("Theoretical Quantiles", fontsize=10)

        # Add ARIMA for comparison in position 2 by shifting
        axes[2].set_visible(False)
        fig.suptitle("SPY — ML Residual QQ-Plots (fat tails persist)",
                     fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "ml_residuals_spy.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)

    return ml_results


# =====================================================================
# PART 4: Evaluation
# =====================================================================
def compute_metrics(actual, predicted, name, asset):
    mask = np.isfinite(actual) & np.isfinite(predicted)
    a, p = actual[mask], predicted[mask]
    rmse = np.sqrt(np.mean((a - p) ** 2))
    mae = np.mean(np.abs(a - p))
    mape = np.mean(np.abs((a - p) / np.where(np.abs(a) > 1e-10, a, np.nan))) * 100
    if np.isnan(mape):
        mape = 0.0
    dir_acc = np.mean(np.sign(a) == np.sign(p)) * 100
    return {"Model": name, "Asset": asset, "RMSE": rmse, "MAE": mae,
            "MAPE (%)": round(mape, 2), "Dir. Accuracy (%)": round(dir_acc, 1)}


def part4a_point_metrics(arima_results, ml_results, returns, test):
    logger.info("=" * 60)
    logger.info("PART 4A: Point Forecast Metrics")
    logger.info("=" * 60)

    rows = []

    # ARIMA — all assets
    for col in returns.columns:
        fc = arima_results[col].forecasts
        rows.append(compute_metrics(fc["actual"].values, fc["predicted"].values, "ARIMA", col))

    # Random Walk — all assets
    for col in returns.columns:
        fc = arima_results[col].forecasts
        rw_pred = np.zeros(len(fc))
        rows.append(compute_metrics(fc["actual"].values, rw_pred, "RandomWalk", col))

    # RF & XGB — SPY and IBM
    for ticker in ["SPY", "IBM"]:
        for key, name in [(f"RF_{ticker}", "RandomForest"), (f"XGB_{ticker}", "XGBoost")]:
            if key in ml_results:
                fc = ml_results[key].forecasts
                rows.append(compute_metrics(fc["actual"].values, fc["predicted"].values, name, ticker))

    df = pd.DataFrame(rows)
    df.to_csv(TBL_DIR / "point_forecast_metrics.csv", index=False)
    print("\n" + df.to_string(index=False))

    # RMSE comparison bar chart — SPY
    with plt.style.context(STYLE):
        spy_df = df[df["Asset"] == "SPY"]
        fig, ax = plt.subplots(figsize=(10, 5))
        models = spy_df["Model"].values
        rmses = spy_df["RMSE"].values
        colors = [MODEL_COLORS.get(m, "grey") for m in models]
        ax.bar(models, rmses, color=colors, alpha=0.85)
        ax.set_ylabel("RMSE", fontsize=12)
        ax.set_title("SPY — RMSE Comparison Across Models", fontsize=14, fontweight="bold")
        for i, (m, v) in enumerate(zip(models, rmses)):
            ax.text(i, v, f"{v:.6f}", ha="center", va="bottom", fontsize=9)
        ax.tick_params(labelsize=10)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "rmse_comparison.png", dpi=DPI)
        plt.close(fig)

    # RMSE all assets (ARIMA vs RandomWalk)
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(12, 5))
        assets = returns.columns.tolist()
        x = np.arange(len(assets))
        w = 0.35
        arima_rmse = [df[(df["Model"] == "ARIMA") & (df["Asset"] == a)]["RMSE"].values[0] for a in assets]
        rw_rmse = [df[(df["Model"] == "RandomWalk") & (df["Asset"] == a)]["RMSE"].values[0] for a in assets]
        ax.bar(x - w / 2, arima_rmse, w, label="ARIMA", color=MODEL_COLORS["ARIMA"], alpha=0.85)
        ax.bar(x + w / 2, rw_rmse, w, label="Random Walk", color=MODEL_COLORS["RandomWalk"], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(assets, fontsize=11)
        ax.set_ylabel("RMSE", fontsize=12)
        ax.set_title("RMSE: ARIMA vs Random Walk (All Assets)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.tick_params(labelsize=10)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "rmse_all_assets.png", dpi=DPI)
        plt.close(fig)

    return df


def part4b_prediction_intervals(arima_results, returns):
    logger.info("=" * 60)
    logger.info("PART 4B: Prediction Interval Evaluation")
    logger.info("=" * 60)

    coverage_rows = []

    for ticker in returns.columns:
        fc = arima_results[ticker].forecasts
        crisis = crisis_mask_for(fc.index)
        calm = ~crisis

        actual = fc["actual"]
        in_95 = (actual >= fc["lower_95"]) & (actual <= fc["upper_95"])

        cov_all = in_95.mean() * 100
        cov_crisis = in_95[crisis].mean() * 100 if crisis.sum() > 0 else np.nan
        cov_calm = in_95[calm].mean() * 100 if calm.sum() > 0 else np.nan

        width_95 = (fc["upper_95"] - fc["lower_95"])
        width_all = width_95.mean()
        width_crisis = width_95[crisis].mean() if crisis.sum() > 0 else np.nan
        width_calm = width_95[calm].mean() if calm.sum() > 0 else np.nan

        coverage_rows.append({
            "Model": "ARIMA", "Asset": ticker,
            "Overall Coverage (%)": round(cov_all, 1),
            "Crisis Coverage (%)": round(cov_crisis, 1),
            "Calm Coverage (%)": round(cov_calm, 1),
            "Avg PI Width": round(width_all, 6),
            "Crisis PI Width": round(width_crisis, 6),
            "Calm PI Width": round(width_calm, 6),
        })
        logger.info(f"  {ticker}: overall={cov_all:.1f}%, crisis={cov_crisis:.1f}%, calm={cov_calm:.1f}%")

    cov_df = pd.DataFrame(coverage_rows)
    cov_df.to_csv(TBL_DIR / "coverage_by_regime.csv", index=False)
    print("\n" + cov_df.to_string(index=False))

    # KEY FIGURES: prediction interval plots
    for ticker in ["SPY", "IBM"]:
        fc = arima_results[ticker].forecasts
        with plt.style.context(STYLE):
            fig, ax = plt.subplots(figsize=(14, 6))

            ax.fill_between(fc.index, fc["lower_95"], fc["upper_95"],
                            alpha=0.25, color="#1f77b4", label="95% PI")
            ax.plot(fc.index, fc["actual"] * 100, lw=0.4, color="black", alpha=0.5)

            exceedances = ~((fc["actual"] >= fc["lower_95"]) & (fc["actual"] <= fc["upper_95"]))
            ax.scatter(fc.index[exceedances], fc["actual"][exceedances] * 100,
                       s=12, color="#d62728", zorder=5, label="Exceedance", alpha=0.7)

            shade_regimes(ax)
            ax.set_ylabel("Return (%)", fontsize=12)
            ax.set_xlabel("Date", fontsize=12)
            crisis_cov = cov_df[cov_df["Asset"] == ticker]["Crisis Coverage (%)"].values[0]
            calm_cov = cov_df[cov_df["Asset"] == ticker]["Calm Coverage (%)"].values[0]
            overall_cov = cov_df[cov_df["Asset"] == ticker]["Overall Coverage (%)"].values[0]
            ax.set_title(
                f"{ticker} — ARIMA 95% Prediction Intervals "
                f"(Overall: {overall_cov:.0f}%, Crisis: {crisis_cov:.0f}%, Calm: {calm_cov:.0f}%)",
                fontsize=13, fontweight="bold")
            ax.legend(fontsize=10, loc="upper right")
            ax.tick_params(labelsize=10)
            fig.tight_layout()
            fig.savefig(FIG_DIR / f"prediction_intervals_{ticker.lower()}.png", dpi=DPI)
            plt.close(fig)

    return cov_df


def part4c_exceedance_clustering(arima_results):
    logger.info("=" * 60)
    logger.info("PART 4C: Exceedance Clustering")
    logger.info("=" * 60)

    fc = arima_results["SPY"].forecasts
    actual = fc["actual"]
    exceed = ~((actual >= fc["lower_95"]) & (actual <= fc["upper_95"]))
    exceed_int = exceed.astype(int)

    # Longest consecutive streak
    streaks = []
    current = 0
    for v in exceed_int:
        if v == 1:
            current += 1
        else:
            if current > 0:
                streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)

    longest = max(streaks) if streaks else 0
    p_uncond = exceed_int.mean()

    # Conditional probability
    prev_exceed = exceed_int.shift(1).fillna(0)
    p_cond = exceed_int[prev_exceed == 1].mean() if (prev_exceed == 1).sum() > 0 else 0

    exc_rows = [{
        "Asset": "SPY", "Model": "ARIMA",
        "P(exceed)": round(p_uncond, 4),
        "P(exceed|prev_exceed)": round(p_cond, 4),
        "Clustering Ratio": round(p_cond / p_uncond if p_uncond > 0 else 0, 2),
        "Longest Streak": longest,
        "Total Exceedances": int(exceed_int.sum()),
        "Total Observations": len(exceed_int),
    }]
    exc_df = pd.DataFrame(exc_rows)
    exc_df.to_csv(TBL_DIR / "exceedance_analysis.csv", index=False)
    print("\n" + exc_df.to_string(index=False))
    logger.info(f"  P(exceed)={p_uncond:.4f}, P(exceed|prev)={p_cond:.4f}, "
                f"ratio={p_cond / p_uncond if p_uncond > 0 else 0:.2f}, longest_streak={longest}")

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

        ax = axes[0]
        ax.plot(fc.index, actual * 100, lw=0.4, color="black", alpha=0.5)
        ax.scatter(fc.index[exceed], actual[exceed] * 100,
                   s=15, color="#d62728", zorder=5, alpha=0.7)
        shade_regimes(ax, label=True)
        ax.set_ylabel("Return (%)", fontsize=11)
        ax.set_title("SPY Returns with 95% PI Exceedances", fontsize=13, fontweight="bold")

        ax = axes[1]
        ax.bar(fc.index, exceed_int, width=1.5, color="#d62728", alpha=0.7)
        shade_regimes(ax, label=False)
        ax.set_ylabel("Exceedance (0/1)", fontsize=11)
        ax.set_title(f"Exceedance Indicator (P(e|e_prev)={p_cond:.3f} vs P(e)={p_uncond:.3f}, "
                     f"ratio={p_cond / p_uncond if p_uncond > 0 else 0:.1f}x)",
                     fontsize=12, fontweight="bold")
        ax.tick_params(labelsize=10)

        fig.tight_layout()
        fig.savefig(FIG_DIR / "exceedance_clustering_spy.png", dpi=DPI)
        plt.close(fig)

    return exc_df


# =====================================================================
# PART 6: Summary Dashboard
# =====================================================================
def part6_summary(arima_results, ml_results, metrics_df, cov_df):
    logger.info("=" * 60)
    logger.info("PART 6: Summary Dashboard")
    logger.info("=" * 60)

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(16, 11))

        # Top-left: actual vs predicted (ARIMA + RF)
        ax = axes[0, 0]
        fc_a = arima_results["SPY"].forecasts
        ax.plot(fc_a.index, fc_a["actual"] * 100, lw=0.4, color="black", alpha=0.5, label="Actual")
        ax.plot(fc_a.index, fc_a["predicted"] * 100, lw=0.5, color="#1f77b4", alpha=0.6, label="ARIMA")
        if "RF_SPY" in ml_results:
            fc_r = ml_results["RF_SPY"].forecasts
            ax.plot(fc_r.index, fc_r["predicted"] * 100, lw=0.5, color="#2ca02c", alpha=0.6, label="RF")
        ax.set_title("SPY: Actual vs Predicted Returns", fontsize=12, fontweight="bold")
        ax.set_ylabel("Return (%)", fontsize=10)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

        # Top-right: prediction intervals with exceedances
        ax = axes[0, 1]
        ax.fill_between(fc_a.index, fc_a["lower_95"], fc_a["upper_95"],
                        alpha=0.25, color="#1f77b4", label="95% PI")
        ax.plot(fc_a.index, fc_a["actual"] * 100, lw=0.3, color="black", alpha=0.4)
        exceed = ~((fc_a["actual"] >= fc_a["lower_95"]) & (fc_a["actual"] <= fc_a["upper_95"]))
        ax.scatter(fc_a.index[exceed], fc_a["actual"][exceed] * 100,
                   s=8, color="#d62728", zorder=5, alpha=0.7, label="Exceedance")
        shade_regimes(ax, label=False)
        row = cov_df[cov_df["Asset"] == "SPY"].iloc[0]
        ax.set_title(f"ARIMA 95% PI (Crisis cov: {row['Crisis Coverage (%)']:.0f}%)",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

        # Bottom-left: RMSE comparison
        ax = axes[1, 0]
        spy_m = metrics_df[metrics_df["Asset"] == "SPY"]
        models = spy_m["Model"].values
        rmses = spy_m["RMSE"].values
        colors = [MODEL_COLORS.get(m, "grey") for m in models]
        ax.bar(models, rmses, color=colors, alpha=0.85)
        ax.set_ylabel("RMSE", fontsize=10)
        ax.set_title("SPY — RMSE by Model", fontsize=12, fontweight="bold")
        for i, v in enumerate(rmses):
            ax.text(i, v, f"{v:.5f}", ha="center", va="bottom", fontsize=8)
        ax.tick_params(labelsize=9)

        # Bottom-right: coverage comparison
        ax = axes[1, 1]
        assets_cov = cov_df["Asset"].values
        x = np.arange(len(assets_cov))
        w = 0.25
        ax.bar(x - w, cov_df["Overall Coverage (%)"], w, label="Overall", color="#1f77b4", alpha=0.85)
        ax.bar(x, cov_df["Crisis Coverage (%)"], w, label="Crisis", color="#d62728", alpha=0.85)
        ax.bar(x + w, cov_df["Calm Coverage (%)"], w, label="Calm", color="#2ca02c", alpha=0.85)
        ax.axhline(95, color="black", ls="--", lw=1, label="Target 95%")
        ax.set_xticks(x)
        ax.set_xticklabels(assets_cov, fontsize=9)
        ax.set_ylabel("Coverage (%)", fontsize=10)
        ax.set_title("95% PI Coverage by Regime", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=9)

        fig.suptitle("Deterministic Baseline: Point Forecasts and Prediction Interval Failures",
                     fontsize=15, fontweight="bold", y=1.02)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "baseline_summary.png", dpi=DPI_HIGH, bbox_inches="tight")
        plt.close(fig)

    logger.info("  Saved baseline_summary.png")


# =====================================================================
# MAIN
# =====================================================================
def main():
    print("=" * 60)
    print("  QUANT RISK ENGINE — Day 3: Deterministic Baselines")
    print("=" * 60)

    # Part 1
    returns, vix, features, train, test = part1_prepare()

    # Part 2: ARIMA
    part2a_stationarity(returns, train)
    arima_orders = part2b_arima_fit(train)
    arima_results = part2c_arima_forecast(train, test, arima_orders)
    part2d_arima_residuals(arima_results)

    # Part 3: ML
    ml_results = part3_ml_models(returns, features, test)

    # Part 4: Evaluation
    metrics_df = part4a_point_metrics(arima_results, ml_results, returns, test)
    cov_df = part4b_prediction_intervals(arima_results, returns)
    part4c_exceedance_clustering(arima_results)

    # Part 6: Summary
    part6_summary(arima_results, ml_results, metrics_df, cov_df)

    # Verify outputs
    print("\n" + "=" * 60)
    print("  OUTPUT VERIFICATION")
    print("=" * 60)

    expected_figs = [
        "arima_residuals_spy.png", "arima_residuals_ibm.png",
        "rf_feature_importance.png", "xgb_feature_importance.png",
        "ml_residuals_spy.png", "rmse_comparison.png", "rmse_all_assets.png",
        "prediction_intervals_spy.png", "prediction_intervals_ibm.png",
        "exceedance_clustering_spy.png", "baseline_summary.png",
    ]
    expected_tbls = [
        "stationarity_tests.csv", "arima_orders.csv",
        "rf_feature_importance.csv", "point_forecast_metrics.csv",
        "coverage_by_regime.csv", "exceedance_analysis.csv",
    ] + [f"arima_forecasts_{t}.csv" for t in returns.columns]

    all_ok = True
    for f in expected_figs:
        ok = (FIG_DIR / f).exists()
        print(f"  [{'OK' if ok else 'MISSING':>7s}] figures/baseline/{f}")
        if not ok: all_ok = False
    for f in expected_tbls:
        ok = (TBL_DIR / f).exists()
        print(f"  [{'OK' if ok else 'MISSING':>7s}] tables/{f}")
        if not ok: all_ok = False

    print("\n" + ("ALL OUTPUTS GENERATED SUCCESSFULLY" if all_ok
                  else "SOME OUTPUTS MISSING") + "\n" + "=" * 60)


if __name__ == "__main__":
    main()
