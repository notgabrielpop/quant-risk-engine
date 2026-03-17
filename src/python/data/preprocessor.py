"""Preprocess raw OHLCV data into multiple aligned datasets.

Produces four datasets:
  - portfolio: IBM, AAPL, JPM, TLT, GLD, SPY on business days (~2004+)
  - extended:  IBM, SPY on business days (~1993+)
  - crypto:    BTC-USD on calendar days
  - vix:       ^VIX reference series on business days

Weekends are filtered BEFORE forward-filling to avoid spurious 0.0 returns.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from python.utils.config import DATA_PROCESSED, DATA_RAW, PORTFOLIO_TICKERS

logger = logging.getLogger(__name__)


def load_raw_prices(raw_dir: Path | None = None) -> dict[str, pd.Series]:
    """Load adjusted close prices from all raw CSVs.

    Returns:
        Dict mapping ticker name -> pd.Series of adjusted close prices.
    """
    if raw_dir is None:
        raw_dir = DATA_RAW

    prices: dict[str, pd.Series] = {}
    csv_files = sorted(raw_dir.glob("*_daily.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    for csv_path in csv_files:
        ticker = csv_path.stem.replace("_daily", "")
        df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)

        if "Adj_Close" in df.columns:
            prices[ticker] = df["Adj_Close"].dropna()
        elif "Close" in df.columns:
            logger.warning(f"{ticker}: No Adj_Close column, using Close")
            prices[ticker] = df["Close"].dropna()
        else:
            logger.warning(f"{ticker}: No price column found, skipping")

    logger.info(f"Loaded {len(prices)} assets from {raw_dir}")
    return prices


def filter_business_days(df: pd.DataFrame) -> pd.DataFrame:
    """Remove weekend rows (Saturday=5, Sunday=6) from a DatetimeIndex DataFrame."""
    mask = df.index.dayofweek < 5
    n_removed = len(df) - mask.sum()
    if n_removed > 0:
        logger.info(f"Filtered {n_removed} weekend rows")
    return df[mask]


def align_and_compute(
    prices: dict[str, pd.Series],
    tickers: list[str],
    business_days_only: bool = True,
    max_ffill_days: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict]]:
    """Align selected tickers, forward-fill, compute log-returns.

    Args:
        prices: All loaded price series.
        tickers: Subset of tickers to align.
        business_days_only: If True, drop weekends before alignment.
        max_ffill_days: Max consecutive NaNs to forward-fill.

    Returns:
        (aligned_prices, log_returns, per_asset_metadata)
    """
    selected = {t: prices[t] for t in tickers if t in prices}
    missing = [t for t in tickers if t not in prices]
    if missing:
        logger.warning(f"Tickers not found in raw data: {missing}")

    raw_df = pd.DataFrame(selected).sort_index()

    if business_days_only:
        raw_df = filter_business_days(raw_df)

    # Per-asset raw stats (before alignment)
    meta: dict[str, dict] = {}
    for col in raw_df.columns:
        series = raw_df[col].dropna()
        meta[col] = {
            "raw_start": str(series.index.min().date()) if len(series) > 0 else None,
            "raw_end": str(series.index.max().date()) if len(series) > 0 else None,
            "raw_observations": len(series),
        }

    # Forward-fill small gaps (holidays that differ across assets)
    n_before_ffill = raw_df.notna().sum()
    raw_df = raw_df.ffill(limit=max_ffill_days)
    n_after_ffill = raw_df.notna().sum()

    for col in raw_df.columns:
        meta[col]["n_filled"] = int(n_after_ffill[col] - n_before_ffill[col])

    # Inner join: keep only rows with ALL assets present
    n_before_drop = len(raw_df)
    aligned = raw_df.dropna()
    n_dropped = n_before_drop - len(aligned)

    for col in aligned.columns:
        meta[col]["n_dropped"] = n_dropped
        meta[col]["aligned_start"] = str(aligned.index.min().date())
        meta[col]["aligned_end"] = str(aligned.index.max().date())
        meta[col]["aligned_observations"] = len(aligned)

    # Log-returns
    log_returns = np.log(aligned / aligned.shift(1)).iloc[1:]  # drop first NaN row

    logger.info(
        f"Aligned {len(aligned.columns)} assets: "
        f"{len(aligned)} dates ({aligned.index.min().date()} to {aligned.index.max().date()}), "
        f"{n_dropped} rows dropped, {len(log_returns)} returns"
    )

    return aligned, log_returns, meta


def build_portfolio_dataset(
    prices: dict[str, pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Portfolio dataset: 6 core assets on business days."""
    tickers = PORTFOLIO_TICKERS  # IBM, AAPL, JPM, TLT, GLD, SPY
    aligned, returns, meta = align_and_compute(prices, tickers, business_days_only=True)
    return aligned, returns, meta


def build_extended_dataset(
    prices: dict[str, pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Extended dataset: IBM + SPY on business days from ~1993."""
    tickers = ["IBM", "SPY"]
    aligned, returns, meta = align_and_compute(prices, tickers, business_days_only=True)
    return aligned, returns, meta


def build_crypto_dataset(
    prices: dict[str, pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """BTC-USD on calendar days (7-day trading)."""
    tickers = ["BTCUSD"]
    aligned, returns, meta = align_and_compute(prices, tickers, business_days_only=False)
    return aligned, returns, meta


def build_vix_reference(
    prices: dict[str, pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """VIX reference series on business days."""
    tickers = ["VIX"]
    aligned, returns, meta = align_and_compute(prices, tickers, business_days_only=True)
    return aligned, returns, meta


def _dataset_summary(name: str, prices: pd.DataFrame, returns: pd.DataFrame, meta: dict) -> dict:
    """Build summary dict for a dataset."""
    return {
        "name": name,
        "assets": meta,
        "n_assets": len(prices.columns),
        "n_price_dates": len(prices),
        "n_return_observations": len(returns),
        "date_range_start": str(prices.index.min().date()),
        "date_range_end": str(prices.index.max().date()),
        "asset_list": list(prices.columns),
    }


def _print_dataset_summary(name: str, summary: dict, meta: dict) -> None:
    """Print a formatted summary block for one dataset."""
    print(f"\n  [{name}]")
    print(f"    Assets:       {summary['asset_list']}")
    print(f"    Price dates:  {summary['n_price_dates']}")
    print(f"    Return obs:   {summary['n_return_observations']}")
    print(f"    Date range:   {summary['date_range_start']} to {summary['date_range_end']}")
    for asset, info in meta.items():
        print(
            f"      {asset:>8s}: {info['raw_observations']:>6d} raw, "
            f"{info.get('n_filled', 0):>3d} filled, "
            f"{info.get('n_dropped', 0):>4d} dropped"
        )


def run_preprocessing(
    raw_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Run the full preprocessing pipeline, producing all four datasets.

    Returns:
        Combined metadata dict for all datasets.
    """
    if raw_dir is None:
        raw_dir = DATA_RAW
    if output_dir is None:
        output_dir = DATA_PROCESSED

    output_dir.mkdir(parents=True, exist_ok=True)
    all_prices = load_raw_prices(raw_dir)

    metadata: dict[str, dict] = {}

    # 1. Portfolio dataset (6 assets, business days, ~2004+)
    port_prices, port_returns, port_meta = build_portfolio_dataset(all_prices)
    port_prices.to_csv(output_dir / "portfolio_prices.csv")
    port_returns.to_csv(output_dir / "portfolio_returns.csv")
    metadata["portfolio"] = _dataset_summary("portfolio", port_prices, port_returns, port_meta)

    # 2. Extended dataset (IBM + SPY, business days, ~1993+)
    ext_prices, ext_returns, ext_meta = build_extended_dataset(all_prices)
    ext_prices.to_csv(output_dir / "extended_prices.csv")
    ext_returns.to_csv(output_dir / "extended_returns.csv")
    metadata["extended"] = _dataset_summary("extended", ext_prices, ext_returns, ext_meta)

    # 3. Crypto dataset (BTC-USD, calendar days)
    btc_prices, btc_returns, btc_meta = build_crypto_dataset(all_prices)
    btc_returns.to_csv(output_dir / "btc_returns.csv")
    metadata["crypto"] = _dataset_summary("crypto", btc_prices, btc_returns, btc_meta)

    # 4. VIX reference (business days)
    vix_prices, vix_returns, vix_meta = build_vix_reference(all_prices)
    vix_prices.to_csv(output_dir / "vix_daily.csv")
    metadata["vix"] = _dataset_summary("vix", vix_prices, vix_returns, vix_meta)

    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    for key in ["portfolio", "extended", "crypto", "vix"]:
        _print_dataset_summary(key, metadata[key], metadata[key]["assets"])
    print("\n" + "=" * 60)

    return metadata
