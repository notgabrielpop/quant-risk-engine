"""Preprocess raw OHLCV data: compute log-returns, align to common dates, export.

Pipeline:
  1. Load all raw CSVs from data/raw/
  2. Extract adjusted close prices
  3. Align all series to common business days (inner join)
  4. Forward-fill gaps up to 3 days, then drop remaining NaN rows
  5. Compute log-returns: r_t = ln(P_t / P_{t-1})
  6. Save prices_aligned.csv, returns_aligned.csv, metadata.json
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from python.utils.config import DATA_PROCESSED, DATA_RAW

logger = logging.getLogger(__name__)


def load_raw_prices(raw_dir: Path | None = None) -> dict[str, pd.Series]:
    """Load adjusted close prices from all raw CSVs.

    Args:
        raw_dir: Directory containing {ticker}_daily.csv files.

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


def align_prices(
    prices: dict[str, pd.Series],
    max_ffill_days: int = 3,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Align all price series to common business days.

    Args:
        prices: Dict of ticker -> price series.
        max_ffill_days: Maximum forward-fill window for missing data.

    Returns:
        Tuple of (aligned price DataFrame, per-asset metadata dict).
    """
    raw_df = pd.DataFrame(prices)
    raw_df = raw_df.sort_index()

    metadata: dict[str, dict] = {}

    for col in raw_df.columns:
        n_total = raw_df[col].notna().sum()
        metadata[col] = {
            "raw_start": str(raw_df[col].first_valid_index().date()) if raw_df[col].first_valid_index() else None,
            "raw_end": str(raw_df[col].last_valid_index().date()) if raw_df[col].last_valid_index() else None,
            "raw_observations": int(n_total),
        }

    # Inner join: keep only dates where ALL assets have data (after ffill)
    aligned = raw_df.dropna(how="all")

    # Forward-fill gaps up to max_ffill_days
    n_before_ffill = aligned.notna().sum()
    aligned = aligned.ffill(limit=max_ffill_days)
    n_after_ffill = aligned.notna().sum()

    for col in aligned.columns:
        metadata[col]["n_missing_filled"] = int(n_after_ffill[col] - n_before_ffill[col])

    # Drop rows that still have NaN after ffill
    n_before_drop = len(aligned)
    aligned = aligned.dropna()
    n_dropped = n_before_drop - len(aligned)

    for col in aligned.columns:
        metadata[col]["n_dropped"] = n_dropped
        metadata[col]["aligned_start"] = str(aligned.index.min().date())
        metadata[col]["aligned_end"] = str(aligned.index.max().date())
        metadata[col]["aligned_observations"] = len(aligned)

    logger.info(
        f"Alignment: {len(raw_df.columns)} assets, "
        f"{len(aligned)} common dates "
        f"({aligned.index.min().date()} to {aligned.index.max().date()}), "
        f"{n_dropped} rows dropped after ffill"
    )

    return aligned, metadata


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log-returns from aligned price series.

    Args:
        prices: DataFrame of aligned adjusted close prices.

    Returns:
        DataFrame of log-returns (first row dropped).
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return log_returns


def run_preprocessing(
    raw_dir: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the full preprocessing pipeline.

    Args:
        raw_dir: Directory with raw CSVs. Defaults to config.DATA_RAW.
        output_dir: Output directory. Defaults to config.DATA_PROCESSED.

    Returns:
        Tuple of (aligned prices DataFrame, log-returns DataFrame, metadata dict).
    """
    if raw_dir is None:
        raw_dir = DATA_RAW
    if output_dir is None:
        output_dir = DATA_PROCESSED

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load raw prices
    prices = load_raw_prices(raw_dir)

    # Step 2: Align to common dates
    aligned_prices, per_asset_meta = align_prices(prices)

    # Step 3: Compute log-returns
    log_returns = compute_log_returns(aligned_prices)

    # Step 4: Build full metadata
    metadata = {
        "assets": per_asset_meta,
        "total": {
            "n_assets": len(aligned_prices.columns),
            "n_common_dates": len(aligned_prices),
            "n_return_observations": len(log_returns),
            "date_range_start": str(aligned_prices.index.min().date()),
            "date_range_end": str(aligned_prices.index.max().date()),
            "asset_list": list(aligned_prices.columns),
        },
    }

    # Step 5: Save outputs
    prices_path = output_dir / "prices_aligned.csv"
    returns_path = output_dir / "returns_aligned.csv"
    meta_path = output_dir / "metadata.json"

    aligned_prices.to_csv(prices_path)
    log_returns.to_csv(returns_path)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved: {prices_path.name}, {returns_path.name}, {meta_path.name}")

    # Step 6: Print summary
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Assets:          {metadata['total']['n_assets']}")
    print(f"Asset list:      {metadata['total']['asset_list']}")
    print(f"Common dates:    {metadata['total']['n_common_dates']}")
    print(f"Return obs:      {metadata['total']['n_return_observations']}")
    print(f"Date range:      {metadata['total']['date_range_start']} to {metadata['total']['date_range_end']}")
    print()
    for asset, info in per_asset_meta.items():
        print(
            f"  {asset:>10s}: {info['raw_observations']:>6d} raw obs, "
            f"{info.get('n_missing_filled', 0):>4d} filled, "
            f"{info.get('n_dropped', 0):>4d} dropped"
        )
    print("=" * 60 + "\n")

    return aligned_prices, log_returns, metadata
