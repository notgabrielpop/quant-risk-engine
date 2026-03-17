"""Download OHLCV daily data for all configured assets via yfinance.

Downloads max available history for each ticker and saves individual
CSVs to data/raw/{ticker}_daily.csv with standardised column names.
Handles download failures gracefully — logs a warning and continues.
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from python.utils.config import DATA_RAW, TICKERS

logger = logging.getLogger(__name__)


def download_ticker(ticker: str, output_dir: Path) -> pd.DataFrame | None:
    """Download full OHLCV history for a single ticker.

    Args:
        ticker: Yahoo Finance ticker symbol.
        output_dir: Directory to save the CSV file.

    Returns:
        DataFrame of daily OHLCV data, or None if download failed.
    """
    safe_name = ticker.replace("^", "").replace("-", "")
    output_path = output_dir / f"{safe_name}_daily.csv"

    try:
        logger.info(f"Downloading {ticker}...")
        data = yf.download(ticker, period="max", auto_adjust=False, progress=False)

        if data.empty:
            logger.warning(f"No data returned for {ticker}")
            return None

        # Flatten MultiIndex columns if present (yfinance >= 0.2.31)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Standardise column names
        rename_map = {
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Adj Close": "Adj_Close",
            "Volume": "Volume",
        }
        data = data.rename(columns=rename_map)

        expected_cols = ["Open", "High", "Low", "Close", "Adj_Close", "Volume"]
        available_cols = [c for c in expected_cols if c in data.columns]
        data = data[available_cols]

        data.index.name = "Date"
        data.to_csv(output_path)

        logger.info(
            f"  {ticker}: {len(data)} rows, "
            f"{data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')} "
            f"-> {output_path.name}"
        )
        return data

    except Exception as e:
        logger.warning(f"Failed to download {ticker}: {e}")
        return None


def download_all(
    tickers: list[str] | None = None,
    output_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Download all configured tickers.

    Args:
        tickers: List of ticker symbols. Defaults to config.TICKERS.
        output_dir: Output directory. Defaults to config.DATA_RAW.

    Returns:
        Dict mapping ticker -> DataFrame for successful downloads.
    """
    if tickers is None:
        tickers = TICKERS
    if output_dir is None:
        output_dir = DATA_RAW

    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = download_ticker(ticker, output_dir)
        if df is not None:
            results[ticker] = df

    n_ok = len(results)
    n_fail = len(tickers) - n_ok
    logger.info(f"Download complete: {n_ok}/{len(tickers)} succeeded")
    if n_fail > 0:
        failed = [t for t in tickers if t not in results]
        logger.warning(f"Failed tickers: {failed}")

    return results
