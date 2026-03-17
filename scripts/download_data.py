"""Standalone script to download all market data.

Usage:
    python scripts/download_data.py
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from python.data.downloader import download_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    """Download all configured market data."""
    print("=" * 60)
    print("QUANT RISK ENGINE — Data Download")
    print("=" * 60)

    results = download_all()

    print(f"\nDownloaded {len(results)} assets:")
    for ticker, df in results.items():
        print(f"  {ticker:>10s}: {len(df):>6d} rows")

    print("\nDone.")


if __name__ == "__main__":
    main()
