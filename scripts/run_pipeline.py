"""Full pipeline runner: download data, preprocess, and validate.

Usage:
    python scripts/run_pipeline.py
"""

import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from python.data.downloader import download_all
from python.data.preprocessor import run_preprocessing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    """Run the full data pipeline."""
    print("=" * 60)
    print("QUANT RISK ENGINE — Full Pipeline")
    print("=" * 60)

    # Step 1: Download
    print("\n[1/2] Downloading market data...")
    download_all()

    # Step 2: Preprocess
    print("\n[2/2] Preprocessing...")
    metadata = run_preprocessing()

    for name in ["portfolio", "extended", "crypto", "vix"]:
        ds = metadata[name]
        print(f"\n  {name}: {ds['n_assets']} assets, {ds['n_return_observations']} returns")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
