"""Central configuration for paths, asset tickers, regime periods, and simulation defaults."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_OPTIONS = PROJECT_ROOT / "data" / "options"
NOTEBOOKS = PROJECT_ROOT / "notebooks"

# Assets
TICKERS = ["IBM", "AAPL", "JPM", "TLT", "GLD", "SPY", "^VIX", "BTC-USD"]
EQUITY_TICKERS = ["IBM", "AAPL", "JPM"]
PORTFOLIO_TICKERS = ["IBM", "AAPL", "JPM", "TLT", "GLD", "SPY"]  # excludes VIX and BTC

# Time periods for regime analysis
REGIMES = {
    "black_monday": ("1987-10-01", "1987-11-30"),
    "dotcom_crash": ("2000-03-01", "2002-10-31"),
    "gfc_2008": ("2008-06-01", "2009-06-30"),
    "covid_crash": ("2020-02-01", "2020-05-31"),
    "calm_2013": ("2013-01-01", "2014-12-31"),
    "calm_2017": ("2017-01-01", "2018-06-30"),
}

# Simulation defaults
DEFAULT_N_PATHS = 1_000_000
DEFAULT_CONFIDENCE_LEVELS = [0.95, 0.99]
