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
EXTENDED_TICKERS = ["IBM", "SPY"]

# Regime periods for the PORTFOLIO dataset (~2004-11 onward, after GLD launch).
# Black Monday (1987) and Dotcom crash (2000-2002) predate GLD and are only
# available in the EXTENDED dataset.
REGIMES = {
    "gfc_2008": ("2008-06-01", "2009-06-30"),
    "euro_debt_2011": ("2011-07-01", "2011-12-31"),
    "covid_crash": ("2020-02-01", "2020-05-31"),
    "rate_hikes_2022": ("2022-01-01", "2022-12-31"),
    "calm_2013": ("2013-01-01", "2014-12-31"),
    "calm_2017": ("2017-01-01", "2018-06-30"),
}

# Additional regimes available only in the EXTENDED dataset (IBM + SPY, ~1993+).
EXTENDED_REGIMES = {
    "dotcom_crash": ("2000-03-01", "2002-10-31"),
    "dotcom_bubble": ("1998-01-01", "2000-03-01"),
    "calm_1990s": ("1995-01-01", "1997-12-31"),
}

# Simulation defaults
DEFAULT_N_PATHS = 1_000_000
DEFAULT_CONFIDENCE_LEVELS = [0.95, 0.99]
