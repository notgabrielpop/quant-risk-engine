"""Placeholder tests for the data acquisition and preprocessing pipeline.

TODO:
  - Test CSV output format and column names
  - Test log-return computation correctness
  - Test alignment with known missing-data patterns
  - Test metadata.json schema
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


def test_config_paths_exist() -> None:
    """Verify config paths resolve to expected locations."""
    from python.utils.config import PROJECT_ROOT, DATA_RAW, DATA_PROCESSED

    assert PROJECT_ROOT.exists()
    assert "quant-risk-engine" in str(PROJECT_ROOT)
    assert DATA_RAW == PROJECT_ROOT / "data" / "raw"
    assert DATA_PROCESSED == PROJECT_ROOT / "data" / "processed"


def test_tickers_configured() -> None:
    """Verify ticker lists are non-empty and consistent."""
    from python.utils.config import TICKERS, EQUITY_TICKERS, PORTFOLIO_TICKERS

    assert len(TICKERS) == 8
    assert all(t in TICKERS for t in EQUITY_TICKERS)
    assert all(t in TICKERS for t in PORTFOLIO_TICKERS)
