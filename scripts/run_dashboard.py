#!/usr/bin/env python3
"""Launch the Quant Risk Engine dashboard."""

import subprocess
import sys
import os

os.chdir(os.path.join(os.path.dirname(__file__), '..'))

try:
    import quant_engine_py
    print("C++ engine available — running in LIVE mode")
except ImportError:
    print("C++ engine not found — running in PRECOMPUTED mode")

subprocess.run([sys.executable, "-m", "streamlit", "run",
                "src/python/visualization/dashboard.py",
                "--server.port", "8501",
                "--theme.base", "light",
                "--theme.primaryColor", "#0066cc"])
