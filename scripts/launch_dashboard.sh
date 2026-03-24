#!/bin/bash
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null
streamlit run src/python/visualization/dashboard.py --theme.base light --theme.primaryColor "#0066cc"
