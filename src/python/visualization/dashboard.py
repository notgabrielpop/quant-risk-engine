"""Quant Risk Engine — Streamlit Dashboard."""

import streamlit as st
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'python'))

st.set_page_config(
    page_title="Quant Risk Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Engine availability check
try:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'build', 'src', 'cpp'))
    import quant_engine_py
    _live = True
except ImportError:
    _live = False

st.markdown("""
<style>
.main-header { font-size: 2.4rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0; }
.sub-header  { font-size: 1.1rem; color: #666; margin-top: 0; }
.stMetric { background: #f8f9fa; border-radius: 8px; padding: 8px; }
</style>
""", unsafe_allow_html=True)

if not _live:
    st.warning("Running in **demo mode** — C++ engine not found. Using pre-computed results.")

st.markdown('<p class="main-header">Quant Risk Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Model Risk Quantification with Stochastic Volatility</p>',
            unsafe_allow_html=True)
st.markdown("---")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Models", "3", "GBM / Heston / Rough Heston")
c2.metric("C++ Engine", "42/42 tests", "Live" if _live else "Precomputed")
c3.metric("Hidden Hedging Risk", "6x", "ES99 GBM vs Heston")
c4.metric("Backtest", "GREEN", "Basel traffic light (Heston)")

st.markdown("---")

st.markdown("""
### Explore the Risk Engine

| Page | Description |
|------|-------------|
| **Model Explorer** | Simulate GBM, Heston, Rough Heston interactively |
| **Model Risk** | Side-by-side three-model comparison |
| **Portfolio Risk** | Multi-asset analysis with vine copulas |
| **Backtest Results** | Walk-forward validation on SPY, IBM, JPM |
| **Dynamic Hedging** | Hidden P&L risk from model misspecification |

Use the **sidebar** to navigate between pages.
""")

with st.sidebar:
    st.markdown("### About")
    st.markdown("**Pop Gabriel-Razvan**")
    st.markdown("University research project")
    st.markdown("---")
    st.markdown("C++20 · OpenMP · pybind11 · Sobol QMC")
    st.markdown("Python · Streamlit · Plotly")
