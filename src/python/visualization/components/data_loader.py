"""Centralized data loading from project outputs."""

import os
import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
TABLE_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'tables')
FIG_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'figures')


def _table(name):
    return os.path.join(TABLE_DIR, name)


@st.cache_data
def load_backtest_master():
    return pd.read_csv(_table('backtest_master_comparison.csv'))


@st.cache_data
def load_backtest_regime():
    return pd.read_csv(_table('backtest_regime_comparison.csv'))


@st.cache_data
def load_portfolio_risk():
    return pd.read_csv(_table('portfolio_risk_comparison.csv'))


@st.cache_data
def load_euler_allocation():
    return pd.read_csv(_table('euler_allocation.csv'))


@st.cache_data
def load_hedging_results():
    return pd.read_csv(_table('hedging_results.csv'))


@st.cache_data
def load_calibrated_params():
    return pd.read_csv(_table('calibrated_params_mle.csv'))


@st.cache_data
def load_calibration_fit():
    path = _table('calibration_fit_comparison.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def load_summary_stats():
    return pd.read_csv(_table('summary_statistics.csv'))


@st.cache_data
def load_regime_copula():
    path = _table('regime_copula_comparison.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def load_coverage_by_regime():
    path = _table('coverage_by_regime.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None
