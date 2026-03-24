"""Page 4: Walk-forward backtesting results."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'python'))

from visualization.components.data_loader import (
    load_backtest_master, load_backtest_regime, load_coverage_by_regime,
)
from visualization.components import plot_factory as pf

st.set_page_config(page_title="Backtest Results", layout="wide")
st.title("Backtest Results")
st.markdown("Walk-forward VaR/ES validation on 1,053 trading days (2022-2025).")

master = load_backtest_master()
regime = load_backtest_regime()
coverage = load_coverage_by_regime()

with st.sidebar:
    st.header("Filters")
    all_assets = master['Asset'].unique().tolist()
    asset = st.selectbox("Asset", all_assets, index=all_assets.index('SPY') if 'SPY' in all_assets else 0)
    all_models = master['Model'].unique().tolist()
    models = st.multiselect("Models", all_models, default=all_models)
    alpha = st.selectbox("Confidence", ["99%", "95%"], index=0)

# Filter
df = master[(master['Asset'] == asset) & (master['Alpha'] == alpha) & (master['Model'].isin(models))]

# Row 1: Pass/Fail summary
st.markdown("### Model Scorecard")

if len(df) > 0:
    # Color-coded pass/fail
    pass_fail = []
    for _, row in df.iterrows():
        kupiec = "PASS" if row['Kupiec_p'] > 0.05 else "FAIL"
        christ = "PASS" if row['Christ_p'] > 0.05 else "FAIL"
        indep = "PASS" if row['Indep_p'] > 0.05 else "FAIL"
        pass_fail.append({
            'Model': row['Model'],
            'Exceedances': f"{row['Exceedances']} ({row['Rate']})",
            'Kupiec': f"{kupiec} (p={row['Kupiec_p']:.3f})",
            'Christoffersen': f"{christ} (p={row['Christ_p']:.3f})",
            'Independence': f"{indep} (p={row['Indep_p']:.3f})",
            'Basel': row['Basel'],
        })
    st.dataframe(pd.DataFrame(pass_fail), use_container_width=True, hide_index=True)

    # Heatmap
    fig = pf.scorecard_heatmap(df, f'Statistical Test Results — {asset} ({alpha})')
    st.plotly_chart(fig, use_container_width=True)

# Key findings callout
heston_row = df[df['Model'] == 'Heston']
gbm_row = df[df['Model'] == 'GBM']
if len(heston_row) > 0 and len(gbm_row) > 0:
    h_kupiec = heston_row.iloc[0]['Kupiec_p']
    g_kupiec = gbm_row.iloc[0]['Kupiec_p']
    h_basel = str(heston_row.iloc[0]['Basel']).lower()
    g_basel = str(gbm_row.iloc[0]['Basel']).lower()

    if h_kupiec > 0.05:
        st.success(f"Heston passes Kupiec (p={h_kupiec:.3f}) — Basel {h_basel.upper()}")
    if g_kupiec <= 0.05:
        st.error(f"GBM fails Kupiec (p={g_kupiec:.3f}) — Basel {g_basel.upper()}")

st.markdown("---")

# Row 2: Regime analysis
st.markdown("### Regime Analysis: Calm vs Crisis")

regime_df = regime[(regime['Asset'] == asset) & (regime['Model'].isin(models))] if regime is not None else pd.DataFrame()

if len(regime_df) > 0:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Exceedance Rates")
        display = regime_df[['Model', 'Calm_days', 'Calm_rate', 'Crisis_days', 'Crisis_rate', 'Target']].copy()
        st.dataframe(display, use_container_width=True, hide_index=True)

    with c2:
        # Bar chart: calm vs crisis
        fig = go.Figure()
        for _, row in regime_df.iterrows():
            calm_pct = float(row['Calm_rate'].replace('%', ''))
            crisis_pct = float(row['Crisis_rate'].replace('%', ''))
            fig.add_trace(go.Bar(name=f"{row['Model']} Calm", x=[row['Model']],
                                 y=[calm_pct], marker_color='#2196F3', opacity=0.7))
            fig.add_trace(go.Bar(name=f"{row['Model']} Crisis", x=[row['Model']],
                                 y=[crisis_pct], marker_color='#dc3545', opacity=0.7))
        target_pct = float(regime_df.iloc[0]['Target'].replace('%', '')) if '%' in str(regime_df.iloc[0]['Target']) else float(regime_df.iloc[0]['Target']) * 100
        fig.add_hline(y=target_pct, line_dash='dash', line_color='black',
                      annotation_text=f'Target {target_pct}%')
        fig.update_layout(title='Exceedance Rate by Regime', barmode='group',
                          yaxis_title='Exceedance Rate (%)', showlegend=False, **pf.LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

# Row 3: detailed statistics
st.markdown("### Detailed Statistics")
if len(df) > 0:
    detail = df[['Model', 'N_days', 'Exceedances', 'Rate', 'Cluster', 'MF_mean', 'MF_p']].copy()
    detail.columns = ['Model', 'Days', 'Exceedances', 'Rate', 'Clustering Ratio', 'MF Residual Mean', 'MF p-value']
    st.dataframe(detail, use_container_width=True, hide_index=True)
