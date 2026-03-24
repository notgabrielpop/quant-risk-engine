"""Page 3: Multi-asset portfolio risk with copulas."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'python'))

from visualization.components.data_loader import (
    load_portfolio_risk, load_euler_allocation, load_regime_copula,
)
from visualization.components import plot_factory as pf

st.set_page_config(page_title="Portfolio Risk", layout="wide")
st.title("Portfolio Risk Analysis")
st.markdown("Multi-asset dependence modeling with vine copulas and Euler allocation.")

port_risk = load_portfolio_risk()
euler = load_euler_allocation()
regime = load_regime_copula()

# Sidebar: weight controls
assets = euler['Asset'].unique().tolist() if 'Asset' in euler.columns else ['IBM', 'AAPL', 'JPM', 'TLT', 'GLD', 'SPY']

with st.sidebar:
    st.header("Portfolio Weights")
    weights = {}
    for a in assets:
        weights[a] = st.slider(a, 0.0, 1.0, 1.0 / len(assets), 0.05)
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    st.markdown(f"_Normalized to sum = 1.0_")

    conf = st.selectbox("Confidence Level", ["95%", "99%"], index=1)

# Row 1: key metrics
equal_df = port_risk[port_risk['Weighting'] == 'Equal'] if 'Weighting' in port_risk.columns else port_risk

c1, c2, c3 = st.columns(3)

vine_row = equal_df[equal_df['Model'].str.contains('Vine', case=False, na=False)]
gauss_row = equal_df[equal_df['Model'].str.contains('Gaussian', case=False, na=False)]
indep_row = equal_df[equal_df['Model'].str.contains('Independent', case=False, na=False)]

es_col = 'ES_99' if conf == '99%' else 'ES_95'
var_col = 'VaR_99' if conf == '99%' else 'VaR_95'

vine_es = vine_row[es_col].values[0] if len(vine_row) > 0 else 0
gauss_es = gauss_row[es_col].values[0] if len(gauss_row) > 0 else 0
indep_es = indep_row[es_col].values[0] if len(indep_row) > 0 else 0

c1.metric(f"Vine Copula ES {conf}", f"{vine_es*100:.2f}%")
div_benefit = (1 - vine_es / indep_es) * 100 if indep_es > 0 else 0
c2.metric("Diversification Benefit", f"{div_benefit:.1f}%", "vs independent")
model_risk_pct = (vine_es / gauss_es - 1) * 100 if gauss_es > 0 else 0
c3.metric("Copula Model Risk", f"{model_risk_pct:+.1f}%", "Vine vs Gaussian")

st.markdown("---")

# Row 2: Euler allocation + risk comparison
c1, c2 = st.columns(2)

with c1:
    euler_eq = euler[euler['Weighting'] == 'Equal'] if 'Weighting' in euler.columns else euler
    if len(euler_eq) > 0:
        fig = pf.euler_pie(
            euler_eq['Asset'].tolist(),
            euler_eq['Weight'].tolist(),
            euler_eq['ES_Contribution'].tolist(),
            'Euler Risk Allocation (Equal Weights)',
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

with c2:
    # Risk comparison bars by dependence model
    risk_models = {}
    for _, row in equal_df.iterrows():
        risk_models[row['Model']] = {
            'var_95': row.get('VaR_95', 0), 'var_99': row.get('VaR_99', 0),
            'es_95': row.get('ES_95', 0), 'es_99': row.get('ES_99', 0),
        }
    if risk_models:
        fig = pf.risk_bars(risk_models, f'Portfolio Risk by Dependence Model')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# Row 3: risk contribution analysis
st.markdown("### Risk Contribution Analysis")
if len(euler_eq) > 0:
    display_df = euler_eq[['Asset', 'Weight', 'ES_Contribution', 'Pct_of_ES']].copy()
    display_df['Weight (%)'] = (display_df['Weight'] * 100).round(1)
    display_df['Risk Contribution (%)'] = (display_df['Pct_of_ES'] * 100).round(1)
    display_df['Risk/Weight Ratio'] = (display_df['Pct_of_ES'] / display_df['Weight']).round(2)
    st.dataframe(
        display_df[['Asset', 'Weight (%)', 'Risk Contribution (%)', 'Risk/Weight Ratio']],
        use_container_width=True, hide_index=True,
    )
    st.info("Assets with **Risk/Weight Ratio > 1** contribute more risk than their weight suggests. "
            "Consider reducing exposure to these assets for better risk-adjusted allocation.")

# Row 4: regime analysis
if regime is not None and len(regime) > 0:
    st.markdown("### Regime Dependence Analysis")
    st.dataframe(regime, use_container_width=True, hide_index=True)
