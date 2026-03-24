"""Page 5: Dynamic hedging simulation and hidden risk."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'python'))

from visualization.components.data_loader import load_hedging_results
from visualization.components import plot_factory as pf

st.set_page_config(page_title="Dynamic Hedging", layout="wide")
st.title("Dynamic Hedging & Hidden Risk")
st.markdown("Quantifying the P&L risk a Black-Scholes hedger doesn't see.")

hedging = load_hedging_results()

with st.sidebar:
    st.header("Scenario")
    scenarios = hedging['Scenario'].unique().tolist()
    scenario = st.selectbox("True Dynamics", scenarios, index=1)
    strategies = hedging['Strategy'].unique().tolist()
    strategy = st.selectbox("Hedge Strategy", strategies, index=0)

# Get selected row + GBM baseline
sel = hedging[(hedging['Scenario'] == scenario) & (hedging['Strategy'] == strategy)]
gbm_row = hedging[(hedging['Scenario'] == 'GBM true') & (hedging['Strategy'] == 'constant')]

if len(sel) == 0:
    st.warning("No data for this combination.")
    st.stop()

sel = sel.iloc[0]
gbm = gbm_row.iloc[0] if len(gbm_row) > 0 else sel

# Row 1: key metrics
c1, c2, c3 = st.columns(3)
c1.metric("Hedge P&L Std", f"${sel['PnL_Std']:.2f}",
          f"{sel['PnL_Std'] / gbm['PnL_Std']:.1f}x vs GBM" if gbm['PnL_Std'] > 0 else "")
c2.metric("ES₉₉", f"${sel['ES_99']:.2f}",
          f"{sel['ES_99'] / gbm['ES_99']:.1f}x vs GBM" if gbm['ES_99'] > 0 else "")
c3.metric("P(loss > 2x premium)", f"{sel['P_loss_2x']*100:.2f}%")

st.markdown("---")

# Row 2: comparison table
st.markdown("### All Scenarios")
display = hedging.copy()
display['PnL_Mean'] = display['PnL_Mean'].round(3)
display['PnL_Std'] = display['PnL_Std'].round(3)
display['Skewness'] = display['Skewness'].round(3)
display['Kurtosis'] = display['Kurtosis'].round(3)
display['VaR_99'] = display['VaR_99'].round(3)
display['ES_99'] = display['ES_99'].round(3)
display['P_loss_2x'] = (display['P_loss_2x'] * 100).round(2).astype(str) + '%'
st.dataframe(display, use_container_width=True, hide_index=True)

st.markdown("---")

# Row 3: comparison bar charts
c1, c2 = st.columns(2)

with c1:
    # ES99 by scenario (constant strategy only)
    const = hedging[hedging['Strategy'] == 'constant']
    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#2ca02c']
    for i, (_, row) in enumerate(const.iterrows()):
        fig.add_trace(go.Bar(
            x=[row['Scenario']], y=[row['ES_99']],
            name=row['Scenario'],
            marker_color=colors[i % len(colors)],
        ))
    fig.update_layout(title='ES₉₉ by True Dynamics (Constant Vol Hedge)',
                      yaxis_title='ES₉₉ ($)', showlegend=False, **pf.LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    # Strategy comparison for selected scenario
    scenario_data = hedging[hedging['Scenario'] == scenario]
    if len(scenario_data) > 1:
        fig = go.Figure()
        for i, (_, row) in enumerate(scenario_data.iterrows()):
            fig.add_trace(go.Bar(
                x=['P&L Std', 'ES₉₉'],
                y=[row['PnL_Std'], row['ES_99']],
                name=row['Strategy'],
                marker_color=colors[i % len(colors)],
            ))
        fig.update_layout(title=f'Strategy Comparison: {scenario}',
                          barmode='group', yaxis_title='$ Value', **pf.LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

# Row 4: Hidden risk callout
st.markdown("---")
hidden = sel['ES_99'] - gbm['ES_99']
if hidden > 0:
    scaled = hidden * 100_000 / 100  # scale to $10M position
    st.warning(
        f"A trader using Black-Scholes on a **$10M SPY** position underestimates "
        f"their worst-case hedging loss by **${scaled:,.0f}** at the 99th percentile. "
        f"This is the **hidden risk** of model misspecification."
    )
else:
    st.info("Under GBM true dynamics, the Black-Scholes hedge works as expected.")

# Option details
with st.expander("Option Details"):
    st.markdown(f"""
    - **Spot**: $100 (ATM)
    - **Strike**: $100
    - **Maturity**: 3 months (63 trading days)
    - **BS vol**: 20%
    - **Premium received**: ${sel['Premium']:.2f}
    - **Rebalancing**: Daily
    - **Paths**: 100,000
    """)
