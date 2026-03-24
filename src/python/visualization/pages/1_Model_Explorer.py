"""Page 1: Interactive single-model simulation."""

import streamlit as st
import numpy as np
import sys, os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'python'))

from visualization.components.engine_wrapper import get_engine, compute_risk_metrics
from visualization.components import plot_factory as pf

st.set_page_config(page_title="Model Explorer", layout="wide")
st.title("Model Explorer")
st.markdown("Simulate and visualize individual stochastic models interactively.")

engine = get_engine()

# ── Sidebar controls ──
with st.sidebar:
    st.header("Parameters")
    model = st.selectbox("Model", ["GBM", "Heston", "Rough Heston"])
    S0 = st.number_input("S₀", value=100.0, min_value=1.0, max_value=1000.0, step=10.0)
    mu = st.slider("μ (drift)", 0.0, 0.15, 0.05, 0.01)
    T = st.selectbox("T (years)", [0.25, 0.5, 1.0, 2.0, 5.0, 10.0], index=2)
    n_paths = st.selectbox("Paths", [10_000, 50_000, 100_000, 500_000], index=2,
                           format_func=lambda x: f"{x:,}")
    n_steps = int(252 * T)

    if model == "GBM":
        sigma = st.slider("σ (volatility)", 0.05, 0.80, 0.20, 0.01)
    else:
        v0 = st.slider("v₀ (initial variance)", 0.005, 0.20, 0.04, 0.005)
        kappa = st.slider("κ (mean reversion)", 0.5, 10.0, 2.0, 0.1)
        theta = st.slider("θ (long-run variance)", 0.005, 0.15, 0.04, 0.005)
        sigma_v = st.slider("σᵥ (vol of vol)", 0.05, 1.0, 0.30, 0.05)
        rho = st.slider("ρ (correlation)", -0.95, -0.10, -0.70, 0.05)
        if model == "Rough Heston":
            H = st.slider("H (Hurst)", 0.05, 0.45, 0.10, 0.01)

# ── Simulate ──
color = pf.COLORS.get(model.lower().replace(' ', '_'), '#1f77b4')

with st.spinner("Simulating..."):
    if model == "GBM":
        result = engine.simulate_gbm(S0, mu, sigma, T, n_paths, n_steps)
    elif model == "Heston":
        result = engine.simulate_heston(S0, mu, v0, kappa, theta, sigma_v, rho, T, n_paths, n_steps)
    else:
        result = engine.simulate_rough_heston(S0, mu, v0, kappa, theta, sigma_v, rho, H, T, n_paths, n_steps)

returns = result['returns']
rm95 = compute_risk_metrics(returns, 0.95)
rm99 = compute_risk_metrics(returns, 0.99)

# ── Layout: 2×2 grid ──
row1_l, row1_r = st.columns(2)

with row1_l:
    fig = pf.distribution_plot({model: returns}, f'{model} Return Distribution', show_var=True)
    st.plotly_chart(fig, use_container_width=True)

with row1_r:
    fig = pf.qq_plot(returns, f'{model} QQ-Plot vs Normal', color=color)
    st.plotly_chart(fig, use_container_width=True)

row2_l, row2_r = st.columns(2)

with row2_l:
    if result.get('prices') is not None:
        fig = pf.price_paths(result['prices'], n_show=20,
                             title=f'{model} Sample Paths', color=color)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Full price paths not available in precomputed mode.")

with row2_r:
    st.markdown("#### Risk Metrics")
    mc1, mc2 = st.columns(2)
    mc1.metric("VaR 95%", f"${rm95['var']*S0:.2f}")
    mc1.metric("VaR 99%", f"${rm99['var']*S0:.2f}")
    mc2.metric("ES 95%", f"${rm95['es']*S0:.2f}")
    mc2.metric("ES 99%", f"${rm99['es']*S0:.2f}")
    st.markdown("---")
    mc3, mc4 = st.columns(2)
    mc3.metric("Skewness", f"{rm99['skewness']:.3f}")
    mc3.metric("Excess Kurtosis", f"{rm99['kurtosis']:.3f}")
    mc4.metric("Mean Return", f"{rm99['mean']*100:.2f}%")
    mc4.metric("Std Dev", f"{rm99['std']*100:.2f}%")
