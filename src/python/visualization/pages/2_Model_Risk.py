"""Page 2: Three-model comparison — model risk quantification."""

import streamlit as st
import numpy as np
import sys, os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'python'))

from visualization.components.engine_wrapper import get_engine, compute_risk_metrics
from visualization.components import plot_factory as pf
from visualization.components.data_loader import load_calibrated_params

st.set_page_config(page_title="Model Risk", layout="wide")
st.title("Model Risk Comparison")
st.markdown("Compare risk estimates across GBM, Heston, and Rough Heston side-by-side.")

engine = get_engine()

# Load calibrated defaults
cal = load_calibrated_params()
h_row = cal[(cal['Asset'] == 'SPY') & (cal['Model'] == 'heston')]
rh_row = cal[(cal['Asset'] == 'SPY') & (cal['Model'] == 'rough_heston')]

h_def = h_row.iloc[0] if len(h_row) > 0 else None
rh_def = rh_row.iloc[0] if len(rh_row) > 0 else None

with st.sidebar:
    st.header("Shared Parameters")
    S0 = st.number_input("S₀", value=100.0, step=10.0)
    mu = st.slider("μ", 0.0, 0.15, 0.05, 0.01)
    T = st.selectbox("T", [0.25, 0.5, 1.0, 2.0, 5.0, 10.0], index=2)
    n_paths = st.selectbox("Paths", [50_000, 100_000, 500_000], index=1,
                           format_func=lambda x: f"{x:,}")
    n_steps = int(252 * T)

    st.header("Heston / RH Parameters")
    v0 = st.slider("v₀", 0.005, 0.20, float(h_def['v0']) if h_def is not None else 0.04, 0.005)
    kappa = st.slider("κ", 0.1, 10.0, float(h_def['kappa']) if h_def is not None else 2.0, 0.1)
    theta = st.slider("θ", 0.005, 0.15, float(h_def['theta']) if h_def is not None else 0.04, 0.005)
    sigma_v = st.slider("σᵥ", 0.05, 1.0, float(h_def['sigma_v']) if h_def is not None else 0.30, 0.05)
    rho = st.slider("ρ", -0.95, -0.10, float(h_def['rho']) if h_def is not None else -0.70, 0.05)
    H = st.slider("H (Rough Heston)", 0.05, 0.25,
                   float(rh_def['H']) if rh_def is not None else 0.10, 0.01)

    run = st.button("Run All Models", type="primary", use_container_width=True)

sigma_bs = np.sqrt(theta)

if run or 'mr_results' not in st.session_state:
    with st.spinner("Simulating 3 models..."):
        gbm = engine.simulate_gbm(S0, mu, sigma_bs, T, n_paths, n_steps)
        hes = engine.simulate_heston(S0, mu, v0, kappa, theta, sigma_v, rho, T, n_paths, n_steps)
        rh = engine.simulate_rough_heston(S0, mu, v0, kappa, theta, sigma_v, rho, H, T, n_paths, n_steps)
    st.session_state['mr_results'] = (gbm, hes, rh)

gbm, hes, rh = st.session_state['mr_results']

# Compute risk
risk = {}
for name, res in [('GBM', gbm), ('Heston', hes), ('Rough Heston', rh)]:
    r95 = compute_risk_metrics(res['returns'], 0.95)
    r99 = compute_risk_metrics(res['returns'], 0.99)
    risk[name] = {
        'var_95': r95['var'] * S0, 'var_99': r99['var'] * S0,
        'es_95': r95['es'] * S0, 'es_99': r99['es'] * S0,
        'skewness': r99['skewness'], 'kurtosis': r99['kurtosis'],
    }

# Row 1: overlaid distributions
fig = pf.distribution_plot(
    {'GBM': gbm['returns'], 'Heston': hes['returns'], 'Rough Heston': rh['returns']},
    'Return Distribution: Three Models Compared',
)
st.plotly_chart(fig, use_container_width=True)

# Row 2: risk bars + metrics
c1, c2 = st.columns([2, 1])
with c1:
    fig = pf.risk_bars(risk, 'Risk Metrics Comparison (per $100 invested)')
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.markdown("#### Model Risk Quantification")
    es99_gbm = risk['GBM']['es_99']
    es99_hes = risk['Heston']['es_99']
    es99_rh = risk['Rough Heston']['es_99']
    model_risk = max(es99_hes, es99_rh) - es99_gbm
    pct = (max(es99_hes, es99_rh) / es99_gbm - 1) * 100 if es99_gbm > 0 else 0

    st.metric("Model Risk (ES₉₉)", f"${model_risk:.2f}", f"per $100 invested")
    st.metric("GBM Underestimation", f"{pct:.0f}%", "vs worst stochastic vol model")
    st.markdown("---")
    for name in ['GBM', 'Heston', 'Rough Heston']:
        st.metric(f"{name} ES₉₉", f"${risk[name]['es_99']:.2f}")

st.info("The difference between GBM and Heston ES99 represents **model risk** — "
        "risk arising from the choice of mathematical framework, not from market movements.")

# Row 3: QQ comparison
c1, c2, c3 = st.columns(3)
for col, (name, res) in zip([c1, c2, c3],
                             [('GBM', gbm), ('Heston', hes), ('Rough Heston', rh)]):
    with col:
        color = pf.COLORS.get(name.lower().replace(' ', '_'), '#1f77b4')
        fig = pf.qq_plot(res['returns'], f'{name} QQ-Plot', color=color)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
