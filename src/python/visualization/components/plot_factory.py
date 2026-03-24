"""Centralized Plotly figure generators with consistent styling."""

import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

COLORS = {
    'gbm': '#1f77b4',
    'heston': '#ff7f0e',
    'rough_heston': '#d62728',
    'arima': '#9467bd',
    'real': '#7f7f7f',
    'normal': '#000000',
}

MODEL_NAMES = {
    'gbm': 'GBM (Black-Scholes)',
    'heston': 'Heston',
    'rough_heston': 'Rough Heston',
    'arima': 'ARIMA',
    'GBM': 'GBM',
    'Heston': 'Heston',
    'Rough Heston': 'Rough Heston',
}

LAYOUT = dict(
    font=dict(family="Inter, sans-serif", size=12),
    plot_bgcolor='white',
    paper_bgcolor='white',
    margin=dict(l=60, r=30, t=50, b=50),
    xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
    yaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
)


def distribution_plot(distributions, title, show_var=False, var_level=0.99):
    """Overlaid histograms from {name: returns_array} dict."""
    fig = go.Figure()
    color_list = list(COLORS.values())
    for i, (name, returns) in enumerate(distributions.items()):
        color = COLORS.get(name.lower().replace(' ', '_'), color_list[i % len(color_list)])
        fig.add_trace(go.Histogram(
            x=returns, nbinsx=200, histnorm='probability density',
            name=name, opacity=0.55,
            marker_color=color,
        ))
        if show_var:
            var_val = -np.percentile(returns, (1 - var_level) * 100)
            fig.add_vline(x=-var_val, line_dash='dash', line_color=color,
                          annotation_text=f'VaR {name}')

    # Normal overlay
    if distributions:
        first = list(distributions.values())[0]
        x = np.linspace(np.percentile(first, 0.5), np.percentile(first, 99.5), 300)
        fig.add_trace(go.Scatter(
            x=x, y=norm.pdf(x, np.mean(first), np.std(first)),
            mode='lines', name='Normal',
            line=dict(color='black', dash='dash', width=1.5),
        ))

    fig.update_layout(title=title, barmode='overlay',
                      xaxis_title='Return', yaxis_title='Density', **LAYOUT)
    return fig


def qq_plot(returns, title, color='#1f77b4'):
    """QQ-plot vs normal."""
    sorted_r = np.sort(returns)
    n = len(sorted_r)
    theoretical = norm.ppf(np.linspace(1/(n+1), n/(n+1), n))
    # Subsample for performance
    idx = np.linspace(0, n-1, min(1000, n), dtype=int)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=theoretical[idx], y=sorted_r[idx],
        mode='markers', marker=dict(size=3, color=color, opacity=0.5),
        name='Data',
    ))
    lims = [min(theoretical[idx].min(), sorted_r[idx].min()),
            max(theoretical[idx].max(), sorted_r[idx].max())]
    fig.add_trace(go.Scatter(
        x=lims, y=lims, mode='lines',
        line=dict(color='black', dash='dash', width=1),
        name='45° line',
    ))
    fig.update_layout(title=title, xaxis_title='Theoretical', yaxis_title='Sample', **LAYOUT)
    return fig


def risk_bars(risk_dict, title='Risk Metrics Comparison'):
    """Grouped bars: {model: {var_95, var_99, es_95, es_99}}."""
    metrics = ['VaR 95%', 'VaR 99%', 'ES 95%', 'ES 99%']
    keys = ['var_95', 'var_99', 'es_95', 'es_99']
    fig = go.Figure()
    color_list = list(COLORS.values())
    for i, (model, vals) in enumerate(risk_dict.items()):
        color = COLORS.get(model.lower().replace(' ', '_'), color_list[i % len(color_list)])
        fig.add_trace(go.Bar(
            name=model,
            x=metrics, y=[vals.get(k, 0) for k in keys],
            marker_color=color,
        ))
    fig.update_layout(title=title, barmode='group', yaxis_title='Loss ($)', **LAYOUT)
    return fig


def price_paths(prices, n_show=20, title='Sample Price Paths', color='#1f77b4'):
    """Plot n_show sample paths from (n_paths, n_steps+1) array."""
    fig = go.Figure()
    if prices is None:
        return fig
    n_paths, n_steps = prices.shape
    t = np.arange(n_steps)
    idx = np.linspace(0, n_paths - 1, min(n_show, n_paths), dtype=int)
    for i in idx:
        fig.add_trace(go.Scatter(
            x=t, y=prices[i], mode='lines',
            line=dict(color=color, width=0.7), opacity=0.4,
            showlegend=False,
        ))
    fig.update_layout(title=title, xaxis_title='Time Step', yaxis_title='Price ($)', **LAYOUT)
    return fig


def scorecard_heatmap(df, title='Model Scorecard'):
    """Heatmap: rows=models, cols=tests. Values are pass/fail."""
    models = df['Model'].unique()
    tests = ['Kupiec_p', 'Christ_p', 'Indep_p', 'Basel']
    test_labels = ['Kupiec', 'Christoffersen', 'Independence', 'Basel']

    z = []
    text = []
    for model in models:
        row_z = []
        row_t = []
        mdf = df[df['Model'] == model].iloc[0] if len(df[df['Model'] == model]) > 0 else None
        for t in tests:
            if mdf is None:
                row_z.append(0)
                row_t.append('N/A')
            elif t == 'Basel':
                val = str(mdf[t]).lower()
                row_z.append(1 if val == 'green' else (0.5 if val == 'yellow' else 0))
                row_t.append(str(mdf[t]))
            else:
                pval = float(mdf[t])
                row_z.append(1 if pval > 0.05 else (0.5 if pval > 0.01 else 0))
                row_t.append(f'p={pval:.3f}')
        z.append(row_z)
        text.append(row_t)

    colorscale = [[0, '#dc3545'], [0.5, '#ffc107'], [1, '#28a745']]
    fig = go.Figure(data=go.Heatmap(
        z=z, x=test_labels, y=list(models), text=text, texttemplate='%{text}',
        colorscale=colorscale, showscale=False,
        zmin=0, zmax=1,
    ))
    fig.update_layout(title=title, **LAYOUT, height=300)
    return fig


def backtest_timeline(dates, returns, var_series, model_name, color='#ff7f0e'):
    """VaR backtest timeline with exceedance markers."""
    fig = go.Figure()
    exc_mask = returns < -var_series
    fig.add_trace(go.Scatter(
        x=dates, y=returns, mode='markers',
        marker=dict(size=3, color='#aaaaaa'), name='Returns',
    ))
    fig.add_trace(go.Scatter(
        x=dates[exc_mask], y=returns[exc_mask], mode='markers',
        marker=dict(size=6, color='red', symbol='x'), name='Exceedances',
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=-var_series, mode='lines',
        line=dict(color=color, width=1.5), name=f'{model_name} VaR',
    ))
    fig.update_layout(title=f'VaR Backtest: {model_name}',
                      xaxis_title='Date', yaxis_title='Return', **LAYOUT)
    return fig


def pnl_distribution(pnl, title, color='#ff7f0e', show_tail=True):
    """Hedge P&L distribution with tail shading."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=pnl, nbinsx=200, histnorm='probability density',
        marker_color=color, opacity=0.7, name='P&L',
    ))
    var99 = -np.percentile(pnl, 1)
    es99_vals = pnl[pnl <= -var99]
    es99 = -np.mean(es99_vals) if len(es99_vals) > 0 else var99

    fig.add_vline(x=0, line_dash='dash', line_color='black', annotation_text='Break-even')
    fig.add_vline(x=-var99, line_dash='dot', line_color='red',
                  annotation_text=f'VaR99=${var99:.2f}')
    fig.add_vline(x=-es99, line_dash='dot', line_color='darkred',
                  annotation_text=f'ES99=${es99:.2f}')

    fig.update_layout(title=title, xaxis_title='P&L ($)', yaxis_title='Density', **LAYOUT)
    return fig


def euler_pie(assets, weights, contributions, title='Euler Allocation'):
    """Side-by-side: weights vs risk contribution."""
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'pie'}, {'type': 'pie'}]],
                        subplot_titles=['Portfolio Weights', 'Risk Contribution'])
    fig.add_trace(go.Pie(labels=assets, values=weights, hole=0.35,
                         textinfo='label+percent'), row=1, col=1)
    fig.add_trace(go.Pie(labels=assets, values=contributions, hole=0.35,
                         textinfo='label+percent'), row=1, col=2)
    fig.update_layout(title=title, **{k: v for k, v in LAYOUT.items() if k != 'xaxis' and k != 'yaxis'})
    return fig
