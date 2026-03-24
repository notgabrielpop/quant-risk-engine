# Building a Quant Risk Engine from Scratch — Part 2: When Diversification Fails and Models Lie

*Pop Gabriel-Razvan | March 2026*

---

Last week I showed that your volatility model lies about risk by 30%. This week: it gets worse.

In Part 1, I built a C++ stochastic volatility simulation engine and demonstrated that switching from the industry-standard Black-Scholes assumption (Geometric Brownian Motion) to Heston stochastic volatility increases Expected Shortfall estimates by 30%. That was the simulation result — a controlled experiment with known parameters. Part 2 is the proof — on real market data, with real portfolio consequences, and with real money at stake.

Here's what we built and what we found.

## 191x Faster Convergence with Sobol QMC

Before we could run any of the analyses below, we needed to solve a computational problem. Standard Monte Carlo converges at O(1/sqrt(N)) — halving the error means quadrupling the paths. For Expected Shortfall at the 99th percentile, where only 1% of simulated paths contribute to the estimate, this inefficiency is compounded. The effective sample size for the tail is N/100, and the resulting estimator has high variance. Running production-quality risk across thousands of positions becomes a bottleneck.

Sobol quasi-random sequences replace pseudorandom samples with low-discrepancy numbers that fill the sample space more uniformly. Instead of clumping randomly (as pseudorandom generators inevitably do), Sobol sequences systematically avoid gaps, achieving near-O(1/N) convergence for smooth integrands. We implemented the Joe-Kuo direction numbers with Brownian bridge path construction, which concentrates the most important dimensions onto the lowest-discrepancy Sobol coordinates.

The result: **191x lower standard error** for mean estimation with the same number of paths. For ES estimation specifically, the improvement is more modest (~1.5x) because tail statistics are less smooth, but still meaningful.

We supplemented QMC with two classical variance reduction techniques:
- **Antithetic variates** exploit the symmetry of the normal distribution — for each path, simulate a mirror path with negated innovations. This achieves an **80% SE reduction** for symmetric functionals.
- **Control variates** use GBM (which has a known analytical solution) as a control for Heston. Both models are driven by identical random numbers, and we subtract the GBM estimation error scaled by an optimal coefficient. This provides a **28% SE reduction** with a correlation of 0.69 between the control and the target.

An interesting side effect: that control variate correlation of 0.69 is itself informative. It measures how far Heston has departed from the constant-volatility assumption. As model risk increases (more complex true dynamics, more extreme parameters), the control becomes less effective. The correlation gap *is* the model risk, expressed as a number between 0 and 1.

## When Diversification Fails — Vine Copulas

This is the finding that surprised me most.

In Part 1's empirical analysis, we documented that equity pair correlations jump from 0.04-0.08 in calm markets to 0.37-0.42 during crashes. But Pearson correlation is a linear, symmetric measure — it cannot distinguish between "stocks move together in rallies" and "stocks crash together." For risk management, only the second scenario matters, and it requires a different mathematical tool.

Enter copulas. Sklar's theorem tells us that any multivariate distribution can be decomposed into marginal distributions and a copula function that captures the pure dependence structure. This separation is powerful: we can combine sophisticated stochastic volatility marginals (from our C++ engine) with realistic multivariate tail dependence (from the copula), without forcing either component to compromise.

We fit a vine copula to 6 assets (SPY, IBM, JPM, AAPL, TLT, GLD) using Student-t bivariate copulas at each node. Vine copulas decompose the 6-dimensional dependence problem into a cascade of bivariate copulas arranged in a tree structure — each pair is modeled separately, and the tree structure is selected automatically by maximizing Kendall's tau across levels.

The key result: **equity tail dependence jumps from 0.01 to 0.49 during crises** — a 49-fold increase. Your portfolio looks diversified in calm markets. During crashes, it isn't.

The portfolio-level numbers tell the story:

- **Vine Copula ES99: 4.34%** — this is the honest tail risk estimate accounting for crash dependence
- **Gaussian Copula ES99: 3.78%** — underestimates by 15% because it assumes the same dependence in tails as in normal markets
- **Independent ES99: 2.62%** — what you'd estimate if you ignored dependence entirely

The Euler risk allocation is even more revealing. Despite holding only 16.7% of the portfolio (equal weight), **JPM contributes 38.1% of tail risk** because it is highly correlated with SPY and AAPL in the left tail. IBM contributes 18.2%, roughly proportional to its weight. GLD provides marginal diversification at 2.3%.

The standout finding: **TLT contributes -2.3% of risk** — it is the only asset that genuinely hedges portfolio tail risk, because government bonds tend to rally during equity crashes. This negative risk contribution means that adding TLT to an equity portfolio doesn't just reduce diversifiable risk — it actively offsets tail losses from other assets. A Gaussian copula would underestimate this benefit because it can't model the asymmetry between upper and lower tail dependence.

## Proof on Real Data — 1,053 Days of Walk-Forward Backtesting

Theory is cheap. Simulations are controlled. Does it work on data the models have never seen?

We ran a walk-forward backtest on **1,053 trading days** (2022-2025) for SPY, IBM, and JPM. The setup mirrors how a production risk system operates: at each date, calibrate model parameters on all prior data using differential evolution, simulate 100,000 one-day-ahead scenarios, compute VaR at the 99th percentile, and compare with the realized return. Full recalibration happens every 60 trading days; the current variance estimate v0 is updated daily from a 20-day realized volatility window.

We tested four models: ARIMA (time-series baseline), GBM (constant volatility), Heston, and Rough Heston. For each, we applied two formal statistical tests — Kupiec (does the exceedance rate match the target?) and Christoffersen (are exceedances independent, or do they cluster?).

The results for SPY at the 99% level:

| Model | Exceedances | Rate | Kupiec Test | Christoffersen | Basel Zone |
|-------|------------|------|-------------|----------------|------------|
| ARIMA | 15 | 1.43% | PASS (p=0.19) | FAIL (p=0.02) | GREEN |
| GBM | 22 | 2.09% | **FAIL** (p=0.002) | **FAIL** (p=0.002) | YELLOW |
| Heston | 14 | 1.33% | PASS (p=0.31) | PASS (p=0.24) | GREEN |
| Rough Heston | 14 | 1.33% | PASS (p=0.31) | PASS (p=0.24) | GREEN |

**GBM is formally rejected.** It produces 22 exceedances where approximately 10.5 are expected, failing both tests. Its VaR is systematically too narrow because constant volatility cannot adapt to regime changes.

ARIMA passes the Kupiec coverage test, but fails Christoffersen — its violations cluster **10.6x** more than expected. Zero exceedances during calm periods, then 9.3% during crises. It fails in bursts, which is arguably more dangerous than a uniform over-count because clustered violations compound into catastrophic multi-day losses.

**Heston and Rough Heston both achieve Basel GREEN** and pass both statistical tests. The stochastic variance process adapts to regime changes: when realized volatility rises, v0 updates upward, widening the one-day VaR automatically.

The regime decomposition is the most striking result: GBM's crisis exceedance rate is **12.4%** — on one in eight crisis days, losses exceed the stated VaR. That's 12x the target. Heston cuts this in half to **6.2%**. Still elevated, but the improvement is dramatic and statistically significant.

## The Dollar Cost — Dynamic Hedging

Here's where abstract risk numbers become concrete money.

A trader sells a 3-month ATM call option on SPY (spot = $100, strike = $100), collects $4.62 in Black-Scholes premium, and delta-hedges daily using the standard BS formula with assumed constant volatility of 20%. If the market truly follows GBM, the hedge is nearly perfect. If it doesn't — if volatility is actually stochastic — systematic hedge errors accumulate because the true delta depends on the instantaneous variance, which the trader ignores.

We simulated 100,000 hedge P&L paths under three "true" market dynamics using empirically calibrated SPY parameters:

| True Market | P&L Std | ES99 | vs GBM |
|------------|---------|------|--------|
| GBM (model is correct) | $0.52 | $0.90 | 1x |
| Heston | $1.72 | **$5.53** | **6x** |
| Rough Heston | $2.34 | **$10.42** | **12x** |

Under GBM, the hedge works as advertised — ES99 is $0.90, well within the $4.62 premium collected. Under Heston, ES99 jumps to $5.53 — already exceeding the premium. Under Rough Heston, where volatility has a rough, fractal-like memory structure that's even harder to track, ES99 reaches **$10.42** — a 12x increase the trader never sees coming.

**On a $10M notional position, this translates to a hidden tail risk of $553,000 to $1,042,000.** This is not a stress scenario — it is the expected 99th-percentile loss under empirically calibrated parameters.

Even a simple improvement — updating the hedge volatility daily from realized variance instead of using constant sigma — reduces Heston ES99 from $5.53 to $4.15 (a 25% improvement). But the residual risk remains substantial. Model risk cannot be hedged away entirely; it can only be acknowledged and managed.

## Try It Yourself

The full risk engine is available as an interactive dashboard at **[quantriskeng.streamlit.app](https://quantriskeng.streamlit.app)**. Five pages let you adjust parameters in real time: simulate individual models, compare three-model risk side by side, explore portfolio copula dependence, review walk-forward backtest results, and analyze dynamic hedging scenarios. The dashboard runs a pure Python simulation backend in the cloud — the same mathematical models, accessible to anyone with a browser.

## The Bottom Line

Three numbers to remember from this project:

1. **30% ES underestimation** from using GBM instead of stochastic volatility
2. **15% portfolio risk underestimation** from using Gaussian instead of vine copulas
3. **6-12x hidden hedging tail risk** for Black-Scholes-assuming traders

Combined, using the standard toolkit (GBM + Gaussian copula + Black-Scholes hedging) can underestimate tail risk by over 40%. This isn't a hypothetical stress scenario — it's the baseline estimate under empirically calibrated parameters, validated on 1,053 days of real market data.

The question was never "What will the price be?" The question is "What does the full landscape of possibilities look like, how confident are we in our map, and what does it cost when the map is wrong?"

---

*The full technical paper with mathematical derivations, statistical test formulations, and additional figures is available as a PDF. Code is open-source at [github.com/notgabrielpop/quant-risk-engine](https://github.com/notgabrielpop/quant-risk-engine). Part 1 (C++ simulation engine, three-model hierarchy, initial model risk results) is linked in the comments.*

*If you're a quant, risk manager, or just curious about how markets really work — the code is open source and the dashboard is live. Pull requests welcome.*

*Built with C++20, OpenMP, pybind11, Sobol QMC, Python, Streamlit, and Plotly.*
