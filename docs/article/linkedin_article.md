Your volatility model is underestimating tail risk by 30%. Here's how I proved it by building a Monte Carlo engine from scratch.

Most risk systems assume constant volatility. The real world doesn't.

---

## The Problem: Point Forecasts Fail When It Matters

I started this project by testing every standard forecasting model on SPY daily returns (2000-2024). ARIMA, Random Forest, XGBoost --- walk-forward, one-step-ahead, expanding window. The result:

| Model | RMSE | Directional Accuracy |
|-------|------|---------------------|
| Random Walk | 0.01117 | --- |
| ARIMA | 0.01121 | 50.5% |
| Random Forest | 0.01117 | 52.9% |
| XGBoost | 0.01126 | 49.7% |

Nothing beats a random walk. But that's not the real failure. The real failure is that ARIMA's 95% prediction intervals only achieve 88% coverage during crises. The exceedances cluster 3.8x more than expected. The model is systematically overconfident during the exact regimes that produce the largest losses.

The question isn't "What will the price be?" It's "What does the full distribution of outcomes look like?"

## Why Markets Aren't Gaussian

Before building the engine, I ran a comprehensive empirical analysis. The findings are striking:

**Fat tails**: JPM daily returns have excess kurtosis of 18.2. Five-sigma events occur 10,000x more often than a normal distribution predicts. On October 13, 2008, SPY returned +13.6% --- an 11.5-sigma event with a Gaussian probability of roughly 10^(-30).

**Volatility clustering**: Large moves predict more large moves. The autocorrelation of absolute returns persists for 60+ trading days. This is the signature of stochastic volatility.

**Tail dependence**: Equity correlations jump from 0.04-0.08 in calm markets to 0.37-0.42 during crashes. Diversification fails exactly when you need it.

## Three Generations of Volatility Models

I built a C++ Monte Carlo engine implementing three models as a progression:

**GBM (Black-Scholes, 1973)**: Constant volatility. The baseline everyone uses. Produces symmetric, light-tailed returns --- none of the features we observe in real data.

**Heston (1993)**: Volatility itself follows a stochastic process. Five parameters: mean-reversion speed, long-run variance, vol-of-vol, leverage correlation (typically -0.7 for equities), and initial variance. The negative correlation between price and volatility produces the skewed, fat-tailed returns we see empirically.

**Rough Heston (Gatheral, Jaisson, Rosenbaum, 2018)**: The state of the art. Replaces Heston's exponential memory with a fractional kernel (Hurst exponent H ~ 0.1). Volatility "remembers" past shocks with a power-law decay instead of exponential --- matching empirical measurements from implied volatility surfaces. For simulation, I approximate the fractional kernel using a sum of 8 exponential factors via spectral quadrature, converting the non-Markovian system into coupled Markovian ODEs.

## The Engine

Built in C++20 with:
- CRTP (compile-time polymorphism) --- all three models share one simulation loop, zero virtual-call overhead
- Struct of Arrays memory layout for cache efficiency
- OpenMP parallelization with deterministic per-thread RNG
- Andersen's QE scheme for Heston (eliminates negative variance by construction)
- Semi-exact factor evolution for Rough Heston
- pybind11 bridge to Python for analysis

Performance on 1M paths x 252 steps:

| Model | Throughput (8 threads) |
|-------|----------------------|
| GBM | 668K paths/sec |
| Heston (QE) | 500K paths/sec |
| Rough Heston | 309K paths/sec |

## The Result: 30% Model Risk

Here's the payoff --- risk metrics from 1 million simulated paths per model:

| Metric | GBM | Heston | Rough Heston |
|--------|-----|--------|--------------|
| VaR 99% | 35.36 | 44.58 | 44.78 |
| ES 99% | 39.52 | 51.31 | 51.81 |
| Skewness | +0.61 | -0.22 | -0.22 |

A risk manager using GBM estimates a 99th-percentile Expected Shortfall of $39.52 on a $100 position. Under stochastic volatility, it's $51.31. That's a 30% gap --- pure model risk, invisible if you only run one model.

Rough Heston adds something Heston can't: the right memory structure. The autocorrelation of absolute returns at lag 1 is 0.20 for Rough Heston vs. 0.08 for classical Heston --- matching the slow power-law decay observed in real data. This will matter when we backtest prediction intervals in Part 2.

## What's Next

Part 2 adds:
- Quasi-Monte Carlo (Sobol sequences) for faster convergence
- Multi-asset portfolio risk with vine copulas
- CUDA GPU acceleration for nested Monte Carlo
- Calibration on real implied volatility surfaces
- Full backtesting with the Christoffersen test

The question is no longer "What will the price be?" but "What does the full landscape of possibilities look like, and how confident are we in our map?"

---

Full technical paper with mathematical details and figures: [PDF version]

Source code: github.com/notgabrielpop/quant-risk-engine

*Pop Gabriel-Razvan | March 2026*
