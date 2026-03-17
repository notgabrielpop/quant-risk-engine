# Quant Risk Engine

Quantitative risk engine for multi-asset portfolio simulation. Compares deterministic vs. probabilistic market models (GBM, Heston, Rough Heston) with a C++20/CUDA Monte Carlo backend and Python analytics layer.

## Project Structure

- `src/cpp/` — C++20 simulation engine (models, Monte Carlo engine, risk metrics, copulas)
- `src/python/` — Python layer (data pipeline, calibration, backtesting, visualization)
- `notebooks/` — Jupyter notebooks for analysis
- `scripts/` — Standalone pipeline scripts
- `tests/` — C++ (Catch2) and Python (pytest) tests
- `docs/` — Paper and references

## Quick Start

```bash
pip install -r requirements.txt
python scripts/download_data.py
python scripts/run_pipeline.py
```

## Models

| Model | Description | Parameters |
|-------|-------------|------------|
| GBM | Geometric Brownian Motion | mu, sigma |
| Heston | Stochastic volatility | kappa, theta, xi, rho, v0 |
| Rough Heston | Fractional stochastic volatility | + Hurst exponent H |

## Building C++ Engine

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```
