#pragma once

/**
 * @file path_data.hpp
 * @brief SoA (Struct-of-Arrays) storage for Monte Carlo paths.
 *
 * Step-major layout: prices[step * n_paths + path_idx] so that all prices
 * at a given timestep are contiguous in memory for cache efficiency and SIMD.
 */

#include <cstddef>
#include <vector>

namespace qre {

struct PathData {
    size_t n_paths;
    size_t n_steps;

    // SoA layout: prices[step * n_paths + path_idx]
    std::vector<double> prices;
    std::vector<double> variances; // For 2D models (Heston)

    PathData(size_t paths, size_t steps, bool has_variance = false)
        : n_paths(paths), n_steps(steps),
          prices(paths * (steps + 1), 0.0) // +1 for initial condition
    {
        if (has_variance) {
            variances.resize(paths * (steps + 1), 0.0);
        }
    }

    double& price(size_t path, size_t step) { return prices[step * n_paths + path]; }
    double  price(size_t path, size_t step) const { return prices[step * n_paths + path]; }

    double& variance(size_t path, size_t step) { return variances[step * n_paths + path]; }
    double  variance(size_t path, size_t step) const { return variances[step * n_paths + path]; }

    const double* terminal_prices() const { return &prices[n_steps * n_paths]; }
    double*       terminal_prices()       { return &prices[n_steps * n_paths]; }
};

// ---------------------------------------------------------------------------
// Model parameter structs
// ---------------------------------------------------------------------------

struct ModelParams {
    double S0      = 100.0;
    double T       = 1.0;
    size_t n_steps = 252;

    double dt() const { return T / static_cast<double>(n_steps); }
};

struct GBMParams : ModelParams {
    double mu    = 0.05;
    double sigma = 0.20;
};

struct HestonParams : ModelParams {
    double mu      = 0.05;
    double v0      = 0.04;
    double kappa   = 2.0;
    double theta   = 0.04;
    double sigma_v = 0.3;
    double rho     = -0.7;
};

struct RoughHestonParams : HestonParams {
    double H = 0.1;
};

} // namespace qre
