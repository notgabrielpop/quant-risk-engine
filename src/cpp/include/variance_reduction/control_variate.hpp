#pragma once

/**
 * @file control_variate.hpp
 * @brief Control variate variance reduction for Monte Carlo simulation.
 *
 * Uses GBM (known analytical mean) as a control variate for Heston/Rough Heston.
 * Both models are simulated with the SAME random numbers to maximize correlation.
 *
 *   Y_cv = Y + beta * (E[C] - C)
 *   beta_opt = Cov(Y, C) / Var(C)
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

namespace qre {

struct CVResult {
    std::vector<double> controlled_prices;  // Variance-reduced terminal prices
    double correlation;       // corr(heston, gbm) — should be high
    double variance_reduction; // Var(standard) / Var(controlled) — > 1 means improvement
    double beta;              // Optimal control coefficient
};

/// Estimate optimal beta = Cov(Y, C) / Var(C) from paired samples.
inline double cv_estimate_beta(const double* target, const double* control, size_t n) {
    double mean_y = 0, mean_c = 0;
    for (size_t i = 0; i < n; ++i) {
        mean_y += target[i];
        mean_c += control[i];
    }
    mean_y /= static_cast<double>(n);
    mean_c /= static_cast<double>(n);

    double cov = 0, var_c = 0;
    for (size_t i = 0; i < n; ++i) {
        double dy = target[i] - mean_y;
        double dc = control[i] - mean_c;
        cov   += dy * dc;
        var_c += dc * dc;
    }
    return (var_c > 0.0) ? (cov / var_c) : 0.0;
}

/// Compute correlation between two arrays.
inline double cv_correlation(const double* x, const double* y, size_t n) {
    double mx = 0, my = 0;
    for (size_t i = 0; i < n; ++i) { mx += x[i]; my += y[i]; }
    mx /= static_cast<double>(n);
    my /= static_cast<double>(n);

    double cov = 0, vx = 0, vy = 0;
    for (size_t i = 0; i < n; ++i) {
        double dx = x[i] - mx;
        double dy = y[i] - my;
        cov += dx * dy;
        vx  += dx * dx;
        vy  += dy * dy;
    }
    double denom = std::sqrt(vx * vy);
    return denom > 0 ? cov / denom : 0.0;
}

/// Apply control variate adjustment to terminal prices.
/// target:    Heston terminal prices
/// control:   GBM terminal prices (simulated with same Z)
/// E_control: GBM analytical mean = S0 * exp(mu * T)
/// Returns CVResult with controlled prices and diagnostics.
inline CVResult apply_control_variate(const double* target, const double* control,
                                       size_t n, double E_control) {
    CVResult result;
    result.beta = cv_estimate_beta(target, control, n);
    result.correlation = cv_correlation(target, control, n);

    result.controlled_prices.resize(n);
    for (size_t i = 0; i < n; ++i) {
        result.controlled_prices[i] = target[i] + result.beta * (E_control - control[i]);
    }

    // Compute variance reduction ratio
    double mean_t = 0, mean_cv = 0;
    for (size_t i = 0; i < n; ++i) {
        mean_t  += target[i];
        mean_cv += result.controlled_prices[i];
    }
    mean_t  /= static_cast<double>(n);
    mean_cv /= static_cast<double>(n);

    double var_t = 0, var_cv = 0;
    for (size_t i = 0; i < n; ++i) {
        double dt = target[i] - mean_t;
        double dc = result.controlled_prices[i] - mean_cv;
        var_t  += dt * dt;
        var_cv += dc * dc;
    }
    result.variance_reduction = (var_cv > 0) ? var_t / var_cv : 1.0;

    return result;
}

} // namespace qre
