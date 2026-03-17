#pragma once

/**
 * @file control_variate.hpp
 * @brief Control variate method for Monte Carlo variance reduction.
 *
 * Uses a correlated random variable with known expectation (the "control")
 * to reduce the variance of the target estimator:
 *
 *   Y_cv = Y - beta * (C - E[C])
 *
 * where beta = -Cov(Y,C)/Var(C) is the optimal coefficient.
 *
 * Common controls: GBM terminal price (known in closed form) when
 * estimating prices under stochastic vol models.
 *
 * TODO:
 *   - Implement optimal beta estimation from pilot run
 *   - Multi-control-variate extension
 */

#include <cstddef>
#include <numeric>
#include <vector>

namespace qre::variance_reduction {

class ControlVariate {
public:
    /// Estimate optimal beta from paired samples.
    [[nodiscard]] static auto estimate_beta(const std::vector<double>& target,
                                            const std::vector<double>& control) -> double {
        const std::size_t n = target.size();
        const double mean_y = std::accumulate(target.begin(), target.end(), 0.0) / static_cast<double>(n);
        const double mean_c = std::accumulate(control.begin(), control.end(), 0.0) / static_cast<double>(n);

        double cov = 0.0, var_c = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double dy = target[i] - mean_y;
            const double dc = control[i] - mean_c;
            cov   += dy * dc;
            var_c += dc * dc;
        }
        return (var_c > 0.0) ? -(cov / var_c) : 0.0;
    }

    /// Apply control variate adjustment.
    [[nodiscard]] static auto adjust(const std::vector<double>& target,
                                     const std::vector<double>& control,
                                     double expected_control,
                                     double beta) -> std::vector<double> {
        std::vector<double> adjusted(target.size());
        for (std::size_t i = 0; i < target.size(); ++i) {
            adjusted[i] = target[i] + beta * (control[i] - expected_control);
        }
        return adjusted;
    }
};

} // namespace qre::variance_reduction
