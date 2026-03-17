#pragma once

/**
 * @file importance_sampling.hpp
 * @brief Importance sampling for rare-event simulation in risk estimation.
 *
 * Shifts the sampling distribution to oversample tail events, then
 * corrects with likelihood ratios. Critical for accurate deep-tail
 * VaR/ES estimation where naive MC has high variance.
 *
 * Uses exponential tilting (Esscher transform) for GBM-like models:
 *   dQ/dP = exp(theta * X - psi(theta))
 *
 * TODO:
 *   - Implement optimal theta selection via cross-entropy method
 *   - Adaptive importance sampling with mixture proposals
 */

#include <cmath>
#include <cstddef>
#include <vector>

namespace qre::variance_reduction {

class ImportanceSampling {
public:
    /// Compute likelihood ratios for exponential tilting.
    /// @param draws  Original standard normal draws
    /// @param theta  Tilting parameter (shift in mean)
    /// @return Vector of likelihood ratios dP/dQ
    [[nodiscard]] static auto likelihood_ratios(const std::vector<double>& draws,
                                                double theta) -> std::vector<double> {
        std::vector<double> ratios(draws.size());
        for (std::size_t i = 0; i < draws.size(); ++i) {
            // For normal(0,1) -> normal(theta, 1):
            // dP/dQ = exp(-theta * z + 0.5 * theta^2)
            ratios[i] = std::exp(-theta * draws[i] + 0.5 * theta * theta);
        }
        return ratios;
    }

    /// Apply IS correction: weighted estimator sum(w_i * f_i) / sum(w_i)
    [[nodiscard]] static auto weighted_mean(const std::vector<double>& values,
                                            const std::vector<double>& weights) -> double {
        double num = 0.0, den = 0.0;
        for (std::size_t i = 0; i < values.size(); ++i) {
            num += weights[i] * values[i];
            den += weights[i];
        }
        return (den > 0.0) ? num / den : 0.0;
    }
};

} // namespace qre::variance_reduction
