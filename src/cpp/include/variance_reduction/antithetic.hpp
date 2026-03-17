#pragma once

/**
 * @file antithetic.hpp
 * @brief Antithetic variates for variance reduction in Monte Carlo.
 *
 * For each random path driven by Z, generates a mirror path driven by -Z.
 * Exploits negative correlation to reduce variance of the estimator by
 * up to 50% for monotone payoffs.
 *
 * TODO:
 *   - Template wrapper that pairs original + antithetic paths
 *   - Automatic variance reduction ratio estimation
 */

#include <cstddef>
#include <vector>

namespace qre::variance_reduction {

class AntitheticVariates {
public:
    /// Given a vector of standard normal draws, return the negated draws.
    [[nodiscard]] static auto negate(const std::vector<double>& draws) -> std::vector<double> {
        std::vector<double> anti(draws.size());
        for (std::size_t i = 0; i < draws.size(); ++i) {
            anti[i] = -draws[i];
        }
        return anti;
    }

    /// Compute the antithetic estimator: (f(Z) + f(-Z)) / 2
    [[nodiscard]] static auto combine(double original, double antithetic) -> double {
        return 0.5 * (original + antithetic);
    }
};

} // namespace qre::variance_reduction
