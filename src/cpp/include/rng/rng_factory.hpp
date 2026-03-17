#pragma once

/**
 * @file rng_factory.hpp
 * @brief Factory abstraction for random number generators.
 *
 * Provides a uniform interface for selecting between pseudo-random
 * (Mersenne Twister) and quasi-random (Sobol) generators at runtime.
 */

#include "sobol.hpp"

#include <cstddef>
#include <memory>
#include <random>
#include <variant>
#include <vector>

namespace qre::rng {

enum class RNGType {
    MersenneTwister,
    Sobol
};

class RNGFactory {
public:
    /// Create an RNG of the specified type
    /// @param type      Generator type
    /// @param dimension Number of dimensions
    /// @param seed      Random seed
    static auto create_normal_samples(RNGType type, std::size_t dimension,
                                      std::size_t n_samples, unsigned seed = 42)
        -> std::vector<std::vector<double>> {
        // TODO: implement generator dispatch
        std::vector<std::vector<double>> samples(n_samples, std::vector<double>(dimension));

        if (type == RNGType::MersenneTwister) {
            std::mt19937_64 rng(seed);
            std::normal_distribution<double> normal(0.0, 1.0);
            for (auto& sample : samples) {
                for (auto& val : sample) {
                    val = normal(rng);
                }
            }
        }
        // TODO: Sobol with inverse normal transform

        return samples;
    }
};

} // namespace qre::rng
