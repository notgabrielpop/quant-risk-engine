#pragma once

/**
 * @file sobol.hpp
 * @brief Sobol quasi-random number generator for variance reduction.
 *
 * Generates low-discrepancy Sobol sequences for quasi-Monte Carlo
 * integration. Provides better convergence rates (O(1/N) vs O(1/sqrt(N)))
 * than pseudo-random generators for smooth integrands.
 *
 * TODO:
 *   - Implement Joe-Kuo direction numbers
 *   - Scrambled Sobol for randomized QMC
 *   - Multi-dimensional generation for correlated assets
 */

#include <cstddef>
#include <cstdint>
#include <vector>

namespace qre::rng {

class SobolGenerator {
public:
    /// @param dimension Number of dimensions (assets/factors)
    /// @param seed      Scrambling seed (0 = no scrambling)
    explicit SobolGenerator(std::size_t dimension, std::uint64_t seed = 0)
        : dimension_{dimension}, seed_{seed}, index_{0} {}

    /// Generate the next point in [0,1)^d
    [[nodiscard]] auto next() -> std::vector<double> {
        // TODO: implement Sobol sequence generation
        ++index_;
        return std::vector<double>(dimension_, 0.5); // placeholder
    }

    /// Skip ahead to the n-th point
    void skip(std::size_t n) { index_ = n; }

    /// Reset to the beginning
    void reset() { index_ = 0; }

    [[nodiscard]] auto dimension() const -> std::size_t { return dimension_; }
    [[nodiscard]] auto index() const -> std::size_t { return index_; }

private:
    std::size_t dimension_;
    std::uint64_t seed_;
    std::size_t index_;
};

} // namespace qre::rng
