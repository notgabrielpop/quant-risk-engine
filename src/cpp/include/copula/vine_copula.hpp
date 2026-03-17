#pragma once

/**
 * @file vine_copula.hpp
 * @brief Vine copula for high-dimensional dependence modelling.
 *
 * Decomposes the joint distribution of d assets into a cascade of
 * bivariate copulas arranged in a vine (tree) structure.
 * Supports both C-vine and D-vine decompositions.
 *
 * TODO:
 *   - C-vine and D-vine tree construction
 *   - Sequential pair-copula estimation
 *   - Simulation via Rosenblatt transform
 *   - AIC/BIC-based copula family selection per edge
 */

#include <cstddef>
#include <vector>

namespace qre::copula {

enum class VineType { CVine, DVine };

struct VineEdge {
    std::size_t node_a;
    std::size_t node_b;
    int family;       // copula family ID
    double parameter; // copula parameter
};

class VineCopula {
public:
    explicit VineCopula(std::size_t dimension, VineType type = VineType::CVine)
        : dimension_{dimension}, type_{type} {}

    /// Fit the vine copula to data (n_samples x dimension matrix).
    void fit(const std::vector<std::vector<double>>& data) {
        // TODO: implement sequential estimation
        (void)data;
    }

    /// Sample n_samples from the fitted vine copula.
    [[nodiscard]] auto sample(std::size_t n_samples) const -> std::vector<std::vector<double>> {
        // TODO: implement Rosenblatt sampling
        return std::vector<std::vector<double>>(n_samples, std::vector<double>(dimension_, 0.5));
    }

    [[nodiscard]] auto dimension() const -> std::size_t { return dimension_; }
    [[nodiscard]] auto type() const -> VineType { return type_; }

private:
    std::size_t dimension_;
    VineType type_;
    std::vector<std::vector<VineEdge>> trees_; // one vector of edges per tree level
};

} // namespace qre::copula
