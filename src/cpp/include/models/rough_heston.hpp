#pragma once

/**
 * @file rough_heston.hpp
 * @brief Rough Heston model (Gatheral, Jaisson, Rosenbaum 2018).
 *
 * Replaces Brownian motion in the variance SDE with fractional BM (H ~ 0.1).
 * Requires Markovian approximation or hybrid discretisation.
 *
 * TODO (Week 5): implement Markovian approximation of fractional kernel.
 */

#include "sde_model.hpp"
#include "../core/path_data.hpp"
#include <algorithm>
#include <cmath>

namespace qre {

class RoughHeston : public SDEModel<RoughHeston> {
public:
    static constexpr bool has_variance_process = true;

    RoughHestonParams params;

    explicit RoughHeston(const RoughHestonParams& p) : params(p) {}

    double drift_impl(double S, double /*v*/, double /*t*/) const {
        return params.mu * S;
    }
    double diffusion_impl(double S, double v, double /*t*/) const {
        return std::sqrt(std::max(v, 0.0)) * S;
    }
    // Placeholder — needs fractional kernel
    double variance_drift_impl(double /*S*/, double v, double /*t*/) const {
        return params.kappa * (params.theta - v);
    }
    double variance_diffusion_impl(double /*S*/, double v, double /*t*/) const {
        return params.sigma_v * std::sqrt(std::max(v, 0.0));
    }
    double correlation_impl() const { return params.rho; }
};

} // namespace qre
