#pragma once

/**
 * @file heston.hpp
 * @brief Heston Stochastic Volatility Model.
 *
 *   dS = mu * S * dt  + sqrt(v) * S * dW1
 *   dv = kappa*(theta-v)*dt + sigma_v*sqrt(v)*dW2
 *   corr(dW1, dW2) = rho
 *
 * TODO (Day 5): implement QE discretisation (Andersen 2008).
 */

#include "sde_model.hpp"
#include "../core/path_data.hpp"
#include <algorithm>
#include <cmath>

namespace qre {

class Heston : public SDEModel<Heston> {
public:
    static constexpr bool has_variance_process = true;

    HestonParams params;

    explicit Heston(const HestonParams& p) : params(p) {}

    double drift_impl(double S, double /*v*/, double /*t*/) const {
        return params.mu * S;
    }
    double diffusion_impl(double S, double v, double /*t*/) const {
        return std::sqrt(std::max(v, 0.0)) * S;
    }
    double variance_drift_impl(double /*S*/, double v, double /*t*/) const {
        return params.kappa * (params.theta - v);
    }
    double variance_diffusion_impl(double /*S*/, double v, double /*t*/) const {
        return params.sigma_v * std::sqrt(std::max(v, 0.0));
    }
    double correlation_impl() const { return params.rho; }
};

} // namespace qre
