#pragma once

/**
 * @file gbm.hpp
 * @brief Geometric Brownian Motion: dS = mu*S*dt + sigma*S*dW
 *
 * Has exact solution: S_T = S_0 * exp((mu - sigma²/2)*T + sigma*sqrt(T)*Z)
 * Used both as a model and as validation / control variate.
 */

#include "sde_model.hpp"
#include "../core/path_data.hpp"
#include <cmath>

namespace qre {

class GBM : public SDEModel<GBM> {
public:
    static constexpr bool has_variance_process = false;

    GBMParams params;

    explicit GBM(const GBMParams& p) : params(p) {}

    double drift_impl(double S, double /*v*/, double /*t*/) const {
        return params.mu * S;
    }
    double diffusion_impl(double S, double /*v*/, double /*t*/) const {
        return params.sigma * S;
    }
    double variance_drift_impl(double, double, double) const { return 0.0; }
    double variance_diffusion_impl(double, double, double) const { return 0.0; }
    double correlation_impl() const { return 0.0; }

    /// Exact single step (zero discretisation error)
    static double exact_step(double S, double mu, double sigma,
                             double dt, double Z) {
        return S * std::exp((mu - 0.5 * sigma * sigma) * dt +
                            sigma * std::sqrt(dt) * Z);
    }

    // ----- Analytical formulas -----
    double analytical_mean() const {
        return params.S0 * std::exp(params.mu * params.T);
    }
    double analytical_log_return_mean() const {
        return (params.mu - 0.5 * params.sigma * params.sigma) * params.T;
    }
    double analytical_log_return_variance() const {
        return params.sigma * params.sigma * params.T;
    }
};

} // namespace qre
