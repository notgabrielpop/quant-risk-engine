#pragma once

/**
 * @file gbm.hpp
 * @brief Geometric Brownian Motion model.
 *
 * dS = mu * S * dt + sigma * S * dW
 *
 * Constant drift and volatility; the baseline model against which
 * stochastic-volatility models are compared.
 */

#include "sde_model.hpp"

namespace qre::models {

class GBM : public SDEModel<GBM> {
public:
    GBM(double mu, double sigma) : mu_{mu}, sigma_{sigma} {}

    [[nodiscard]] auto drift_impl(double /*t*/, double x) const -> double {
        return mu_ * x;
    }

    [[nodiscard]] auto diffusion_impl(double /*t*/, double x) const -> double {
        return sigma_ * x;
    }

    [[nodiscard]] auto mu() const -> double { return mu_; }
    [[nodiscard]] auto sigma() const -> double { return sigma_; }

private:
    double mu_;
    double sigma_;
};

} // namespace qre::models
