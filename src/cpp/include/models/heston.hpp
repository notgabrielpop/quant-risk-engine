#pragma once

/**
 * @file heston.hpp
 * @brief Heston stochastic volatility model.
 *
 * System of SDEs:
 *   dS = mu * S * dt + sqrt(V) * S * dW_1
 *   dV = kappa * (theta - V) * dt + xi * sqrt(V) * dW_2
 *   corr(dW_1, dW_2) = rho
 *
 * Requires two-dimensional simulation (price + variance).
 */

#include <algorithm>
#include <cmath>

namespace qre::models {

struct HestonParams {
    double mu;     // drift
    double kappa;  // mean-reversion speed of variance
    double theta;  // long-run variance
    double xi;     // vol-of-vol
    double rho;    // correlation between price and variance Brownians
    double v0;     // initial variance
};

class Heston {
public:
    explicit Heston(const HestonParams& p) : p_{p} {}

    /// Price drift: mu * S
    [[nodiscard]] auto price_drift(double /*t*/, double s, double /*v*/) const -> double {
        return p_.mu * s;
    }

    /// Price diffusion: sqrt(V) * S
    [[nodiscard]] auto price_diffusion(double /*t*/, double s, double v) const -> double {
        return std::sqrt(std::max(v, 0.0)) * s;
    }

    /// Variance drift: kappa * (theta - V)
    [[nodiscard]] auto var_drift(double /*t*/, double v) const -> double {
        return p_.kappa * (p_.theta - v);
    }

    /// Variance diffusion: xi * sqrt(V)
    [[nodiscard]] auto var_diffusion(double /*t*/, double v) const -> double {
        return p_.xi * std::sqrt(std::max(v, 0.0));
    }

    /// Euler step for the full (S, V) system with correlated Brownians
    struct State { double s; double v; };

    [[nodiscard]] auto step(double t, State state, double dt,
                            double z1, double z2) const -> State {
        const double dW1 = z1 * std::sqrt(dt);
        const double dW2 = (p_.rho * z1 + std::sqrt(1.0 - p_.rho * p_.rho) * z2) * std::sqrt(dt);

        double new_v = state.v + var_drift(t, state.v) * dt + var_diffusion(t, state.v) * dW2;
        new_v = std::max(new_v, 0.0); // absorbing boundary at zero

        double new_s = state.s + price_drift(t, state.s, state.v) * dt
                       + price_diffusion(t, state.s, state.v) * dW1;

        return {new_s, new_v};
    }

    [[nodiscard]] auto params() const -> const HestonParams& { return p_; }

private:
    HestonParams p_;
};

} // namespace qre::models
