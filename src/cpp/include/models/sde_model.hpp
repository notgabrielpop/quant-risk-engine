#pragma once

/**
 * @file sde_model.hpp
 * @brief CRTP base class for stochastic differential equation models.
 *
 * Provides a compile-time polymorphic interface for SDE models used in
 * Monte Carlo simulation. Derived classes must implement:
 *   - drift_impl(double t, double x) -> double
 *   - diffusion_impl(double t, double x) -> double
 */

#include <cmath>
#include <cstddef>

namespace qre::models {

template <typename Derived>
class SDEModel {
public:
    /// Drift coefficient mu(t, X_t)
    [[nodiscard]] auto drift(double t, double x) const -> double {
        return static_cast<const Derived*>(this)->drift_impl(t, x);
    }

    /// Diffusion coefficient sigma(t, X_t)
    [[nodiscard]] auto diffusion(double t, double x) const -> double {
        return static_cast<const Derived*>(this)->diffusion_impl(t, x);
    }

    /// Single Euler-Maruyama step: X_{t+dt} = X_t + mu*dt + sigma*dW
    [[nodiscard]] auto euler_step(double t, double x, double dt, double dW) const -> double {
        return x + drift(t, x) * dt + diffusion(t, x) * dW;
    }

    /// Milstein step (for scalar diffusion): adds 0.5 * sigma * sigma' * (dW^2 - dt)
    [[nodiscard]] auto milstein_step(double t, double x, double dt, double dW,
                                     double dsigma_dx) const -> double {
        const double sig = diffusion(t, x);
        return x + drift(t, x) * dt + sig * dW
               + 0.5 * sig * dsigma_dx * (dW * dW - dt);
    }

protected:
    SDEModel() = default;
    ~SDEModel() = default;
    SDEModel(const SDEModel&) = default;
    SDEModel& operator=(const SDEModel&) = default;
};

} // namespace qre::models
