#pragma once

/**
 * @file rough_heston.hpp
 * @brief Rough Heston model via Markovian approximation of the fractional kernel.
 *
 * The variance process is:
 *   v(t) = v0 + integral_0^t K(t-s) [kappa(theta - v(s)) ds + sigma_v sqrt(v(s)) dW2(s)]
 *
 * where K(t) = t^(H-1/2) / Gamma(H+1/2),  H ≈ 0.1  (rough).
 *
 * Markovian approximation: K(t) ≈ Σ_i w_i exp(-lambda_i t) introduces auxiliary
 * factors Y_i with  v(t) = v0 + Σ Y_i(t),  each satisfying:
 *   dY_i = [-lambda_i Y_i + w_i kappa (theta - v)] dt + w_i sigma_v sqrt(v) dW2
 *
 * Reference: Abi Jaber & El Euch (2019), Bayer, Friz & Gatheral (2016).
 */

#include "sde_model.hpp"
#include "fractional_kernel.hpp"
#include "../core/path_data.hpp"
#include <algorithm>
#include <cmath>

namespace qre {

class RoughHeston : public SDEModel<RoughHeston> {
public:
    static constexpr bool has_variance_process = true;

    RoughHestonParams params;
    KernelApproximation kernel;

    explicit RoughHeston(const RoughHestonParams& p,
                         int n_factors = 8,
                         double lambda_min = 0.01,
                         double lambda_max = 100.0)
        : params(p), kernel(p.H, n_factors, lambda_min, lambda_max) {}

    // CRTP SDE interface (for generic use — dedicated simulator preferred)
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

    int n_factors() const { return kernel.n_factors; }
};

} // namespace qre
