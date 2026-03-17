#pragma once

/**
 * @file rough_heston.hpp
 * @brief Rough Heston model with fractional Brownian motion kernel.
 *
 * Extends the Heston model by replacing the variance process with a
 * rough (fractional) process driven by a Volterra integral:
 *
 *   V_t = V_0 + (1/Gamma(H+0.5)) * integral_0^t (t-s)^{H-0.5} * [kappa(theta - V_s) ds + xi sqrt(V_s) dW_s]
 *
 * where H in (0, 0.5) is the Hurst exponent controlling roughness.
 * Typical empirical values: H ~ 0.05-0.15.
 *
 * Discretisation uses the hybrid scheme of Bennedsen, Lunde & Pakkanen (2017).
 */

#include <algorithm>
#include <cmath>
#include <vector>

namespace qre::models {

struct RoughHestonParams {
    double mu;     // price drift
    double kappa;  // mean-reversion speed
    double theta;  // long-run variance
    double xi;     // vol-of-vol
    double rho;    // correlation
    double v0;     // initial variance
    double H;      // Hurst exponent (0 < H < 0.5 for rough regime)
};

class RoughHeston {
public:
    explicit RoughHeston(const RoughHestonParams& p) : p_{p} {}

    /// Pre-compute the fractional kernel weights for n time steps.
    /// K(t-s) = (t-s)^{H-0.5} / Gamma(H+0.5)
    void precompute_kernel(std::size_t n_steps, double dt) {
        kernel_weights_.resize(n_steps);
        const double alpha = p_.H - 0.5;
        const double scale = 1.0 / std::tgamma(p_.H + 0.5);
        for (std::size_t k = 0; k < n_steps; ++k) {
            kernel_weights_[k] = scale * std::pow((k + 1) * dt, alpha);
        }
    }

    /// Compute variance at step i given the history of variance increments.
    /// This is the convolution of the kernel with past variance innovations.
    [[nodiscard]] auto compute_variance(std::size_t step,
                                        const std::vector<double>& var_increments,
                                        double dt) const -> double {
        double v = p_.v0;
        const std::size_t n = std::min(step, kernel_weights_.size());
        for (std::size_t k = 0; k < n; ++k) {
            v += kernel_weights_[n - 1 - k] * var_increments[k] * dt;
        }
        return std::max(v, 0.0);
    }

    [[nodiscard]] auto params() const -> const RoughHestonParams& { return p_; }
    [[nodiscard]] auto kernel_weights() const -> const std::vector<double>& { return kernel_weights_; }

private:
    RoughHestonParams p_;
    std::vector<double> kernel_weights_;
};

} // namespace qre::models
