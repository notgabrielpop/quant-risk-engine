#pragma once

/**
 * @file fractional_kernel.hpp
 * @brief Markovian approximation of the fractional kernel K(t) = t^(H-1/2) / Gamma(H+1/2).
 *
 * Uses the spectral representation of completely monotone kernels:
 *   t^{-(1/2-H)} = (1/Gamma(1/2-H)) * integral_0^inf exp(-lambda*t) * lambda^{-H-1/2} d_lambda
 *
 * Discretized via midpoint quadrature on a log-uniform lambda grid.
 * All weights are guaranteed positive by construction.
 */

#include <cmath>
#include <cstddef>
#include <vector>

namespace qre {

struct KernelApproximation {
    int    n_factors;
    double H;
    std::vector<double> weights; // wᵢ
    std::vector<double> rates;   // λᵢ

    KernelApproximation() : n_factors(0), H(0.5) {}

    KernelApproximation(double H_in, int n_factors_in,
                        double lambda_min = 0.1, double lambda_max = 200.0)
        : n_factors(n_factors_in), H(H_in),
          weights(n_factors_in, 0.0), rates(n_factors_in, 0.0)
    {
        // Spectral representation:
        // K(t) = t^(H-1/2) / Gamma(H+1/2)
        //      = integral_0^inf g(lambda) exp(-lambda t) d_lambda
        //
        // where g(lambda) = lambda^(-H-1/2) / (Gamma(1/2-H) * Gamma(H+1/2))
        //
        // Midpoint quadrature in log-lambda space:
        //   w_i = g(lambda_i) * lambda_i * delta_u
        //       = lambda_i^(1/2-H) * delta_u / (Gamma(1/2-H) * Gamma(H+1/2))

        double alpha   = 0.5 - H;                    // 0.4 for H=0.1
        double log_min = std::log(lambda_min);
        double log_max = std::log(lambda_max);
        double delta_u = (log_max - log_min) / static_cast<double>(n_factors);

        // Gamma(1/2-H) * Gamma(H+1/2) = Gamma(alpha) * Gamma(1-alpha) = pi / sin(pi * alpha)
        // (reflection formula, valid for 0 < alpha < 1)
        double gamma_product = M_PI / std::sin(M_PI * alpha);

        // Midpoint rule: lambda_i at center of i-th interval in log-space
        for (int i = 0; i < n_factors; ++i) {
            double u = log_min + (static_cast<double>(i) + 0.5) * delta_u;
            double lam = std::exp(u);
            rates[i]   = lam;
            weights[i] = std::pow(lam, alpha) * delta_u / gamma_product;
        }
    }

    /// Exact kernel: K(t) = t^(H-0.5) / Gamma(H+0.5)
    double exact_kernel(double t) const {
        return std::pow(t, H - 0.5) / std::tgamma(H + 0.5);
    }

    /// Approximated kernel: Σ wᵢ exp(-λᵢ t)
    double approx_kernel(double t) const {
        double sum = 0.0;
        for (int i = 0; i < n_factors; ++i)
            sum += weights[i] * std::exp(-rates[i] * t);
        return sum;
    }

    /// Relative L2 error on [dt, T]
    double approximation_error(double dt, double T, int n_points = 500) const {
        double err2 = 0.0, norm2 = 0.0;
        double log_dt = std::log(dt), log_T = std::log(T);
        for (int j = 0; j < n_points; ++j) {
            double frac = static_cast<double>(j) / static_cast<double>(n_points - 1);
            double t = std::exp(log_dt + frac * (log_T - log_dt));
            double ex = exact_kernel(t);
            double ap = approx_kernel(t);
            double d  = ex - ap;
            err2  += d * d;
            norm2 += ex * ex;
        }
        return std::sqrt(err2 / norm2);
    }
};

} // namespace qre
