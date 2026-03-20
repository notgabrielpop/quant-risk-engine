#pragma once

/**
 * @file importance_sampling.hpp
 * @brief Importance sampling for tail risk estimation.
 *
 * Shifts the sampling distribution to oversample extreme losses, then
 * corrects with likelihood ratios. Uses exponential tilting (drift shift)
 * on the price Brownian motion.
 *
 * For GBM: shift all Z_i by mu_IS (negative for left tail)
 *   log-weight per step: -mu_IS * Z_i - 0.5 * mu_IS^2
 *   Total log-weight: sum over steps
 *
 * For Heston: shift only Z1 (price Brownian), leave Z2 (variance) unchanged.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

namespace qre {

struct ISResult {
    std::vector<double> terminal_prices;
    std::vector<double> weights;    // Likelihood ratio weights
    double ess;                     // Effective sample size
    double ess_ratio;               // ESS / N
};

/// Compute effective sample size from weights.
/// ESS = (sum w_i)^2 / sum(w_i^2)
inline double compute_ess(const double* weights, size_t n) {
    double sum_w = 0, sum_w2 = 0;
    for (size_t i = 0; i < n; ++i) {
        sum_w  += weights[i];
        sum_w2 += weights[i] * weights[i];
    }
    return (sum_w2 > 0) ? (sum_w * sum_w) / sum_w2 : 0.0;
}

/// Compute weighted Expected Shortfall at level alpha.
/// Uses the self-normalized IS estimator.
inline double compute_weighted_es(const double* prices, const double* weights,
                                   size_t n, double S0, double alpha) {
    // Create (loss, weight) pairs
    struct LW { double loss; double w; };
    std::vector<LW> lw(n);
    for (size_t i = 0; i < n; ++i) {
        lw[i] = {S0 - prices[i], weights[i]};  // loss = S0 - S_T
    }

    // Sort by loss descending (worst losses first)
    std::sort(lw.begin(), lw.end(), [](const LW& a, const LW& b) {
        return a.loss > b.loss;
    });

    // Find the 1-alpha tail in weighted space
    double total_w = 0;
    for (size_t i = 0; i < n; ++i) total_w += lw[i].w;

    double tail_prob = 1.0 - alpha;
    double cumul_w = 0;
    double es_num = 0;
    double es_den = 0;

    for (size_t i = 0; i < n; ++i) {
        double w_frac = lw[i].w / total_w;
        if (cumul_w + w_frac <= tail_prob + 1e-12) {
            es_num += lw[i].w * lw[i].loss;
            es_den += lw[i].w;
            cumul_w += w_frac;
        } else {
            // Partial inclusion of the boundary observation
            double remaining = tail_prob - cumul_w;
            if (remaining > 0 && w_frac > 0) {
                double frac = remaining / w_frac;
                es_num += lw[i].w * frac * lw[i].loss;
                es_den += lw[i].w * frac;
            }
            break;
        }
    }

    return (es_den > 0) ? es_num / es_den : 0.0;
}

/// Compute weighted mean of prices.
inline double compute_weighted_mean(const double* prices, const double* weights, size_t n) {
    double num = 0, den = 0;
    for (size_t i = 0; i < n; ++i) {
        num += weights[i] * prices[i];
        den += weights[i];
    }
    return (den > 0) ? num / den : 0.0;
}

} // namespace qre
