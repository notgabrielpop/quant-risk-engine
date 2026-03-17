#pragma once

/**
 * @file var.hpp
 * @brief VaR, Expected Shortfall, and distribution summary from terminal prices.
 *
 * Uses std::nth_element for O(N) partial sort instead of full O(N log N) sort.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

namespace qre {

struct RiskMetrics {
    double mean_price   = 0;
    double std_price    = 0;
    double skewness     = 0;
    double kurtosis     = 0;
    double var_95       = 0;
    double var_99       = 0;
    double es_95        = 0;
    double es_99        = 0;
    double min_price    = 0;
    double max_price    = 0;
    double median_price = 0;
};

/// Compute VaR at confidence level alpha from a P&L vector.
/// P&L should be (terminal - S0) or returns.  VaR returned as positive loss.
inline double compute_var(std::vector<double>& pnl, double alpha) {
    const size_t n   = pnl.size();
    const size_t idx = static_cast<size_t>(static_cast<double>(n) * (1.0 - alpha));
    std::nth_element(pnl.begin(), pnl.begin() + static_cast<long>(idx), pnl.end());
    return -pnl[idx];
}

/// Compute Expected Shortfall (average loss beyond VaR).
inline double compute_es(std::vector<double>& pnl, double alpha) {
    const size_t n      = pnl.size();
    const size_t cutoff = static_cast<size_t>(static_cast<double>(n) * (1.0 - alpha));
    if (cutoff == 0) return -pnl[0];
    std::partial_sort(pnl.begin(), pnl.begin() + static_cast<long>(cutoff), pnl.end());
    double sum = 0.0;
    for (size_t i = 0; i < cutoff; ++i) sum += pnl[i];
    return -(sum / static_cast<double>(cutoff));
}

/// Compute full risk metrics from an array of terminal prices.
inline RiskMetrics compute_risk_metrics(const double* terminal, size_t n,
                                        double S0) {
    RiskMetrics rm;

    // P&L and moments
    std::vector<double> pnl(n);
    double sum = 0, sum2 = 0, sum3 = 0, sum4 = 0;
    double mn = terminal[0], mx = terminal[0];

    for (size_t i = 0; i < n; ++i) {
        double p = terminal[i];
        pnl[i] = p - S0;
        sum  += p;
        mn = std::min(mn, p);
        mx = std::max(mx, p);
    }

    rm.mean_price = sum / static_cast<double>(n);
    rm.min_price  = mn;
    rm.max_price  = mx;

    for (size_t i = 0; i < n; ++i) {
        double d  = terminal[i] - rm.mean_price;
        double d2 = d * d;
        sum2 += d2;
        sum3 += d2 * d;
        sum4 += d2 * d2;
    }

    double var = sum2 / static_cast<double>(n);
    rm.std_price = std::sqrt(var);
    if (var > 0) {
        rm.skewness = (sum3 / static_cast<double>(n)) / (var * rm.std_price);
        rm.kurtosis = (sum4 / static_cast<double>(n)) / (var * var) - 3.0;
    }

    // Median
    std::vector<double> sorted(terminal, terminal + n);
    std::nth_element(sorted.begin(), sorted.begin() + static_cast<long>(n / 2), sorted.end());
    rm.median_price = sorted[n / 2];

    // VaR and ES
    rm.var_95 = compute_var(pnl, 0.95);
    // Need fresh pnl copy for ES since nth_element/partial_sort modifies order
    std::vector<double> pnl2(n);
    for (size_t i = 0; i < n; ++i) pnl2[i] = terminal[i] - S0;
    rm.es_95 = compute_es(pnl2, 0.95);

    std::vector<double> pnl3(n);
    for (size_t i = 0; i < n; ++i) pnl3[i] = terminal[i] - S0;
    rm.var_99 = compute_var(pnl3, 0.99);

    std::vector<double> pnl4(n);
    for (size_t i = 0; i < n; ++i) pnl4[i] = terminal[i] - S0;
    rm.es_99 = compute_es(pnl4, 0.99);

    return rm;
}

} // namespace qre
