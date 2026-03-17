#pragma once

/**
 * @file var.hpp
 * @brief Value at Risk (VaR) computation from simulated P&L distributions.
 *
 * Supports historical, parametric, and Monte Carlo VaR at arbitrary
 * confidence levels. Operates on terminal P&L vectors produced by SimEngine.
 */

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace qre::risk {

class VaR {
public:
    /// Compute VaR at the given confidence level from a P&L distribution.
    /// @param pnl        Vector of profit/loss values
    /// @param confidence Confidence level (e.g. 0.99)
    /// @return VaR as a positive loss amount
    [[nodiscard]] static auto compute(std::vector<double> pnl, double confidence) -> double {
        if (pnl.empty()) {
            throw std::invalid_argument("P&L vector must not be empty");
        }
        if (confidence <= 0.0 || confidence >= 1.0) {
            throw std::invalid_argument("Confidence must be in (0, 1)");
        }

        std::sort(pnl.begin(), pnl.end());
        const auto idx = static_cast<std::size_t>((1.0 - confidence) * static_cast<double>(pnl.size()));
        return -pnl[idx]; // VaR is reported as positive loss
    }
};

} // namespace qre::risk
