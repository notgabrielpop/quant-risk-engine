#pragma once

/**
 * @file expected_shortfall.hpp
 * @brief Expected Shortfall (CVaR) — the average loss beyond VaR.
 *
 * ES is a coherent risk measure (unlike VaR) and captures tail risk.
 * ES_alpha = E[L | L > VaR_alpha]
 */

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace qre::risk {

class ExpectedShortfall {
public:
    /// Compute ES at the given confidence level.
    /// @param pnl        Vector of profit/loss values
    /// @param confidence Confidence level (e.g. 0.99)
    /// @return ES as a positive loss amount
    [[nodiscard]] static auto compute(std::vector<double> pnl, double confidence) -> double {
        if (pnl.empty()) {
            throw std::invalid_argument("P&L vector must not be empty");
        }

        std::sort(pnl.begin(), pnl.end());
        const auto cutoff = static_cast<std::size_t>((1.0 - confidence) * static_cast<double>(pnl.size()));
        if (cutoff == 0) {
            return -pnl.front();
        }

        const double tail_sum = std::accumulate(pnl.begin(), pnl.begin() + static_cast<long>(cutoff), 0.0);
        return -(tail_sum / static_cast<double>(cutoff));
    }
};

} // namespace qre::risk
