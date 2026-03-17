#pragma once

/**
 * @file euler_allocation.hpp
 * @brief Euler capital allocation for portfolio risk decomposition.
 *
 * Decomposes portfolio-level risk measures (VaR, ES) into per-asset
 * contributions using the Euler (gradient) allocation principle:
 *
 *   rho_i = w_i * partial(rho) / partial(w_i)
 *
 * where sum(rho_i) = rho(portfolio) by Euler's theorem for
 * homogeneous-of-degree-1 risk measures.
 *
 * TODO:
 *   - Implement finite-difference gradient estimation
 *   - Kernel-smoothed conditional expectation for ES allocation
 */

#include <cstddef>
#include <vector>

namespace qre::risk {

struct AllocationResult {
    std::vector<double> contributions;  ///< Per-asset risk contribution
    double total_risk;                  ///< Portfolio-level risk measure
};

class EulerAllocator {
public:
    /// Compute Euler allocation of ES across assets.
    /// @param portfolio_pnl  Total portfolio P&L paths (n_paths)
    /// @param asset_pnls     Per-asset P&L paths (n_assets x n_paths)
    /// @param confidence     Confidence level
    [[nodiscard]] static auto allocate_es(
        const std::vector<double>& portfolio_pnl,
        const std::vector<std::vector<double>>& asset_pnls,
        double confidence) -> AllocationResult {
        // TODO: implement kernel-smoothed ES allocation
        AllocationResult result;
        result.contributions.resize(asset_pnls.size(), 0.0);
        result.total_risk = 0.0;
        return result;
    }
};

} // namespace qre::risk
