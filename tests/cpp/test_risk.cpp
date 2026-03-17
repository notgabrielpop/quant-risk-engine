/**
 * @file test_risk.cpp
 * @brief Tests for VaR and Expected Shortfall computation.
 */

#include "risk/var.hpp"
#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

extern void register_test(const char*, std::function<bool()>);

namespace {

bool test_var_es_basic() {
    // Uniform P&L: -100 to +100, expect VaR_95 ≈ 90, ES_95 > VaR_95
    size_t n = 10'000;
    std::vector<double> terminal(n);
    double S0 = 100.0;
    for (size_t i = 0; i < n; ++i)
        terminal[i] = S0 + (-100.0 + 200.0 * static_cast<double>(i) / static_cast<double>(n - 1));

    auto rm = qre::compute_risk_metrics(terminal.data(), n, S0);

    std::printf("    VaR95=%.2f, VaR99=%.2f, ES95=%.2f, ES99=%.2f\n",
                rm.var_95, rm.var_99, rm.es_95, rm.es_99);
    std::printf("    mean=%.2f, std=%.2f, median=%.2f, min=%.2f, max=%.2f\n",
                rm.mean_price, rm.std_price, rm.median_price, rm.min_price, rm.max_price);

    bool ok = true;
    ok &= rm.var_95 > 80 && rm.var_95 < 100;
    ok &= rm.es_95 > rm.var_95;
    ok &= rm.es_99 > rm.var_99;
    ok &= std::abs(rm.mean_price - S0) < 1.0;
    return ok;
}

struct Register {
    Register() {
        register_test("Risk: VaR/ES basic", test_var_es_basic);
    }
} _reg;

} // anonymous namespace
