/**
 * @file test_gbm.cpp
 * @brief Unit tests for GBM model and CPU simulation engine.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "models/gbm.hpp"
#include "engine/sim_engine.hpp"
#include "engine/cpu_engine.hpp"
#include "risk/var.hpp"
#include "risk/expected_shortfall.hpp"

#include <cmath>
#include <numeric>

using namespace qre;

TEST_CASE("GBM drift and diffusion", "[gbm]") {
    models::GBM gbm(0.05, 0.2);

    REQUIRE(gbm.mu() == 0.05);
    REQUIRE(gbm.sigma() == 0.2);

    SECTION("drift is mu * x") {
        REQUIRE(gbm.drift_impl(0.0, 100.0) == 5.0);
    }

    SECTION("diffusion is sigma * x") {
        REQUIRE(gbm.diffusion_impl(0.0, 100.0) == 20.0);
    }
}

TEST_CASE("GBM Euler step", "[gbm]") {
    models::GBM gbm(0.05, 0.2);
    const double dt = 1.0 / 252.0;
    const double dW = 0.01;

    double result = gbm.euler_step(0.0, 100.0, dt, dW);
    double expected = 100.0 + 0.05 * 100.0 * dt + 0.2 * 100.0 * dW;
    REQUIRE_THAT(result, Catch::Matchers::WithinRel(expected, 1e-12));
}

TEST_CASE("CPU SimEngine produces correct number of paths", "[engine]") {
    models::GBM gbm(0.05, 0.2);
    engine::SimConfig config;
    config.n_paths = 10'000;
    config.n_steps = 252;
    config.s0 = 100.0;

    engine::SimEngine<engine::Backend::CPU, models::GBM> eng(gbm, config);
    auto result = eng.run();

    REQUIRE(result.terminal_prices.size() == 10'000);
    REQUIRE(result.elapsed_ms > 0.0);
}

TEST_CASE("VaR computation", "[risk]") {
    // Uniform P&L from -100 to 100
    std::vector<double> pnl(1000);
    for (int i = 0; i < 1000; ++i) {
        pnl[i] = -100.0 + 200.0 * static_cast<double>(i) / 999.0;
    }

    double var95 = risk::VaR::compute(pnl, 0.95);
    // 5th percentile of [-100, 100] ≈ -90, so VaR ≈ 90
    REQUIRE(var95 > 80.0);
    REQUIRE(var95 < 100.0);
}

TEST_CASE("Expected Shortfall > VaR", "[risk]") {
    std::vector<double> pnl(10000);
    for (int i = 0; i < 10000; ++i) {
        pnl[i] = -100.0 + 200.0 * static_cast<double>(i) / 9999.0;
    }

    double var99 = risk::VaR::compute(pnl, 0.99);
    double es99  = risk::ExpectedShortfall::compute(pnl, 0.99);
    REQUIRE(es99 >= var99);
}
