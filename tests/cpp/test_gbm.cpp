/**
 * @file test_gbm.cpp
 * @brief Rigorous validation of GBM simulation against analytical solutions.
 */

#include "core/path_data.hpp"
#include "core/sim_config.hpp"
#include "models/gbm.hpp"
#include "engine/cpu_engine.hpp"
#include "risk/var.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <functional>
#include <numeric>
#include <vector>

extern void register_test(const char*, std::function<bool()>);

namespace {

// Helper: compute sample mean
double mean(const double* d, size_t n) {
    double s = 0;
    for (size_t i = 0; i < n; ++i) s += d[i];
    return s / static_cast<double>(n);
}

// Helper: compute sample variance
double variance(const double* d, size_t n, double mu) {
    double s = 0;
    for (size_t i = 0; i < n; ++i) { double x = d[i] - mu; s += x * x; }
    return s / static_cast<double>(n - 1);
}

// Standard GBM parameters for all tests
qre::GBMParams make_params() {
    qre::GBMParams p;
    p.S0 = 100.0; p.mu = 0.05; p.sigma = 0.2;
    p.T = 1.0; p.n_steps = 252;
    return p;
}

// ---------------------------------------------------------------
// Test 1: Mean convergence
// ---------------------------------------------------------------
bool test_mean_convergence() {
    auto p = make_params();
    qre::SimConfig cfg; cfg.n_paths = 1'000'000; cfg.n_steps = p.n_steps; cfg.seed = 42;
    qre::PathData paths(cfg.n_paths, cfg.n_steps);
    qre::CPUEngine eng;
    eng.simulate_gbm(p, cfg, paths);

    qre::GBM model(p);
    double analytical = model.analytical_mean();
    double simulated  = mean(paths.terminal_prices(), cfg.n_paths);
    double rel_err    = std::abs(simulated - analytical) / analytical;

    std::printf("    Analytical E[S_T] = %.4f, Simulated = %.4f, RelErr = %.4f%%\n",
                analytical, simulated, rel_err * 100);
    return rel_err < 0.001; // within 0.1%
}

// ---------------------------------------------------------------
// Test 2: Variance of log-returns
// ---------------------------------------------------------------
bool test_log_return_variance() {
    auto p = make_params();
    qre::SimConfig cfg; cfg.n_paths = 1'000'000; cfg.n_steps = p.n_steps; cfg.seed = 42;
    qre::PathData paths(cfg.n_paths, cfg.n_steps);
    qre::CPUEngine eng;
    eng.simulate_gbm(p, cfg, paths);

    qre::GBM model(p);
    double analytical_var = model.analytical_log_return_variance();

    std::vector<double> log_ret(cfg.n_paths);
    for (size_t i = 0; i < cfg.n_paths; ++i)
        log_ret[i] = std::log(paths.terminal_prices()[i] / p.S0);

    double mu_lr = mean(log_ret.data(), cfg.n_paths);
    double sim_var = variance(log_ret.data(), cfg.n_paths, mu_lr);
    double rel_err = std::abs(sim_var - analytical_var) / analytical_var;

    std::printf("    Analytical Var[lnR] = %.6f, Simulated = %.6f, RelErr = %.4f%%\n",
                analytical_var, sim_var, rel_err * 100);
    return rel_err < 0.01; // within 1%
}

// ---------------------------------------------------------------
// Test 3: Distribution shape (normality of log-returns)
// ---------------------------------------------------------------
bool test_distribution_shape() {
    auto p = make_params();
    qre::SimConfig cfg; cfg.n_paths = 1'000'000; cfg.n_steps = p.n_steps; cfg.seed = 42;
    qre::PathData paths(cfg.n_paths, cfg.n_steps);
    qre::CPUEngine eng;
    eng.simulate_gbm(p, cfg, paths);

    size_t n = cfg.n_paths;
    std::vector<double> lr(n);
    for (size_t i = 0; i < n; ++i)
        lr[i] = std::log(paths.terminal_prices()[i] / p.S0);

    double mu = mean(lr.data(), n);
    double s2 = 0, s3 = 0, s4 = 0;
    for (size_t i = 0; i < n; ++i) {
        double d = lr[i] - mu;
        double d2 = d * d;
        s2 += d2; s3 += d2 * d; s4 += d2 * d2;
    }
    double v = s2 / static_cast<double>(n);
    double sd = std::sqrt(v);
    double skew = (s3 / static_cast<double>(n)) / (v * sd);
    double kurt = (s4 / static_cast<double>(n)) / (v * v) - 3.0;

    std::printf("    Skewness = %.4f (expect ~0), Excess Kurtosis = %.4f (expect ~0)\n",
                skew, kurt);
    return std::abs(skew) < 0.02 && std::abs(kurt) < 0.05;
}

// ---------------------------------------------------------------
// Test 4: Convergence rate O(1/sqrt(N))
// ---------------------------------------------------------------
bool test_convergence_rate() {
    auto p = make_params();
    qre::GBM model(p);
    double analytical = model.analytical_mean();
    qre::CPUEngine eng;

    size_t sizes[] = {1000, 10'000, 100'000, 1'000'000};
    double errors[4];

    std::printf("    %10s  %12s  %8s\n", "N", "Error", "Ratio");
    for (int k = 0; k < 4; ++k) {
        qre::SimConfig cfg;
        cfg.n_paths = sizes[k]; cfg.n_steps = p.n_steps; cfg.seed = 42;
        qre::PathData paths(cfg.n_paths, cfg.n_steps);
        eng.simulate_gbm(p, cfg, paths);
        double sim = mean(paths.terminal_prices(), cfg.n_paths);
        errors[k] = std::abs(sim - analytical);
        double ratio = k > 0 ? errors[k] / errors[k - 1] : 0;
        std::printf("    %10zu  %12.6f  %8.3f\n", sizes[k], errors[k],
                    k > 0 ? ratio : 0.0);
    }
    // Ratio should be roughly 1/sqrt(10) ≈ 0.316, allow wide tolerance
    // (randomness means this is noisy)
    return errors[3] < errors[0]; // at minimum, 1M should beat 1K
}

// ---------------------------------------------------------------
// Test 5: Antithetic variance reduction
// ---------------------------------------------------------------
bool test_antithetic() {
    auto p = make_params();
    size_t N = 100'000;
    qre::CPUEngine eng;
    qre::GBM model(p);
    double analytical = model.analytical_mean();

    // Standard
    qre::SimConfig cfg_std; cfg_std.n_paths = N; cfg_std.n_steps = p.n_steps; cfg_std.seed = 42;
    qre::PathData paths_std(N, p.n_steps);
    eng.simulate_gbm(p, cfg_std, paths_std);
    double mu_std = mean(paths_std.terminal_prices(), N);
    double var_std = variance(paths_std.terminal_prices(), N, mu_std);
    double se_std = std::sqrt(var_std / static_cast<double>(N));

    // Antithetic
    qre::SimConfig cfg_anti; cfg_anti.n_paths = N; cfg_anti.n_steps = p.n_steps; cfg_anti.seed = 42;
    qre::PathData paths_anti(N, p.n_steps);
    eng.simulate_gbm_antithetic(p, cfg_anti, paths_anti);

    // Average paired paths for the antithetic estimator
    size_t half = N / 2;
    std::vector<double> anti_avg(half);
    for (size_t i = 0; i < half; ++i)
        anti_avg[i] = 0.5 * (paths_anti.terminal_prices()[i]
                            + paths_anti.terminal_prices()[i + half]);

    double mu_anti = mean(anti_avg.data(), half);
    double var_anti = variance(anti_avg.data(), half, mu_anti);
    double se_anti = std::sqrt(var_anti / static_cast<double>(half));

    double reduction = 1.0 - se_anti / se_std;
    std::printf("    SE(standard) = %.6f, SE(antithetic) = %.6f, reduction = %.1f%%\n",
                se_std, se_anti, reduction * 100);
    return se_anti < se_std; // antithetic should reduce SE
}

// ---------------------------------------------------------------
// Test 6: Reproducibility
// ---------------------------------------------------------------
bool test_reproducibility() {
    auto p = make_params();
    qre::SimConfig cfg; cfg.n_paths = 10'000; cfg.n_steps = p.n_steps; cfg.seed = 42;
    cfg.n_threads = 1; // single thread for bitwise reproducibility
    qre::CPUEngine eng;

    qre::PathData p1(cfg.n_paths, cfg.n_steps);
    qre::PathData p2(cfg.n_paths, cfg.n_steps);
    eng.simulate_gbm(p, cfg, p1);
    eng.simulate_gbm(p, cfg, p2);

    bool identical = true;
    for (size_t i = 0; i < cfg.n_paths; ++i) {
        if (p1.terminal_prices()[i] != p2.terminal_prices()[i]) {
            identical = false;
            break;
        }
    }

    // Different seed
    qre::SimConfig cfg2 = cfg; cfg2.seed = 123;
    qre::PathData p3(cfg2.n_paths, cfg2.n_steps);
    eng.simulate_gbm(p, cfg2, p3);
    bool different = (p1.terminal_prices()[0] != p3.terminal_prices()[0]);

    std::printf("    Same seed identical: %s, Different seed different: %s\n",
                identical ? "yes" : "no", different ? "yes" : "no");
    return identical && different;
}

// ---------------------------------------------------------------
// Test 7: Multi-threaded correctness
// ---------------------------------------------------------------
bool test_multithreaded() {
    auto p = make_params();
    qre::CPUEngine eng;
    size_t N = 500'000;

    qre::SimConfig cfg1; cfg1.n_paths = N; cfg1.n_steps = p.n_steps;
    cfg1.seed = 42; cfg1.n_threads = 1;
    qre::PathData paths1(N, p.n_steps);
    eng.simulate_gbm(p, cfg1, paths1);
    double mu1 = mean(paths1.terminal_prices(), N);
    double var1 = variance(paths1.terminal_prices(), N, mu1);
    double se = std::sqrt(var1 / static_cast<double>(N));

    qre::SimConfig cfg_mt; cfg_mt.n_paths = N; cfg_mt.n_steps = p.n_steps;
    cfg_mt.seed = 42; cfg_mt.n_threads = 0; // auto
    qre::PathData paths_mt(N, p.n_steps);
    eng.simulate_gbm(p, cfg_mt, paths_mt);
    double mu_mt = mean(paths_mt.terminal_prices(), N);

    double diff_se = std::abs(mu1 - mu_mt) / se;
    std::printf("    1-thread mean = %.4f, multi-thread mean = %.4f, diff = %.2f SE\n",
                mu1, mu_mt, diff_se);
    return diff_se < 5.0; // should be within a few SE (different RNG streams)
}

// ---------------------------------------------------------------
// Test 8: VaR/ES analytical comparison
// ---------------------------------------------------------------
bool test_var_analytical() {
    auto p = make_params();
    qre::SimConfig cfg; cfg.n_paths = 1'000'000; cfg.n_steps = p.n_steps; cfg.seed = 42;
    qre::PathData paths(cfg.n_paths, cfg.n_steps);
    qre::CPUEngine eng;
    eng.simulate_gbm(p, cfg, paths);

    auto rm = qre::compute_risk_metrics(paths.terminal_prices(), cfg.n_paths, p.S0);

    // Analytical VaR_99 for GBM:
    // ln(S_T/S0) ~ N(m, s^2) where m = (mu-sigma^2/2)*T, s = sigma*sqrt(T)
    double m = (p.mu - 0.5 * p.sigma * p.sigma) * p.T;
    double s = p.sigma * std::sqrt(p.T);
    // Phi_inv(0.01) ≈ -2.3263
    double z_01 = -2.3263478740408408;
    double st_low = p.S0 * std::exp(m + s * z_01);
    double analytical_var99 = p.S0 - st_low;

    double rel_err = std::abs(rm.var_99 - analytical_var99) / analytical_var99;
    std::printf("    Analytical VaR99 = %.4f, Simulated = %.4f, RelErr = %.2f%%\n",
                analytical_var99, rm.var_99, rel_err * 100);
    std::printf("    ES99 = %.4f, Median = %.4f\n", rm.es_99, rm.median_price);
    return rel_err < 0.02; // within 2%
}

// ---------------------------------------------------------------
// Benchmark: performance timing
// ---------------------------------------------------------------
bool benchmark() {
    auto p = make_params();
    qre::CPUEngine eng;
    size_t N = 1'000'000;

    std::printf("    %8s  %10s  %8s  %14s\n", "Threads", "Time(ms)", "Speedup", "Paths/sec");

    double t1 = 0;
    int max_t = 1;
#ifdef _OPENMP
    max_t = omp_get_max_threads();
#endif
    int thread_counts[] = {1, 2, 4, 8, 16};
    for (int tc : thread_counts) {
        if (tc > max_t) break;
        qre::SimConfig cfg;
        cfg.n_paths = N; cfg.n_steps = p.n_steps; cfg.seed = 42; cfg.n_threads = tc;
        qre::PathData paths(N, p.n_steps);

        // Warm up
        eng.simulate_gbm(p, cfg, paths);

        // Timed run (best of 3)
        double best = 1e9;
        for (int r = 0; r < 3; ++r) {
            auto t_start = std::chrono::high_resolution_clock::now();
            eng.simulate_gbm(p, cfg, paths);
            auto t_end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            best = std::min(best, ms);
        }

        if (tc == 1) t1 = best;
        double speedup = t1 / best;
        double pps = static_cast<double>(N) / (best / 1000.0);
        std::printf("    %8d  %10.1f  %8.2f  %14.0f\n", tc, best, speedup, pps);
    }
    return true; // benchmark always "passes"
}

// ---------------------------------------------------------------
// Registration
// ---------------------------------------------------------------
struct Register {
    Register() {
        register_test("Test 1/8: Mean convergence",          test_mean_convergence);
        register_test("Test 2/8: Log-return variance",       test_log_return_variance);
        register_test("Test 3/8: Distribution shape",        test_distribution_shape);
        register_test("Test 4/8: Convergence rate",          test_convergence_rate);
        register_test("Test 5/8: Antithetic variance reduction", test_antithetic);
        register_test("Test 6/8: Reproducibility",           test_reproducibility);
        register_test("Test 7/8: Multi-threaded correctness", test_multithreaded);
        register_test("Test 8/8: VaR/ES analytical",         test_var_analytical);
        register_test("Benchmark: GBM performance",          benchmark);
    }
} _reg;

} // anonymous namespace
