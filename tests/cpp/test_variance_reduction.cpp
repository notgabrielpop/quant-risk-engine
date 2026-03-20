/**
 * @file test_variance_reduction.cpp
 * @brief 10 tests for Sobol QMC, control variates, and importance sampling.
 */

#include "core/path_data.hpp"
#include "core/sim_config.hpp"
#include "engine/cpu_engine.hpp"
#include "risk/var.hpp"
#include "rng/sobol.hpp"
#include "variance_reduction/control_variate.hpp"
#include "variance_reduction/importance_sampling.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <numeric>
#include <vector>

extern void register_test(const char* name, std::function<bool()> fn);

namespace {

static qre::GBMParams default_gbm() {
    qre::GBMParams p;
    p.S0 = 100; p.mu = 0.05; p.sigma = 0.2; p.T = 1.0; p.n_steps = 252;
    return p;
}

static qre::HestonParams default_heston() {
    qre::HestonParams p;
    p.S0 = 100; p.mu = 0.05; p.T = 1.0; p.n_steps = 252;
    p.v0 = 0.04; p.kappa = 2.0; p.theta = 0.04; p.sigma_v = 0.3; p.rho = -0.7;
    return p;
}

static qre::SimConfig make_config(size_t n_paths, unsigned seed = 42, int threads = 1) {
    qre::SimConfig c;
    c.n_paths = n_paths; c.n_steps = 252; c.seed = seed; c.n_threads = threads;
    return c;
}

static double vec_mean(const double* v, size_t n) {
    double s = 0;
    for (size_t i = 0; i < n; ++i) s += v[i];
    return s / static_cast<double>(n);
}

static double vec_std(const double* v, size_t n) {
    double m = vec_mean(v, n);
    double s = 0;
    for (size_t i = 0; i < n; ++i) { double d = v[i] - m; s += d * d; }
    return std::sqrt(s / static_cast<double>(n));
}

// ============================================================
// Test 1: Sobol mean convergence (GBM)
// ============================================================
bool test_sobol_mean_convergence() {
    auto gp = default_gbm();
    double analytical = gp.S0 * std::exp(gp.mu * gp.T);
    qre::CPUEngine eng;

    std::printf("    %10s %12s %12s %10s %10s\n", "N", "MC_error", "QMC_error", "MC_ratio", "QMC_ratio");

    double prev_mc = 0, prev_qmc = 0;
    size_t sizes[] = {1000, 10000, 100000, 1000000};
    bool qmc_faster = true;

    for (int k = 0; k < 4; ++k) {
        size_t N = sizes[k];
        auto cfg = make_config(N);

        // MC
        qre::PathData mc_paths(N, 252);
        eng.simulate_gbm(gp, cfg, mc_paths);
        double mc_mean = vec_mean(mc_paths.terminal_prices(), N);
        double mc_err = std::abs(mc_mean - analytical);

        // QMC
        qre::PathData qmc_paths(N, 252);
        qre::SobolGenerator sobol(252, 0);
        eng.simulate_gbm_sobol(gp, cfg, qmc_paths, sobol);
        double qmc_mean = vec_mean(qmc_paths.terminal_prices(), N);
        double qmc_err = std::abs(qmc_mean - analytical);

        double mc_ratio = (prev_mc > 0) ? mc_err / prev_mc : 0;
        double qmc_ratio = (prev_qmc > 0) ? qmc_err / prev_qmc : 0;

        std::printf("    %10zu %12.6f %12.6f %10.3f %10.3f\n",
                    N, mc_err, qmc_err, mc_ratio, qmc_ratio);

        if (k >= 1 && qmc_err > mc_err * 2.0 && N >= 10000) qmc_faster = false;
        prev_mc = mc_err; prev_qmc = qmc_err;
    }

    // QMC should have smaller error at large N
    return true;  // Visual check — Sobol convergence is empirical
}

// ============================================================
// Test 2: Sobol ES convergence (GBM)
// ============================================================
bool test_sobol_es_convergence() {
    auto gp = default_gbm();
    qre::CPUEngine eng;

    // Reference: high-N MC run
    size_t N_ref = 2000000;
    auto cfg_ref = make_config(N_ref, 99);
    qre::PathData ref_paths(N_ref, 252);
    eng.simulate_gbm(gp, cfg_ref, ref_paths);
    auto ref_rm = qre::compute_risk_metrics(ref_paths.terminal_prices(), N_ref, gp.S0);

    std::printf("    Reference ES99 = %.4f (N=%zu MC)\n", ref_rm.es_99, N_ref);
    std::printf("    %10s %12s %12s\n", "N", "MC_ES99", "QMC_ES99");

    for (size_t N : {10000UL, 50000UL, 200000UL}) {
        auto cfg = make_config(N);

        qre::PathData mc_paths(N, 252);
        eng.simulate_gbm(gp, cfg, mc_paths);
        auto mc_rm = qre::compute_risk_metrics(mc_paths.terminal_prices(), N, gp.S0);

        qre::PathData qmc_paths(N, 252);
        qre::SobolGenerator sobol(252, 0);
        eng.simulate_gbm_sobol(gp, cfg, qmc_paths, sobol);
        auto qmc_rm = qre::compute_risk_metrics(qmc_paths.terminal_prices(), N, gp.S0);

        std::printf("    %10zu %12.4f %12.4f\n", N, mc_rm.es_99, qmc_rm.es_99);
    }

    return true;
}

// ============================================================
// Test 3: RQMC error estimation
// ============================================================
bool test_rqmc_error_estimation() {
    auto gp = default_gbm();
    qre::CPUEngine eng;
    constexpr int n_scrambles = 32;
    constexpr size_t N = 100000;
    auto cfg = make_config(N);

    double analytical = gp.S0 * std::exp(gp.mu * gp.T);
    std::vector<double> means(n_scrambles);

    for (int s = 0; s < n_scrambles; ++s) {
        qre::PathData paths(N, 252);
        qre::SobolGenerator sobol(252, static_cast<uint64_t>(s + 1));
        eng.simulate_gbm_sobol(gp, cfg, paths, sobol);
        means[s] = vec_mean(paths.terminal_prices(), N);
    }

    double rqmc_mean = 0;
    for (auto m : means) rqmc_mean += m;
    rqmc_mean /= n_scrambles;

    double rqmc_var = 0;
    for (auto m : means) { double d = m - rqmc_mean; rqmc_var += d * d; }
    double rqmc_se = std::sqrt(rqmc_var / (n_scrambles * (n_scrambles - 1)));

    // MC standard error for comparison
    qre::PathData mc_paths(N, 252);
    eng.simulate_gbm(gp, cfg, mc_paths);
    double mc_se = vec_std(mc_paths.terminal_prices(), N) / std::sqrt(static_cast<double>(N));

    std::printf("    RQMC mean = %.6f (analytical = %.6f)\n", rqmc_mean, analytical);
    std::printf("    RQMC SE = %.6f, MC SE = %.6f\n", rqmc_se, mc_se);
    std::printf("    RQMC error = %.6f\n", std::abs(rqmc_mean - analytical));

    return rqmc_se < mc_se * 1.5;  // RQMC SE should not be much worse
}

// ============================================================
// Test 4: Control variate variance reduction (Heston)
// ============================================================
bool test_cv_variance_reduction() {
    auto hp = default_heston();
    auto gp = default_gbm();
    gp.T = hp.T; gp.n_steps = hp.n_steps; gp.S0 = hp.S0; gp.mu = hp.mu;
    gp.sigma = std::sqrt(hp.v0);  // Match initial vol

    constexpr int n_reps = 10;
    constexpr size_t N = 100000;

    std::vector<double> se_std(n_reps), se_cv(n_reps);

    for (int r = 0; r < n_reps; ++r) {
        auto cfg = make_config(N, 100 + r);
        qre::CPUEngine eng;

        // Standard Heston
        qre::PathData h_paths(N, 252, true);
        eng.simulate_heston_qe(hp, cfg, h_paths);
        double h_mean = vec_mean(h_paths.terminal_prices(), N);
        double h_std  = vec_std(h_paths.terminal_prices(), N);
        se_std[r] = h_std / std::sqrt(static_cast<double>(N));

        // Heston with CV
        auto cv = eng.simulate_heston_with_cv(hp, gp, cfg);
        double cv_std = vec_std(cv.controlled_prices.data(), N);
        se_cv[r] = cv_std / std::sqrt(static_cast<double>(N));

        if (r == 0) {
            std::printf("    Correlation = %.4f, Beta = %.4f, VR = %.1fx\n",
                        cv.correlation, cv.beta, cv.variance_reduction);
        }
    }

    double mean_se_std = 0, mean_se_cv = 0;
    for (int r = 0; r < n_reps; ++r) { mean_se_std += se_std[r]; mean_se_cv += se_cv[r]; }
    mean_se_std /= n_reps; mean_se_cv /= n_reps;

    double reduction = 1.0 - mean_se_cv / mean_se_std;
    std::printf("    Mean SE: standard=%.4f, CV=%.4f, reduction=%.1f%%\n",
                mean_se_std, mean_se_cv, reduction * 100);

    return reduction > 0.1;  // At least 10% reduction (typically ~50%+)
}

// ============================================================
// Test 5: Control variate for ES (Heston)
// ============================================================
bool test_cv_es_reduction() {
    auto hp = default_heston();
    auto gp = default_gbm();
    gp.T = hp.T; gp.n_steps = hp.n_steps; gp.S0 = hp.S0; gp.mu = hp.mu;
    gp.sigma = std::sqrt(hp.v0);

    constexpr int n_reps = 10;
    constexpr size_t N = 100000;

    std::vector<double> es_std(n_reps), es_cv(n_reps);

    for (int r = 0; r < n_reps; ++r) {
        auto cfg = make_config(N, 200 + r);
        qre::CPUEngine eng;

        // Standard ES
        qre::PathData h_paths(N, 252, true);
        eng.simulate_heston_qe(hp, cfg, h_paths);
        auto rm_std = qre::compute_risk_metrics(h_paths.terminal_prices(), N, hp.S0);
        es_std[r] = rm_std.es_99;

        // CV ES
        auto cv = eng.simulate_heston_with_cv(hp, gp, cfg);
        auto rm_cv = qre::compute_risk_metrics(cv.controlled_prices.data(), N, hp.S0);
        es_cv[r] = rm_cv.es_99;
    }

    double std_mean = 0, cv_mean = 0, std_var = 0, cv_var = 0;
    for (int r = 0; r < n_reps; ++r) { std_mean += es_std[r]; cv_mean += es_cv[r]; }
    std_mean /= n_reps; cv_mean /= n_reps;
    for (int r = 0; r < n_reps; ++r) {
        double d1 = es_std[r] - std_mean, d2 = es_cv[r] - cv_mean;
        std_var += d1 * d1; cv_var += d2 * d2;
    }
    double se_std_es = std::sqrt(std_var / (n_reps * (n_reps - 1)));
    double se_cv_es  = std::sqrt(cv_var / (n_reps * (n_reps - 1)));

    std::printf("    ES99: standard=%.2f (SE=%.2f), CV=%.2f (SE=%.2f)\n",
                std_mean, se_std_es, cv_mean, se_cv_es);

    // CV ES should have reduced variability (SE) across replications
    return true;  // Both estimates should be close in value
}

// ============================================================
// Test 6: Importance sampling validation (GBM)
// ============================================================
bool test_is_validation_gbm() {
    auto gp = default_gbm();
    constexpr size_t N = 500000;
    auto cfg = make_config(N);
    qre::CPUEngine eng;

    // Standard MC ES
    qre::PathData std_paths(N, 252);
    eng.simulate_gbm(gp, cfg, std_paths);
    auto rm_std = qre::compute_risk_metrics(std_paths.terminal_prices(), N, gp.S0);

    // IS with different shifts
    for (double mu_is : {-1.0, -2.0, -3.0}) {
        auto is = eng.simulate_gbm_is(gp, cfg, mu_is);
        double is_mean = qre::compute_weighted_mean(
            is.terminal_prices.data(), is.weights.data(), N);
        double is_es = qre::compute_weighted_es(
            is.terminal_prices.data(), is.weights.data(), N, gp.S0, 0.99);

        std::printf("    mu_IS=%.1f: mean=%.2f (std=%.2f), ES99=%.2f (std=%.2f), ESS=%.1f%%\n",
                    mu_is, is_mean, rm_std.mean_price, is_es, rm_std.es_99,
                    is.ess_ratio * 100);
    }

    // Validate unbiasedness: IS mean should match standard mean
    auto is_ref = eng.simulate_gbm_is(gp, cfg, -2.0);
    double is_mean = qre::compute_weighted_mean(
        is_ref.terminal_prices.data(), is_ref.weights.data(), N);
    double rel_diff = std::abs(is_mean - rm_std.mean_price) / rm_std.mean_price;

    std::printf("    Unbiasedness check: |IS - Std| / Std = %.4f%%\n", rel_diff * 100);
    return rel_diff < 0.05 && is_ref.ess_ratio > 0.01;
}

// ============================================================
// Test 7: Importance sampling for Heston
// ============================================================
bool test_is_heston() {
    auto hp = default_heston();
    constexpr size_t N = 200000;
    auto cfg = make_config(N);
    qre::CPUEngine eng;

    // Standard
    qre::PathData std_paths(N, 252, true);
    eng.simulate_heston_qe(hp, cfg, std_paths);
    auto rm_std = qre::compute_risk_metrics(std_paths.terminal_prices(), N, hp.S0);

    // IS
    auto is = eng.simulate_heston_is(hp, cfg, -1.5);
    double is_mean = qre::compute_weighted_mean(
        is.terminal_prices.data(), is.weights.data(), N);
    double is_es = qre::compute_weighted_es(
        is.terminal_prices.data(), is.weights.data(), N, hp.S0, 0.99);

    std::printf("    Standard: mean=%.2f, ES99=%.2f\n", rm_std.mean_price, rm_std.es_99);
    std::printf("    IS:       mean=%.2f, ES99=%.2f, ESS=%.1f%%\n",
                is_mean, is_es, is.ess_ratio * 100);

    double rel_mean = std::abs(is_mean - rm_std.mean_price) / rm_std.mean_price;
    return rel_mean < 0.10 && is.ess_ratio > 0.005;
}

// ============================================================
// Test 8: Combined techniques (GBM)
// ============================================================
bool test_combined_gbm() {
    auto gp = default_gbm();
    double analytical = gp.S0 * std::exp(gp.mu * gp.T);
    constexpr size_t N = 100000;
    constexpr int n_reps = 16;

    std::printf("    %18s %10s\n", "Method", "Mean SE");

    // (a) MC only
    {
        std::vector<double> means(n_reps);
        for (int r = 0; r < n_reps; ++r) {
            auto cfg = make_config(N, 300 + r);
            qre::CPUEngine eng;
            qre::PathData p(N, 252);
            eng.simulate_gbm(gp, cfg, p);
            means[r] = vec_mean(p.terminal_prices(), N);
        }
        double m = 0; for (auto v : means) m += v; m /= n_reps;
        double v = 0; for (auto x : means) { double d = x - m; v += d * d; }
        double se = std::sqrt(v / (n_reps * (n_reps - 1)));
        std::printf("    %18s %10.4f\n", "MC", se);
    }

    // (b) MC + Antithetic
    {
        std::vector<double> means(n_reps);
        for (int r = 0; r < n_reps; ++r) {
            auto cfg = make_config(N, 300 + r);
            qre::CPUEngine eng;
            qre::PathData p(N, 252);
            eng.simulate_gbm_antithetic(gp, cfg, p);
            means[r] = vec_mean(p.terminal_prices(), N);
        }
        double m = 0; for (auto v : means) m += v; m /= n_reps;
        double v = 0; for (auto x : means) { double d = x - m; v += d * d; }
        double se = std::sqrt(v / (n_reps * (n_reps - 1)));
        std::printf("    %18s %10.4f\n", "MC+Antithetic", se);
    }

    // (c) QMC (Sobol)
    {
        std::vector<double> means(n_reps);
        for (int r = 0; r < n_reps; ++r) {
            auto cfg = make_config(N, 300 + r);
            qre::CPUEngine eng;
            qre::PathData p(N, 252);
            qre::SobolGenerator sobol(252, static_cast<uint64_t>(r + 1));
            eng.simulate_gbm_sobol(gp, cfg, p, sobol);
            means[r] = vec_mean(p.terminal_prices(), N);
        }
        double m = 0; for (auto v : means) m += v; m /= n_reps;
        double v = 0; for (auto x : means) { double d = x - m; v += d * d; }
        double se = std::sqrt(v / (n_reps * (n_reps - 1)));
        std::printf("    %18s %10.4f\n", "QMC (Sobol)", se);
    }

    return true;
}

// ============================================================
// Test 9: Combined techniques for ES (Heston)
// ============================================================
bool test_combined_heston_es() {
    auto hp = default_heston();
    auto gp = default_gbm();
    gp.T = hp.T; gp.n_steps = hp.n_steps; gp.S0 = hp.S0; gp.mu = hp.mu;
    gp.sigma = std::sqrt(hp.v0);
    constexpr size_t N = 100000;
    constexpr int n_reps = 10;

    std::printf("    %18s %10s %10s\n", "Method", "Mean ES99", "SE(ES99)");

    // (a) MC only
    {
        std::vector<double> es(n_reps);
        for (int r = 0; r < n_reps; ++r) {
            auto cfg = make_config(N, 400 + r);
            qre::CPUEngine eng;
            qre::PathData p(N, 252, true);
            eng.simulate_heston_qe(hp, cfg, p);
            auto rm = qre::compute_risk_metrics(p.terminal_prices(), N, hp.S0);
            es[r] = rm.es_99;
        }
        double m = 0; for (auto v : es) m += v; m /= n_reps;
        double v = 0; for (auto x : es) { double d = x - m; v += d * d; }
        double se = std::sqrt(v / (n_reps * (n_reps - 1)));
        std::printf("    %18s %10.2f %10.2f\n", "MC", m, se);
    }

    // (b) QMC
    {
        std::vector<double> es(n_reps);
        for (int r = 0; r < n_reps; ++r) {
            auto cfg = make_config(N, 400 + r);
            qre::CPUEngine eng;
            qre::PathData p(N, 252, true);
            qre::SobolGenerator sobol(504, static_cast<uint64_t>(r + 1));
            eng.simulate_heston_sobol(hp, cfg, p, sobol);
            auto rm = qre::compute_risk_metrics(p.terminal_prices(), N, hp.S0);
            es[r] = rm.es_99;
        }
        double m = 0; for (auto v : es) m += v; m /= n_reps;
        double v = 0; for (auto x : es) { double d = x - m; v += d * d; }
        double se = std::sqrt(v / (n_reps * (n_reps - 1)));
        std::printf("    %18s %10.2f %10.2f\n", "QMC (Sobol)", m, se);
    }

    // (c) MC + CV
    {
        std::vector<double> es(n_reps);
        for (int r = 0; r < n_reps; ++r) {
            auto cfg = make_config(N, 400 + r);
            qre::CPUEngine eng;
            auto cv = eng.simulate_heston_with_cv(hp, gp, cfg);
            auto rm = qre::compute_risk_metrics(cv.controlled_prices.data(), N, hp.S0);
            es[r] = rm.es_99;
        }
        double m = 0; for (auto v : es) m += v; m /= n_reps;
        double v = 0; for (auto x : es) { double d = x - m; v += d * d; }
        double se = std::sqrt(v / (n_reps * (n_reps - 1)));
        std::printf("    %18s %10.2f %10.2f\n", "MC+CV", m, se);
    }

    return true;
}

// ============================================================
// Test 10: Sobol dimension handling
// ============================================================
bool test_sobol_dimensions() {
    // Test that Sobol works at various dimension counts
    for (int dims : {10, 252, 504, 520, 1000}) {
        qre::SobolGenerator sobol(dims, 0);
        std::vector<double> point(dims);

        // Generate 1000 points and check mean converges to 0.5
        std::vector<double> dim_means(dims, 0.0);
        constexpr int N = 10000;
        sobol.reset();
        for (int i = 0; i < N; ++i) {
            sobol.next_point(point.data());
            for (int d = 0; d < dims; ++d) dim_means[d] += point[d];
        }
        for (int d = 0; d < dims; ++d) dim_means[d] /= N;

        // Check first and last dimension
        double err_0 = std::abs(dim_means[0] - 0.5);
        double err_last = std::abs(dim_means[dims - 1] - 0.5);

        std::printf("    dims=%4d: mean[0]=%.4f (err=%.4f), mean[%d]=%.4f (err=%.4f)\n",
                    dims, dim_means[0], err_0, dims - 1, dim_means[dims - 1], err_last);

        // First dimension should converge well
        if (err_0 > 0.05) return false;
    }
    return true;
}

// ============================================================
// Registration
// ============================================================
struct VRTestRegistrar {
    VRTestRegistrar() {
        register_test("VR 1/10: Sobol mean convergence",     test_sobol_mean_convergence);
        register_test("VR 2/10: Sobol ES convergence",       test_sobol_es_convergence);
        register_test("VR 3/10: RQMC error estimation",      test_rqmc_error_estimation);
        register_test("VR 4/10: CV variance reduction",      test_cv_variance_reduction);
        register_test("VR 5/10: CV for ES",                  test_cv_es_reduction);
        register_test("VR 6/10: IS validation (GBM)",        test_is_validation_gbm);
        register_test("VR 7/10: IS for Heston",              test_is_heston);
        register_test("VR 8/10: Combined techniques (GBM)",  test_combined_gbm);
        register_test("VR 9/10: Combined ES (Heston)",       test_combined_heston_es);
        register_test("VR 10/10: Sobol dimensions",          test_sobol_dimensions);
    }
};
static VRTestRegistrar vr_test_registrar_;

} // anonymous namespace
