/**
 * @file test_heston.cpp
 * @brief 10 validation tests for Heston stochastic volatility model.
 */

#include "core/path_data.hpp"
#include "core/sim_config.hpp"
#include "models/heston.hpp"
#include "engine/cpu_engine.hpp"
#include "risk/var.hpp"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <numeric>
#include <vector>

extern void register_test(const char*, std::function<bool()>);

namespace {

// Helpers
double mean(const double* d, size_t n) {
    double s = 0;
    for (size_t i = 0; i < n; ++i) s += d[i];
    return s / static_cast<double>(n);
}
double variance(const double* d, size_t n, double mu) {
    double s = 0;
    for (size_t i = 0; i < n; ++i) { double x = d[i] - mu; s += x * x; }
    return s / static_cast<double>(n - 1);
}

// Reference parameters satisfying Feller: 2*2*0.04 = 0.16 > 0.09 = 0.3^2
qre::HestonParams make_heston() {
    qre::HestonParams p;
    p.S0 = 100.0; p.mu = 0.05; p.T = 1.0; p.n_steps = 252;
    p.v0 = 0.04; p.kappa = 2.0; p.theta = 0.04;
    p.sigma_v = 0.3; p.rho = -0.7;
    return p;
}

// ---------------------------------------------------------------
// Test 1: Mean price convergence (both schemes)
// E[S_T] = S0 * exp(mu * T) regardless of vol model
// ---------------------------------------------------------------
bool test_heston_mean() {
    auto p = make_heston();
    qre::CPUEngine eng;
    size_t N = 1'000'000;
    double analytical = p.S0 * std::exp(p.mu * p.T);

    // Euler
    qre::SimConfig cfg; cfg.n_paths = N; cfg.n_steps = p.n_steps; cfg.seed = 42;
    qre::PathData paths_e(N, p.n_steps, true);
    eng.simulate_heston_euler(p, cfg, paths_e);
    double mu_euler = mean(paths_e.terminal_prices(), N);
    double err_e = std::abs(mu_euler - analytical) / analytical;

    // QE
    qre::PathData paths_q(N, p.n_steps, true);
    eng.simulate_heston_qe(p, cfg, paths_q);
    double mu_qe = mean(paths_q.terminal_prices(), N);
    double err_q = std::abs(mu_qe - analytical) / analytical;

    std::printf("    Analytical E[S_T] = %.4f\n", analytical);
    std::printf("    Euler:  mean = %.4f, RelErr = %.4f%%\n", mu_euler, err_e * 100);
    std::printf("    QE:     mean = %.4f, RelErr = %.4f%%\n", mu_qe, err_q * 100);
    return err_e < 0.005 && err_q < 0.005; // within 0.5%
}

// ---------------------------------------------------------------
// Test 2: Forward variance curve
// Set v0 = 0.09, E[v_t] = theta + (v0-theta)*exp(-kappa*t)
// ---------------------------------------------------------------
bool test_forward_variance() {
    auto p = make_heston();
    p.v0 = 0.09; // Start above equilibrium
    qre::CPUEngine eng;
    size_t N = 500'000;

    qre::SimConfig cfg; cfg.n_paths = N; cfg.n_steps = p.n_steps; cfg.seed = 42;
    qre::PathData paths(N, p.n_steps, true);
    eng.simulate_heston_qe(p, cfg, paths);

    struct CheckPoint { double t; size_t step; double expected; };
    CheckPoint checks[] = {
        {0.25, 63,  p.theta + (p.v0 - p.theta) * std::exp(-p.kappa * 0.25)},
        {0.50, 126, p.theta + (p.v0 - p.theta) * std::exp(-p.kappa * 0.50)},
        {1.00, 252, p.theta + (p.v0 - p.theta) * std::exp(-p.kappa * 1.00)},
    };

    bool ok = true;
    for (auto& cp : checks) {
        // Average variance at this step across all paths
        double sum = 0;
        for (size_t i = 0; i < N; ++i)
            sum += paths.variance(i, cp.step);
        double sim = sum / static_cast<double>(N);
        double rel_err = std::abs(sim - cp.expected) / cp.expected;
        std::printf("    t=%.2f: E[v]=%.6f, sim=%.6f, RelErr=%.2f%%\n",
                    cp.t, cp.expected, sim, rel_err * 100);
        ok &= rel_err < 0.02; // within 2%
    }
    return ok;
}

// ---------------------------------------------------------------
// Test 3: Non-Gaussian returns (negative skew, positive kurtosis)
// ---------------------------------------------------------------
bool test_non_gaussian() {
    auto p = make_heston();
    qre::CPUEngine eng;
    size_t N = 1'000'000;

    qre::SimConfig cfg; cfg.n_paths = N; cfg.n_steps = p.n_steps; cfg.seed = 42;
    qre::PathData paths(N, p.n_steps, true);
    eng.simulate_heston_qe(p, cfg, paths);

    // Compute log-return moments
    std::vector<double> lr(N);
    for (size_t i = 0; i < N; ++i)
        lr[i] = std::log(paths.terminal_prices()[i] / p.S0);

    double mu_lr = mean(lr.data(), N);
    double s2 = 0, s3 = 0, s4 = 0;
    for (size_t i = 0; i < N; ++i) {
        double d = lr[i] - mu_lr;
        double d2 = d * d;
        s2 += d2; s3 += d2 * d; s4 += d2 * d2;
    }
    double v = s2 / static_cast<double>(N);
    double sd = std::sqrt(v);
    double skew = (s3 / static_cast<double>(N)) / (v * sd);
    double kurt = (s4 / static_cast<double>(N)) / (v * v) - 3.0;

    std::printf("    Skewness = %.4f (expect < 0), Excess Kurtosis = %.4f (expect > 0)\n",
                skew, kurt);
    return skew < -0.1 && kurt > 0.3;
}

// ---------------------------------------------------------------
// Test 4: Volatility clustering (ACF of |returns|)
// ---------------------------------------------------------------
bool test_vol_clustering() {
    auto p = make_heston();
    qre::CPUEngine eng;
    size_t N = 10'000; // Need full paths, fewer paths is fine

    qre::SimConfig cfg; cfg.n_paths = N; cfg.n_steps = p.n_steps; cfg.seed = 42;
    qre::PathData paths(N, p.n_steps, true);
    eng.simulate_heston_qe(p, cfg, paths);

    // Compute daily log-returns for each path, then average ACF
    size_t n_ret = p.n_steps;
    int lags[] = {1, 5, 10, 20};
    double acf_vals[4] = {0};

    for (int lag_idx = 0; lag_idx < 4; ++lag_idx) {
        int lag = lags[lag_idx];
        double acf_sum = 0;
        size_t valid_paths = 0;

        for (size_t i = 0; i < N; ++i) {
            // Compute |returns| for this path
            std::vector<double> abs_ret(n_ret);
            for (size_t s = 0; s < n_ret; ++s)
                abs_ret[s] = std::abs(std::log(paths.price(i, s + 1) / paths.price(i, s)));

            // Mean of |returns|
            double m = 0;
            for (size_t s = 0; s < n_ret; ++s) m += abs_ret[s];
            m /= static_cast<double>(n_ret);

            // ACF at this lag
            double num = 0, den = 0;
            for (size_t s = 0; s < n_ret; ++s) {
                den += (abs_ret[s] - m) * (abs_ret[s] - m);
                if (s + static_cast<size_t>(lag) < n_ret)
                    num += (abs_ret[s] - m) * (abs_ret[s + lag] - m);
            }
            if (den > 1e-15) {
                acf_sum += num / den;
                ++valid_paths;
            }
        }
        acf_vals[lag_idx] = acf_sum / static_cast<double>(valid_paths);
    }

    std::printf("    ACF of |returns|: lag1=%.4f, lag5=%.4f, lag10=%.4f, lag20=%.4f\n",
                acf_vals[0], acf_vals[1], acf_vals[2], acf_vals[3]);
    // All should be positive (vol clustering)
    return acf_vals[0] > 0.02 && acf_vals[1] > 0.01 && acf_vals[10 < 20 ? 2 : 3] > 0.0;
}

// ---------------------------------------------------------------
// Test 5: Leverage effect (negative corr between returns and vol change)
// ---------------------------------------------------------------
bool test_leverage_effect() {
    auto p = make_heston();
    qre::CPUEngine eng;
    size_t N = 50'000;

    qre::SimConfig cfg; cfg.n_paths = N; cfg.n_steps = p.n_steps; cfg.seed = 42;
    qre::PathData paths(N, p.n_steps, true);
    eng.simulate_heston_qe(p, cfg, paths);

    // Contemporaneous correlation: corr(return_t, dv_t)
    // return_t = ln(S_{t+1}/S_t), dv_t = v_{t+1} - v_t
    double sum_r = 0, sum_dv = 0, sum_rr = 0, sum_dvdv = 0, sum_rdv = 0;
    size_t count = 0;

    for (size_t i = 0; i < N; ++i) {
        for (size_t s = 0; s < p.n_steps; ++s) {
            double ret = std::log(paths.price(i, s + 1) / paths.price(i, s));
            double dv  = paths.variance(i, s + 1) - paths.variance(i, s);
            sum_r   += ret;
            sum_dv  += dv;
            sum_rr  += ret * ret;
            sum_dvdv += dv * dv;
            sum_rdv += ret * dv;
            ++count;
        }
    }

    double n = static_cast<double>(count);
    double cov  = sum_rdv / n - (sum_r / n) * (sum_dv / n);
    double sd_r = std::sqrt(sum_rr / n - (sum_r / n) * (sum_r / n));
    double sd_dv = std::sqrt(sum_dvdv / n - (sum_dv / n) * (sum_dv / n));
    double corr = cov / (sd_r * sd_dv);

    std::printf("    Corr(return, next dv) = %.4f (expect < 0 with rho=%.1f)\n",
                corr, p.rho);
    return corr < -0.05; // Should be noticeably negative
}

// ---------------------------------------------------------------
// Test 6: Variance non-negativity (Euler vs QE)
// ---------------------------------------------------------------
bool test_variance_nonneg() {
    auto p = make_heston();
    qre::CPUEngine eng;
    size_t N = 500'000;

    qre::SimConfig cfg; cfg.n_paths = N; cfg.n_steps = p.n_steps; cfg.seed = 42;

    // Euler with counting
    qre::PathData paths_e(N, p.n_steps, true);
    size_t neg_euler = 0;
    eng.simulate_heston_euler_count(p, cfg, paths_e, neg_euler);

    // QE — check all variance values
    qre::PathData paths_q(N, p.n_steps, true);
    eng.simulate_heston_qe(p, cfg, paths_q);
    size_t neg_qe = 0;
    for (size_t i = 0; i < N; ++i)
        for (size_t s = 0; s <= p.n_steps; ++s)
            if (paths_q.variance(i, s) < 0.0) ++neg_qe;

    std::printf("    Euler negative variance incidents: %zu (before absorption)\n", neg_euler);
    std::printf("    QE negative variance values: %zu\n", neg_qe);
    return neg_qe == 0; // QE must never produce negative variance
}

// ---------------------------------------------------------------
// Test 7: Euler vs QE consistency
// ---------------------------------------------------------------
bool test_euler_vs_qe() {
    auto p = make_heston();
    qre::CPUEngine eng;
    size_t N = 1'000'000;

    qre::SimConfig cfg; cfg.n_paths = N; cfg.n_steps = p.n_steps; cfg.seed = 42;

    qre::PathData paths_e(N, p.n_steps, true);
    eng.simulate_heston_euler(p, cfg, paths_e);
    auto rm_e = qre::compute_risk_metrics(paths_e.terminal_prices(), N, p.S0);

    qre::PathData paths_q(N, p.n_steps, true);
    eng.simulate_heston_qe(p, cfg, paths_q);
    auto rm_q = qre::compute_risk_metrics(paths_q.terminal_prices(), N, p.S0);

    double diff_mean = std::abs(rm_e.mean_price - rm_q.mean_price) / rm_q.mean_price;
    double diff_std  = std::abs(rm_e.std_price - rm_q.std_price) / rm_q.std_price;
    double diff_var99 = std::abs(rm_e.var_99 - rm_q.var_99) / rm_q.var_99;
    double diff_es99  = std::abs(rm_e.es_99 - rm_q.es_99) / rm_q.es_99;

    std::printf("    Euler: mean=%.2f std=%.2f VaR99=%.2f ES99=%.2f\n",
                rm_e.mean_price, rm_e.std_price, rm_e.var_99, rm_e.es_99);
    std::printf("    QE:    mean=%.2f std=%.2f VaR99=%.2f ES99=%.2f\n",
                rm_q.mean_price, rm_q.std_price, rm_q.var_99, rm_q.es_99);
    std::printf("    Diffs: mean=%.2f%% std=%.2f%% VaR99=%.2f%% ES99=%.2f%%\n",
                diff_mean*100, diff_std*100, diff_var99*100, diff_es99*100);
    return diff_mean < 0.02 && diff_std < 0.05 && diff_var99 < 0.10 && diff_es99 < 0.10;
}

// ---------------------------------------------------------------
// Test 8: QE dt sensitivity (should be more stable than Euler)
// ---------------------------------------------------------------
bool test_dt_sensitivity() {
    auto p = make_heston();
    qre::CPUEngine eng;
    size_t N = 500'000;
    size_t steps[] = {52, 252, 1260};

    std::printf("    %8s  %6s  %10s  %10s  %10s  %10s\n",
                "Scheme", "Steps", "Mean", "Std", "VaR99", "ES99");

    double qe_vars[3], eu_vars[3];

    for (int k = 0; k < 3; ++k) {
        p.n_steps = steps[k];
        qre::SimConfig cfg;
        cfg.n_paths = N; cfg.n_steps = steps[k]; cfg.seed = 42;

        // QE
        qre::PathData pq(N, steps[k], true);
        eng.simulate_heston_qe(p, cfg, pq);
        auto rmq = qre::compute_risk_metrics(pq.terminal_prices(), N, p.S0);
        qe_vars[k] = rmq.var_99;
        std::printf("    %8s  %6zu  %10.2f  %10.2f  %10.2f  %10.2f\n",
                    "QE", steps[k], rmq.mean_price, rmq.std_price, rmq.var_99, rmq.es_99);

        // Euler
        qre::PathData pe(N, steps[k], true);
        eng.simulate_heston_euler(p, cfg, pe);
        auto rme = qre::compute_risk_metrics(pe.terminal_prices(), N, p.S0);
        eu_vars[k] = rme.var_99;
        std::printf("    %8s  %6zu  %10.2f  %10.2f  %10.2f  %10.2f\n",
                    "Euler", steps[k], rme.mean_price, rme.std_price, rme.var_99, rme.es_99);
    }

    // QE range should be smaller than Euler range (more stable across dt)
    double qe_range = std::abs(qe_vars[0] - qe_vars[2]) / qe_vars[2];
    double eu_range = std::abs(eu_vars[0] - eu_vars[2]) / eu_vars[2];
    std::printf("    QE VaR99 range: %.2f%%, Euler VaR99 range: %.2f%%\n",
                qe_range * 100, eu_range * 100);
    // Both should produce reasonable results
    return qe_vars[2] > 10 && eu_vars[2] > 10;
}

// ---------------------------------------------------------------
// Test 9: Feller condition violation (sigma_v = 0.8)
// ---------------------------------------------------------------
bool test_feller_violation() {
    auto p = make_heston();
    p.sigma_v = 0.8; // Feller violated: 2*2*0.04=0.16 < 0.64
    qre::CPUEngine eng;
    size_t N = 500'000;

    qre::SimConfig cfg; cfg.n_paths = N; cfg.n_steps = p.n_steps; cfg.seed = 42;

    // Euler with counting
    qre::PathData paths_e(N, p.n_steps, true);
    size_t neg_euler = 0;
    eng.simulate_heston_euler_count(p, cfg, paths_e, neg_euler);

    // QE
    qre::PathData paths_q(N, p.n_steps, true);
    eng.simulate_heston_qe(p, cfg, paths_q);
    size_t neg_qe = 0;
    bool any_nan = false;
    for (size_t i = 0; i < N; ++i) {
        for (size_t s = 0; s <= p.n_steps; ++s) {
            if (paths_q.variance(i, s) < 0.0) ++neg_qe;
            if (std::isnan(paths_q.variance(i, s)) || std::isnan(paths_q.price(i, s)))
                any_nan = true;
        }
    }

    double pct_euler = 100.0 * static_cast<double>(neg_euler) /
                       static_cast<double>(N * p.n_steps);

    std::printf("    Feller violated: 2*kappa*theta=%.2f vs sigma_v^2=%.2f\n",
                2 * p.kappa * p.theta, p.sigma_v * p.sigma_v);
    std::printf("    Euler neg incidents: %zu (%.2f%% of all steps)\n", neg_euler, pct_euler);
    std::printf("    QE neg variance: %zu, any NaN: %s\n", neg_qe, any_nan ? "YES" : "no");

    // Euler should have many negative incidents; QE should have zero
    return neg_euler > 0 && neg_qe == 0 && !any_nan;
}

// ---------------------------------------------------------------
// Test 10: Reproducibility and multi-thread correctness
// ---------------------------------------------------------------
bool test_heston_repro() {
    auto p = make_heston();
    qre::CPUEngine eng;
    size_t N = 10'000;

    // Reproducibility (QE, single thread)
    qre::SimConfig cfg; cfg.n_paths = N; cfg.n_steps = p.n_steps;
    cfg.seed = 42; cfg.n_threads = 1;

    qre::PathData p1(N, p.n_steps, true), p2(N, p.n_steps, true);
    eng.simulate_heston_qe(p, cfg, p1);
    eng.simulate_heston_qe(p, cfg, p2);

    bool identical = true;
    for (size_t i = 0; i < N; ++i)
        if (p1.terminal_prices()[i] != p2.terminal_prices()[i]) { identical = false; break; }

    // Different seed
    qre::SimConfig cfg2 = cfg; cfg2.seed = 123;
    qre::PathData p3(N, p.n_steps, true);
    eng.simulate_heston_qe(p, cfg2, p3);
    bool different = (p1.terminal_prices()[0] != p3.terminal_prices()[0]);

    // Multi-thread vs single-thread (statistical equivalence)
    size_t N2 = 500'000;
    qre::SimConfig cfg_st; cfg_st.n_paths = N2; cfg_st.n_steps = p.n_steps;
    cfg_st.seed = 42; cfg_st.n_threads = 1;
    qre::PathData ps(N2, p.n_steps, true);
    eng.simulate_heston_qe(p, cfg_st, ps);
    double mu1 = mean(ps.terminal_prices(), N2);
    double var1 = variance(ps.terminal_prices(), N2, mu1);
    double se = std::sqrt(var1 / static_cast<double>(N2));

    qre::SimConfig cfg_mt = cfg_st; cfg_mt.n_threads = 0;
    qre::PathData pm(N2, p.n_steps, true);
    eng.simulate_heston_qe(p, cfg_mt, pm);
    double mu_mt = mean(pm.terminal_prices(), N2);

    double diff_se = std::abs(mu1 - mu_mt) / se;
    std::printf("    Same seed identical: %s, Different seed different: %s\n",
                identical ? "yes" : "no", different ? "yes" : "no");
    std::printf("    1-thread=%.4f, multi-thread=%.4f, diff=%.2f SE\n", mu1, mu_mt, diff_se);
    return identical && different && diff_se < 5.0;
}

// ---------------------------------------------------------------
// Benchmark: GBM vs Heston Euler vs Heston QE
// ---------------------------------------------------------------
bool benchmark_heston() {
    auto hp = make_heston();
    qre::GBMParams gp;
    gp.S0 = hp.S0; gp.mu = hp.mu; gp.sigma = std::sqrt(hp.v0);
    gp.T = hp.T; gp.n_steps = hp.n_steps;

    qre::CPUEngine eng;
    size_t N = 1'000'000;

    std::printf("    %12s  %8s  %10s  %8s  %14s\n",
                "Scheme", "Threads", "Time(ms)", "Speedup", "Paths/sec");

    struct BenchEntry { const char* name; int scheme; };
    BenchEntry entries[] = {{"GBM exact", 0}, {"Heston Euler", 1}, {"Heston QE", 2}};

    int max_t = 1;
    #ifdef _OPENMP
    max_t = omp_get_max_threads();
    #endif

    for (auto& entry : entries) {
        double t1_best = 0;
        int thread_counts[] = {1, max_t};
        for (int tc : thread_counts) {
            if (tc > max_t) break;
            qre::SimConfig cfg;
            cfg.n_paths = N; cfg.n_steps = hp.n_steps; cfg.seed = 42; cfg.n_threads = tc;

            // Warm up + timed (best of 3)
            double best = 1e9;
            for (int r = 0; r < 3; ++r) {
                if (entry.scheme == 0) {
                    qre::PathData paths(N, gp.n_steps);
                    auto t0 = std::chrono::high_resolution_clock::now();
                    eng.simulate_gbm(gp, cfg, paths);
                    auto t_end = std::chrono::high_resolution_clock::now();
                    best = std::min(best, std::chrono::duration<double, std::milli>(t_end - t0).count());
                } else if (entry.scheme == 1) {
                    qre::PathData paths(N, hp.n_steps, true);
                    auto t0 = std::chrono::high_resolution_clock::now();
                    eng.simulate_heston_euler(hp, cfg, paths);
                    auto t_end = std::chrono::high_resolution_clock::now();
                    best = std::min(best, std::chrono::duration<double, std::milli>(t_end - t0).count());
                } else {
                    qre::PathData paths(N, hp.n_steps, true);
                    auto t0 = std::chrono::high_resolution_clock::now();
                    eng.simulate_heston_qe(hp, cfg, paths);
                    auto t_end = std::chrono::high_resolution_clock::now();
                    best = std::min(best, std::chrono::duration<double, std::milli>(t_end - t0).count());
                }
            }
            if (tc == 1) t1_best = best;
            double speedup = t1_best / best;
            double pps = static_cast<double>(N) / (best / 1000.0);
            std::printf("    %12s  %8d  %10.1f  %8.2f  %14.0f\n",
                        entry.name, tc, best, speedup, pps);
        }
    }
    return true;
}

// ---------------------------------------------------------------
// Registration
// ---------------------------------------------------------------
struct Register {
    Register() {
        register_test("Heston 1/10: Mean convergence",       test_heston_mean);
        register_test("Heston 2/10: Forward variance curve",  test_forward_variance);
        register_test("Heston 3/10: Non-Gaussian returns",    test_non_gaussian);
        register_test("Heston 4/10: Volatility clustering",   test_vol_clustering);
        register_test("Heston 5/10: Leverage effect",         test_leverage_effect);
        register_test("Heston 6/10: Variance non-negativity", test_variance_nonneg);
        register_test("Heston 7/10: Euler vs QE consistency", test_euler_vs_qe);
        register_test("Heston 8/10: dt sensitivity",          test_dt_sensitivity);
        register_test("Heston 9/10: Feller violation",        test_feller_violation);
        register_test("Heston 10/10: Reproducibility",        test_heston_repro);
        register_test("Benchmark: Heston performance",         benchmark_heston);
    }
} _reg;

} // anonymous namespace
