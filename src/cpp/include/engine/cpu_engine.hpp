#pragma once

/**
 * @file cpu_engine.hpp
 * @brief CPU Monte Carlo engine: GBM, Heston (Euler/QE), Rough Heston (Markov), OpenMP.
 *        Supports MC (Mersenne Twister) and QMC (Sobol), control variates, importance sampling.
 */

#include "../core/path_data.hpp"
#include "../core/sim_config.hpp"
#include "../models/fractional_kernel.hpp"
#include "../rng/sobol.hpp"
#include "../rng/inverse_normal.hpp"
#include "../variance_reduction/control_variate.hpp"
#include "../variance_reduction/importance_sampling.hpp"
#include <cmath>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#else
inline int omp_get_max_threads() { return 1; }
inline int omp_get_thread_num()  { return 0; }
#endif

namespace qre {

class CPUEngine {
public:

    // -----------------------------------------------------------------
    // GBM — exact log-normal solution (zero discretisation error)
    // -----------------------------------------------------------------
    void simulate_gbm(const GBMParams& params, const SimConfig& config,
                      PathData& paths) {
        const double dt         = params.dt();
        const double drift_term = (params.mu - 0.5 * params.sigma * params.sigma) * dt;
        const double vol_term   = params.sigma * std::sqrt(dt);
        const int nt = config.n_threads > 0 ? config.n_threads : omp_get_max_threads();

        for (size_t i = 0; i < config.n_paths; ++i)
            paths.price(i, 0) = params.S0;

        #pragma omp parallel num_threads(nt)
        {
            const auto tid = static_cast<unsigned long long>(omp_get_thread_num());
            std::mt19937_64 rng(config.seed + tid * 1000003ULL);
            std::normal_distribution<double> normal(0.0, 1.0);

            #pragma omp for schedule(static)
            for (size_t i = 0; i < config.n_paths; ++i) {
                double S = params.S0;
                for (size_t s = 0; s < config.n_steps; ++s) {
                    S *= std::exp(drift_term + vol_term * normal(rng));
                    paths.price(i, s + 1) = S;
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // GBM with antithetic variates
    // -----------------------------------------------------------------
    void simulate_gbm_antithetic(const GBMParams& params,
                                 const SimConfig& config,
                                 PathData& paths) {
        const size_t half       = config.n_paths / 2;
        const double dt         = params.dt();
        const double drift_term = (params.mu - 0.5 * params.sigma * params.sigma) * dt;
        const double vol_term   = params.sigma * std::sqrt(dt);
        const int nt = config.n_threads > 0 ? config.n_threads : omp_get_max_threads();

        for (size_t i = 0; i < config.n_paths; ++i)
            paths.price(i, 0) = params.S0;

        #pragma omp parallel num_threads(nt)
        {
            const auto tid = static_cast<unsigned long long>(omp_get_thread_num());
            std::mt19937_64 rng(config.seed + tid * 1000003ULL);
            std::normal_distribution<double> normal(0.0, 1.0);

            #pragma omp for schedule(static)
            for (size_t i = 0; i < half; ++i) {
                double S_pos = params.S0;
                double S_neg = params.S0;
                for (size_t s = 0; s < config.n_steps; ++s) {
                    double Z = normal(rng);
                    S_pos *= std::exp(drift_term + vol_term * Z);
                    S_neg *= std::exp(drift_term + vol_term * (-Z));
                    paths.price(i,        s + 1) = S_pos;
                    paths.price(i + half, s + 1) = S_neg;
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // Heston Euler-Maruyama with absorption and log-price step
    // -----------------------------------------------------------------
    void simulate_heston_euler(const HestonParams& params,
                               const SimConfig& config,
                               PathData& paths) {
        const double dt       = params.dt();
        const double sqrt_dt  = std::sqrt(dt);
        const double rho      = params.rho;
        const double sqrt_1r2 = std::sqrt(1.0 - rho * rho);
        const int nt = config.n_threads > 0 ? config.n_threads : omp_get_max_threads();

        for (size_t i = 0; i < config.n_paths; ++i) {
            paths.price(i, 0)    = params.S0;
            paths.variance(i, 0) = params.v0;
        }

        #pragma omp parallel num_threads(nt)
        {
            const auto tid = static_cast<unsigned long long>(omp_get_thread_num());
            std::mt19937_64 rng(config.seed + tid * 1000003ULL);
            std::normal_distribution<double> normal(0.0, 1.0);

            #pragma omp for schedule(static)
            for (size_t i = 0; i < config.n_paths; ++i) {
                double S = params.S0;
                double v = params.v0;

                for (size_t s = 0; s < config.n_steps; ++s) {
                    double Z1 = normal(rng);
                    double Z2 = normal(rng);
                    double dW1 = Z1;
                    double dW2 = rho * Z1 + sqrt_1r2 * Z2;

                    double v_pos = std::max(v, 0.0);
                    double sqrt_v = std::sqrt(v_pos);

                    // Variance: Euler-Maruyama with absorption
                    double v_new = v + params.kappa * (params.theta - v) * dt
                                     + params.sigma_v * sqrt_v * sqrt_dt * dW2;
                    v_new = std::max(v_new, 0.0);

                    // Price: exact log-price step (keeps S > 0)
                    S *= std::exp((params.mu - 0.5 * v_pos) * dt
                                 + sqrt_v * sqrt_dt * dW1);

                    v = v_new;
                    paths.price(i, s + 1)    = S;
                    paths.variance(i, s + 1) = v;
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // Heston Euler-Maruyama — count negative variance incidents
    // (diagnostic version, returns count via output parameter)
    // -----------------------------------------------------------------
    void simulate_heston_euler_count(const HestonParams& params,
                                     const SimConfig& config,
                                     PathData& paths,
                                     size_t& neg_var_count) {
        const double dt       = params.dt();
        const double sqrt_dt  = std::sqrt(dt);
        const double rho      = params.rho;
        const double sqrt_1r2 = std::sqrt(1.0 - rho * rho);
        const int nt = config.n_threads > 0 ? config.n_threads : omp_get_max_threads();

        for (size_t i = 0; i < config.n_paths; ++i) {
            paths.price(i, 0)    = params.S0;
            paths.variance(i, 0) = params.v0;
        }

        size_t total_neg = 0;

        #pragma omp parallel num_threads(nt) reduction(+:total_neg)
        {
            const auto tid = static_cast<unsigned long long>(omp_get_thread_num());
            std::mt19937_64 rng(config.seed + tid * 1000003ULL);
            std::normal_distribution<double> normal(0.0, 1.0);

            #pragma omp for schedule(static)
            for (size_t i = 0; i < config.n_paths; ++i) {
                double S = params.S0;
                double v = params.v0;

                for (size_t s = 0; s < config.n_steps; ++s) {
                    double Z1 = normal(rng);
                    double Z2 = normal(rng);
                    double dW1 = Z1;
                    double dW2 = rho * Z1 + sqrt_1r2 * Z2;

                    double v_pos = std::max(v, 0.0);
                    double sqrt_v = std::sqrt(v_pos);

                    double v_new = v + params.kappa * (params.theta - v) * dt
                                     + params.sigma_v * sqrt_v * sqrt_dt * dW2;
                    if (v_new < 0.0) ++total_neg;
                    v_new = std::max(v_new, 0.0);

                    S *= std::exp((params.mu - 0.5 * v_pos) * dt
                                 + sqrt_v * sqrt_dt * dW1);

                    v = v_new;
                    paths.price(i, s + 1)    = S;
                    paths.variance(i, s + 1) = v;
                }
            }
        }
        neg_var_count = total_neg;
    }

    // -----------------------------------------------------------------
    // Heston QE scheme (Andersen 2008)
    // -----------------------------------------------------------------
    void simulate_heston_qe(const HestonParams& params,
                            const SimConfig& config,
                            PathData& paths) {
        const double dt      = params.dt();
        const double exp_kdt = std::exp(-params.kappa * dt);
        const double rho     = params.rho;
        const double kappa   = params.kappa;
        const double theta   = params.theta;
        const double sigma_v = params.sigma_v;
        const double mu      = params.mu;
        const int nt = config.n_threads > 0 ? config.n_threads : omp_get_max_threads();

        // Andersen's log-price coefficients (gamma1 = gamma2 = 0.5, trapezoidal)
        constexpr double gamma1 = 0.5;
        constexpr double gamma2 = 0.5;
        const double K0 = -rho * kappa * theta * dt / sigma_v;
        const double K1 = gamma1 * dt * (kappa * rho / sigma_v - 0.5) - rho / sigma_v;
        const double K2 = gamma2 * dt * (kappa * rho / sigma_v - 0.5) + rho / sigma_v;
        const double K3 = gamma1 * dt * (1.0 - rho * rho);
        const double K4 = gamma2 * dt * (1.0 - rho * rho);

        constexpr double psi_threshold = 1.5;

        for (size_t i = 0; i < config.n_paths; ++i) {
            paths.price(i, 0)    = params.S0;
            paths.variance(i, 0) = params.v0;
        }

        #pragma omp parallel num_threads(nt)
        {
            const auto tid = static_cast<unsigned long long>(omp_get_thread_num());
            std::mt19937_64 rng(config.seed + tid * 1000003ULL);
            std::normal_distribution<double> normal(0.0, 1.0);
            std::uniform_real_distribution<double> uniform(0.0, 1.0);

            #pragma omp for schedule(static)
            for (size_t i = 0; i < config.n_paths; ++i) {
                double S    = params.S0;
                double lnS  = std::log(S);
                double v    = params.v0;

                for (size_t s = 0; s < config.n_steps; ++s) {
                    double U1 = uniform(rng);
                    double Z1 = normal(rng);

                    // --- Variance step (QE scheme) ---
                    // Conditional moments: E[v_{t+dt} | v_t] and Var[v_{t+dt} | v_t]
                    double m  = theta + (v - theta) * exp_kdt;
                    double s2 = v * (sigma_v * sigma_v * exp_kdt / kappa) * (1.0 - exp_kdt)
                              + (theta * sigma_v * sigma_v / (2.0 * kappa))
                                * (1.0 - exp_kdt) * (1.0 - exp_kdt);

                    double v_new;

                    // Protect against m <= 0 (shouldn't happen with valid params but be safe)
                    if (m <= 1e-15) {
                        v_new = 0.0;
                    } else {
                        double psi = s2 / (m * m);

                        if (psi <= psi_threshold) {
                            // Regime 1: Quadratic — non-central chi-squared approximation
                            double inv_psi = 1.0 / psi;
                            double b2 = 2.0 * inv_psi - 1.0
                                       + std::sqrt(2.0 * inv_psi)
                                         * std::sqrt(std::max(2.0 * inv_psi - 1.0, 0.0));
                            double a  = m / (1.0 + b2);
                            double b  = std::sqrt(b2);
                            // Use inverse normal CDF of U1
                            double Zv = inv_normal_cdf(U1);
                            v_new = a * (b + Zv) * (b + Zv);
                        } else {
                            // Regime 2: Exponential — mass at zero + exponential tail
                            double p    = (psi - 1.0) / (psi + 1.0);
                            double beta = (1.0 - p) / m;
                            if (U1 <= p) {
                                v_new = 0.0;
                            } else {
                                v_new = std::log((1.0 - p) / (1.0 - U1)) / beta;
                            }
                        }
                    }

                    // --- Price step (Andersen log-price formula) ---
                    double vol_term = K3 * v + K4 * v_new;
                    vol_term = std::max(vol_term, 0.0);

                    lnS += mu * dt + K0 + K1 * v + K2 * v_new
                         + std::sqrt(vol_term) * Z1;
                    S = std::exp(lnS);

                    v = v_new;
                    paths.price(i, s + 1)    = S;
                    paths.variance(i, s + 1) = v;
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // Rough Heston via Markovian factor approximation
    // v(t) = v0 + Σ Y_i(t),  dY_i = [-λ_i Y_i + w_i κ(θ-v)] dt + w_i σ_v √v dW2
    // -----------------------------------------------------------------
    void simulate_rough_heston(const RoughHestonParams& params,
                               const KernelApproximation& kernel,
                               const SimConfig& config,
                               PathData& paths) {
        const int    nf       = kernel.n_factors;
        const double dt       = params.dt();
        const double sqrt_dt  = std::sqrt(dt);
        const double rho      = params.rho;
        const double sqrt_1r2 = std::sqrt(1.0 - rho * rho);
        const double kappa    = params.kappa;
        const double theta    = params.theta;
        const double sigma_v  = params.sigma_v;
        const double mu       = params.mu;
        const double v0       = params.v0;
        const int nt = config.n_threads > 0 ? config.n_threads : omp_get_max_threads();

        // Precompute per-factor exponential decay
        std::vector<double> exp_neg_ldt(nf);
        for (int f = 0; f < nf; ++f)
            exp_neg_ldt[f] = std::exp(-kernel.rates[f] * dt);

        for (size_t i = 0; i < config.n_paths; ++i) {
            paths.price(i, 0)    = params.S0;
            paths.variance(i, 0) = v0;
        }

        #pragma omp parallel num_threads(nt)
        {
            const auto tid = static_cast<unsigned long long>(omp_get_thread_num());
            std::mt19937_64 rng(config.seed + tid * 1000003ULL);
            std::normal_distribution<double> normal(0.0, 1.0);

            // Per-thread factor storage
            std::vector<double> Y(nf, 0.0);

            #pragma omp for schedule(static)
            for (size_t i = 0; i < config.n_paths; ++i) {
                double S = params.S0;
                double v = v0;

                // Reset factors to zero for each path
                for (int f = 0; f < nf; ++f) Y[f] = 0.0;

                for (size_t s = 0; s < config.n_steps; ++s) {
                    double Z1 = normal(rng);
                    double Z2 = normal(rng);
                    double dW1 = Z1;
                    double dW2 = rho * Z1 + sqrt_1r2 * Z2;

                    double v_pos  = std::max(v, 0.0);
                    double sqrt_v = std::sqrt(v_pos);

                    // Update each Markovian factor (semi-exact: exact for decay)
                    double mean_rev_drive = kappa * (theta - v);
                    double diff_drive     = sigma_v * sqrt_v * sqrt_dt * dW2;

                    double sum_Y = 0.0;
                    for (int f = 0; f < nf; ++f) {
                        double w  = kernel.weights[f];
                        double el = exp_neg_ldt[f];
                        double lam = kernel.rates[f];

                        // Exact decay + Euler for drift & diffusion
                        Y[f] = Y[f] * el
                             + w * mean_rev_drive * (1.0 - el) / lam
                             + w * diff_drive;
                        sum_Y += Y[f];
                    }

                    // Reconstruct variance
                    v = v0 + sum_Y;
                    v = std::max(v, 0.0);

                    // Price step (log-price for stability)
                    S *= std::exp((mu - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * dW1);

                    paths.price(i, s + 1)    = S;
                    paths.variance(i, s + 1) = v;
                }
            }
        }
    }

    // =================================================================
    // Sobol QMC simulation methods
    // =================================================================

    // -----------------------------------------------------------------
    // GBM — Sobol QMC (pre-generate Sobol normals, then parallel sim)
    // -----------------------------------------------------------------
    void simulate_gbm_sobol(const GBMParams& params, const SimConfig& config,
                            PathData& paths, SobolGenerator& sobol) {
        const size_t n  = config.n_paths;
        const size_t ns = config.n_steps;
        const double dt         = params.dt();
        const double drift_term = (params.mu - 0.5 * params.sigma * params.sigma) * dt;
        const double vol_term   = params.sigma * std::sqrt(dt);
        const int nt = config.n_threads > 0 ? config.n_threads : omp_get_max_threads();

        // Pre-generate Sobol normal matrix: [n_paths][n_steps]
        auto Z = sobol.generate_normal_matrix(n);

        for (size_t i = 0; i < n; ++i) paths.price(i, 0) = params.S0;

        #pragma omp parallel for num_threads(nt) schedule(static)
        for (size_t i = 0; i < n; ++i) {
            double S = params.S0;
            for (size_t s = 0; s < ns; ++s) {
                S *= std::exp(drift_term + vol_term * Z[i * ns + s]);
                paths.price(i, s + 1) = S;
            }
        }
    }

    // -----------------------------------------------------------------
    // Heston QE — Sobol QMC (2 * n_steps dimensions)
    // -----------------------------------------------------------------
    void simulate_heston_sobol(const HestonParams& params, const SimConfig& config,
                               PathData& paths, SobolGenerator& sobol) {
        const size_t n  = config.n_paths;
        const size_t ns = config.n_steps;
        const double dt      = params.dt();
        const double exp_kdt = std::exp(-params.kappa * dt);
        const double rho     = params.rho;
        const double kappa   = params.kappa;
        const double theta   = params.theta;
        const double sigma_v = params.sigma_v;
        const double mu      = params.mu;
        const int nt = config.n_threads > 0 ? config.n_threads : omp_get_max_threads();
        const int n_dims = static_cast<int>(2 * ns);

        constexpr double gamma1 = 0.5, gamma2 = 0.5;
        const double K0 = -rho * kappa * theta * dt / sigma_v;
        const double K1 = gamma1 * dt * (kappa * rho / sigma_v - 0.5) - rho / sigma_v;
        const double K2 = gamma2 * dt * (kappa * rho / sigma_v - 0.5) + rho / sigma_v;
        const double K3 = gamma1 * dt * (1.0 - rho * rho);
        const double K4 = gamma2 * dt * (1.0 - rho * rho);
        constexpr double psi_threshold = 1.5;

        // Pre-generate Sobol normals: dims 0..ns-1 = Z1 (price), ns..2*ns-1 = Zunif (variance QE)
        // For QE we need a uniform for variance, so we generate uniform Sobol and only transform
        // the price dimensions to normal. But it's simpler to generate all as uniform,
        // then transform price dims to normal.
        std::vector<double> U(n * n_dims);
        {
            std::vector<double> point(n_dims);
            sobol.reset();
            for (size_t i = 0; i < n; ++i) {
                sobol.next_point(point.data());
                for (int d = 0; d < n_dims; ++d)
                    U[i * n_dims + d] = point[d];
            }
        }

        for (size_t i = 0; i < n; ++i) {
            paths.price(i, 0)    = params.S0;
            paths.variance(i, 0) = params.v0;
        }

        #pragma omp parallel for num_threads(nt) schedule(static)
        for (size_t i = 0; i < n; ++i) {
            double S    = params.S0;
            double lnS  = std::log(S);
            double v    = params.v0;

            for (size_t s = 0; s < ns; ++s) {
                // Z1 for price (transform uniform to normal)
                double u_price = U[i * n_dims + s];
                if (u_price < 1e-10) u_price = 1e-10;
                if (u_price > 1.0 - 1e-10) u_price = 1.0 - 1e-10;
                double Z1 = inverse_normal_cdf(u_price);

                // U1 for variance QE (keep as uniform)
                double U1 = U[i * n_dims + ns + s];
                if (U1 < 1e-10) U1 = 1e-10;
                if (U1 > 1.0 - 1e-10) U1 = 1.0 - 1e-10;

                // --- Variance step (QE scheme) ---
                double m  = theta + (v - theta) * exp_kdt;
                double s2 = v * (sigma_v * sigma_v * exp_kdt / kappa) * (1.0 - exp_kdt)
                          + (theta * sigma_v * sigma_v / (2.0 * kappa))
                            * (1.0 - exp_kdt) * (1.0 - exp_kdt);
                double v_new;
                if (m <= 1e-15) {
                    v_new = 0.0;
                } else {
                    double psi = s2 / (m * m);
                    if (psi <= psi_threshold) {
                        double inv_psi = 1.0 / psi;
                        double b2 = 2.0 * inv_psi - 1.0
                                   + std::sqrt(2.0 * inv_psi)
                                     * std::sqrt(std::max(2.0 * inv_psi - 1.0, 0.0));
                        double a  = m / (1.0 + b2);
                        double b  = std::sqrt(b2);
                        double Zv = inverse_normal_cdf(U1);
                        v_new = a * (b + Zv) * (b + Zv);
                    } else {
                        double p    = (psi - 1.0) / (psi + 1.0);
                        double beta = (1.0 - p) / m;
                        if (U1 <= p) v_new = 0.0;
                        else         v_new = std::log((1.0 - p) / (1.0 - U1)) / beta;
                    }
                }

                // --- Price step ---
                double vol_term = K3 * v + K4 * v_new;
                vol_term = std::max(vol_term, 0.0);
                lnS += mu * dt + K0 + K1 * v + K2 * v_new + std::sqrt(vol_term) * Z1;
                S = std::exp(lnS);

                v = v_new;
                paths.price(i, s + 1)    = S;
                paths.variance(i, s + 1) = v;
            }
        }
    }

    // =================================================================
    // Control Variate simulation methods
    // =================================================================

    // -----------------------------------------------------------------
    // Heston QE + GBM with shared random numbers for CV
    // -----------------------------------------------------------------
    CVResult simulate_heston_with_cv(const HestonParams& hp, const GBMParams& gp,
                                     const SimConfig& config) {
        const size_t n  = config.n_paths;
        const size_t ns = config.n_steps;
        const double dt      = hp.dt();
        const double exp_kdt = std::exp(-hp.kappa * dt);
        const double rho     = hp.rho;
        const double kappa   = hp.kappa;
        const double theta   = hp.theta;
        const double sigma_v = hp.sigma_v;
        const double mu_h    = hp.mu;
        const int nt = config.n_threads > 0 ? config.n_threads : omp_get_max_threads();

        // GBM precomputed
        const double gbm_dt    = gp.dt();
        const double gbm_drift = (gp.mu - 0.5 * gp.sigma * gp.sigma) * gbm_dt;
        const double gbm_vol   = gp.sigma * std::sqrt(gbm_dt);

        // QE coefficients
        constexpr double gamma1 = 0.5, gamma2 = 0.5;
        const double K0 = -rho * kappa * theta * dt / sigma_v;
        const double K1 = gamma1 * dt * (kappa * rho / sigma_v - 0.5) - rho / sigma_v;
        const double K2 = gamma2 * dt * (kappa * rho / sigma_v - 0.5) + rho / sigma_v;
        const double K3 = gamma1 * dt * (1.0 - rho * rho);
        const double K4 = gamma2 * dt * (1.0 - rho * rho);
        constexpr double psi_threshold = 1.5;

        std::vector<double> heston_terminal(n);
        std::vector<double> gbm_terminal(n);

        #pragma omp parallel num_threads(nt)
        {
            const auto tid = static_cast<unsigned long long>(omp_get_thread_num());
            std::mt19937_64 rng(config.seed + tid * 1000003ULL);
            std::normal_distribution<double> normal(0.0, 1.0);
            std::uniform_real_distribution<double> uniform(0.0, 1.0);

            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                double S_h  = hp.S0, lnS_h = std::log(hp.S0), v = hp.v0;
                double S_g  = gp.S0;

                for (size_t s = 0; s < ns; ++s) {
                    double U1 = uniform(rng);
                    double Z1 = normal(rng);  // Shared between Heston price and GBM

                    // --- GBM with shared Z1 ---
                    S_g *= std::exp(gbm_drift + gbm_vol * Z1);

                    // --- Heston QE variance step ---
                    double m  = theta + (v - theta) * exp_kdt;
                    double s2 = v * (sigma_v * sigma_v * exp_kdt / kappa) * (1.0 - exp_kdt)
                              + (theta * sigma_v * sigma_v / (2.0 * kappa))
                                * (1.0 - exp_kdt) * (1.0 - exp_kdt);
                    double v_new;
                    if (m <= 1e-15) {
                        v_new = 0.0;
                    } else {
                        double psi = s2 / (m * m);
                        if (psi <= psi_threshold) {
                            double inv_psi = 1.0 / psi;
                            double b2 = 2.0 * inv_psi - 1.0
                                       + std::sqrt(2.0 * inv_psi)
                                         * std::sqrt(std::max(2.0 * inv_psi - 1.0, 0.0));
                            double a  = m / (1.0 + b2);
                            double b  = std::sqrt(b2);
                            double Zv = inv_normal_cdf(U1);
                            v_new = a * (b + Zv) * (b + Zv);
                        } else {
                            double p    = (psi - 1.0) / (psi + 1.0);
                            double beta = (1.0 - p) / m;
                            if (U1 <= p) v_new = 0.0;
                            else         v_new = std::log((1.0 - p) / (1.0 - U1)) / beta;
                        }
                    }

                    double vol_term = K3 * v + K4 * v_new;
                    vol_term = std::max(vol_term, 0.0);
                    lnS_h += mu_h * dt + K0 + K1 * v + K2 * v_new + std::sqrt(vol_term) * Z1;
                    S_h = std::exp(lnS_h);
                    v = v_new;
                }
                heston_terminal[i] = S_h;
                gbm_terminal[i]    = S_g;
            }
        }

        double E_gbm = gp.S0 * std::exp(gp.mu * gp.T);
        return apply_control_variate(heston_terminal.data(), gbm_terminal.data(), n, E_gbm);
    }

    // =================================================================
    // Importance Sampling simulation methods
    // =================================================================

    // -----------------------------------------------------------------
    // GBM with importance sampling (drift shift on Z)
    // mu_is: total shift on terminal distribution (e.g., -2.33 for 1% tail)
    //        internally scaled to per-step: theta = mu_is / sqrt(n_steps)
    // -----------------------------------------------------------------
    ISResult simulate_gbm_is(const GBMParams& params, const SimConfig& config,
                             double mu_is) {
        const size_t n  = config.n_paths;
        const size_t ns = config.n_steps;
        const double dt         = params.dt();
        const double drift_term = (params.mu - 0.5 * params.sigma * params.sigma) * dt;
        const double vol_term   = params.sigma * std::sqrt(dt);
        const int nt = config.n_threads > 0 ? config.n_threads : omp_get_max_threads();

        // Scale: total shift distributed across steps
        const double theta = mu_is / std::sqrt(static_cast<double>(ns));

        ISResult result;
        result.terminal_prices.resize(n);
        result.weights.resize(n);

        #pragma omp parallel num_threads(nt)
        {
            const auto tid = static_cast<unsigned long long>(omp_get_thread_num());
            std::mt19937_64 rng(config.seed + tid * 1000003ULL);
            std::normal_distribution<double> normal(0.0, 1.0);

            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                double S = params.S0;
                double log_w = 0.0;

                for (size_t s = 0; s < ns; ++s) {
                    double Z = normal(rng);
                    double Z_shifted = Z + theta;

                    S *= std::exp(drift_term + vol_term * Z_shifted);

                    // Log-likelihood ratio per step: -theta * Z - 0.5 * theta^2
                    log_w += -theta * Z - 0.5 * theta * theta;
                }
                result.terminal_prices[i] = S;
                result.weights[i] = std::exp(log_w);
            }
        }

        result.ess = compute_ess(result.weights.data(), n);
        result.ess_ratio = result.ess / static_cast<double>(n);
        return result;
    }

    // -----------------------------------------------------------------
    // Heston QE with importance sampling (drift shift on Z1 only)
    // mu_is: total shift (e.g., -2.33 for 1% tail), scaled to per-step internally
    // -----------------------------------------------------------------
    ISResult simulate_heston_is(const HestonParams& params, const SimConfig& config,
                                double mu_is) {
        const size_t n  = config.n_paths;
        const size_t ns = config.n_steps;
        const double dt      = params.dt();
        const double exp_kdt = std::exp(-params.kappa * dt);
        const double rho     = params.rho;
        const double kappa   = params.kappa;
        const double theta   = params.theta;
        const double sigma_v = params.sigma_v;
        const double mu      = params.mu;
        const int nt = config.n_threads > 0 ? config.n_threads : omp_get_max_threads();

        // Scale total shift to per-step
        const double theta_is = mu_is / std::sqrt(static_cast<double>(ns));

        constexpr double gamma1 = 0.5, gamma2 = 0.5;
        const double K0 = -rho * kappa * theta * dt / sigma_v;
        const double K1 = gamma1 * dt * (kappa * rho / sigma_v - 0.5) - rho / sigma_v;
        const double K2 = gamma2 * dt * (kappa * rho / sigma_v - 0.5) + rho / sigma_v;
        const double K3 = gamma1 * dt * (1.0 - rho * rho);
        const double K4 = gamma2 * dt * (1.0 - rho * rho);
        constexpr double psi_threshold = 1.5;

        ISResult result;
        result.terminal_prices.resize(n);
        result.weights.resize(n);

        #pragma omp parallel num_threads(nt)
        {
            const auto tid = static_cast<unsigned long long>(omp_get_thread_num());
            std::mt19937_64 rng(config.seed + tid * 1000003ULL);
            std::normal_distribution<double> normal(0.0, 1.0);
            std::uniform_real_distribution<double> uniform(0.0, 1.0);

            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                double S    = params.S0;
                double lnS  = std::log(S);
                double v    = params.v0;
                double log_w = 0.0;

                for (size_t s = 0; s < ns; ++s) {
                    double U1 = uniform(rng);
                    double Z1_orig = normal(rng);
                    double Z1 = Z1_orig + theta_is;  // Shifted

                    // Log-likelihood for this step
                    log_w += -theta_is * Z1_orig - 0.5 * theta_is * theta_is;

                    // --- Variance step (QE) ---
                    double m_val = theta + (v - theta) * exp_kdt;
                    double s2 = v * (sigma_v * sigma_v * exp_kdt / kappa) * (1.0 - exp_kdt)
                              + (theta * sigma_v * sigma_v / (2.0 * kappa))
                                * (1.0 - exp_kdt) * (1.0 - exp_kdt);
                    double v_new;
                    if (m_val <= 1e-15) {
                        v_new = 0.0;
                    } else {
                        double psi = s2 / (m_val * m_val);
                        if (psi <= psi_threshold) {
                            double inv_psi = 1.0 / psi;
                            double b2 = 2.0 * inv_psi - 1.0
                                       + std::sqrt(2.0 * inv_psi)
                                         * std::sqrt(std::max(2.0 * inv_psi - 1.0, 0.0));
                            double a  = m_val / (1.0 + b2);
                            double b  = std::sqrt(b2);
                            double Zv = inv_normal_cdf(U1);
                            v_new = a * (b + Zv) * (b + Zv);
                        } else {
                            double p    = (psi - 1.0) / (psi + 1.0);
                            double beta = (1.0 - p) / m_val;
                            if (U1 <= p) v_new = 0.0;
                            else         v_new = std::log((1.0 - p) / (1.0 - U1)) / beta;
                        }
                    }

                    double vol_term = K3 * v + K4 * v_new;
                    vol_term = std::max(vol_term, 0.0);
                    lnS += mu * dt + K0 + K1 * v + K2 * v_new + std::sqrt(vol_term) * Z1;
                    S = std::exp(lnS);
                    v = v_new;
                }
                result.terminal_prices[i] = S;
                result.weights[i] = std::exp(log_w);
            }
        }

        result.ess = compute_ess(result.weights.data(), n);
        result.ess_ratio = result.ess / static_cast<double>(n);
        return result;
    }

    // -----------------------------------------------------------------
    // Generic Euler-Maruyama for 2D SDE (price + variance)
    // -----------------------------------------------------------------
    template <typename Model>
    void simulate_euler_2d(const Model& model, double S0, double v0,
                           double T, const SimConfig& config,
                           PathData& paths) {
        const double dt         = T / static_cast<double>(config.n_steps);
        const double sqrt_dt    = std::sqrt(dt);
        const double rho        = model.correlation();
        const double sqrt_1_r2  = std::sqrt(1.0 - rho * rho);
        const int nt = config.n_threads > 0 ? config.n_threads : omp_get_max_threads();

        for (size_t i = 0; i < config.n_paths; ++i) {
            paths.price(i, 0)    = S0;
            paths.variance(i, 0) = v0;
        }

        #pragma omp parallel num_threads(nt)
        {
            const auto tid = static_cast<unsigned long long>(omp_get_thread_num());
            std::mt19937_64 rng(config.seed + tid * 1000003ULL);
            std::normal_distribution<double> normal(0.0, 1.0);

            #pragma omp for schedule(static)
            for (size_t i = 0; i < config.n_paths; ++i) {
                double S = S0;
                double v = v0;
                for (size_t s = 0; s < config.n_steps; ++s) {
                    double Z1 = normal(rng);
                    double Z2 = normal(rng);
                    double dW1 = Z1;
                    double dW2 = rho * Z1 + sqrt_1_r2 * Z2;
                    double t   = static_cast<double>(s) * dt;

                    S += model.drift(S, v, t) * dt
                       + model.diffusion(S, v, t) * sqrt_dt * dW1;

                    v += model.variance_drift(S, v, t) * dt
                       + model.variance_diffusion(S, v, t) * sqrt_dt * dW2;
                    v = std::max(v, 0.0);

                    paths.price(i, s + 1)    = S;
                    paths.variance(i, s + 1) = v;
                }
            }
        }
    }

private:
    // Rational approximation of the inverse normal CDF (Beasley-Springer-Moro)
    static double inv_normal_cdf(double u) {
        // Clamp to avoid infinities
        if (u <= 0.0) u = 1e-15;
        if (u >= 1.0) u = 1.0 - 1e-15;

        static constexpr double a[] = {
            -3.969683028665376e+01,  2.209460984245205e+02,
            -2.759285104469687e+02,  1.383577518672690e+02,
            -3.066479806614716e+01,  2.506628277459239e+00
        };
        static constexpr double b[] = {
            -5.447609879822406e+01,  1.615858368580409e+02,
            -1.556989798598866e+02,  6.680131188771972e+01,
            -1.328068155288572e+01
        };
        static constexpr double c[] = {
            -7.784894002430293e-03, -3.223964580411365e-01,
            -2.400758277161838e+00, -2.549732539343734e+00,
             4.374664141464968e+00,  2.938163982698783e+00
        };
        static constexpr double d[] = {
             7.784695709041462e-03,  3.224671290700398e-01,
             2.445134137142996e+00,  3.754408661907416e+00
        };

        constexpr double p_low  = 0.02425;
        constexpr double p_high = 1.0 - p_low;

        double q, r;

        if (u < p_low) {
            // Lower tail
            q = std::sqrt(-2.0 * std::log(u));
            return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                    ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
        } else if (u <= p_high) {
            // Central region
            q = u - 0.5;
            r = q * q;
            return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
                   (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
        } else {
            // Upper tail
            q = std::sqrt(-2.0 * std::log(1.0 - u));
            return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                     ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
        }
    }
};

} // namespace qre
