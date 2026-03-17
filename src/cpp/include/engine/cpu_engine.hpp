#pragma once

/**
 * @file cpu_engine.hpp
 * @brief CPU Monte Carlo engine: exact GBM, Heston Euler-Maruyama, Heston QE, OpenMP.
 */

#include "../core/path_data.hpp"
#include "../core/sim_config.hpp"
#include <cmath>
#include <random>

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
