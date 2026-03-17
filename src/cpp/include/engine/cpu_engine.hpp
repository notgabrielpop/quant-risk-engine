#pragma once

/**
 * @file cpu_engine.hpp
 * @brief CPU Monte Carlo engine: exact GBM, Euler-Maruyama 2D, OpenMP.
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
    // paths must hold 2 * half paths; first half normal, second antithetic
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
    // Euler-Maruyama for 2D SDE (price + variance), correlated BMs
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
};

} // namespace qre
