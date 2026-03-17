#pragma once

/**
 * @file cpu_engine.hpp
 * @brief CPU backend for SimEngine using SIMD intrinsics and OpenMP.
 *
 * Specializes SimEngine<Backend::CPU, Model> to run Monte Carlo paths
 * in parallel on the CPU. Uses OpenMP for thread-level parallelism.
 */

#include "sim_engine.hpp"
#include "../models/sde_model.hpp"

#include <chrono>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace qre::engine {

template <typename Model>
class SimEngine<Backend::CPU, Model> {
public:
    explicit SimEngine(const Model& model, const SimConfig& config = {})
        : model_{model}, config_{config} {}

    auto run() -> SimResult {
        auto start = std::chrono::high_resolution_clock::now();

        SimResult result;
        result.terminal_prices.resize(config_.n_paths);

        #pragma omp parallel
        {
            // Per-thread RNG for reproducibility
            #ifdef _OPENMP
            const unsigned thread_seed = config_.seed + static_cast<unsigned>(omp_get_thread_num());
            #else
            const unsigned thread_seed = config_.seed;
            #endif
            std::mt19937_64 rng(thread_seed);
            std::normal_distribution<double> normal(0.0, 1.0);

            #pragma omp for schedule(static)
            for (std::size_t i = 0; i < config_.n_paths; ++i) {
                double x = config_.s0;
                for (std::size_t step = 0; step < config_.n_steps; ++step) {
                    const double t  = step * config_.dt;
                    const double dW = normal(rng) * std::sqrt(config_.dt);
                    x = model_.euler_step(t, x, config_.dt, dW);
                }
                result.terminal_prices[i] = x;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        result.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        return result;
    }

    auto run_with_paths() -> SimResult {
        // TODO: implement full-path storage variant
        return run();
    }

private:
    Model model_;
    SimConfig config_;
};

} // namespace qre::engine
