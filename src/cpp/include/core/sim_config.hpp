#pragma once

/**
 * @file sim_config.hpp
 * @brief Simulation configuration: paths, threading, variance reduction.
 */

#include <cstddef>

namespace qre {

enum class Backend  { CPU, CUDA };
enum class RNGType  { MersenneTwister, Sobol };

struct SimConfig {
    size_t       n_paths   = 1'000'000;
    size_t       n_steps   = 252;
    Backend      backend   = Backend::CPU;
    RNGType      rng_type  = RNGType::MersenneTwister;
    unsigned int seed      = 42;
    int          n_threads = 0; // 0 = auto-detect

    bool antithetic      = false;
    bool control_variate = false;
};

} // namespace qre
