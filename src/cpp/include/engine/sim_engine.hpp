#pragma once

/**
 * @file sim_engine.hpp
 * @brief Template-based simulation engine with backend dispatch.
 *
 * Provides a compile-time dispatch mechanism that selects between
 * CPU (SIMD+OpenMP) and CUDA backends via the Backend enum.
 * The primary specializations are in cpu_engine.hpp and cuda_engine.cuh.
 */

#include <cstddef>
#include <vector>

namespace qre::engine {

/// Computation backend selector
enum class Backend {
    CPU,   ///< SIMD + OpenMP parallel backend
    CUDA   ///< NVIDIA GPU backend
};

/// Simulation configuration
struct SimConfig {
    std::size_t n_paths  = 1'000'000;  ///< Number of Monte Carlo paths
    std::size_t n_steps  = 252;        ///< Time steps per path
    double      dt       = 1.0 / 252;  ///< Time step size (years)
    double      s0       = 100.0;      ///< Initial asset price
    unsigned    seed     = 42;         ///< RNG seed
};

/// Result of a simulation run
struct SimResult {
    std::vector<double> terminal_prices;  ///< Terminal values for each path
    std::vector<std::vector<double>> paths; ///< Full paths (optional, can be empty)
    double elapsed_ms = 0.0;              ///< Wall-clock time in milliseconds
};

/**
 * Primary template — intentionally left undefined.
 * Specializations in cpu_engine.hpp and cuda_engine.cuh provide the implementation.
 *
 * @tparam B     Backend to dispatch to
 * @tparam Model SDE model type (must satisfy the SDEModel CRTP interface)
 */
template <Backend B, typename Model>
class SimEngine {
public:
    explicit SimEngine(const Model& model, const SimConfig& config = {});

    /// Run the Monte Carlo simulation and return results.
    auto run() -> SimResult;

    /// Run and store full paths (higher memory usage).
    auto run_with_paths() -> SimResult;
};

} // namespace qre::engine
