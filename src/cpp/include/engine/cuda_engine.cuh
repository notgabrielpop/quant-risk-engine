#pragma once

/**
 * @file cuda_engine.cuh
 * @brief CUDA GPU backend for SimEngine (placeholder).
 *
 * Will implement GPU-accelerated Monte Carlo simulation using
 * cuRAND for random number generation and custom CUDA kernels
 * for path evolution.
 *
 * TODO:
 *   - Implement cuRAND-based path generation kernel
 *   - Device memory management with RAII wrappers
 *   - Async stream-based execution for overlapping H2D/D2H transfers
 *   - Support for multi-GPU execution
 */

#ifdef QRE_CUDA_ENABLED

#include "sim_engine.hpp"

namespace qre::engine {

template <typename Model>
class SimEngine<Backend::CUDA, Model> {
public:
    explicit SimEngine(const Model& model, const SimConfig& config = {})
        : model_{model}, config_{config} {}

    auto run() -> SimResult {
        // TODO: launch CUDA kernel
        SimResult result;
        return result;
    }

    auto run_with_paths() -> SimResult {
        // TODO: implement with full path storage on device
        return run();
    }

private:
    Model model_;
    SimConfig config_;
};

} // namespace qre::engine

#endif // QRE_CUDA_ENABLED
