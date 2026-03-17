/**
 * @file pybind_module.cpp
 * @brief pybind11 Python bindings for the quant risk engine.
 *
 * Exposes the C++ simulation engine, models, and risk metrics to Python.
 * Build requires pybind11 to be available.
 */

#ifdef __has_include
#if __has_include(<pybind11/pybind11.h>)
#define HAS_PYBIND11
#endif
#endif

#ifdef HAS_PYBIND11

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "models/gbm.hpp"
#include "engine/sim_engine.hpp"
#include "engine/cpu_engine.hpp"
#include "risk/var.hpp"
#include "risk/expected_shortfall.hpp"

namespace py = pybind11;

PYBIND11_MODULE(qre_python, m) {
    m.doc() = "Quant Risk Engine — C++20/CUDA Monte Carlo simulation with Python bindings";

    // GBM model
    py::class_<qre::models::GBM>(m, "GBM")
        .def(py::init<double, double>(), py::arg("mu"), py::arg("sigma"))
        .def("mu", &qre::models::GBM::mu)
        .def("sigma", &qre::models::GBM::sigma);

    // SimConfig
    py::class_<qre::engine::SimConfig>(m, "SimConfig")
        .def(py::init<>())
        .def_readwrite("n_paths", &qre::engine::SimConfig::n_paths)
        .def_readwrite("n_steps", &qre::engine::SimConfig::n_steps)
        .def_readwrite("dt", &qre::engine::SimConfig::dt)
        .def_readwrite("s0", &qre::engine::SimConfig::s0)
        .def_readwrite("seed", &qre::engine::SimConfig::seed);

    // SimResult
    py::class_<qre::engine::SimResult>(m, "SimResult")
        .def_readonly("terminal_prices", &qre::engine::SimResult::terminal_prices)
        .def_readonly("elapsed_ms", &qre::engine::SimResult::elapsed_ms);

    // CPU SimEngine for GBM
    using GBMEngine = qre::engine::SimEngine<qre::engine::Backend::CPU, qre::models::GBM>;
    py::class_<GBMEngine>(m, "GBMCPUEngine")
        .def(py::init<const qre::models::GBM&, const qre::engine::SimConfig&>())
        .def("run", &GBMEngine::run);

    // Risk metrics
    m.def("compute_var", &qre::risk::VaR::compute, py::arg("pnl"), py::arg("confidence"));
    m.def("compute_es", &qre::risk::ExpectedShortfall::compute, py::arg("pnl"), py::arg("confidence"));
}

#endif // HAS_PYBIND11
