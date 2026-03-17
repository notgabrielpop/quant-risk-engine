/**
 * @file pybind_module.cpp
 * @brief pybind11 Python bindings for the quant risk engine.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "core/path_data.hpp"
#include "core/sim_config.hpp"
#include "models/gbm.hpp"
#include "engine/cpu_engine.hpp"
#include "risk/var.hpp"

namespace py = pybind11;

// Return terminal prices as a numpy array (no copy — data owned by PathData capsule)
static py::array_t<double> simulate_gbm(const qre::GBMParams& params,
                                         const qre::SimConfig& config) {
    auto paths = std::make_shared<qre::PathData>(config.n_paths, config.n_steps);
    qre::CPUEngine eng;
    eng.simulate_gbm(params, config, *paths);

    // Create numpy array that shares ownership with the PathData
    return py::array_t<double>(
        static_cast<py::ssize_t>(config.n_paths),
        paths->terminal_prices(),
        py::capsule(new std::shared_ptr<qre::PathData>(paths),
                    [](void* p) { delete static_cast<std::shared_ptr<qre::PathData>*>(p); })
    );
}

// Return full paths as a (n_paths, n_steps+1) numpy array
static py::array_t<double> simulate_gbm_paths(const qre::GBMParams& params,
                                                const qre::SimConfig& config) {
    auto paths = std::make_shared<qre::PathData>(config.n_paths, config.n_steps);
    qre::CPUEngine eng;
    eng.simulate_gbm(params, config, *paths);

    // SoA layout is step-major: prices[step * n_paths + path]
    // We need to transpose to path-major for Python: (n_paths, n_steps+1)
    size_t n = config.n_paths;
    size_t s = config.n_steps + 1;
    auto result = py::array_t<double>({static_cast<py::ssize_t>(n),
                                        static_cast<py::ssize_t>(s)});
    auto buf = result.mutable_unchecked<2>();
    for (size_t step = 0; step < s; ++step)
        for (size_t path = 0; path < n; ++path)
            buf(path, step) = paths->price(path, step);

    return result;
}

static qre::RiskMetrics compute_risk(py::array_t<double> terminal, double S0) {
    auto buf = terminal.unchecked<1>();
    return qre::compute_risk_metrics(buf.data(0), static_cast<size_t>(buf.shape(0)), S0);
}

PYBIND11_MODULE(quant_engine_py, m) {
    m.doc() = "Quant Risk Engine — C++20 Monte Carlo simulation";

    // GBMParams
    py::class_<qre::GBMParams>(m, "GBMParams")
        .def(py::init<>())
        .def_readwrite("S0",      &qre::GBMParams::S0)
        .def_readwrite("mu",      &qre::GBMParams::mu)
        .def_readwrite("sigma",   &qre::GBMParams::sigma)
        .def_readwrite("T",       &qre::GBMParams::T)
        .def_readwrite("n_steps", &qre::GBMParams::n_steps);

    // SimConfig
    py::class_<qre::SimConfig>(m, "SimConfig")
        .def(py::init<>())
        .def_readwrite("n_paths",   &qre::SimConfig::n_paths)
        .def_readwrite("n_steps",   &qre::SimConfig::n_steps)
        .def_readwrite("seed",      &qre::SimConfig::seed)
        .def_readwrite("n_threads", &qre::SimConfig::n_threads)
        .def_readwrite("antithetic", &qre::SimConfig::antithetic);

    // RiskMetrics
    py::class_<qre::RiskMetrics>(m, "RiskMetrics")
        .def(py::init<>())
        .def_readonly("mean_price",   &qre::RiskMetrics::mean_price)
        .def_readonly("std_price",    &qre::RiskMetrics::std_price)
        .def_readonly("skewness",     &qre::RiskMetrics::skewness)
        .def_readonly("kurtosis",     &qre::RiskMetrics::kurtosis)
        .def_readonly("var_95",       &qre::RiskMetrics::var_95)
        .def_readonly("var_99",       &qre::RiskMetrics::var_99)
        .def_readonly("es_95",        &qre::RiskMetrics::es_95)
        .def_readonly("es_99",        &qre::RiskMetrics::es_99)
        .def_readonly("min_price",    &qre::RiskMetrics::min_price)
        .def_readonly("max_price",    &qre::RiskMetrics::max_price)
        .def_readonly("median_price", &qre::RiskMetrics::median_price);

    // Simulation functions
    m.def("simulate_gbm", &simulate_gbm,
          py::arg("params"), py::arg("config"),
          "Simulate GBM and return terminal prices as numpy array");

    m.def("simulate_gbm_paths", &simulate_gbm_paths,
          py::arg("params"), py::arg("config"),
          "Simulate GBM and return full paths as (n_paths, n_steps+1) array");

    m.def("compute_risk", &compute_risk,
          py::arg("terminal_prices"), py::arg("S0"),
          "Compute VaR, ES, and distribution metrics from terminal prices");
}
