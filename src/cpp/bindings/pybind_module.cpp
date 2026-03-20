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
#include "models/heston.hpp"
#include "models/rough_heston.hpp"
#include "engine/cpu_engine.hpp"
#include "risk/var.hpp"

namespace py = pybind11;

// ---------- Helpers: SoA → numpy ----------

static py::array_t<double> pathdata_to_numpy(const qre::PathData& paths,
                                              size_t n, size_t steps) {
    size_t cols = steps + 1;
    auto result = py::array_t<double>({static_cast<py::ssize_t>(n),
                                        static_cast<py::ssize_t>(cols)});
    auto buf = result.mutable_unchecked<2>();
    for (size_t step = 0; step < cols; ++step)
        for (size_t path = 0; path < n; ++path)
            buf(path, step) = paths.price(path, step);
    return result;
}

static py::array_t<double> variance_to_numpy(const qre::PathData& paths,
                                              size_t n, size_t steps) {
    size_t cols = steps + 1;
    auto result = py::array_t<double>({static_cast<py::ssize_t>(n),
                                        static_cast<py::ssize_t>(cols)});
    auto buf = result.mutable_unchecked<2>();
    for (size_t step = 0; step < cols; ++step)
        for (size_t path = 0; path < n; ++path)
            buf(path, step) = paths.variance(path, step);
    return result;
}

static py::array_t<double> vec_to_numpy(const std::vector<double>& v) {
    auto result = py::array_t<double>(static_cast<py::ssize_t>(v.size()));
    auto buf = result.mutable_unchecked<1>();
    for (size_t i = 0; i < v.size(); ++i) buf(i) = v[i];
    return result;
}

// ---------- GBM ----------

static py::array_t<double> simulate_gbm(const qre::GBMParams& params,
                                         const qre::SimConfig& config) {
    auto paths = std::make_shared<qre::PathData>(config.n_paths, config.n_steps);
    qre::CPUEngine eng;
    eng.simulate_gbm(params, config, *paths);
    return py::array_t<double>(
        static_cast<py::ssize_t>(config.n_paths),
        paths->terminal_prices(),
        py::capsule(new std::shared_ptr<qre::PathData>(paths),
                    [](void* p) { delete static_cast<std::shared_ptr<qre::PathData>*>(p); })
    );
}

static py::array_t<double> simulate_gbm_paths(const qre::GBMParams& params,
                                                const qre::SimConfig& config) {
    auto paths = std::make_shared<qre::PathData>(config.n_paths, config.n_steps);
    qre::CPUEngine eng;
    eng.simulate_gbm(params, config, *paths);
    return pathdata_to_numpy(*paths, config.n_paths, config.n_steps);
}

// ---------- GBM Sobol ----------

static py::array_t<double> simulate_gbm_sobol(const qre::GBMParams& params,
                                                const qre::SimConfig& config,
                                                uint64_t scramble_seed) {
    auto paths = std::make_shared<qre::PathData>(config.n_paths, config.n_steps);
    qre::SobolGenerator sobol(static_cast<int>(config.n_steps), scramble_seed);
    qre::CPUEngine eng;
    eng.simulate_gbm_sobol(params, config, *paths, sobol);
    return py::array_t<double>(
        static_cast<py::ssize_t>(config.n_paths),
        paths->terminal_prices(),
        py::capsule(new std::shared_ptr<qre::PathData>(paths),
                    [](void* p) { delete static_cast<std::shared_ptr<qre::PathData>*>(p); })
    );
}

// ---------- Heston ----------

static py::array_t<double> simulate_heston(const qre::HestonParams& params,
                                            const qre::SimConfig& config,
                                            bool use_qe) {
    auto paths = std::make_shared<qre::PathData>(config.n_paths, config.n_steps, true);
    qre::CPUEngine eng;
    if (use_qe) eng.simulate_heston_qe(params, config, *paths);
    else        eng.simulate_heston_euler(params, config, *paths);
    return py::array_t<double>(
        static_cast<py::ssize_t>(config.n_paths),
        paths->terminal_prices(),
        py::capsule(new std::shared_ptr<qre::PathData>(paths),
                    [](void* p) { delete static_cast<std::shared_ptr<qre::PathData>*>(p); })
    );
}

static py::tuple simulate_heston_full(const qre::HestonParams& params,
                                       const qre::SimConfig& config,
                                       bool use_qe) {
    qre::PathData paths(config.n_paths, config.n_steps, true);
    qre::CPUEngine eng;
    if (use_qe) eng.simulate_heston_qe(params, config, paths);
    else        eng.simulate_heston_euler(params, config, paths);
    return py::make_tuple(pathdata_to_numpy(paths, config.n_paths, config.n_steps),
                          variance_to_numpy(paths, config.n_paths, config.n_steps));
}

// ---------- Heston Sobol ----------

static py::array_t<double> simulate_heston_sobol(const qre::HestonParams& params,
                                                   const qre::SimConfig& config,
                                                   uint64_t scramble_seed) {
    auto paths = std::make_shared<qre::PathData>(config.n_paths, config.n_steps, true);
    qre::SobolGenerator sobol(static_cast<int>(2 * config.n_steps), scramble_seed);
    qre::CPUEngine eng;
    eng.simulate_heston_sobol(params, config, *paths, sobol);
    return py::array_t<double>(
        static_cast<py::ssize_t>(config.n_paths),
        paths->terminal_prices(),
        py::capsule(new std::shared_ptr<qre::PathData>(paths),
                    [](void* p) { delete static_cast<std::shared_ptr<qre::PathData>*>(p); })
    );
}

// ---------- Heston with Control Variate ----------

static py::dict simulate_heston_with_cv(const qre::HestonParams& hp,
                                         const qre::GBMParams& gp,
                                         const qre::SimConfig& config) {
    qre::CPUEngine eng;
    auto cv = eng.simulate_heston_with_cv(hp, gp, config);
    py::dict result;
    result["controlled_prices"] = vec_to_numpy(cv.controlled_prices);
    result["correlation"] = cv.correlation;
    result["variance_reduction"] = cv.variance_reduction;
    result["beta"] = cv.beta;
    return result;
}

// ---------- GBM with Importance Sampling ----------

static py::dict simulate_gbm_is(const qre::GBMParams& params,
                                  const qre::SimConfig& config,
                                  double mu_is) {
    qre::CPUEngine eng;
    auto is = eng.simulate_gbm_is(params, config, mu_is);
    py::dict result;
    result["terminal_prices"] = vec_to_numpy(is.terminal_prices);
    result["weights"] = vec_to_numpy(is.weights);
    result["ess"] = is.ess;
    result["ess_ratio"] = is.ess_ratio;
    return result;
}

// ---------- Heston with Importance Sampling ----------

static py::dict simulate_heston_is(const qre::HestonParams& params,
                                    const qre::SimConfig& config,
                                    double mu_is) {
    qre::CPUEngine eng;
    auto is = eng.simulate_heston_is(params, config, mu_is);
    py::dict result;
    result["terminal_prices"] = vec_to_numpy(is.terminal_prices);
    result["weights"] = vec_to_numpy(is.weights);
    result["ess"] = is.ess;
    result["ess_ratio"] = is.ess_ratio;
    return result;
}

// ---------- Rough Heston ----------

static py::array_t<double> simulate_rough_heston(const qre::RoughHestonParams& params,
                                                   const qre::SimConfig& config,
                                                   int n_factors) {
    qre::RoughHeston model(params, n_factors);
    auto paths = std::make_shared<qre::PathData>(config.n_paths, config.n_steps, true);
    qre::CPUEngine eng;
    eng.simulate_rough_heston(params, model.kernel, config, *paths);
    return py::array_t<double>(
        static_cast<py::ssize_t>(config.n_paths),
        paths->terminal_prices(),
        py::capsule(new std::shared_ptr<qre::PathData>(paths),
                    [](void* p) { delete static_cast<std::shared_ptr<qre::PathData>*>(p); })
    );
}

static py::tuple simulate_rough_heston_full(const qre::RoughHestonParams& params,
                                             const qre::SimConfig& config,
                                             int n_factors) {
    qre::RoughHeston model(params, n_factors);
    qre::PathData paths(config.n_paths, config.n_steps, true);
    qre::CPUEngine eng;
    eng.simulate_rough_heston(params, model.kernel, config, paths);
    return py::make_tuple(pathdata_to_numpy(paths, config.n_paths, config.n_steps),
                          variance_to_numpy(paths, config.n_paths, config.n_steps));
}

// ---------- Risk ----------

static qre::RiskMetrics compute_risk(py::array_t<double> terminal, double S0) {
    auto buf = terminal.unchecked<1>();
    return qre::compute_risk_metrics(buf.data(0), static_cast<size_t>(buf.shape(0)), S0);
}

// ---------- Module ----------

PYBIND11_MODULE(quant_engine_py, m) {
    m.doc() = "Quant Risk Engine — C++20 Monte Carlo simulation";

    py::class_<qre::GBMParams>(m, "GBMParams")
        .def(py::init<>())
        .def_readwrite("S0",      &qre::GBMParams::S0)
        .def_readwrite("mu",      &qre::GBMParams::mu)
        .def_readwrite("sigma",   &qre::GBMParams::sigma)
        .def_readwrite("T",       &qre::GBMParams::T)
        .def_readwrite("n_steps", &qre::GBMParams::n_steps);

    py::class_<qre::HestonParams>(m, "HestonParams")
        .def(py::init<>())
        .def_readwrite("S0",      &qre::HestonParams::S0)
        .def_readwrite("mu",      &qre::HestonParams::mu)
        .def_readwrite("T",       &qre::HestonParams::T)
        .def_readwrite("n_steps", &qre::HestonParams::n_steps)
        .def_readwrite("v0",      &qre::HestonParams::v0)
        .def_readwrite("kappa",   &qre::HestonParams::kappa)
        .def_readwrite("theta",   &qre::HestonParams::theta)
        .def_readwrite("sigma_v", &qre::HestonParams::sigma_v)
        .def_readwrite("rho",     &qre::HestonParams::rho);

    py::class_<qre::RoughHestonParams>(m, "RoughHestonParams")
        .def(py::init<>())
        .def_readwrite("S0",      &qre::RoughHestonParams::S0)
        .def_readwrite("mu",      &qre::RoughHestonParams::mu)
        .def_readwrite("T",       &qre::RoughHestonParams::T)
        .def_readwrite("n_steps", &qre::RoughHestonParams::n_steps)
        .def_readwrite("v0",      &qre::RoughHestonParams::v0)
        .def_readwrite("kappa",   &qre::RoughHestonParams::kappa)
        .def_readwrite("theta",   &qre::RoughHestonParams::theta)
        .def_readwrite("sigma_v", &qre::RoughHestonParams::sigma_v)
        .def_readwrite("rho",     &qre::RoughHestonParams::rho)
        .def_readwrite("H",       &qre::RoughHestonParams::H);

    py::class_<qre::SimConfig>(m, "SimConfig")
        .def(py::init<>())
        .def_readwrite("n_paths",    &qre::SimConfig::n_paths)
        .def_readwrite("n_steps",    &qre::SimConfig::n_steps)
        .def_readwrite("seed",       &qre::SimConfig::seed)
        .def_readwrite("n_threads",  &qre::SimConfig::n_threads)
        .def_readwrite("antithetic", &qre::SimConfig::antithetic);

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

    // Standard MC
    m.def("simulate_gbm", &simulate_gbm,
          py::arg("params"), py::arg("config"));
    m.def("simulate_gbm_paths", &simulate_gbm_paths,
          py::arg("params"), py::arg("config"));
    m.def("simulate_heston", &simulate_heston,
          py::arg("params"), py::arg("config"), py::arg("use_qe") = true);
    m.def("simulate_heston_full", &simulate_heston_full,
          py::arg("params"), py::arg("config"), py::arg("use_qe") = true);
    m.def("simulate_rough_heston", &simulate_rough_heston,
          py::arg("params"), py::arg("config"), py::arg("n_factors") = 8);
    m.def("simulate_rough_heston_full", &simulate_rough_heston_full,
          py::arg("params"), py::arg("config"), py::arg("n_factors") = 8);

    // QMC (Sobol)
    m.def("simulate_gbm_sobol", &simulate_gbm_sobol,
          py::arg("params"), py::arg("config"), py::arg("scramble_seed") = 0);
    m.def("simulate_heston_sobol", &simulate_heston_sobol,
          py::arg("params"), py::arg("config"), py::arg("scramble_seed") = 0);

    // Control Variate
    m.def("simulate_heston_with_cv", &simulate_heston_with_cv,
          py::arg("heston_params"), py::arg("gbm_params"), py::arg("config"));

    // Importance Sampling
    m.def("simulate_gbm_is", &simulate_gbm_is,
          py::arg("params"), py::arg("config"), py::arg("mu_is") = -2.0);
    m.def("simulate_heston_is", &simulate_heston_is,
          py::arg("params"), py::arg("config"), py::arg("mu_is") = -2.0);

    // Risk
    m.def("compute_risk", &compute_risk,
          py::arg("terminal_prices"), py::arg("S0"));
}
