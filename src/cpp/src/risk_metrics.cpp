/**
 * @file risk_metrics.cpp
 * @brief Compiled risk metric utilities (non-template code).
 *
 * The core compute_risk_metrics is header-only (inline), but this TU
 * exists so the quant_engine library target has at least one .cpp file.
 */

#include "risk/var.hpp"

// Explicit instantiation anchor — keeps the linker happy when the library
// is built as a static lib rather than header-only.
namespace qre {
namespace detail {
    volatile int risk_metrics_anchor = 0;
}
}
