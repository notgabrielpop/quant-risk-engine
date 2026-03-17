#pragma once

/**
 * @file sde_model.hpp
 * @brief CRTP base class for SDE models.
 *
 * Provides compile-time polymorphism with zero virtual dispatch overhead.
 * All models must implement:
 *   drift_impl(S, v, t)              -> double
 *   diffusion_impl(S, v, t)          -> double
 *   variance_drift_impl(S, v, t)     -> double  (if 2D)
 *   variance_diffusion_impl(S, v, t) -> double  (if 2D)
 *   correlation_impl()               -> double  (if 2D)
 *   static constexpr bool has_variance_process
 */

namespace qre {

template <typename Derived>
struct SDEModel {

    double drift(double S, double v, double t) const {
        return static_cast<const Derived*>(this)->drift_impl(S, v, t);
    }

    double diffusion(double S, double v, double t) const {
        return static_cast<const Derived*>(this)->diffusion_impl(S, v, t);
    }

    double variance_drift(double S, double v, double t) const {
        return static_cast<const Derived*>(this)->variance_drift_impl(S, v, t);
    }

    double variance_diffusion(double S, double v, double t) const {
        return static_cast<const Derived*>(this)->variance_diffusion_impl(S, v, t);
    }

    static constexpr bool has_variance = Derived::has_variance_process;

    double correlation() const {
        return static_cast<const Derived*>(this)->correlation_impl();
    }
};

} // namespace qre
