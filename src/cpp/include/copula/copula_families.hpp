#pragma once

/**
 * @file copula_families.hpp
 * @brief Archimedean copula families: Clayton, Gumbel, Frank.
 *
 * Each copula captures a different dependence structure:
 *   - Clayton: strong lower-tail dependence (crashes)
 *   - Gumbel:  strong upper-tail dependence
 *   - Frank:   symmetric dependence (no tail dependence)
 *
 * Used as building blocks for vine copula constructions.
 *
 * TODO:
 *   - Implement generator functions and their inverses
 *   - Conditional distribution functions for vine sampling
 *   - MLE parameter fitting
 */

#include <cmath>
#include <stdexcept>

namespace qre::copula {

/// Bivariate Clayton copula: C(u,v) = (u^{-theta} + v^{-theta} - 1)^{-1/theta}
class Clayton {
public:
    explicit Clayton(double theta) : theta_{theta} {
        if (theta <= 0.0) throw std::invalid_argument("Clayton theta must be > 0");
    }

    [[nodiscard]] auto cdf(double u, double v) const -> double {
        return std::pow(std::pow(u, -theta_) + std::pow(v, -theta_) - 1.0,
                        -1.0 / theta_);
    }

    [[nodiscard]] auto theta() const -> double { return theta_; }

    /// Lower tail dependence coefficient: lambda_L = 2^{-1/theta}
    [[nodiscard]] auto lower_tail_dependence() const -> double {
        return std::pow(2.0, -1.0 / theta_);
    }

private:
    double theta_;
};

/// Bivariate Gumbel copula: C(u,v) = exp(-[(-ln u)^theta + (-ln v)^theta]^{1/theta})
class Gumbel {
public:
    explicit Gumbel(double theta) : theta_{theta} {
        if (theta < 1.0) throw std::invalid_argument("Gumbel theta must be >= 1");
    }

    [[nodiscard]] auto cdf(double u, double v) const -> double {
        const double a = std::pow(-std::log(u), theta_);
        const double b = std::pow(-std::log(v), theta_);
        return std::exp(-std::pow(a + b, 1.0 / theta_));
    }

    [[nodiscard]] auto theta() const -> double { return theta_; }

    /// Upper tail dependence coefficient: lambda_U = 2 - 2^{1/theta}
    [[nodiscard]] auto upper_tail_dependence() const -> double {
        return 2.0 - std::pow(2.0, 1.0 / theta_);
    }

private:
    double theta_;
};

/// Bivariate Frank copula (symmetric, no tail dependence)
class Frank {
public:
    explicit Frank(double theta) : theta_{theta} {
        if (theta == 0.0) throw std::invalid_argument("Frank theta must be != 0");
    }

    [[nodiscard]] auto cdf(double u, double v) const -> double {
        const double e1 = std::exp(-theta_ * u) - 1.0;
        const double e2 = std::exp(-theta_ * v) - 1.0;
        const double et = std::exp(-theta_) - 1.0;
        return -std::log(1.0 + (e1 * e2) / et) / theta_;
    }

    [[nodiscard]] auto theta() const -> double { return theta_; }

private:
    double theta_;
};

} // namespace qre::copula
