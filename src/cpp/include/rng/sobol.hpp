#pragma once

/**
 * @file sobol.hpp
 * @brief Sobol quasi-random sequence generator with Gray code optimization.
 *
 * Generates low-discrepancy Sobol sequences using Joe-Kuo direction numbers.
 * Loads direction numbers from file (new-joe-kuo-6.21201, up to 21201 dimensions).
 * Supports Owen's digital shift scrambling for randomized QMC.
 *
 * Key properties:
 *   - O(1) per point via Gray code (XOR with single direction number)
 *   - Inverse normal CDF transform for Gaussian variates
 *   - Digital shift scrambling for error estimation
 */

#include "inverse_normal.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace qre {

class SobolGenerator {
public:
    static constexpr int MAX_BITS = 30;  // 2^30 > 10^9 points

    /**
     * @param n_dims        Number of dimensions
     * @param scramble_seed Seed for digital shift scrambling (0 = no scrambling)
     * @param jk_file       Path to Joe-Kuo direction numbers file (optional)
     */
    SobolGenerator(int n_dims, uint64_t scramble_seed = 0,
                   const std::string& jk_file = "")
        : n_dims_(n_dims), index_(0),
          direction_(n_dims, std::vector<uint32_t>(MAX_BITS, 0)),
          state_(n_dims, 0),
          shift_(n_dims, 0)
    {
        assert(n_dims > 0);
        init_direction_numbers(jk_file);
        if (scramble_seed != 0) init_scramble(scramble_seed);
    }

    /// Generate the next Sobol point as uniform [0,1)^d
    void next_point(double* output) {
        if (index_ == 0) {
            for (int d = 0; d < n_dims_; ++d)
                output[d] = to_double(state_[d] ^ shift_[d]);
            ++index_;
            return;
        }

        // Gray code: find rightmost zero bit of index_
        uint32_t c = rightmost_zero_bit(static_cast<uint32_t>(index_));
        if (c >= MAX_BITS) c = MAX_BITS - 1;

        for (int d = 0; d < n_dims_; ++d) {
            state_[d] ^= direction_[d][c];
            output[d] = to_double(state_[d] ^ shift_[d]);
        }
        ++index_;
    }

    /// Generate the next point and transform to N(0,1) via inverse CDF
    void next_normal_point(double* output) {
        next_point(output);
        for (int d = 0; d < n_dims_; ++d) {
            double u = output[d];
            if (u < 1e-10) u = 1e-10;
            if (u > 1.0 - 1e-10) u = 1.0 - 1e-10;
            output[d] = inverse_normal_cdf(u);
        }
    }

    /// Skip to a specific index
    void skip_to(size_t target) {
        for (int d = 0; d < n_dims_; ++d) state_[d] = 0;
        uint32_t g = static_cast<uint32_t>(target) ^ (static_cast<uint32_t>(target) >> 1);
        for (int bit = 0; bit < MAX_BITS; ++bit) {
            if ((g >> bit) & 1u) {
                for (int d = 0; d < n_dims_; ++d)
                    state_[d] ^= direction_[d][bit];
            }
        }
        index_ = target;
    }

    void reset() {
        for (int d = 0; d < n_dims_; ++d) state_[d] = 0;
        index_ = 0;
    }

    void scramble(uint64_t seed) {
        init_scramble(seed);
        reset();
    }

    int    dimensions()    const { return n_dims_; }
    size_t current_index() const { return index_; }

    /// Pre-generate a matrix of normal Sobol points: [n_points * n_dims]
    std::vector<double> generate_normal_matrix(size_t n_points) {
        std::vector<double> matrix(n_points * n_dims_);
        std::vector<double> point(n_dims_);
        reset();
        for (size_t i = 0; i < n_points; ++i) {
            next_normal_point(point.data());
            for (int d = 0; d < n_dims_; ++d)
                matrix[i * n_dims_ + d] = point[d];
        }
        return matrix;
    }

private:
    int    n_dims_;
    size_t index_;
    std::vector<std::vector<uint32_t>> direction_;
    std::vector<uint32_t> state_;
    std::vector<uint32_t> shift_;

    static double to_double(uint32_t x) {
        return static_cast<double>(x) / static_cast<double>(1u << MAX_BITS);
    }

    static uint32_t rightmost_zero_bit(uint32_t n) {
        uint32_t pos = 0;
        uint32_t val = n;
        while (val & 1u) { val >>= 1; ++pos; }
        return pos;
    }

    void init_scramble(uint64_t seed) {
        std::mt19937 rng(static_cast<unsigned>(seed));
        uint32_t mask = (1u << MAX_BITS) - 1;
        for (int d = 0; d < n_dims_; ++d)
            shift_[d] = rng() & mask;
    }

    void init_direction_numbers(const std::string& jk_file) {
        // Dimension 0: van der Corput sequence (base 2)
        for (int bit = 0; bit < MAX_BITS; ++bit)
            direction_[0][bit] = 1u << (MAX_BITS - 1 - bit);

        if (n_dims_ <= 1) return;

        // Try to load Joe-Kuo direction numbers from file
        std::string path = jk_file;
        if (path.empty()) {
            // Try standard locations
            static const char* candidates[] = {
                "data/joe-kuo-directions.txt",
                "../data/joe-kuo-directions.txt",
                "../../data/joe-kuo-directions.txt",
                "../../../data/joe-kuo-directions.txt",
            };
            for (auto& c : candidates) {
                std::ifstream test(c);
                if (test.good()) { path = c; break; }
            }
        }

        if (!path.empty()) {
            if (load_joe_kuo(path)) return;
        }

        // Fallback: generate basic direction numbers algorithmically
        init_fallback_directions();
    }

    bool load_joe_kuo(const std::string& filename) {
        std::ifstream infile(filename);
        if (!infile.is_open()) return false;

        // Skip header line
        std::string line;
        std::getline(infile, line);

        int dim_loaded = 0;
        while (std::getline(infile, line) && dim_loaded < n_dims_ - 1) {
            std::istringstream iss(line);
            unsigned d_val, s_val, a_val;
            if (!(iss >> d_val >> s_val >> a_val)) continue;

            int dim = dim_loaded + 1;  // dim 0 is van der Corput
            int s = static_cast<int>(s_val);

            // Read initial direction numbers m_1, ..., m_s
            std::vector<uint32_t> m(s);
            for (int i = 0; i < s; ++i) {
                unsigned mi;
                if (!(iss >> mi)) mi = 1;
                m[i] = mi;
            }

            // Compute direction numbers v[i] = m[i] * 2^(MAX_BITS - i) for i < s
            std::vector<uint32_t> v(MAX_BITS, 0);
            for (int i = 0; i < s && i < MAX_BITS; ++i)
                v[i] = m[i] << (MAX_BITS - 1 - i);

            // Recurrence for i >= s:
            // v[i] = c1*v[i-1] ^ c2*v[i-2] ^ ... ^ cs*v[i-s] ^ (v[i-s] >> s)
            // where c_j are bits of 'a' plus implicit leading/trailing 1s
            // The primitive polynomial is P(x) = x^s + a_{s-1}*x^{s-1} + ... + a_1*x + 1
            // The coefficients are bits of 'a' reading from MSB to LSB
            for (int i = s; i < MAX_BITS; ++i) {
                uint32_t val = v[i - s] ^ (v[i - s] >> s);
                for (int j = 1; j < s; ++j) {
                    if ((a_val >> (s - 1 - j)) & 1u)
                        val ^= v[i - j];
                }
                v[i] = val;
            }

            for (int bit = 0; bit < MAX_BITS; ++bit)
                direction_[dim][bit] = v[bit];

            ++dim_loaded;
        }

        // If we didn't load enough dimensions, fill remaining with fallback
        if (dim_loaded < n_dims_ - 1) {
            for (int dim = dim_loaded + 1; dim < n_dims_; ++dim) {
                // Simple fallback for missing dimensions
                for (int bit = 0; bit < MAX_BITS; ++bit) {
                    uint32_t hash = static_cast<uint32_t>(dim * 2654435761u + bit * 340573321u);
                    direction_[dim][bit] = (hash & ((1u << MAX_BITS) - 1)) | (1u << (MAX_BITS - 1));
                }
            }
        }

        return true;
    }

    void init_fallback_directions() {
        // Simple fallback: use bit-reversal permutations
        // Not true Sobol but provides some stratification
        for (int dim = 1; dim < n_dims_; ++dim) {
            for (int bit = 0; bit < MAX_BITS; ++bit) {
                // Create direction numbers that at least have the right structure
                uint32_t base = 1u << (MAX_BITS - 1 - bit);
                uint32_t scramble = static_cast<uint32_t>((dim * 2654435761u) >> (32 - MAX_BITS));
                direction_[dim][bit] = base ^ (scramble & ((1u << (MAX_BITS - bit)) - 1));
                // Ensure MSB is set for proper normalization
                direction_[dim][bit] |= (1u << (MAX_BITS - 1 - bit));
            }
        }
    }
};

} // namespace qre
