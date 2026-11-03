// n-tuple after-state value function (docs/superpowers/specs/2026-11-03-ntuple-learner-design.md).
//
// V(a) = sum over tuples i and the 8 board symmetries s of w[off_i + index_i(s(board))]
//      + sum over groups g of w_fit[cooldown][g][bucket(legal placements of g's variants)]
//      + sum over the 2 remaining pieces p of w_rem[group(p)][bucket(legal placements of p)].
// Bit j of a tuple index is the board bit at the tuple's j-th cell (cells ascending). Shared weights are
// read and written through relaxed std::atomic_ref, so lock-free multithreaded training is well defined.
#pragma once

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "engine.hpp"

namespace bpa {

inline constexpr int FIT_BUCKETS = 8;
inline constexpr int MAX_TUPLE_CELLS = 20;
inline constexpr int MAX_TUPLES = 256;

struct Afterstate {
    Board board = 0;  // after placement and line clears
    int remaining[2] = {0, 0};  // the two pieces left in the tray
    bool cooldown = false;  // the next spawn is restricted to groups 0..7
};

// 0, 1, 2, 3-4, 5-8, 9-16, 17-32, 33+ -> 0..7
inline int fit_bucket(int n) { return n < 3 ? n : std::min(7, std::bit_width(static_cast<unsigned>(n - 1)) + 1); }

// The 8 symmetries of the 10x10 board, from row-wise tables for a horizontal flip and a transpose.
class Symmetries {
public:
    Symmetries() {
        for (int row = 0; row < 1024; row++) {
            for (int c = 0; c < N; c++) {
                if (row >> c & 1) {
                    rev_[row] |= static_cast<uint16_t>(1 << (N - 1 - c));
                    spread_[row] |= bit(c * N);
                }
            }
        }
    }

    void all(Board b, Board out[8]) const {
        const Board h = hflip(b), t = transpose(b), th = hflip(t);
        out[0] = b, out[1] = h, out[2] = vflip(b), out[3] = vflip(h);
        out[4] = t, out[5] = th, out[6] = vflip(t), out[7] = vflip(th);
    }

private:
    uint16_t rev_[1024] = {};
    Board spread_[1024] = {};

    static unsigned row(Board b, int r) { return static_cast<unsigned>(b >> (N * r)) & 0x3FF; }
    Board hflip(Board b) const {
        Board out = 0;
        for (int r = 0; r < N; r++) out |= static_cast<Board>(rev_[row(b, r)]) << (N * r);
        return out;
    }
    static Board vflip(Board b) {
        Board out = 0;
        for (int r = 0; r < N; r++) out |= static_cast<Board>(row(b, r)) << (N * (N - 1 - r));
        return out;
    }
    Board transpose(Board b) const {
        Board out = 0;
        for (int r = 0; r < N; r++) out |= spread_[row(b, r)] << r;
        return out;
    }
};

inline const Symmetries& symmetries() {
    static const Symmetries s;
    return s;
}

inline uint64_t extract_bits(uint64_t x, uint64_t mask) {
#ifdef __BMI2__
    return _pext_u64(x, mask);
#else
    uint64_t out = 0;
    for (int j = 0; mask; mask &= mask - 1, j++) out |= (x >> std::countr_zero(mask) & 1) << j;
    return out;
#endif
}

class NTuple {
public:
    NTuple(const Tables& t, std::vector<std::vector<int>> tuples) : t_(t), tuples_(std::move(tuples)) {
        if (tuples_.size() > MAX_TUPLES) throw std::invalid_argument("at most 256 tuples");
        uint64_t offset = 0;
        for (auto& cells : tuples_) {
            if (cells.empty() || cells.size() > MAX_TUPLE_CELLS) throw std::invalid_argument("tuples need 1..20 cells");
            Mask m{0, 0, 0, static_cast<uint32_t>(offset)};
            for (int c : cells) {
                if (c < 0 || c >= CELLS) throw std::invalid_argument("tuple cell outside 0..99");
                if (c < 64) m.lo |= uint64_t{1} << c;
                else m.hi |= uint64_t{1} << (c - 64);
            }
            if (std::popcount(m.lo) + std::popcount(m.hi) != static_cast<int>(cells.size()))
                throw std::invalid_argument("duplicate cell in a tuple");
            std::sort(cells.begin(), cells.end());
            m.lo_bits = std::popcount(m.lo);
            masks_.push_back(m);
            offset += uint64_t{1} << cells.size();
            if (offset > (uint64_t{1} << 31)) throw std::invalid_argument("too many tuple weights");
        }
        fit_offset_ = static_cast<uint32_t>(offset);
        rem_offset_ = fit_offset_ + 2 * t_.n_groups * FIT_BUCKETS;
        w_.assign(rem_offset_ + t_.n_groups * FIT_BUCKETS, 0.0f);
    }

    size_t n_weights() const { return w_.size(); }
    int n_lookups() const { return 8 * static_cast<int>(masks_.size()) + t_.n_groups + 2; }
    float* weights() { return w_.data(); }
    const Tables& tables() const { return t_; }
    const std::vector<std::vector<int>>& tuples() const { return tuples_; }

    double value(const Afterstate& a) const {
        uint32_t idx[MAX_LOOKUPS];
        const int n = lookups(a, idx);
        double v = 0;
        for (int k = 0; k < n; k++) v += load(idx[k]);
        return v;
    }

    // TD step: every looked-up weight moves by alpha * (target - V(a)) / n_lookups.
    void update(const Afterstate& a, double target, float alpha) {
        uint32_t idx[MAX_LOOKUPS];
        const int n = lookups(a, idx);
        double v = 0;
        for (int k = 0; k < n; k++) v += load(idx[k]);
        const auto step = static_cast<float>(alpha * (target - v) / n);
        for (int k = 0; k < n; k++) {
            std::atomic_ref<float> w(w_[idx[k]]);
            w.store(w.load(std::memory_order_relaxed) + step, std::memory_order_relaxed);
        }
    }

private:
    struct Mask {
        uint64_t lo, hi;
        int lo_bits;
        uint32_t offset;
    };
    static constexpr int MAX_LOOKUPS = 8 * MAX_TUPLES + MAX_GROUPS + 2;

    Tables t_;
    std::vector<std::vector<int>> tuples_;
    std::vector<Mask> masks_;
    uint32_t fit_offset_ = 0, rem_offset_ = 0;
    std::vector<float> w_;

    float load(uint32_t i) const {
        return std::atomic_ref<float>(const_cast<float&>(w_[i])).load(std::memory_order_relaxed);
    }

    int lookups(const Afterstate& a, uint32_t* idx) const {
        int k = 0;
        Board boards[8];
        symmetries().all(a.board, boards);
        for (const Mask& m : masks_) {
            for (const Board b : boards) {
                const uint64_t lo = extract_bits(static_cast<uint64_t>(b), m.lo);
                const uint64_t hi = extract_bits(static_cast<uint64_t>(b >> 64), m.hi);
                idx[k++] = m.offset + static_cast<uint32_t>(lo | hi << m.lo_bits);
            }
        }
        int count[MAX_PIECES], group_count[MAX_GROUPS] = {};
        for (int p = 0; p < t_.n_pieces; p++) {
            const Board legal = legal_positions(t_, a.board, p);
            count[p] = std::popcount(static_cast<uint64_t>(legal)) + std::popcount(static_cast<uint64_t>(legal >> 64));
            group_count[t_.group[p]] += count[p];
        }
        const int base = fit_offset_ + (a.cooldown ? t_.n_groups : 0) * FIT_BUCKETS;
        for (int g = 0; g < t_.n_groups; g++) idx[k++] = base + g * FIT_BUCKETS + fit_bucket(group_count[g]);
        for (const int p : a.remaining) idx[k++] = rem_offset_ + t_.group[p] * FIT_BUCKETS + fit_bucket(count[p]);
        return k;
    }
};

}  // namespace bpa
