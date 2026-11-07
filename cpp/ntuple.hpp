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
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <mutex>
#include <limits>
#include <stdexcept>
#include <thread>
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

// The part of V that depends only on the board: the tuple sum and the placement counts. The search reuses it
// for every (cooldown, remaining) combination of one board (docs/superpowers/specs/2026-11-05-ntuple-search-design.md).
struct BoardEval {
    double tuples = 0;  // sum of the 8 * n_tuples tuple weights, from 0 in lookup order
    int count[MAX_PIECES] = {};  // legal placements of each piece
    int group_count[MAX_GROUPS] = {};  // summed over each group's pieces
};

// What the children of one board reuse (NTuple::child_eval): its tuple lookups and its pieces' legal positions.
struct BoardParts {
    std::vector<uint32_t> idx;  // weight index of each tuple lookup, in board_eval's order
    std::vector<float> w;  // the weight read at each of them
    Board legal[MAX_PIECES] = {};  // legal_positions of each piece
};

// 0, 1, 2, 3-4, 5-8, 9-16, 17-32, 33+ -> 0..7
inline int fit_bucket(int n) { return n < 3 ? n : std::min(7, std::bit_width(static_cast<unsigned>(n - 1)) + 1); }

inline int popcount128(Board b) { return std::popcount(static_cast<uint64_t>(b)) + std::popcount(static_cast<uint64_t>(b >> 64)); }

// b shifted towards higher bits by d (d < 0: towards lower bits), |d| < 128
inline Board shift(Board b, int d) { return d >= 0 ? b << d : b >> -d; }

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
        init_children();
    }

    size_t n_weights() const { return w_.size(); }
    int n_lookups() const { return 8 * static_cast<int>(masks_.size()) + t_.n_groups + 2; }
    float* weights() { return w_.data(); }
    const Tables& tables() const { return t_; }
    const std::vector<std::vector<int>>& tuples() const { return tuples_; }

    BoardEval board_eval(Board board) const {
        BoardEval e;
        Board boards[8];
        symmetries().all(board, boards);
        double sum = 0;  // a local: summing into e.tuples kept it in memory across the atomic loads (35% slower)
        for (const Mask& m : masks_)
            for (const Board b : boards) sum += load(tuple_index(m, b));
        e.tuples = sum;
        placement_counts(board, e.count, e.group_count);
        return e;
    }

    BoardParts parts(Board board) const {
        BoardParts p;
        Board boards[8];
        symmetries().all(board, boards);
        p.idx.reserve(8 * masks_.size());
        p.w.reserve(8 * masks_.size());
        for (const Mask& m : masks_) {
            for (const Board b : boards) {
                p.idx.push_back(tuple_index(m, b));
                p.w.push_back(load(p.idx.back()));
            }
        }
        for (int q = 0; q < t_.n_pieces; q++) p.legal[q] = legal_positions(t_, board, q);
        return p;
    }

    // board_eval(placed.board), bit for bit, where placed = apply(t, board, piece, pos) and parent = parts(board).
    // Without a line clear the child is the parent plus the piece's cells, so only the lookups that read those
    // cells change, and a piece's legal positions lose those the new cells block; the sum keeps board_eval's order.
    BoardEval child_eval(const BoardParts& parent, int piece, int pos, const Placed& placed) const {
        if (placed.lines) return board_eval(placed.board);
        const int n = static_cast<int>(parent.w.size());
        float w[8 * MAX_TUPLES];
        std::copy_n(parent.w.data(), n, w);
        Board img[8];  // the piece at pos under each symmetry
        for (int s = 0; s < 8; s++) img[s] = shift(piece_sym_[piece][s], sym_shift_[pos][s]);
        uint64_t touched[TOUCH_WORDS] = {};
        for (int k = 0; k < t_.size[piece]; k++) {
            const uint64_t* tc = &touch_[(pos + t_.offsets[piece][k]) * touch_words_];
            for (int j = 0; j < touch_words_; j++) touched[j] |= tc[j];
        }
        for (int j = 0; j < touch_words_; j++) {
            for (uint64_t bits = touched[j]; bits; bits &= bits - 1) {
                const int k = 64 * j + std::countr_zero(bits);
                const Mask& m = masks_[k >> 3];
                const Board b = img[k & 7];  // its cells were empty in the parent: their bits add to the index
                const uint64_t lo = extract_bits(static_cast<uint64_t>(b), m.lo);
                const uint64_t hi = extract_bits(static_cast<uint64_t>(b >> 64), m.hi);
                w[k] = load(parent.idx[k] + static_cast<uint32_t>(lo | hi << m.lo_bits));
            }
        }
        BoardEval e;
        double sum = 0;
        for (int k = 0; k < n; k++) sum += w[k];
        e.tuples = sum;
        const int d = pos - block_center_;
        for (int p = 0; p < t_.n_pieces; p++) {
            e.count[p] = popcount128(parent.legal[p] & ~shift(blocks_[piece][p], d));
            e.group_count[t_.group[p]] += e.count[p];
        }
        return e;
    }

    // The tuple sum plus the fit weights: the part of V shared by every remaining pair.
    double fit_prefix(const BoardEval& e, bool cooldown) const {
        double v = e.tuples;
        const int base = fit_offset_ + (cooldown ? t_.n_groups : 0) * FIT_BUCKETS;
        for (int g = 0; g < t_.n_groups; g++) v += load(base + g * FIT_BUCKETS + fit_bucket(e.group_count[g]));
        return v;
    }

    float rem_weight(const BoardEval& e, int piece) const { return load(rem_index(piece, e.count)); }

    // V from a board's BoardEval: the same lookups and summation order as value(Afterstate), so bit-identical.
    // Callers that add the remaining weights to fit_prefix one at a time, in this order, get the same double.
    double value(const BoardEval& e, bool cooldown, int rem0, int rem1) const {
        double v = fit_prefix(e, cooldown);
        v += rem_weight(e, rem0);
        v += rem_weight(e, rem1);
        return v;
    }

    double value(const Afterstate& a) const { return value(board_eval(a.board), a.cooldown, a.remaining[0], a.remaining[1]); }

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

    // Temporal-coherence step sizes (Beal & Smith 1999; Jaskowski 2018 for 2048 n-tuple networks). Per weight,
    // E sums the steps it was asked to take and A their sizes, so |E| / A is 1 while the steps agree in sign and
    // falls towards 0 while they cancel. Stored as (E, A) pairs of doubles: a float A stops growing once it is
    // 2^24 steps, after a few million moves for the fit weights read on every move.
    void enable_tc() {
        if (tc_.empty()) tc_.assign(2 * w_.size(), 0.0);
    }
    bool tc_enabled() const { return !tc_.empty(); }
    double* tc_state() { return tc_.data(); }

    // TD step with temporal-coherence rates: with delta = (target - V(a)) / n_lookups, each looked-up weight
    // moves by alpha * rate * delta, rate = |E| / A (1 while A is 0), then E += delta and A += |delta|.
    void update_tc(const Afterstate& a, double target, float alpha) {
        uint32_t idx[MAX_LOOKUPS];
        const int n = lookups(a, idx);
        double v = 0;
        for (int k = 0; k < n; k++) v += load(idx[k]);
        const double delta = (target - v) / n;
        constexpr auto relaxed = std::memory_order_relaxed;
        for (int k = 0; k < n; k++) {
            std::atomic_ref<float> w(w_[idx[k]]);
            std::atomic_ref<double> e(tc_[2 * size_t{idx[k]}]), s(tc_[2 * size_t{idx[k]} + 1]);
            const double E = e.load(relaxed), A = s.load(relaxed);
            const double rate = A > 0 ? std::abs(E) / A : 1.0;
            w.store(w.load(relaxed) + static_cast<float>(alpha * rate * delta), relaxed);
            e.store(E + delta, relaxed);
            s.store(A + std::abs(delta), relaxed);
        }
    }

    // update_tc once TC is enabled, update otherwise
    void learn(const Afterstate& a, double target, float alpha) {
        if (tc_enabled()) update_tc(a, target, alpha);
        else update(a, target, alpha);
    }

private:
    struct Mask {
        uint64_t lo, hi;
        int lo_bits;
        uint32_t offset;
    };
    static constexpr int MAX_LOOKUPS = 8 * MAX_TUPLES + MAX_GROUPS + 2;
    static constexpr int TOUCH_WORDS = (8 * MAX_TUPLES + 63) / 64;

    Tables t_;
    std::vector<std::vector<int>> tuples_;
    std::vector<Mask> masks_;
    uint32_t fit_offset_ = 0, rem_offset_ = 0;
    std::vector<float> w_;
    std::vector<double> tc_;  // temporal coherence: (E, A) per weight, empty until enable_tc
    // child_eval's tables
    Board piece_sym_[MAX_PIECES][8] = {};  // each piece at position 0 under the 8 symmetries
    int sym_shift_[CELLS][8] = {};  // moving a piece from 0 to pos translates its image: shift(piece_sym_, this)
    int touch_words_ = 0;
    std::vector<uint64_t> touch_;  // bit k of row c: lookup k (board_eval's order) reads cell c
    int block_center_ = 0;  // the largest cell offset of any piece
    Board blocks_[MAX_PIECES][MAX_PIECES] = {};  // x at pos blocks p at shift(blocks_[x][p], pos - block_center_)

    void init_children() {
        for (int pos = 0; pos < CELLS; pos++) {
            const int r = pos / N, c = pos % N;  // symmetries().all's order: id, h, v, vh, t, ht, vt, vht
            const int d[8] = {N * r + c, N * r - c, -N * r + c, -N * r - c, N * c + r, N * c - r, -N * c + r, -N * c - r};
            std::copy(d, d + 8, sym_shift_[pos]);
        }
        for (int p = 0; p < t_.n_pieces; p++) symmetries().all(t_.shape[p], piece_sym_[p]);
        touch_words_ = (8 * static_cast<int>(masks_.size()) + 63) / 64;
        touch_.assign(CELLS * touch_words_, 0);
        for (int c = 0; c < CELLS; c++) {
            Board img[8];
            symmetries().all(bit(c), img);
            for (size_t i = 0; i < masks_.size(); i++) {
                for (int s = 0; s < 8; s++) {
                    if (!((static_cast<uint64_t>(img[s]) & masks_[i].lo) | (static_cast<uint64_t>(img[s] >> 64) & masks_[i].hi)))
                        continue;
                    const size_t k = 8 * i + s;
                    touch_[c * touch_words_ + k / 64] |= uint64_t{1} << (k % 64);
                }
            }
        }
        // legal_positions: p is blocked at q when q + off is filled for an offset off of p, so x at pos blocks
        // p at pos + o - off for each offset o of x: the union over off of shape[x] << (pos - off)
        for (int p = 0; p < t_.n_pieces; p++)
            for (int k = 0; k < t_.size[p]; k++) block_center_ = std::max(block_center_, t_.offsets[p][k]);
        for (int x = 0; x < t_.n_pieces; x++)
            for (int p = 0; p < t_.n_pieces; p++)
                for (int k = 0; k < t_.size[p]; k++) blocks_[x][p] |= t_.shape[x] << (block_center_ - t_.offsets[p][k]);
    }

    float load(uint32_t i) const {
        return std::atomic_ref<float>(const_cast<float&>(w_[i])).load(std::memory_order_relaxed);
    }

    static uint32_t tuple_index(const Mask& m, Board b) {
        const uint64_t lo = extract_bits(static_cast<uint64_t>(b), m.lo);
        const uint64_t hi = extract_bits(static_cast<uint64_t>(b >> 64), m.hi);
        return m.offset + static_cast<uint32_t>(lo | hi << m.lo_bits);
    }

    // count[p] = legal placements of piece p; group_count[g] += count[p] for p in g (zeroed by the caller)
    void placement_counts(Board board, int* count, int* group_count) const {
        for (int p = 0; p < t_.n_pieces; p++) {
            const Board legal = legal_positions(t_, board, p);
            count[p] = std::popcount(static_cast<uint64_t>(legal)) + std::popcount(static_cast<uint64_t>(legal >> 64));
            group_count[t_.group[p]] += count[p];
        }
    }

    uint32_t rem_index(int piece, const int* count) const {
        return rem_offset_ + t_.group[piece] * FIT_BUCKETS + fit_bucket(count[piece]);
    }

    int lookups(const Afterstate& a, uint32_t* idx) const {
        int k = 0;
        Board boards[8];
        symmetries().all(a.board, boards);
        for (const Mask& m : masks_)
            for (const Board b : boards) idx[k++] = tuple_index(m, b);
        int count[MAX_PIECES], group_count[MAX_GROUPS] = {};
        placement_counts(a.board, count, group_count);
        const int base = fit_offset_ + (a.cooldown ? t_.n_groups : 0) * FIT_BUCKETS;
        for (int g = 0; g < t_.n_groups; g++) idx[k++] = base + g * FIT_BUCKETS + fit_bucket(group_count[g]);
        for (const int p : a.remaining) idx[k++] = rem_index(p, count);
        return k;
    }
};

// ---- playing and training ---------------------------------------------------------------------

struct Choice {
    int slot = 0, pos = 0, points = 0;
    Afterstate after;
    double v = 0;  // V(after)
    double q = 0;  // points + gamma * v
};

// Greedy 1-ply move: argmax over legal moves of points + gamma * V(after-state), first maximum in engine
// order (slot-major, position ascending). Returns false only if the state has no legal move; with
// non-finite values it still returns a legal move, and callers check std::isfinite(out.q).
inline bool best_move(const NTuple& nt, const State& s, float gamma, Choice& out) {
    const Tables& t = nt.tables();
    Afterstate a;
    a.cooldown = in_cooldown(t, s.history);  // the spawn after this move uses the current history
    bool found = false;
    double best = -std::numeric_limits<double>::infinity();
    for (int slot = 0; slot < 3; slot++) {
        const int piece = s.pending[slot];
        a.remaining[0] = s.pending[slot == 0 ? 1 : 0];
        a.remaining[1] = s.pending[slot == 2 ? 1 : 2];
        for_each_bit(legal_positions(t, s.board, piece), [&](int pos) {
            const Placed p = apply(t, s.board, piece, pos);
            a.board = p.board;
            const double v = nt.value(a), q = p.points + gamma * v;
            if (!found || q > best) best = q, found = true, out = {slot, pos, p.points, a, v, q};
        });
    }
    return found;
}

// Runs f(i) for i in [0, n) on n threads; the first exception thrown by any thread is rethrown after all join,
// so a failing worker raises in Python instead of calling std::terminate.
template <class F>
inline void parallel(int n, F&& f) {
    std::exception_ptr error;
    std::mutex m;
    {
        std::vector<std::jthread> pool;
        for (int i = 0; i < n; i++) {
            pool.emplace_back([&, i] {
                try {
                    f(i);
                } catch (...) {
                    std::lock_guard lock(m);
                    if (!error) error = std::current_exception();
                }
            });
        }
    }
    if (error) std::rethrow_exception(error);
}

inline void check_finite(double q) {
    if (!std::isfinite(q))
        throw std::runtime_error("n-tuple values diverged (non-finite value); lower alpha or use fewer threads");
}

// Per-thread seeds: the (i + 1)-th output of a splitmix64 stream started at ``seed``.
inline std::vector<uint64_t> stream_seeds(uint64_t seed, size_t n) {
    std::vector<uint64_t> out(n);
    for (uint64_t& s : out) s = splitmix64(seed);
    return out;
}

struct GameOutcome {
    int64_t score = 0, moves = 0, lines = 0;
    bool truncated = false;
};

// Greedy 1-ply games; game i uses its own random stream, so results do not depend on the thread count.
inline std::vector<GameOutcome> evaluate(const NTuple& nt, int games, uint64_t seed, int64_t max_moves, int threads,
                                         float gamma) {
    if (threads < 1) throw std::invalid_argument("threads must be >= 1");
    if (games < 0) throw std::invalid_argument("games must be >= 0");
    const std::vector<uint64_t> seeds = stream_seeds(seed, games);
    std::vector<GameOutcome> out(games);
    std::atomic<int> next{0};
    parallel(threads, [&](int) {
                for (int g; (g = next.fetch_add(1)) < games;) {
                    Xoshiro256 rng(seeds[g]);
                    State s;
                    reset(nt.tables(), s, rng);
                    Choice c;
                    bool truncated = false;
                    while (best_move(nt, s, gamma, c)) {
                        check_finite(c.q);
                        if (s.moves >= max_moves) {
                            truncated = true;
                            break;
                        }
                        step(nt.tables(), s, c.slot, c.pos, rng);
                    }
                    out[g] = {s.score, s.moves, s.lines, truncated};
                }
    });
    return out;
}

}  // namespace bpa
