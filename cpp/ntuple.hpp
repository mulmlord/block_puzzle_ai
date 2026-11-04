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

struct TrainStats {
    uint64_t moves = 0;
    double seconds = 0;
    uint64_t games = 0, deaths = 0, fresh_games = 0, restarts = 0;
    double fresh_score_sum = 0, fresh_moves_sum = 0, value_sum = 0;

    double mean_score() const { return fresh_games ? fresh_score_sum / fresh_games : 0; }
    double mean_moves() const { return fresh_games ? fresh_moves_sum / fresh_games : 0; }
    double value_mean() const { return moves ? value_sum / moves : 0; }
};

// Lock-free multithreaded TD(0) on after-states with restarts near game overs. Every thread keeps its game,
// snapshot ring, restart pool and random stream across run() calls, so long games continue between chunks.
class Trainer {
public:
    static constexpr int SNAPSHOT_EVERY = 50, RING = 10, POOL = 4096;
    static constexpr int64_t POOL_WINDOW = 500;

    Trainer(NTuple& nt, float gamma, float alpha, double restart_frac, int threads, uint64_t seed)
        : alpha(alpha), nt_(nt), gamma_(gamma), restart_frac_(restart_frac) {
        if (threads < 1) throw std::invalid_argument("threads must be >= 1");
        if (restart_frac < 0 || restart_frac > 1) throw std::invalid_argument("restart_frac must be in [0, 1]");
        for (uint64_t s : stream_seeds(seed, threads)) workers_.emplace_back(s);
    }

    float alpha;

    TrainStats run(uint64_t moves) {
        const auto n = static_cast<uint64_t>(workers_.size());
        std::vector<TrainStats> per(n);
        const auto start = std::chrono::steady_clock::now();
        parallel(static_cast<int>(n), [&](int i) { per[i] = work(workers_[i], moves / n + (i < moves % n)); });
        TrainStats total;
        total.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        for (const TrainStats& st : per) {
            total.moves += st.moves, total.games += st.games, total.deaths += st.deaths;
            total.fresh_games += st.fresh_games, total.restarts += st.restarts;
            total.fresh_score_sum += st.fresh_score_sum, total.fresh_moves_sum += st.fresh_moves_sum;
            total.value_sum += st.value_sum;
        }
        return total;
    }

private:
    struct Snapshot {
        State s;
        uint64_t game = 0;
    };
    struct Worker {
        explicit Worker(uint64_t seed) : rng(seed) {}
        Xoshiro256 rng;
        State s;
        Choice choice;  // the move chosen in s
        bool started = false, fresh = true;
        uint64_t game = 0, steps = 0;
        Snapshot ring[RING];
        int ring_size = 0, ring_head = 0;
        std::vector<State> pool;
        size_t pool_head = 0;
    };

    NTuple& nt_;
    float gamma_;
    double restart_frac_;
    std::vector<Worker> workers_;

    void new_game(Worker& w, TrainStats& st) {
        const Tables& t = nt_.tables();
        w.game++;
        w.ring_size = w.ring_head = 0;
        const bool from_pool = !w.pool.empty() && w.rng.randrange(1u << 30) < restart_frac_ * (1u << 30);
        if (from_pool) {
            w.s = w.pool[w.rng.randrange(static_cast<uint32_t>(w.pool.size()))];
            st.restarts++;
        } else {
            reset(t, w.s, w.rng);
        }
        w.fresh = !from_pool;
        if (!best_move(nt_, w.s, gamma_, w.choice)) {  // unreachable (pool states have a legal move), but never play a stale move
            reset(t, w.s, w.rng);
            w.fresh = true;
            best_move(nt_, w.s, gamma_, w.choice);  // an empty board always fits a piece
        }
        check_finite(w.choice.q);
    }

    void game_over(Worker& w, TrainStats& st) {
        st.games++, st.deaths++;
        if (w.fresh) st.fresh_games++, st.fresh_score_sum += w.s.score, st.fresh_moves_sum += w.s.moves;
        for (int k = 0; k < w.ring_size; k++) {
            const Snapshot& snap = w.ring[k];
            if (snap.game != w.game || snap.s.moves < w.s.moves - POOL_WINDOW) continue;
            if (w.pool.size() < POOL) w.pool.push_back(snap.s);
            else w.pool[w.pool_head] = snap.s, w.pool_head = (w.pool_head + 1) % POOL;
        }
    }

    TrainStats work(Worker& w, uint64_t moves) {
        const Tables& t = nt_.tables();
        TrainStats st;
        if (!w.started) new_game(w, st), w.started = true;
        for (uint64_t m = 0; m < moves; m++) {
            const Afterstate a = w.choice.after;
            step(t, w.s, w.choice.slot, w.choice.pos, w.rng);
            Choice next;
            if (best_move(nt_, w.s, gamma_, next)) {
                check_finite(next.q);
                if (w.steps++ % SNAPSHOT_EVERY == 0) {  // only states with a legal move may become restart states
                    w.ring[w.ring_head] = {w.s, w.game};
                    w.ring_head = (w.ring_head + 1) % RING;
                    w.ring_size = std::min(w.ring_size + 1, RING);
                }
                nt_.update(a, next.q, alpha);
                w.choice = next;
            } else {
                nt_.update(a, 0.0, alpha);
                game_over(w, st);
                new_game(w, st);
            }
            st.value_sum += w.choice.v;
        }
        st.moves = moves;
        return st;
    }
};

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
