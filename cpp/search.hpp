// Expectimax search over the n-tuple after-state value (docs/superpowers/specs/2026-11-05-ntuple-search-design.md).
//
//   Q1(m)          = r(m) + gamma * V(a_m)                       (best_move's score)
//   E(s, m, R)     = r(m) + gamma * sum_x P(x) * B(s'(m, x), R)  (P from s's score and history, x ascending)
//   B(s, [])       = max_m Q1(m), or 0 with no legal move
//   B(s, [k, R'])  = max over the top k moves of s by Q1 of E(s, m, R'), or 0 with no legal move
//
// The root keeps its top widths[0] moves by Q1 and plays the one with the highest E(s, m, widths[1:]).
// Ties in Q1 keep engine order (slot-major, position ascending); k = 0 keeps every move. No random numbers.
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <mutex>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "engine.hpp"
#include "ntuple.hpp"

namespace bpa {

struct Tier {
    double below;  // the tier applies when the root's best Q1 is below this (+inf: every move)
    std::vector<int> widths;  // moves kept per spawn level (0 = all); depth = widths.size() + 1
};

inline void check_widths(const std::vector<int>& widths) {
    if (widths.empty()) throw std::invalid_argument("search widths: at least one level");
    for (int w : widths)
        if (w < 0) throw std::invalid_argument("search widths must be >= 0 (0 = all moves)");
}

// Valid tiers: strictly decreasing below, non-decreasing depth, valid widths.
inline void check_tiers(const std::vector<Tier>& tiers) {
    for (size_t i = 0; i < tiers.size(); i++) {
        if (std::isnan(tiers[i].below)) throw std::invalid_argument("search tier: below is NaN");
        check_widths(tiers[i].widths);
        if (i == 0) continue;
        if (!(tiers[i].below < tiers[i - 1].below)) throw std::invalid_argument("search tiers need strictly decreasing below");
        if (tiers[i].widths.size() < tiers[i - 1].widths.size())
            throw std::invalid_argument("search tiers need non-decreasing depth");
    }
}

struct Scored {
    int slot, pos, points;
    double q1, score;
};

namespace search_detail {

struct Cand {
    int slot, pos, points;
    Board board;  // after placement and clears
    int rem0, rem1;  // the two pieces left, in slot order
    double v, q1;
};

// A placement of a pending piece of a child state. Its board is the same for every spawned piece x, so the
// board's evaluation is shared: V = pre[cooldown] + rem_weight(x), summed in value()'s order.
struct Known {
    int pos, points;
    Board board;
    BoardEval e;
    double pre[2];  // fit_prefix(e, cooldown) + rem_weight(e, other pending piece)
};

// Every legal move of the root with Q1, in engine order, computed exactly as best_move does.
inline void root_moves(const NTuple& nt, const State& s, float gamma, std::vector<Cand>& out) {
    const Tables& t = nt.tables();
    const bool cooldown = in_cooldown(t, s.history);
    out.clear();
    const BoardParts parts = nt.parts(s.board);
    for (int slot = 0; slot < 3; slot++) {
        const int piece = s.pending[slot];
        const int r0 = s.pending[slot == 0 ? 1 : 0], r1 = s.pending[slot == 2 ? 1 : 2];
        for_each_bit(parts.legal[piece], [&](int pos) {
            const Placed p = apply(t, s.board, piece, pos);
            const double v = nt.value(nt.child_eval(parts, piece, pos, p), cooldown, r0, r1);
            out.push_back({slot, pos, p.points, p.board, r0, r1, v, p.points + gamma * v});
        });
    }
}

inline void known_placements(const NTuple& nt, Board board, const BoardParts& parts, int piece, int other,
                             std::vector<Known>& out) {
    const Tables& t = nt.tables();
    out.clear();
    for_each_bit(parts.legal[piece], [&](int pos) {
        const Placed p = apply(t, board, piece, pos);
        Known k{pos, p.points, p.board, nt.child_eval(parts, piece, pos, p), {}};
        const float w = nt.rem_weight(k.e, other);
        for (int cd = 0; cd < 2; cd++) {
            double v = nt.fit_prefix(k.e, cd);
            v += w;
            k.pre[cd] = v;
        }
        out.push_back(k);
    });
}

// The moves of the child with board ``board`` (parts: nt.parts(board)) and pending (rem0, rem1, x), in engine
// order, as emit(Cand). k0 and k1 hold the placements of rem0 and rem1; with skip_slot1 the slot-1 moves are left
// out (they repeat slot 0 when rem0 == rem1, which only a max may ignore).
template <class F>
inline void child_moves(const NTuple& nt, Board board, const BoardParts& parts, int rem0, int rem1, int x, bool cooldown,
                        float gamma, const std::vector<Known>& k0, const std::vector<Known>& k1, bool skip_slot1, F&& emit) {
    for (const Known& k : k0) {
        double v = k.pre[cooldown];
        v += nt.rem_weight(k.e, x);
        emit(Cand{0, k.pos, k.points, k.board, rem1, x, v, k.points + gamma * v});
    }
    if (!skip_slot1) {
        for (const Known& k : k1) {
            double v = k.pre[cooldown];
            v += nt.rem_weight(k.e, x);
            emit(Cand{1, k.pos, k.points, k.board, rem0, x, v, k.points + gamma * v});
        }
    }
    const Tables& t = nt.tables();
    for_each_bit(parts.legal[x], [&](int pos) {
        const Placed p = apply(t, board, x, pos);
        const double v = nt.value(nt.child_eval(parts, x, pos, p), cooldown, rem0, rem1);
        emit(Cand{2, pos, p.points, p.board, rem0, rem1, v, p.points + gamma * v});
    });
}

// Indices of the top k moves by Q1 (k = 0 or k >= size: all), ties in engine order.
inline std::vector<int> top_k(const std::vector<Cand>& moves, int k) {
    std::vector<int> idx(moves.size());
    std::iota(idx.begin(), idx.end(), 0);
    for (const Cand& c : moves) check_finite(c.q1);  // the sort needs an order
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return moves[a].q1 > moves[b].q1; });
    if (k > 0 && k < static_cast<int>(idx.size())) idx.resize(k);
    return idx;
}

inline double expectation(const NTuple& nt, Board after, int rem0, int rem1, int64_t parent_score,
                          const int parent_history[3], int points, float gamma, std::span<const int> widths);

// The part of E(parent, m, widths) shared by every spawned piece x: the move m with after-state board ``after``,
// remaining (rem0, rem1) and reward ``points``, the spawn distribution and the placements of rem0 and rem1.
struct Expect {
    Board after = 0;
    int rem0 = 0, rem1 = 0, points = 0;
    int64_t score = 0;  // after m
    int parent_history[3] = {};
    std::span<const int> widths;
    bool skip_slot1 = false;
    double p[MAX_PIECES] = {};
    BoardParts parts;
    std::vector<Known> k0, k1;
};

inline void expect_init(const NTuple& nt, Expect& e, Board after, int rem0, int rem1, int64_t parent_score,
                        const int parent_history[3], int points, std::span<const int> widths) {
    spawn_distribution(nt.tables(), parent_score, parent_history, e.p);
    e.after = after, e.rem0 = rem0, e.rem1 = rem1, e.points = points, e.score = parent_score + points;
    std::copy_n(parent_history, 3, e.parent_history);
    e.widths = widths;
    e.skip_slot1 = widths.empty() && rem0 == rem1;
    e.parts = nt.parts(after);
    known_placements(nt, after, e.parts, rem0, rem1, e.k0);
    if (!e.skip_slot1) known_placements(nt, after, e.parts, rem1, rem0, e.k1);
}

// B of the child where x spawned (x must have P(x) > 0): 0 when it has no legal move.
inline double spawned_value(const NTuple& nt, const Expect& e, int x, float gamma) {
    const Tables& t = nt.tables();
    const int history[3] = {t.group[x], e.parent_history[0], e.parent_history[1]};
    const bool cooldown = in_cooldown(t, history);
    double b = 0;
    bool found = false;
    if (e.widths.empty()) {
        child_moves(nt, e.after, e.parts, e.rem0, e.rem1, x, cooldown, gamma, e.k0, e.k1, e.skip_slot1, [&](const Cand& c) {
            if (!found || c.q1 > b) b = c.q1, found = true;
        });
        return b;
    }
    std::vector<Cand> moves;
    child_moves(nt, e.after, e.parts, e.rem0, e.rem1, x, cooldown, gamma, e.k0, e.k1, false,
                [&](const Cand& c) { moves.push_back(c); });
    for (const int i : top_k(moves, e.widths[0])) {
        const Cand& c = moves[i];
        const double v = expectation(nt, c.board, c.rem0, c.rem1, e.score, history, c.points, gamma, e.widths.subspan(1));
        if (!found || v > b) b = v, found = true;
    }
    return b;
}

// E from the B of each spawned piece (b[x], read only where P(x) > 0), summed in piece order.
inline double expect_total(const Expect& e, const Tables& t, const double* b, float gamma) {
    double total = 0;
    for (int x = 0; x < t.n_pieces; x++)
        if (e.p[x] != 0) total += e.p[x] * b[x];
    return e.points + gamma * total;
}

// E(parent, m, widths) for the move m with after-state board ``after``, remaining (rem0, rem1) and reward ``points``.
inline double expectation(const NTuple& nt, Board after, int rem0, int rem1, int64_t parent_score,
                          const int parent_history[3], int points, float gamma, std::span<const int> widths) {
    Expect e;
    expect_init(nt, e, after, rem0, rem1, parent_score, parent_history, points, widths);
    double b[MAX_PIECES];
    for (int x = 0; x < nt.tables().n_pieces; x++)
        if (e.p[x] != 0) b[x] = spawned_value(nt, e, x, gamma);
    return expect_total(e, nt.tables(), b, gamma);
}

// The root's kept moves (top widths[0] by Q1), in engine order.
inline std::vector<int> kept_root_moves(const std::vector<Cand>& root, int k) {
    std::vector<int> kept = top_k(root, k);
    std::sort(kept.begin(), kept.end());
    return kept;
}

}  // namespace search_detail

// Every kept root move with its Q1 and its search score E(s, m, widths[1:]), in engine order; empty with no legal move.
inline std::vector<Scored> search_scores(const NTuple& nt, const State& s, float gamma, const std::vector<int>& widths) {
    using namespace search_detail;
    check_widths(widths);
    std::vector<Cand> root;
    root_moves(nt, s, gamma, root);
    std::vector<Scored> out;
    if (root.empty()) return out;
    for (const int i : kept_root_moves(root, widths[0])) {
        const Cand& c = root[i];
        const double e = expectation(nt, c.board, c.rem0, c.rem1, s.score, s.history, c.points, gamma,
                                     std::span<const int>(widths).subspan(1));
        out.push_back({c.slot, c.pos, c.points, c.q1, e});
    }
    return out;
}

// Lets idle threads join the searches of running games (evaluate_search, once fewer games than threads are
// left). A driver publishes a loop of n tasks in its slot; it and any idle thread take tasks one at a time. Each
// task writes its own output, so the results do not depend on which thread ran what.
class Helpers {
public:
    explicit Helpers(int slots) : slots_(slots) {}

    // f(i) for every i in [0, n), on the caller and on idle threads; returns when all have run, rethrowing the
    // first exception of any task.
    template <class F>
    void run(int slot, size_t n, F& f) {
        Job job;
        job.n = n;
        job.ctx = &f;
        job.call = [](void* c, size_t i) { (*static_cast<F*>(c))(i); };
        Slot& s = slots_[slot];
        s.job.store(&job);
        while (take(job)) {}
        while (job.done.load() < n) pause();
        s.job.store(nullptr);
        while (s.visitors.load() > 0) pause();  // a thread that saw the job may still be reading job.next
        if (job.error) std::rethrow_exception(job.error);
    }

    // An idle thread's loop: runs tasks of any slot until done() is true.
    template <class Done>
    void help(Done&& done) {
        for (int idle = 0; !done();) {
            bool worked = false;
            for (Slot& s : slots_) {
                if (!s.job.load(std::memory_order_relaxed)) continue;
                s.visitors.fetch_add(1);  // seq_cst with run()'s store and load: run waits for us, or we see nullptr
                if (Job* job = s.job.load()) worked |= take(*job);
                s.visitors.fetch_sub(1);
            }
            if (worked) idle = 0;
            else if (++idle < 64) pause();
            else std::this_thread::yield();
        }
    }

private:
    struct Job {
        size_t n = 0;
        std::atomic<size_t> next{0}, done{0};
        void* ctx = nullptr;
        void (*call)(void*, size_t) = nullptr;
        std::atomic<bool> failed{false};
        std::exception_ptr error;  // written by the first task that throws
    };
    struct alignas(64) Slot {
        std::atomic<Job*> job{nullptr};
        std::atomic<int> visitors{0};
    };
    std::vector<Slot> slots_;

    // Runs one task of the job; false when none was left.
    static bool take(Job& job) {
        if (job.next.load(std::memory_order_relaxed) >= job.n) return false;
        const size_t i = job.next.fetch_add(1);
        if (i >= job.n) return false;
        try {
            job.call(job.ctx, i);
        } catch (...) {
            if (!job.failed.exchange(true)) job.error = std::current_exception();
        }
        job.done.fetch_add(1);
        return true;
    }

    static void pause() {
#if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#endif
    }
};

// f(i) for i in [0, n): shared with idle threads through helpers, or on this thread alone without them.
template <class F>
inline void run_tasks(Helpers* helpers, int slot, size_t n, F&& f) {
    if (helpers && n > 1) {
        helpers->run(slot, n, f);
    } else {
        for (size_t i = 0; i < n; i++) f(i);
    }
}

// The searched move: the last tier whose below exceeds the root's best Q1 picks the widths, and the kept move
// with the highest search score is played (first in engine order on ties). With no such tier it plays
// best_move's move, bit for bit. tier is the index used (-1: 1-ply); changed says the move differs from
// best_move's. out.q is the search score (Q1 at tier -1); out.after and out.v are the chosen move's; best_q1,
// if given, receives the root's best Q1. Returns false with no legal move. Tiers must pass check_tiers. With
// helpers, idle threads may take part of the search (from this thread's slot); the result is the same.
inline bool search_move(const NTuple& nt, const State& s, float gamma, const std::vector<Tier>& tiers, Choice& out,
                        int& tier, bool& changed, double* best_q1 = nullptr, Helpers* helpers = nullptr, int slot = 0) {
    using namespace search_detail;
    std::vector<Cand> root;
    root_moves(nt, s, gamma, root);
    if (root.empty()) return false;
    int one = 0;  // best_move's choice: the first maximum of Q1
    for (int i = 1; i < static_cast<int>(root.size()); i++)
        if (root[i].q1 > root[one].q1) one = i;
    if (best_q1) *best_q1 = root[one].q1;
    tier = -1;
    for (int i = static_cast<int>(tiers.size()) - 1; i >= 0; i--) {
        if (tiers[i].below > root[one].q1) {
            tier = i;
            break;
        }
    }
    int pick = one;
    double q = root[one].q1;
    if (tier >= 0) {
        // expectation() of each kept move, as tasks: first the parts shared by the spawned pieces, then one per piece
        const Tables& t = nt.tables();
        const std::vector<int>& widths = tiers[tier].widths;
        const std::vector<int> kept = kept_root_moves(root, widths[0]);
        std::vector<Expect> ex(kept.size());
        run_tasks(helpers, slot, kept.size(), [&](size_t j) {
            const Cand& c = root[kept[j]];
            expect_init(nt, ex[j], c.board, c.rem0, c.rem1, s.score, s.history, c.points,
                        std::span<const int>(widths).subspan(1));
        });
        std::vector<std::pair<int, int>> tasks;  // (kept move, spawned piece)
        for (size_t j = 0; j < kept.size(); j++)
            for (int x = 0; x < t.n_pieces; x++)
                if (ex[j].p[x] != 0) tasks.emplace_back(static_cast<int>(j), x);
        std::vector<std::array<double, MAX_PIECES>> b(kept.size());
        run_tasks(helpers, slot, tasks.size(), [&](size_t i) {
            const auto [j, x] = tasks[i];
            b[j][x] = spawned_value(nt, ex[j], x, gamma);
        });
        bool found = false;
        for (size_t j = 0; j < kept.size(); j++) {
            const double e = expect_total(ex[j], t, b[j].data(), gamma);
            if (!found || e > q) q = e, pick = kept[j], found = true;
        }
    }
    const Cand& c = root[pick];
    out.slot = c.slot, out.pos = c.pos, out.points = c.points;
    out.after.board = c.board, out.after.remaining[0] = c.rem0, out.after.remaining[1] = c.rem1;
    out.after.cooldown = in_cooldown(nt.tables(), s.history);
    out.v = c.v, out.q = q;
    changed = pick != one;
    return true;
}

// ---- evaluation ---------------------------------------------------------------------------------

struct GameSearch {
    GameOutcome outcome;
    std::vector<uint64_t> searched, changed;  // moves per tier that used it / where it changed the 1-ply move
    double seconds = 0;
    std::vector<double> last_q1;  // the root's best Q1 of the last <= 20 moves played, oldest first
};

// Searched games, scheduled, seeded and capped like evaluate(): game g uses stream_seeds(seed, games)[g], and a game
// is truncated when it has a legal move at max_moves. Games listed in ``skip`` are not played. on_game(g, result)
// runs once per finished game, under a mutex. An exception in a worker or in on_game stops every game before its
// next move and is rethrown. Returns the games played, in game order.
inline std::vector<std::pair<int, GameSearch>> evaluate_search(const NTuple& nt, int games, uint64_t seed, int64_t max_moves,
                                                               int threads, float gamma, const std::vector<Tier>& tiers,
                                                               const std::vector<int>& skip,
                                                               const std::function<void(int, const GameSearch&)>& on_game) {
    if (threads < 1) throw std::invalid_argument("threads must be >= 1");
    if (games < 0) throw std::invalid_argument("games must be >= 0");
    check_tiers(tiers);
    std::vector<char> skipped(games, 0);
    for (const int g : skip) {
        if (g < 0 || g >= games) throw std::invalid_argument("skip: game index out of range");
        skipped[g] = 1;
    }
    std::vector<std::pair<int, GameSearch>> out;
    for (int g = 0; g < games; g++)
        if (!skipped[g]) out.push_back({g, {}});
    const std::vector<uint64_t> seeds = stream_seeds(seed, games);
    const Tables& t = nt.tables();
    std::atomic<size_t> next{0}, finished{0};
    std::atomic<bool> stop{false};
    std::mutex m;
    Helpers helpers(threads);  // a thread with no game left helps the others
    parallel(threads, [&](int worker) {
        try {
            for (size_t i; (i = next.fetch_add(1)) < out.size();) {
                const auto start = std::chrono::steady_clock::now();
                const int g = out[i].first;
                GameSearch r;
                r.searched.assign(tiers.size(), 0);
                r.changed.assign(tiers.size(), 0);
                std::array<double, 20> ring;
                size_t played = 0;
                Xoshiro256 rng(seeds[g]);
                State s;
                reset(t, s, rng);
                Choice c;
                int tier;
                bool changed, truncated = false;
                double q1;
                for (;;) {
                    if (stop.load(std::memory_order_relaxed)) return;
                    if (s.moves >= max_moves) {  // checked first, so the move at the cap is not searched
                        truncated = has_legal_move(t, s);
                        break;
                    }
                    if (!search_move(nt, s, gamma, tiers, c, tier, changed, &q1, &helpers, worker)) break;
                    check_finite(c.q);
                    if (tier >= 0) r.searched[tier]++, r.changed[tier] += changed;
                    ring[played++ % ring.size()] = q1;
                    step(t, s, c.slot, c.pos, rng);
                }
                for (size_t k = played > ring.size() ? played - ring.size() : 0; k < played; k++)
                    r.last_q1.push_back(ring[k % ring.size()]);
                r.outcome = {s.score, s.moves, s.lines, truncated};
                r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                out[i].second = std::move(r);
                {
                    std::lock_guard lock(m);
                    if (stop.load()) return;
                    if (on_game) on_game(g, out[i].second);
                }
                finished.fetch_add(1);
            }
            helpers.help([&] { return finished.load() == out.size() || stop.load(); });
        } catch (...) {
            stop.store(true);
            throw;
        }
    });
    return out;
}

}  // namespace bpa
