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
    for (int slot = 0; slot < 3; slot++) {
        const int piece = s.pending[slot];
        const int r0 = s.pending[slot == 0 ? 1 : 0], r1 = s.pending[slot == 2 ? 1 : 2];
        for_each_bit(legal_positions(t, s.board, piece), [&](int pos) {
            const Placed p = apply(t, s.board, piece, pos);
            const double v = nt.value(nt.board_eval(p.board), cooldown, r0, r1);
            out.push_back({slot, pos, p.points, p.board, r0, r1, v, p.points + gamma * v});
        });
    }
}

inline void known_placements(const NTuple& nt, Board board, int piece, int other, std::vector<Known>& out) {
    const Tables& t = nt.tables();
    out.clear();
    for_each_bit(legal_positions(t, board, piece), [&](int pos) {
        const Placed p = apply(t, board, piece, pos);
        Known k{pos, p.points, p.board, nt.board_eval(p.board), {}};
        const float w = nt.rem_weight(k.e, other);
        for (int cd = 0; cd < 2; cd++) {
            double v = nt.fit_prefix(k.e, cd);
            v += w;
            k.pre[cd] = v;
        }
        out.push_back(k);
    });
}

// The moves of the child with board ``board`` and pending (rem0, rem1, x), in engine order, as emit(Cand).
// k0 and k1 hold the placements of rem0 and rem1; with skip_slot1 the slot-1 moves are left out (they
// repeat slot 0 when rem0 == rem1, which only a max may ignore).
template <class F>
inline void child_moves(const NTuple& nt, Board board, int rem0, int rem1, int x, bool cooldown, float gamma,
                        const std::vector<Known>& k0, const std::vector<Known>& k1, bool skip_slot1, F&& emit) {
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
    for_each_bit(legal_positions(t, board, x), [&](int pos) {
        const Placed p = apply(t, board, x, pos);
        const double v = nt.value(nt.board_eval(p.board), cooldown, rem0, rem1);
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

// E(parent, m, widths) for the move m with after-state board ``after``, remaining (rem0, rem1) and reward ``points``.
inline double expectation(const NTuple& nt, Board after, int rem0, int rem1, int64_t parent_score,
                          const int parent_history[3], int points, float gamma, std::span<const int> widths) {
    const Tables& t = nt.tables();
    double p[MAX_PIECES];
    spawn_distribution(t, parent_score, parent_history, p);
    const bool last = widths.empty();
    const bool skip_slot1 = last && rem0 == rem1;
    std::vector<Known> k0, k1;
    known_placements(nt, after, rem0, rem1, k0);
    if (!skip_slot1) known_placements(nt, after, rem1, rem0, k1);
    const int64_t score = parent_score + points;
    std::vector<Cand> moves;
    double total = 0;
    for (int x = 0; x < t.n_pieces; x++) {
        if (p[x] == 0) continue;
        const int history[3] = {t.group[x], parent_history[0], parent_history[1]};
        const bool cooldown = in_cooldown(t, history);
        double b = 0;  // B(child): 0 when the child has no legal move
        if (last) {
            bool found = false;
            child_moves(nt, after, rem0, rem1, x, cooldown, gamma, k0, k1, skip_slot1, [&](const Cand& c) {
                if (!found || c.q1 > b) b = c.q1, found = true;
            });
        } else {
            moves.clear();
            child_moves(nt, after, rem0, rem1, x, cooldown, gamma, k0, k1, false, [&](const Cand& c) { moves.push_back(c); });
            bool found = false;
            for (const int i : top_k(moves, widths[0])) {
                const Cand& c = moves[i];
                const double e = expectation(nt, c.board, c.rem0, c.rem1, score, history, c.points, gamma, widths.subspan(1));
                if (!found || e > b) b = e, found = true;
            }
        }
        total += p[x] * b;
    }
    return points + gamma * total;
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

// The searched move: the last tier whose below exceeds the root's best Q1 picks the widths, and the kept move
// with the highest search score is played (first in engine order on ties). With no such tier it plays
// best_move's move, bit for bit. tier is the index used (-1: 1-ply); changed says the move differs from
// best_move's. out.q is the search score (Q1 at tier -1); out.after and out.v are the chosen move's; best_q1,
// if given, receives the root's best Q1. Returns false with no legal move. Tiers must pass check_tiers.
inline bool search_move(const NTuple& nt, const State& s, float gamma, const std::vector<Tier>& tiers, Choice& out,
                        int& tier, bool& changed, double* best_q1 = nullptr) {
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
        const std::vector<int>& widths = tiers[tier].widths;
        bool found = false;
        for (const int i : kept_root_moves(root, widths[0])) {
            const Cand& c = root[i];
            const double e = expectation(nt, c.board, c.rem0, c.rem1, s.score, s.history, c.points, gamma,
                                         std::span<const int>(widths).subspan(1));
            if (!found || e > q) q = e, pick = i, found = true;
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
    std::atomic<size_t> next{0};
    std::atomic<bool> stop{false};
    std::mutex m;
    parallel(threads, [&](int) {
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
                    if (!search_move(nt, s, gamma, tiers, c, tier, changed, &q1)) break;
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
                std::lock_guard lock(m);
                if (stop.load()) return;
                if (on_game) on_game(g, out[i].second);
            }
        } catch (...) {
            stop.store(true);
            throw;
        }
    });
    return out;
}

}  // namespace bpa
