// Self-play TD(0) training of the n-tuple value (docs/superpowers/specs/2026-11-03-ntuple-learner-design.md), with
// optional search-backed targets (docs/superpowers/specs/2026-11-07-search-targets-design.md).
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "engine.hpp"
#include "ntuple.hpp"
#include "search.hpp"

namespace bpa {

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
// With search_widths, every step is search_learn's (search-backed targets) and search_play picks the searched move;
// without, the target is the next state's best Q1, as before.
class Trainer {
public:
    static constexpr int SNAPSHOT_EVERY = 50, RING = 10, POOL = 4096;
    static constexpr int64_t POOL_WINDOW = 500;

    Trainer(NTuple& nt, float gamma, float alpha, double restart_frac, int threads, uint64_t seed,
            std::vector<int> search_widths = {}, bool search_play = false)
        : alpha(alpha), nt_(nt), gamma_(gamma), restart_frac_(restart_frac), widths_(std::move(search_widths)),
          search_play_(search_play) {
        if (threads < 1) throw std::invalid_argument("threads must be >= 1");
        if (restart_frac < 0 || restart_frac > 1) throw std::invalid_argument("restart_frac must be in [0, 1]");
        if (!widths_.empty()) check_widths(widths_);
        if (search_play && widths_.empty()) throw std::invalid_argument("search_play needs search_widths");
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
    std::vector<int> widths_;  // empty: 1-ply targets
    bool search_play_;
    std::vector<Worker> workers_;

    // The move to play in s without learning: best_move's, or with search targets pick_scored's. False with no
    // legal move.
    bool choose(const State& s, Choice& out) const {
        if (widths_.empty()) return best_move(nt_, s, gamma_, out);
        const std::vector<Scored> kept = search_scores(nt_, s, gamma_, widths_);
        if (kept.empty()) return false;
        out = pick_scored(nt_, s, kept, search_play_);
        return true;
    }

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
        if (!choose(w.s, w.choice)) {  // unreachable (pool states have a legal move), but never play a stale move
            reset(t, w.s, w.rng);
            w.fresh = true;
            choose(w.s, w.choice);  // an empty board always fits a piece
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
            const bool alive = widths_.empty() ? best_move(nt_, w.s, gamma_, next)
                                               : search_learn(nt_, a, w.s, gamma_, widths_, alpha, search_play_, next);
            if (alive) {
                check_finite(next.q);
                if (w.steps++ % SNAPSHOT_EVERY == 0) {  // only states with a legal move may become restart states
                    w.ring[w.ring_head] = {w.s, w.game};
                    w.ring_head = (w.ring_head + 1) % RING;
                    w.ring_size = std::min(w.ring_size + 1, RING);
                }
                if (widths_.empty()) nt_.learn(a, next.q, alpha);  // search_learn has learned already
                w.choice = next;
            } else {
                if (widths_.empty()) nt_.learn(a, 0.0, alpha);
                game_over(w, st);
                new_game(w, st);
            }
            st.value_sum += w.choice.v;
        }
        st.moves = moves;
        return st;
    }
};

}  // namespace bpa
