// Python bindings: block_puzzle_ai._engine (wrapped by block_puzzle_ai.engine)
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <optional>
#include <stdexcept>
#include <tuple>
#include <variant>
#include <vector>

#include "engine.hpp"
#include "ntuple.hpp"
#include "rng.hpp"
#include "search.hpp"
#include "trainer.hpp"

namespace nb = nanobind;
using bpa::Tables;

namespace {

uint32_t check_n(uint32_t n) {
    if (n == 0) throw std::invalid_argument("randrange: n must be >= 1");
    return n;
}

// history as passed from Python: exactly 3 group ids, -1 = empty
void read_history(const Tables& t, const std::vector<int>& history, int out[3]) {
    if (history.size() != 3) throw std::invalid_argument("history must have 3 entries (-1 = empty)");
    for (int i = 0; i < 3; i++) {
        if (history[i] < -1 || history[i] >= t.n_groups) throw std::invalid_argument("history entry out of range");
        out[i] = history[i];
    }
}

int64_t check_score(int64_t score) {
    if (score < 0) throw std::invalid_argument("score must be >= 0");
    return score;
}

// A state as passed from Python; moves and lines start at 0.
bpa::State make_state(const Tables& t, uint64_t lo, uint64_t hi, const std::vector<int>& pending, int64_t score,
                      const std::vector<int>& history) {
    if (hi >> (bpa::CELLS - 64)) throw std::invalid_argument("board has bits beyond cell 99");
    if (pending.size() != 3) throw std::invalid_argument("pending must have 3 piece ids");
    for (int p : pending)
        if (p < 0 || p >= t.n_pieces) throw std::invalid_argument("pending piece id out of range");
    bpa::State s;
    read_history(t, history, s.history);
    s.board = static_cast<bpa::Board>(hi) << 64 | lo;
    std::copy(pending.begin(), pending.end(), s.pending);
    s.score = check_score(score);
    return s;
}

using PyTiers = std::vector<std::tuple<double, std::vector<int>>>;  // (below, widths); below may be inf

std::vector<bpa::Tier> read_tiers(const PyTiers& tiers) {
    std::vector<bpa::Tier> out;
    for (const auto& [below, widths] : tiers) out.push_back({below, widths});
    bpa::check_tiers(out);
    return out;
}

// One game with its own tables and random source (PyRandom replays game.Game(seed) exactly).
class Game {
public:
    using Move = std::tuple<int, int, int>;  // slot, row, col
    using Afterstate = std::tuple<int, int, int, uint64_t, uint64_t, int, int>;  // move, board lo/hi, points, lines

    Game(const Tables& t, uint64_t seed, bool fast_rng) : t_(t) {
        if (fast_rng) rng_.emplace<bpa::Xoshiro256>(seed);
        else rng_.emplace<bpa::PyRandom>(seed);
        reset();
    }

    void reset() {
        std::visit([&](auto& rng) { bpa::reset(t_, s_, rng); }, rng_);
    }

    void set_state(uint64_t lo, uint64_t hi, const std::vector<int>& pending, int64_t score, const std::vector<int>& history) {
        s_ = make_state(t_, lo, hi, pending, score, history);
    }

    std::tuple<int, int, bool> step(int slot, int row, int col) {
        if (done()) throw std::invalid_argument("game is over");
        if (row < 0 || row >= bpa::N || col < 0 || col >= bpa::N) throw std::invalid_argument("move off the board");
        const bpa::Placed a = std::visit([&](auto& rng) { return bpa::step(t_, s_, slot, row * bpa::N + col, rng); }, rng_);
        return {a.points, a.lines, done()};
    }

    std::vector<Move> legal_moves() const {
        std::vector<Move> moves;
        for (int slot = 0; slot < 3; slot++)
            bpa::for_each_bit(bpa::legal_positions(t_, s_.board, s_.pending[slot]),
                              [&](int pos) { moves.emplace_back(slot, pos / bpa::N, pos % bpa::N); });
        return moves;
    }

    std::vector<Afterstate> afterstates() const {
        std::vector<Afterstate> out;
        for (int slot = 0; slot < 3; slot++) {
            const int piece = s_.pending[slot];
            bpa::for_each_bit(bpa::legal_positions(t_, s_.board, piece), [&](int pos) {
                const bpa::Placed a = bpa::apply(t_, s_.board, piece, pos);
                out.emplace_back(slot, pos / bpa::N, pos % bpa::N, static_cast<uint64_t>(a.board),
                                 static_cast<uint64_t>(a.board >> 64), a.points, a.lines);
            });
        }
        return out;
    }

    std::vector<double> spawn_distribution() const {
        std::vector<double> out(t_.n_pieces);
        bpa::spawn_distribution(t_, s_.score, s_.history, out.data());
        return out;
    }

    uint64_t board_lo() const { return static_cast<uint64_t>(s_.board); }
    uint64_t board_hi() const { return static_cast<uint64_t>(s_.board >> 64); }
    std::vector<int> pending() const { return {s_.pending, s_.pending + 3}; }
    std::vector<int> history() const { return {s_.history, s_.history + 3}; }
    int64_t score() const { return s_.score; }
    int64_t moves() const { return s_.moves; }
    int64_t lines() const { return s_.lines; }
    bool done() const { return !bpa::has_legal_move(t_, s_); }

private:
    Tables t_;
    bpa::State s_;
    std::variant<bpa::PyRandom, bpa::Xoshiro256> rng_{std::in_place_type<bpa::Xoshiro256>, 0};
};

bpa::Afterstate make_afterstate(const Tables& t, uint64_t lo, uint64_t hi, const std::vector<int>& remaining, bool cooldown) {
    if (hi >> (bpa::CELLS - 64)) throw std::invalid_argument("board has bits beyond cell 99");
    if (remaining.size() != 2) throw std::invalid_argument("remaining must hold 2 piece ids");
    bpa::Afterstate a;
    a.board = static_cast<bpa::Board>(hi) << 64 | lo;
    for (int i = 0; i < 2; i++) {
        if (remaining[i] < 0 || remaining[i] >= t.n_pieces) throw std::invalid_argument("remaining piece id out of range");
        a.remaining[i] = remaining[i];
    }
    a.cooldown = cooldown;
    return a;
}

}  // namespace

NB_MODULE(_engine, m) {
    m.doc() = "C++ engine for the block puzzle (see docs/superpowers/specs/2026-11-03-cpp-engine-design.md)";

    nb::class_<bpa::PyRandom>(m, "PyRandom", "Bit-exact port of CPython's random.Random")
        .def(nb::init<uint64_t>(), nb::arg("seed"))
        .def("randrange", [](bpa::PyRandom& r, uint32_t n) { return r.randrange(check_n(n)); }, nb::arg("n"))
        .def(
            "getrandbits",
            [](bpa::PyRandom& r, int k) {
                if (k < 1 || k > 32) throw std::invalid_argument("getrandbits: k must be in 1..32");
                return r.getrandbits(k);
            },
            nb::arg("k"));

    nb::class_<bpa::Xoshiro256>(m, "Xoshiro256", "xoshiro256** seeded through splitmix64")
        .def(nb::init<uint64_t>(), nb::arg("seed"))
        .def("randrange", [](bpa::Xoshiro256& r, uint32_t n) { return r.randrange(check_n(n)); }, nb::arg("n"));

    nb::class_<Tables>(m, "Tables", "Piece and spawn tables; cells[p] = offsets r * 10 + c, ids = all_pieces() order")
        .def(
            "__init__",
            [](Tables* self, const std::vector<std::vector<int>>& cells, const std::vector<int>& groups,
               const std::vector<int>& hard_groups) { new (self) Tables(bpa::make_tables(cells, groups, hard_groups)); },
            nb::arg("cells"), nb::arg("groups"), nb::arg("hard_groups"))
        .def_ro("n_pieces", &Tables::n_pieces)
        .def_ro("n_groups", &Tables::n_groups);

    nb::class_<Game>(m, "Game", "One game; PyRandom (default) replays game.Game(seed), fast_rng uses xoshiro256**")
        .def(nb::init<const Tables&, uint64_t, bool>(), nb::arg("tables"), nb::arg("seed"), nb::arg("fast_rng") = false)
        .def("reset", &Game::reset)
        .def("set_state", &Game::set_state, nb::arg("board_lo"), nb::arg("board_hi"), nb::arg("pending"),
             nb::arg("score"), nb::arg("history"), "moves and lines restart at 0")
        .def("step", &Game::step, nb::arg("slot"), nb::arg("row"), nb::arg("col"), "-> (points, lines, done)")
        .def("legal_moves", &Game::legal_moves, "(slot, row, col), slot-major then position ascending")
        .def("afterstates", &Game::afterstates, "(slot, row, col, board_lo, board_hi, points, lines) per legal move")
        .def("spawn_distribution", &Game::spawn_distribution)
        .def_prop_ro("board_lo", &Game::board_lo)
        .def_prop_ro("board_hi", &Game::board_hi)
        .def_prop_ro("pending", &Game::pending)
        .def_prop_ro("history", &Game::history)
        .def_prop_ro("score", &Game::score)
        .def_prop_ro("moves", &Game::moves)
        .def_prop_ro("lines_cleared", &Game::lines)
        .def_prop_ro("done", &Game::done);

    nb::class_<bpa::NTuple>(m, "NTuple", "n-tuple after-state value function over shared float32 weights")
        .def(nb::init<const Tables&, std::vector<std::vector<int>>>(), nb::arg("tables"), nb::arg("tuples"))
        .def_prop_ro("n_weights", &bpa::NTuple::n_weights)
        .def_prop_ro("n_lookups", &bpa::NTuple::n_lookups)
        .def_prop_ro("tuples", &bpa::NTuple::tuples)
        .def_prop_ro(
            "weights",
            [](bpa::NTuple& n) {
                return nb::ndarray<nb::numpy, float, nb::ndim<1>>(n.weights(), {n.n_weights()}, nb::handle());
            },
            nb::rv_policy::reference_internal, "writable float32 view of the weights (keeps the model alive)")
        .def(
            "value",
            [](const bpa::NTuple& n, uint64_t lo, uint64_t hi, const std::vector<int>& remaining, bool cooldown) {
                return n.value(make_afterstate(n.tables(), lo, hi, remaining, cooldown));
            },
            nb::arg("board_lo"), nb::arg("board_hi"), nb::arg("remaining"), nb::arg("cooldown"))
        .def(
            "update",
            [](bpa::NTuple& n, uint64_t lo, uint64_t hi, const std::vector<int>& remaining, bool cooldown, double target,
               float alpha) { n.update(make_afterstate(n.tables(), lo, hi, remaining, cooldown), target, alpha); },
            nb::arg("board_lo"), nb::arg("board_hi"), nb::arg("remaining"), nb::arg("cooldown"), nb::arg("target"),
            nb::arg("alpha"), "TD step: each looked-up weight += alpha * (target - V) / n_lookups")
        .def("enable_tc", &bpa::NTuple::enable_tc, "allocate temporal-coherence state; the trainer then uses update_tc")
        .def_prop_ro("tc_enabled", &bpa::NTuple::tc_enabled)
        .def_prop_ro(
            "tc_state",
            [](bpa::NTuple& n) {
                if (!n.tc_enabled()) throw std::runtime_error("temporal coherence is not enabled");
                return nb::ndarray<nb::numpy, double, nb::ndim<2>>(n.tc_state(), {n.n_weights(), 2}, nb::handle());
            },
            nb::rv_policy::reference_internal, "writable float64 view of the (E, A) pair per weight")
        .def(
            "update_tc",
            [](bpa::NTuple& n, uint64_t lo, uint64_t hi, const std::vector<int>& remaining, bool cooldown, double target,
               float alpha) {
                if (!n.tc_enabled()) throw std::runtime_error("temporal coherence is not enabled");
                n.update_tc(make_afterstate(n.tables(), lo, hi, remaining, cooldown), target, alpha);
            },
            nb::arg("board_lo"), nb::arg("board_hi"), nb::arg("remaining"), nb::arg("cooldown"), nb::arg("target"),
            nb::arg("alpha"), "TD step scaled per weight by |E| / A (1 before its first step)");

    nb::class_<bpa::Trainer>(m, "Trainer", "lock-free multithreaded TD(0) self-play with restarts near game overs")
        .def(nb::init<bpa::NTuple&, float, float, double, int, uint64_t, std::vector<int>, bool>(), nb::arg("ntuple"),
             nb::arg("gamma"), nb::arg("alpha"), nb::arg("restart_frac"), nb::arg("threads"), nb::arg("seed"),
             nb::arg("search_widths") = std::vector<int>{}, nb::arg("search_play") = false, nb::keep_alive<1, 2>(),
             "search_widths: search-backed targets (search_learn) on every step; empty: the next state's best Q1")
        .def_rw("alpha", &bpa::Trainer::alpha)
        .def(
            "run",
            [](bpa::Trainer& tr, uint64_t moves) {
                bpa::TrainStats st;
                {
                    nb::gil_scoped_release release;
                    st = tr.run(moves);
                }
                nb::dict d;
                d["moves"] = st.moves, d["seconds"] = st.seconds, d["games"] = st.games, d["deaths"] = st.deaths;
                d["fresh_games"] = st.fresh_games, d["mean_score"] = st.mean_score(), d["mean_moves"] = st.mean_moves();
                d["restarts"] = st.restarts, d["value_mean"] = st.value_mean();
                return d;
            },
            nb::arg("moves"), "play this many moves in total across the threads, learning as it goes");

    m.def(
        "evaluate_ntuple",
        [](const bpa::NTuple& nt, int games, uint64_t seed, int64_t max_moves, int threads, float gamma) {
            std::vector<bpa::GameOutcome> res;
            {
                nb::gil_scoped_release release;
                res = bpa::evaluate(nt, games, seed, max_moves, threads, gamma);
            }
            std::vector<std::tuple<int64_t, int64_t, int64_t, bool>> out;
            for (const auto& r : res) out.emplace_back(r.score, r.moves, r.lines, r.truncated);
            return out;
        },
        nb::arg("ntuple"), nb::arg("games"), nb::arg("seed"), nb::arg("max_moves"), nb::arg("threads"),
        nb::arg("gamma") = 1.0f, "greedy 1-ply games -> (score, moves, lines, truncated) per game, in game order");

    m.def(
        "search_scores",
        [](const bpa::NTuple& nt, uint64_t lo, uint64_t hi, const std::vector<int>& pending, int64_t score,
           const std::vector<int>& history, float gamma, const std::vector<int>& widths) {
            const bpa::State s = make_state(nt.tables(), lo, hi, pending, score, history);
            bpa::check_widths(widths);
            std::vector<bpa::Scored> res;
            {
                nb::gil_scoped_release release;
                res = bpa::search_scores(nt, s, gamma, widths);
            }
            std::vector<std::tuple<int, int, int, double, double>> out;
            for (const auto& r : res) out.emplace_back(r.slot, r.pos, r.points, r.q1, r.score);
            return out;
        },
        nb::arg("ntuple"), nb::arg("board_lo"), nb::arg("board_hi"), nb::arg("pending"), nb::arg("score"),
        nb::arg("history"), nb::arg("gamma"), nb::arg("widths"),
        "kept root moves (top widths[0] by Q1) in engine order -> (slot, pos, points, q1, search score)");

    m.def(
        "search_learn",
        [](bpa::NTuple& nt, uint64_t prev_lo, uint64_t prev_hi, const std::vector<int>& prev_remaining, bool prev_cooldown,
           uint64_t lo, uint64_t hi, const std::vector<int>& pending, int64_t score, const std::vector<int>& history,
           float gamma, const std::vector<int>& widths, float alpha, bool play_search) -> std::optional<std::tuple<int, int>> {
            const bpa::Afterstate prev = make_afterstate(nt.tables(), prev_lo, prev_hi, prev_remaining, prev_cooldown);
            const bpa::State s = make_state(nt.tables(), lo, hi, pending, score, history);
            bpa::check_widths(widths);
            bpa::Choice c;
            bool found;
            {
                nb::gil_scoped_release release;
                found = bpa::search_learn(nt, prev, s, gamma, widths, alpha, play_search, c);
            }
            if (!found) return std::nullopt;
            return std::make_tuple(c.slot, c.pos);
        },
        nb::arg("ntuple"), nb::arg("prev_lo"), nb::arg("prev_hi"), nb::arg("prev_remaining"), nb::arg("prev_cooldown"),
        nb::arg("board_lo"), nb::arg("board_hi"), nb::arg("pending"), nb::arg("score"), nb::arg("history"), nb::arg("gamma"),
        nb::arg("widths"), nb::arg("alpha"), nb::arg("play_search"),
        "one search-backed TD step at a state reached from the after-state prev -> the move to play (slot, pos), or None");

    m.def(
        "search_move",
        [](const bpa::NTuple& nt, uint64_t lo, uint64_t hi, const std::vector<int>& pending, int64_t score,
           const std::vector<int>& history, float gamma,
           const PyTiers& tiers) -> std::optional<std::tuple<int, int, double, int, bool>> {
            const bpa::State s = make_state(nt.tables(), lo, hi, pending, score, history);
            const std::vector<bpa::Tier> t = read_tiers(tiers);
            bpa::Choice c;
            int tier = -1;
            bool changed = false, found;
            {
                nb::gil_scoped_release release;
                found = bpa::search_move(nt, s, gamma, t, c, tier, changed);
            }
            if (!found) return std::nullopt;
            return std::make_tuple(c.slot, c.pos, c.q, tier, changed);
        },
        nb::arg("ntuple"), nb::arg("board_lo"), nb::arg("board_hi"), nb::arg("pending"), nb::arg("score"),
        nb::arg("history"), nb::arg("gamma"), nb::arg("tiers"),
        "searched move -> (slot, pos, q, tier, changed), or None with no legal move; tier -1 = 1-ply");

    m.def(
        "check_search_tiers", [](const PyTiers& tiers) { read_tiers(tiers); }, nb::arg("tiers"),
        "raise ValueError unless below strictly decreases, depth does not decrease and widths are >= 0");

    m.def(
        "evaluate_ntuple_search",
        [](const bpa::NTuple& nt, int games, uint64_t seed, int64_t max_moves, int threads, float gamma, const PyTiers& tiers,
           const std::vector<int>& skip, nb::object on_game) {
            using Row = std::tuple<int, int64_t, int64_t, int64_t, bool, std::vector<uint64_t>, std::vector<uint64_t>, double,
                                   std::vector<double>>;
            auto row = [](int g, const bpa::GameSearch& r) {
                return Row{g, r.outcome.score, r.outcome.moves, r.outcome.lines, r.outcome.truncated, r.searched, r.changed,
                           r.seconds, r.last_q1};
            };
            const std::vector<bpa::Tier> t = read_tiers(tiers);
            std::function<void(int, const bpa::GameSearch&)> callback;
            if (!on_game.is_none()) {
                callback = [&](int g, const bpa::GameSearch& r) {
                    nb::gil_scoped_acquire acquire;
                    std::apply([&](auto&&... a) { on_game(a...); }, row(g, r));
                };
            }
            std::vector<std::pair<int, bpa::GameSearch>> res;
            {
                nb::gil_scoped_release release;
                res = bpa::evaluate_search(nt, games, seed, max_moves, threads, gamma, t, skip, callback);
            }
            std::vector<Row> out;
            for (const auto& [g, r] : res) out.push_back(row(g, r));
            return out;
        },
        nb::arg("ntuple"), nb::arg("games"), nb::arg("seed"), nb::arg("max_moves"), nb::arg("threads"), nb::arg("gamma"),
        nb::arg("tiers"), nb::arg("skip"), nb::arg("on_game").none(),
        "searched games -> (game, score, moves, lines, truncated, searched, changed, seconds, last_q1) per game played, in "
        "game order; on_game(*row) is called as each game finishes");

    m.def(
        "random_play",
        [](const Tables& t, int threads, uint64_t moves_per_thread, uint64_t seed) {
            const bpa::PlayStats st = bpa::play_random(t, threads, moves_per_thread, seed, false);
            return std::make_tuple(st.moves, st.games, st.seconds);
        },
        nb::arg("tables"), nb::arg("threads"), nb::arg("moves_per_thread"), nb::arg("seed"),
        nb::call_guard<nb::gil_scoped_release>(), "uniformly random play on N threads -> (moves, games, seconds)");

    m.def(
        "afterstate_play",
        [](const Tables& t, int threads, uint64_t moves_per_thread, uint64_t seed) {
            const bpa::PlayStats st = bpa::play_random(t, threads, moves_per_thread, seed, true);
            return std::make_tuple(st.moves, st.games, st.afterstates, st.seconds);
        },
        nb::arg("tables"), nb::arg("threads"), nb::arg("moves_per_thread"), nb::arg("seed"),
        nb::call_guard<nb::gil_scoped_release>(),
        "random play that also applies every legal move first -> (moves, games, afterstates, seconds)");

    m.def(
        "spawn_sequence",
        [](const Tables& t, uint64_t seed, int64_t score, const std::vector<int>& history, int n) {
            int h[3];
            read_history(t, history, h);
            bpa::PyRandom rng(seed);
            std::vector<int> pieces(std::max(n, 0));
            for (int& piece : pieces) piece = bpa::spawn(t, check_score(score), h, rng);
            return pieces;
        },
        nb::arg("tables"), nb::arg("seed"), nb::arg("score"), nb::arg("history"), nb::arg("n"),
        "n spawns at a fixed score with PyRandom(seed); the history evolves as in the game");

    m.def(
        "spawn_distribution",
        [](const Tables& t, int64_t score, const std::vector<int>& history) {
            int h[3];
            read_history(t, history, h);
            std::vector<double> out(t.n_pieces);
            bpa::spawn_distribution(t, check_score(score), h, out.data());
            return out;
        },
        nb::arg("tables"), nb::arg("score"), nb::arg("history"), "exact probability of each piece id spawning next");
}
