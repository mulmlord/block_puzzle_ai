// Python bindings: block_puzzle_ai._engine (wrapped by block_puzzle_ai.engine)
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>
#include <vector>

#include "engine.hpp"
#include "rng.hpp"

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
