// Python bindings: block_puzzle_ai._engine
#include <nanobind/nanobind.h>

#include <stdexcept>

#include "rng.hpp"

namespace nb = nanobind;

namespace {

uint32_t check_n(uint32_t n) {
    if (n == 0) throw std::invalid_argument("randrange: n must be >= 1");
    return n;
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
}
