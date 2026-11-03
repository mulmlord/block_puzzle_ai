"""Parity tests of the C++ engine (block_puzzle_ai._engine) against game.py and batch_env.py."""

import random

import pytest

from block_puzzle_ai import _engine

SEEDS = [0, 1, 42, 999, 2**32 - 1, 2**32 + 5, 2**40 + 123, 2**64 - 1]


@pytest.mark.parametrize("seed", SEEDS)
def test_pyrandom_matches_cpython(seed):
    ours, ref = _engine.PyRandom(seed), random.Random(seed)
    for n in [1000, 12, 1, 8, 2, 4, 9, 7, 3] * 200:  # interleaved like randTiles
        assert ours.randrange(n) == ref.randrange(n)
    assert [ours.getrandbits(k) for k in (1, 5, 31, 32)] == [ref.getrandbits(k) for k in (1, 5, 31, 32)]


def test_xoshiro_is_deterministic_and_in_range():
    a, b = _engine.Xoshiro256(7), _engine.Xoshiro256(7)
    draws = [a.randrange(10) for _ in range(100_000)]
    assert draws[:1000] == [b.randrange(10) for _ in range(1000)]
    counts = [draws.count(i) for i in range(10)]
    assert min(counts) > 9_500 and max(counts) < 10_500
