"""Parity tests of the C++ engine (block_puzzle_ai._engine) against game.py and batch_env.py."""

import random

import pytest
import torch

from block_puzzle_ai import _engine
from block_puzzle_ai.batch_env import spawn_distribution as torch_spawn_distribution
from block_puzzle_ai.batch_env import tables_for
from block_puzzle_ai.engine import cpp_tables, spawn_sequence
from block_puzzle_ai.game import PieceGenerator
from block_puzzle_ai.pieces import all_pieces, load_piece_table

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


# scores around each spawn threshold, where r = 999 crosses it, and where float32 rounds the score
SCORES = [0, 1, 2999, 3000, 3001, 3002, 3003, 3004, 5999, 6001, 6005, 6006, 6007,
          11999, 12001, 12011, 12012, 12013, 100_000, 16_777_217, 40_000_001]  # fmt: skip
HISTORIES = [[], [0], [8], [3, 10], [1, 2, 11], [4, 5, 6]]


@pytest.mark.parametrize("score", SCORES)
def test_spawn_sequence_matches_reference(score):
    ids = {p: i for i, p in enumerate(all_pieces())}
    for seed, hist in enumerate(HISTORIES):
        gen = PieceGenerator(random.Random(seed), history=list(hist))
        assert spawn_sequence(seed, score, hist, 500) == [ids[gen.spawn(score)] for _ in range(500)]


@pytest.mark.parametrize("seed", [2**32 + 5, 2**64 - 1])
def test_spawn_sequence_big_seeds(seed):
    ids = {p: i for i, p in enumerate(all_pieces())}
    gen = PieceGenerator(random.Random(seed))
    assert spawn_sequence(seed, 100_000, [], 500) == [ids[gen.spawn(100_000)] for _ in range(500)]


@pytest.mark.parametrize("score", SCORES)
def test_spawn_distribution_matches_batch_env(score):
    t = tables_for("cpu")
    for hist in HISTORIES:
        h = (hist + [-1, -1, -1])[:3]
        expected = torch_spawn_distribution(t, torch.tensor([score]), torch.tensor([h]))[0].tolist()
        got = _engine.spawn_distribution(cpp_tables(), score, h)
        assert max(abs(a - b) for a, b in zip(got, expected, strict=True)) < 1e-6
        assert abs(sum(got) - 1) < 1e-9


@pytest.mark.parametrize(
    "cells, groups",
    [
        ([[0]] * 33, list(range(16)) + [15] * 17),  # 33 pieces
        ([[0]] * 17, list(range(17))),  # 17 groups
        ([list(range(10))], [0]),  # 10 cells
        ([[100]], [0]),  # offset off the board
        ([[0], [0]], [1, 0]),  # groups not non-decreasing from 0
    ],
)
def test_tables_reject_bad_input(cells, groups):
    with pytest.raises(ValueError):
        _engine.Tables(cells, groups, [])
