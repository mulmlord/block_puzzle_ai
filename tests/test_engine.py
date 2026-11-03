"""Parity tests of the C++ engine (block_puzzle_ai._engine) against game.py and batch_env.py."""

import random

import pytest
import torch

from block_puzzle_ai import _engine
from block_puzzle_ai.batch_env import spawn_distribution as torch_spawn_distribution
from block_puzzle_ai.batch_env import tables_for
from block_puzzle_ai.engine import CppGame, cpp_tables, spawn_sequence
from block_puzzle_ai.game import Game, Move, PieceGenerator, clear_full_lines, placements, points_for
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


def test_afterstates_match_reference_on_dense_boards():
    rng, pieces, max_lines = random.Random(7), all_pieces(), 0
    for i in range(1000):
        density = 0.3 + 0.6 * i / 999
        board = sum(1 << c for c in range(100) if rng.random() < density)
        board, _, _ = clear_full_lines(board)  # reachable boards have no full line before a move
        pending = [rng.choice(pieces) for _ in range(3)]
        ref = Game(0)
        ref.board, ref.pending = board, list(pending)
        expected = []
        for m in ref.legal_moves():
            piece = pending[m.slot]
            pl = next(p for p in placements(piece) if (p.row, p.col) == (m.row, m.col))
            after, lines, _ = clear_full_lines(board | pl.mask, pl.lines)
            expected.append((m, after, points_for(piece.size, lines), lines))
            max_lines = max(max_lines, lines)
        cpp = CppGame(0)
        cpp.set_state(board, pending, 0, [])
        assert cpp.legal_moves() == [m for m, *_ in expected]
        assert cpp.afterstates() == expected
        assert cpp.done == (not expected)
    assert max_lines >= 3  # the boards did produce simultaneous row and column clears


def test_set_state_rejects_bad_input():
    g, p = CppGame(0), all_pieces()[:3]
    for args in [(1 << 100, p, 0, []), (0, p, -1, []), (0, p, 0, [16]), (0, p, 0, [-2])]:
        with pytest.raises(ValueError):
            g.set_state(*args)
    raw = _engine.Game(cpp_tables(), 0)
    for pending in ([28, 0, 0], [-1, 0, 0], [0, 0]):
        with pytest.raises(ValueError):
            raw.set_state(0, 0, pending, 0, [-1, -1, -1])


def assert_same(ref: Game, cpp: CppGame):
    assert cpp.board == ref.board
    assert cpp.pending == ref.pending
    assert (cpp.score, cpp.moves, cpp.lines_cleared, cpp.done) == (ref.score, ref.moves, ref.lines_cleared, ref.done)
    assert cpp.history == (ref.generator.history[:3] + [-1, -1, -1])[:3]


@pytest.mark.parametrize("block", range(10))
def test_trajectory_parity(block):
    for seed in range(100 * block, 100 * block + 100):
        ref, cpp, pick = Game(seed), CppGame(seed), random.Random(10_000 + seed)
        for _ in range(2):  # the second game checks that reset keeps the spawn history
            assert_same(ref, cpp)
            while not ref.done:
                moves = ref.legal_moves()
                assert cpp.legal_moves() == moves
                move = pick.choice(moves)
                r = ref.step(move)
                assert cpp.step(move) == (r.points, r.lines, r.done)
                assert_same(ref, cpp)
            ref.reset()
            cpp.reset()


DOT, SQUARE3 = load_piece_table()[0][0], load_piece_table()[8][0]  # as in tests/test_game.py
CHECKER = sum(1 << (r * 10 + c) for r in range(10) for c in range(10) if (r + c) % 2 == 0)


def test_step_errors():
    g = CppGame(3)
    g.set_state(1, [DOT] * 3, 0, [])  # cell (0, 0) filled
    with pytest.raises(ValueError):
        g.step(Move(0, 0, 0))  # occupied
    with pytest.raises(ValueError):
        g.step(Move(0, 10, 0))  # off the board
    g.set_state(CHECKER, [SQUARE3] * 3, 0, [])  # no two empty cells are adjacent, and no line is full
    assert g.done and g.legal_moves() == []
    with pytest.raises(ValueError):
        g.step(Move(0, 0, 1))


def test_fast_rng_game_is_deterministic():
    a, b = CppGame(5, fast_rng=True), CppGame(5, fast_rng=True)
    pick = random.Random(0)
    while not a.done:
        move = pick.choice(a.legal_moves())
        assert a.step(move) == b.step(move)
    assert a.board == b.board and a.moves > 0
