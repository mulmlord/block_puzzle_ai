"""Tests of the n-tuple value function, trainer and CLI (block_puzzle_ai.ntuple, cpp/ntuple.hpp)."""

import random

import numpy as np
import pytest

from block_puzzle_ai import _engine
from block_puzzle_ai.engine import CppGame, cpp_tables
from block_puzzle_ai.game import placements
from block_puzzle_ai.ntuple import (
    fit_bucket,
    make_model,
    shape_cells,
    shapes_to_tuples,
    transform_board,
    transform_cell,
    transform_piece,
    value,
)
from block_puzzle_ai.pieces import all_pieces


def orbit(cells):
    return {frozenset(r * 10 + c for r, c in (transform_cell(t, i // 10, i % 10) for i in cells)) for t in range(8)}


@pytest.mark.parametrize("name, count", [("window4", 10), ("line", 5)])
def test_orbits(name, count):
    reps = shapes_to_tuples([name])
    assert len(reps) == count and all(r == sorted(r) for r in reps)
    cells = shape_cells(name)
    h, w = 1 + max(r for r, _ in cells), 1 + max(c for _, c in cells)
    for r0 in range(11 - h):
        for c0 in range(11 - w):
            inst = frozenset((r0 + r) * 10 + c0 + c for r, c in cells)
            assert sum(inst in orbit(rep) for rep in reps) == 1
    assert len(shapes_to_tuples(["window4", "line"])) == 15


def test_transforms_form_a_group_and_map_pieces():
    rng = random.Random(0)
    board = rng.getrandbits(100)
    assert len({transform_board(t, board) for t in range(8)}) == 8
    for p in range(len(all_pieces())):
        images = {transform_piece(t, p) for t in range(8)}
        assert {all_pieces()[i].group for i in images} == {all_pieces()[p].group}


def test_fit_bucket_and_bad_shape():
    assert [fit_bucket(n) for n in (0, 1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33, 99)] == [0, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
    with pytest.raises(ValueError):
        shape_cells("blob")


def n_placements(piece, board):
    return sum(not p.mask & board for p in placements(piece))


def brute_value(tuples, w, board, remaining, cooldown):
    pieces, v, off = all_pieces(), 0.0, 0
    for cells in tuples:
        for t in range(8):
            b = transform_board(t, board)
            v += w[off + sum((b >> c & 1) << j for j, c in enumerate(cells))]
        off += 1 << len(cells)
    for g in range(12):
        count = sum(n_placements(p, board) for p in pieces if p.group == g)
        v += w[off + (cooldown * 12 + g) * 8 + fit_bucket(count)]
    for p in remaining:
        v += w[off + 192 + pieces[p].group * 8 + fit_bucket(n_placements(pieces[p], board))]
    return v


def sample_afterstates(n, seed):
    """n/2 after-states from random play, n/2 random boards; (board, remaining ids, cooldown)."""
    rng, out = random.Random(seed), []
    game = CppGame(seed)
    while len(out) < n // 2:
        if game.done:
            game.reset()
        move, after, _, _ = rng.choice(game.afterstates())
        ids = [all_pieces().index(p) for p in game.pending]
        out.append((after, [ids[i] for i in range(3) if i != move.slot], rng.random() < 0.3))
        game.step(move)
    while len(out) < n:
        density = rng.uniform(0.2, 0.7)
        board = sum(1 << c for c in range(100) if rng.random() < density)
        out.append((board, [rng.randrange(28), rng.randrange(28)], rng.random() < 0.3))
    return out


@pytest.fixture
def random_model():
    tuples = shapes_to_tuples(["window4", "line"])
    model = make_model(tuples)
    model.weights[:] = np.random.default_rng(0).normal(size=model.n_weights).astype(np.float32)
    return tuples, model


def test_value_matches_brute_force(random_model):
    tuples, model = random_model
    assert model.n_weights == 10 * 65536 + 5 * 1024 + 288 and model.n_lookups == 8 * 15 + 14
    for board, rem, cd in sample_afterstates(60, 1):
        expected = brute_value(tuples, model.weights, board, rem, cd)
        assert value(model, board, rem, cd) == pytest.approx(expected, rel=1e-5, abs=1e-3)


def test_value_is_symmetric(random_model):
    _, model = random_model
    for board, rem, cd in sample_afterstates(200, 2):
        v = value(model, board, rem, cd)
        for t in range(8):
            image = value(model, transform_board(t, board), [transform_piece(t, p) for p in rem], cd)
            assert image == pytest.approx(v, rel=1e-5, abs=1e-3)


@pytest.mark.parametrize("tuples", [[list(range(21))], [[100]], [[3, 3]], [[]]])
def test_ntuple_rejects_bad_tuples(tuples):
    with pytest.raises(ValueError):
        _engine.NTuple(cpp_tables(), tuples)


def test_value_rejects_bad_remaining(random_model):
    _, model = random_model
    for rem in ([0], [0, 28], [-1, 0]):
        with pytest.raises(ValueError):
            model.value(0, 0, rem, False)


def test_update_moves_toward_target(random_model):
    _, model = random_model
    for board, rem, cd in sample_afterstates(20, 3):
        before = value(model, board, rem, cd)
        model.update(board & (2**64 - 1), board >> 64, rem, cd, before + 50.0, 0.1)
        after = value(model, board, rem, cd)
        assert before < after < before + 50.0


def test_update_alpha_one_lands_on_target():
    tuples = [[0, 1, 2, 3, 10, 11, 12, 13]]
    model, rng = make_model(tuples), random.Random(4)
    while True:  # a board whose 8 symmetric indices are distinct, with remaining pieces from different groups
        board = rng.getrandbits(100)
        idx = {sum((transform_board(t, board) >> c & 1) << j for j, c in enumerate(tuples[0])) for t in range(8)}
        if len(idx) == 8:
            break
    rem = [0, 27]
    assert all_pieces()[0].group != all_pieces()[27].group
    model.update(board & (2**64 - 1), board >> 64, rem, False, 123.5, 1.0)
    assert value(model, board, rem, False) == pytest.approx(123.5, abs=1e-3)


def test_update_precision_at_large_values():
    model = make_model(shapes_to_tuples(["window4", "line"]))
    model.weights[:] = 1e8 / model.n_lookups
    board, rem = 0, [0, 1]
    before = value(model, board, rem, False)
    model.update(0, 0, rem, False, before + 1000.0, 0.1)
    assert value(model, board, rem, False) - before >= 50.0  # nominal step: 0.1 * 1000 = 100
