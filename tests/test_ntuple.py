"""Tests of the n-tuple value function, trainer and CLI (block_puzzle_ai.ntuple, cpp/ntuple.hpp)."""

import random

import pytest

from block_puzzle_ai.ntuple import fit_bucket, shape_cells, shapes_to_tuples, transform_board, transform_cell, transform_piece
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
