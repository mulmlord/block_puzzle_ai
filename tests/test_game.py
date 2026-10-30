import random

import pytest

from block_puzzle_ai.game import (
    COL_MASKS,
    N,
    ROW_MASKS,
    Game,
    Move,
    PieceGenerator,
    allowed_groups,
    points_for,
    spawn_level,
)
from block_puzzle_ai.pieces import HARD_GROUPS, all_pieces, load_piece_table

DOT = load_piece_table()[0][0]
SQUARE3 = load_piece_table()[8][0]


class ScriptedRng:
    """randrange returns scripted values in order, so tests control every rand() call."""

    def __init__(self, values):
        self.values = list(values)
        self.calls = []

    def randrange(self, n):
        self.calls.append(n)
        return self.values.pop(0) % n


def bit(r, c):
    return 1 << (r * N + c)


def make_game(pending, board=0, score=0, rng_values=(0,) * 64):
    game = Game(generator=PieceGenerator(ScriptedRng(rng_values)))
    game.board, game.pending, game.score, game.done = board, list(pending), score, False
    return game


def test_piece_table_matches_extracted_counts():
    table = load_piece_table("new")
    assert [len(g) for g in table] == [1, 2, 2, 2, 2, 4, 1, 4, 1, 4, 4, 1]
    assert [len(p.cells) for p in all_pieces()] == [1, 2, 2, 3, 3, 4, 4, 5, 5, 3, 3, 3, 3, 4, 5, 5, 5, 5, 9, 4, 4, 4, 4, 5, 5, 5, 5, 5]
    assert len(load_piece_table("old")) == 12


@pytest.mark.parametrize("lines,expected", [(0, 10), (1, 110), (2, 310), (3, 610), (4, 1010), (5, 1510), (6, 2110)])
def test_points_match_emulated_crush_formula(lines, expected):
    assert points_for(1, lines) == expected
    assert points_for(9, lines) == expected + 80


def test_row_and_column_clear_simultaneously_shared_cell_once():
    board = (ROW_MASKS[3] | COL_MASKS[5]) & ~bit(3, 5)
    game = make_game([DOT, DOT, DOT], board=board)
    result = game.step(Move(0, 3, 5))
    assert result.lines == 2
    assert bin(result.cleared).count("1") == 19
    assert result.points == 10 + 300
    assert game.board == 0


def test_no_gravity_after_clear():
    keep = bit(2, 0) | bit(9, 9)
    game = make_game([DOT, DOT, DOT], board=(ROW_MASKS[4] & ~bit(4, 7)) | keep)
    game.step(Move(0, 4, 7))
    assert game.board == keep


def test_illegal_moves_rejected():
    game = make_game([SQUARE3, DOT, DOT], board=bit(1, 1))
    assert not game.fits(SQUARE3, 0, 0)  # overlap
    assert not game.fits(SQUARE3, 8, 0)  # out of bounds
    assert game.fits(SQUARE3, 2, 2)
    with pytest.raises(ValueError):
        game.step(Move(0, 0, 0))


def test_pending_slots_shift_left_and_new_piece_appends():
    a, b, c = all_pieces()[0], all_pieces()[1], all_pieces()[3]
    game = make_game([a, b, c])
    result = game.step(Move(0, 0, 0))
    assert game.pending[:2] == [b, c]
    assert game.pending[2] == result.spawned


def test_spawn_uses_score_before_the_move():
    # r = 999 -> level = int(0.999 * score); score 5000 -> level 4995 -> 10 allowed groups
    rng = ScriptedRng([999, 0, 0] + [0] * 10)
    game = make_game([DOT, DOT, DOT], score=5000)
    game.generator = PieceGenerator(rng)
    game.step(Move(0, 0, 0))
    assert rng.calls[:2] == [1000, 10]


def test_spawn_level_float32_threshold():
    assert spawn_level(3003, 999) == 2999
    assert spawn_level(3004, 999) == 3000
    assert allowed_groups(spawn_level(3003, 999), 12) == 9
    assert allowed_groups(spawn_level(3004, 999), 12) == 10
    assert allowed_groups(12000, 12) == 12


def test_hard_piece_cooldown_and_low_score_groups():
    gen = PieceGenerator(random.Random(0))
    groups = [gen.spawn(0).group for _ in range(5000)]
    assert max(groups) == 8  # groups 9..11 need score >= ~3004
    for i, g in enumerate(groups):
        if g in HARD_GROUPS:
            assert all(h < 8 for h in groups[i + 1 : i + 4])

    gen = PieceGenerator(random.Random(1))
    assert {gen.spawn(10**6).group for _ in range(5000)} == set(range(12))


def test_game_over_when_no_pending_piece_fits():
    full_but_one = ((1 << 100) - 1) & ~bit(0, 0) & ~bit(9, 9)
    game = make_game([SQUARE3, SQUARE3, SQUARE3], board=full_but_one)
    assert not game.has_legal_move()
    game = make_game([SQUARE3, SQUARE3, DOT], board=full_but_one)
    assert game.has_legal_move()


def test_random_games_terminate_and_keep_invariants():
    rng = random.Random(7)
    for seed in range(5):
        game = Game(seed=seed)
        while not game.done:
            moves = game.legal_moves()
            assert moves
            before = game.score
            result = game.step(rng.choice(moves))
            assert game.score - before == result.points
            assert all((game.colors[i] >= 0) == bool(game.board >> i & 1) for i in range(N * N))
        assert not game.legal_moves()
