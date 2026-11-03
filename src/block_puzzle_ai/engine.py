"""Python face of the C++ engine (``block_puzzle_ai._engine``, sources in ``cpp/``).

The engine is a single-game twin of ``game.py`` for CPU-heavy work (n-tuple training, search). With
the default PyRandom source it replays ``game.Game(seed)`` exactly; tests/test_engine.py checks that.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache

from block_puzzle_ai import _engine
from block_puzzle_ai.game import N, Move
from block_puzzle_ai.pieces import HARD_GROUPS, Piece, all_pieces

_LOW64 = (1 << 64) - 1
_BOARD = (1 << N * N) - 1


@cache
def cpp_tables(table: str = "new") -> _engine.Tables:
    pieces = all_pieces(table)
    return _engine.Tables([[r * N + c for r, c in p.cells] for p in pieces], [p.group for p in pieces], sorted(HARD_GROUPS))


def _history3(history: Sequence[int]) -> list[int]:
    """The 3 most recent spawned groups, most recent first, padded with -1 like batch_env."""
    h = list(history)[:3]
    return h + [-1] * (3 - len(h))


def spawn_sequence(seed: int, score: int, history: Sequence[int], n: int, table: str = "new") -> list[int]:
    """Piece ids of n spawns at a fixed score, drawn like PieceGenerator(random.Random(seed))."""
    return _engine.spawn_sequence(cpp_tables(table), seed, score, _history3(history), n)


class CppGame:
    """game.Game-like view of one C++ game. Boards are 100-bit ints and pieces are pieces.Piece."""

    def __init__(self, seed: int, table: str = "new", fast_rng: bool = False):
        self._pieces = all_pieces(table)
        self._ids = {p: i for i, p in enumerate(self._pieces)}
        self._game = _engine.Game(cpp_tables(table), seed, fast_rng)

    @property
    def board(self) -> int:
        return self._game.board_lo | self._game.board_hi << 64

    @property
    def pending(self) -> list[Piece]:
        return [self._pieces[i] for i in self._game.pending]

    @property
    def score(self) -> int:
        return self._game.score

    @property
    def moves(self) -> int:
        return self._game.moves

    @property
    def lines_cleared(self) -> int:
        return self._game.lines_cleared

    @property
    def history(self) -> list[int]:
        return self._game.history

    @property
    def done(self) -> bool:
        return self._game.done

    def set_state(self, board: int, pending: Sequence[Piece], score: int, history: Sequence[int]) -> None:
        """Moves and lines restart at 0. Raises ValueError on boards beyond 100 cells or bad ids."""
        if board < 0 or board & ~_BOARD:
            raise ValueError("board must be a 100-bit int")
        self._game.set_state(board & _LOW64, board >> 64, [self._ids[p] for p in pending], score, _history3(history))

    def legal_moves(self) -> list[Move]:
        return [Move(*m) for m in self._game.legal_moves()]

    def afterstates(self) -> list[tuple[Move, int, int, int]]:
        """(move, board after placement and clears, points, lines) for every legal move."""
        return [(Move(s, r, c), lo | hi << 64, points, lines) for s, r, c, lo, hi, points, lines in self._game.afterstates()]

    def spawn_distribution(self) -> list[float]:
        """Probability of each piece id being spawned by the next move."""
        return self._game.spawn_distribution()
