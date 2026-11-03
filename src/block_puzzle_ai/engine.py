"""Python face of the C++ engine (``block_puzzle_ai._engine``, sources in ``cpp/``).

The engine is a single-game twin of ``game.py`` for CPU-heavy work (n-tuple training, search). With
the default PyRandom source it replays ``game.Game(seed)`` exactly; tests/test_engine.py checks that.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache

from block_puzzle_ai import _engine
from block_puzzle_ai.game import N
from block_puzzle_ai.pieces import HARD_GROUPS, all_pieces


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
