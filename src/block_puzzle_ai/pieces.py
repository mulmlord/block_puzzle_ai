"""Piece table recovered from PendingTileManager::init() (see docs/REVERSE_ENGINEERING.md)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from importlib import resources

# Groups the RNG treats as "hard": after one of these, the next 3 picks are restricted to groups 0..7.
HARD_GROUPS = frozenset({8, 10, 11})


@dataclass(frozen=True)
class Piece:
    group: int
    variant: int
    cells: tuple[tuple[int, int], ...]  # (row, col) relative to the top-left of the bounding box

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def height(self) -> int:
        return 1 + max(r for r, _ in self.cells)

    @property
    def width(self) -> int:
        return 1 + max(c for _, c in self.cells)


@cache
def load_piece_table(variant: str = "new") -> tuple[tuple[Piece, ...], ...]:
    """Groups of rotation variants. variant='new' is the table fresh installs get (UserDefault isOld=false)."""
    raw = json.loads(resources.files("block_puzzle_ai.data").joinpath("pieces.json").read_text())
    return tuple(
        tuple(Piece(gi, vi, tuple((r, c) for r, c in t["cells"])) for vi, t in enumerate(tiles))
        for gi, tiles in enumerate(raw["variants"][variant])
    )


@cache
def all_pieces(variant: str = "new") -> tuple[Piece, ...]:
    """Flat list of every (group, rotation) of a table; the index is used as a piece id."""
    return tuple(p for group in load_piece_table(variant) for p in group)
