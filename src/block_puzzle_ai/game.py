"""Reference simulator of the classic-mode rules recovered in docs/RULES.md.

The board is a 100-bit Python int; bit ``r * 10 + c`` is cell (row r, col c), row 0 = top.
This module favours clarity; ``batch_env`` is the vectorized GPU twin used for training and
is tested against this implementation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from functools import cache
from typing import NamedTuple, Protocol

import numpy as np

from block_puzzle_ai.pieces import HARD_GROUPS, Piece, load_piece_table

N = 10
ROW_MASKS = tuple(((1 << N) - 1) << (r * N) for r in range(N))
COL_MASKS = tuple(sum(1 << (r * N + c) for r in range(N)) for c in range(N))
LINE_MASKS = ROW_MASKS + COL_MASKS  # 0..9 rows, 10..19 columns
SPAWN_THRESHOLDS = (2999, 5999, 11999)  # randTiles level cut-offs, see RULES.md §2


class Move(NamedTuple):
    slot: int  # index into Game.pending
    row: int  # board row of the piece's bounding-box top-left
    col: int


@dataclass(frozen=True)
class Placement:
    row: int
    col: int
    mask: int
    lines: tuple[int, ...]  # LINE_MASKS indices this placement intersects


@dataclass(frozen=True)
class StepResult:
    points: int
    lines: int
    cleared: int  # bitmask of removed cells
    placed: int  # bitmask of the piece's cells
    spawned: Piece
    done: bool


def points_for(n_cells: int, lines: int) -> int:
    """checkToCrush / crushPendingTiles: 10 per placed cell + 50*L*(L+1) for L completed lines."""
    return 10 * n_cells + 50 * lines * (lines + 1)


@cache
def placements(piece: Piece) -> tuple[Placement, ...]:
    result = []
    for row in range(N - piece.height + 1):
        for col in range(N - piece.width + 1):
            mask = 0
            for r, c in piece.cells:
                mask |= 1 << ((row + r) * N + col + c)
            rows = range(row, row + piece.height)
            cols = range(N + col, N + col + piece.width)
            result.append(Placement(row, col, mask, tuple(rows) + tuple(cols)))
    return tuple(result)


def clear_full_lines(board: int, candidate_lines: tuple[int, ...] = tuple(range(2 * N))) -> tuple[int, int, int]:
    """Rows and columns are detected on the same board, then removed together (shared cells once)."""
    cleared = 0
    lines = 0
    for i in candidate_lines:
        m = LINE_MASKS[i]
        if board & m == m:
            cleared |= m
            lines += 1
    return board & ~cleared, lines, cleared


def spawn_level(score: int, r: int) -> int:
    """v = (int)(r / 1000.0f * score) in float32, as in PendingTileManager::randTiles."""
    return int(np.float32(r) / np.float32(1000.0) * np.float32(score))


def allowed_groups(level: int, n_groups: int) -> int:
    for i, threshold in enumerate(SPAWN_THRESHOLDS):
        if level <= threshold:
            return n_groups - 3 + i
    return n_groups


class RandomSource(Protocol):
    def randrange(self, n: int) -> int: ...


@dataclass
class PieceGenerator:
    """PendingTileManager::randTiles. The 3-entry history persists across games, like in the app."""

    rng: RandomSource
    table: str = "new"
    history: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.groups = load_piece_table(self.table)

    def spawn(self, score: int) -> Piece:
        del self.history[3:]
        level = spawn_level(score, self.rng.randrange(1000))
        group = self.rng.randrange(allowed_groups(level, len(self.groups)))
        if any(h in HARD_GROUPS for h in self.history):
            group = self.rng.randrange(8)
        self.history.insert(0, group)
        variants = self.groups[group]
        return variants[self.rng.randrange(len(variants))]


class Game:
    def __init__(self, seed: int | None = None, table: str = "new", generator: PieceGenerator | None = None):
        self.generator = generator or PieceGenerator(random.Random(seed), table)
        self.reset()

    def reset(self) -> None:
        self.board = 0
        self.colors = [-1] * (N * N)
        self.score = 0
        self.moves = 0
        self.lines_cleared = 0
        self.pending: list[Piece] = [self.generator.spawn(0) for _ in range(3)]
        self.done = not self.has_legal_move()

    def fits(self, piece: Piece, row: int, col: int) -> bool:
        if not (0 <= row <= N - piece.height and 0 <= col <= N - piece.width):
            return False
        return all(not self.board >> ((row + r) * N + col + c) & 1 for r, c in piece.cells)

    def legal_moves(self) -> list[Move]:
        return [
            Move(slot, p.row, p.col)
            for slot, piece in enumerate(self.pending)
            for p in placements(piece)
            if not p.mask & self.board
        ]

    def has_legal_move(self) -> bool:
        return any(not p.mask & self.board for piece in self.pending for p in placements(piece))

    def step(self, move: Move) -> StepResult:
        if self.done:
            raise ValueError("game is over")
        piece = self.pending[move.slot]
        if not self.fits(piece, move.row, move.col):
            raise ValueError(f"illegal move {move}")
        placement = next(p for p in placements(piece) if (p.row, p.col) == (move.row, move.col))

        board = self.board | placement.mask
        for r, c in piece.cells:
            self.colors[(move.row + r) * N + move.col + c] = piece.group
        # onTouchEnded: remove the placed piece, then bornPendingNode with the score *before* this move
        del self.pending[move.slot]
        spawned = self.generator.spawn(self.score)
        self.pending.append(spawned)

        self.board, lines, cleared = clear_full_lines(board, placement.lines)
        for i in range(N * N):
            if cleared >> i & 1:
                self.colors[i] = -1
        points = points_for(piece.size, lines)
        self.score += points
        self.moves += 1
        self.lines_cleared += lines
        self.done = not self.has_legal_move()
        return StepResult(points, lines, cleared, placement.mask, spawned, self.done)

    def board_rows(self) -> list[list[int]]:
        return [[self.board >> (r * N + c) & 1 for c in range(N)] for r in range(N)]
