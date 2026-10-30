"""Vectorized twin of game.Game: B independent games advanced in lock-step with torch (CPU or GPU).

Cells are a (B, 100) bool tensor (index r * 10 + c), pieces are ids into pieces.all_pieces(table),
and a move is (slot, pos) with pos = row * 10 + col of the piece's bounding-box top-left.
tests/test_batch_env.py checks it against the reference implementation in game.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import torch

from block_puzzle_ai.game import N, Game, Move, placements
from block_puzzle_ai.pieces import HARD_GROUPS, all_pieces, load_piece_table

CELLS = N * N
# pending order after removing a slot: the other two keep their order, the new piece goes last
REMAINING = torch.tensor([[1, 2], [0, 2], [0, 1]])


class Tables:
    """Per-piece placement masks and spawn tables as tensors on one device."""

    def __init__(self, device: torch.device, table: str = "new"):
        pieces = all_pieces(table)
        groups = load_piece_table(table)
        n = len(pieces)
        mask = torch.zeros(n, CELLS, CELLS, dtype=torch.bool)
        valid = torch.zeros(n, CELLS, dtype=torch.bool)
        for pid, piece in enumerate(pieces):
            for pl in placements(piece):
                pos = pl.row * N + pl.col
                valid[pid, pos] = True
                for r, c in piece.cells:
                    mask[pid, pos, (pl.row + r) * N + pl.col + c] = True
        self.n_pieces = n
        self.n_groups = len(groups)
        self.mask = mask.to(device)  # (P, pos, cell)
        self.valid = valid.to(device)  # (P, pos)
        self.cells = torch.tensor([p.size for p in pieces], device=device)
        self.group = torch.tensor([p.group for p in pieces], device=device)
        self.group_size = torch.tensor([len(g) for g in groups], device=device)
        self.group_offset = torch.cumsum(self.group_size, 0) - self.group_size
        self.hard = torch.tensor([g in HARD_GROUPS for g in range(len(groups))], device=device)
        self.remaining = REMAINING.to(device)


@cache
def tables_for(device: str, table: str = "new") -> Tables:
    return Tables(torch.device(device), table)


def spawn_from_uniforms(
    t: Tables, score: torch.Tensor, history: torch.Tensor, u: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """randTiles for a batch. u: (B, 4) float64 uniforms for r, group, forced-small group, variant.

    history: (B, 3) most-recent-first group ids, -1 = empty. Returns (piece ids, new history).
    """
    r = (u[:, 0] * 1000).long().clamp(max=999)
    level = (r.float() / 1000.0 * score.float()).long()  # float32 like the game
    limit = t.n_groups - 3 + (level > 2999).long() + (level > 5999).long() + (level > 11999).long()
    group = (u[:, 1] * limit).long().minimum(limit - 1)
    in_cooldown = (t.hard[history.clamp(min=0)] & (history >= 0)).any(1)
    group = torch.where(in_cooldown, (u[:, 2] * 8).long().clamp(max=7), group)
    size = t.group_size[group]
    variant = (u[:, 3] * size).long().minimum(size - 1)
    new_history = torch.cat([group[:, None], history[:, :2]], 1)
    return t.group_offset[group] + variant, new_history


@dataclass
class Candidates:
    """Every (slot, pos) move of every game, evaluated in one shot."""

    legal: torch.Tensor  # (B, 3, 100) bool
    after: torch.Tensor  # (B, 3, 100, 100) bool board after placement and clears
    points: torch.Tensor  # (B, 3, 100) long
    lines: torch.Tensor  # (B, 3, 100) long

    @property
    def terminal(self) -> torch.Tensor:
        return ~self.legal.flatten(1).any(1)


class BatchGame:
    def __init__(self, n: int, device: str | torch.device = "cpu", seed: int | None = None, table: str = "new"):
        self.device = torch.device(device)
        self.t = tables_for(str(self.device), table)
        self.n = n
        self.rng = torch.Generator(self.device)
        self.rng.manual_seed(seed if seed is not None else torch.seed())
        self.board = torch.zeros(n, CELLS, dtype=torch.bool, device=self.device)
        self.pending = torch.zeros(n, 3, dtype=torch.long, device=self.device)
        self.score = torch.zeros(n, dtype=torch.long, device=self.device)
        self.moves = torch.zeros(n, dtype=torch.long, device=self.device)
        self.lines_cleared = torch.zeros(n, dtype=torch.long, device=self.device)
        self.history = torch.full((n, 3), -1, dtype=torch.long, device=self.device)
        self.reset()

    def _spawn(self, idx: torch.Tensor) -> torch.Tensor:
        u = torch.rand(len(idx), 4, generator=self.rng, device=self.device, dtype=torch.float64)
        piece, self.history[idx] = spawn_from_uniforms(self.t, self.score[idx], self.history[idx], u)
        return piece

    def reset(self, idx: torch.Tensor | None = None) -> None:
        """restart(): empty board, score 0, three fresh pieces. The spawn history is kept."""
        if idx is None:
            idx = torch.arange(self.n, device=self.device)
        elif idx.dtype == torch.bool:
            idx = idx.nonzero().squeeze(1)
        self.board[idx] = False
        self.score[idx] = 0
        self.moves[idx] = 0
        self.lines_cleared[idx] = 0
        for slot in range(3):
            self.pending[idx, slot] = self._spawn(idx)

    def candidates(self) -> Candidates:
        b = self.n
        mask = self.t.mask[self.pending]  # (B, 3, pos, cell)
        board = self.board[:, None, None, :]
        legal = self.t.valid[self.pending] & ~(mask & board).any(-1)
        grid = (mask | board).view(b, 3, CELLS, N, N)
        rows = grid.all(-1)
        cols = grid.all(-2)
        lines = rows.sum(-1) + cols.sum(-1)
        cleared = rows[..., :, None] | cols[..., None, :]
        after = (grid & ~cleared).view(b, 3, CELLS, CELLS)
        points = 10 * self.t.cells[self.pending][..., None] + 50 * lines * (lines + 1)
        return Candidates(legal, after, points, lines)

    def step(self, cand: Candidates, slot: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Apply one legal move per game (taken from ``cand``); returns the points scored."""
        b = torch.arange(self.n, device=self.device)
        if not bool(cand.legal[b, slot, pos].all()):
            raise ValueError("illegal move in batch")
        points = cand.points[b, slot, pos]
        self.board = cand.after[b, slot, pos].clone()
        remaining = self.pending.gather(1, self.t.remaining[slot])
        spawned = self._spawn(b)  # uses the score before this move, like bornPendingNode
        self.pending = torch.cat([remaining, spawned[:, None]], 1)
        self.score += points
        self.moves += 1
        self.lines_cleared += cand.lines[b, slot, pos]
        return points

    # ---- interop with the reference simulator ----------------------------------------------
    @classmethod
    def from_game(cls, game: Game, device: str | torch.device = "cpu") -> BatchGame:
        env = cls(1, device, seed=0, table=game.generator.table)
        ids = {(p.group, p.variant): i for i, p in enumerate(all_pieces(game.generator.table))}
        env.board[0] = torch.tensor([bool(game.board >> i & 1) for i in range(CELLS)])
        env.pending[0] = torch.tensor([ids[(p.group, p.variant)] for p in game.pending])
        env.score[0] = game.score
        env.moves[0] = game.moves
        hist = (game.generator.history[:3] + [-1, -1, -1])[:3]
        env.history[0] = torch.tensor(hist)
        return env


def move_from_index(slot: int, pos: int) -> Move:
    return Move(int(slot), int(pos) // N, int(pos) % N)
