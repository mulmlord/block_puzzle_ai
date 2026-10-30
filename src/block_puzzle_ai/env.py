"""Gymnasium-style single-game environment on top of the reference simulator.

Training in this repo uses the vectorized BatchGame directly; this wrapper is for interoperability
with standard RL libraries (same reset/step signatures as gymnasium, no dependency on it).

Action space: Discrete(300), action = slot * 100 + row * 10 + col (top-left of the piece's bounding box).
Observation: {"board": int8[10, 10], "pieces": int8[3, 5, 5], "piece_ids": int64[3], "score": int64}.
"""

from __future__ import annotations

import numpy as np

from block_puzzle_ai.game import N, Game, Move
from block_puzzle_ai.pieces import all_pieces

N_ACTIONS = 3 * N * N


class BlockPuzzleEnv:
    def __init__(self, table: str = "new", max_moves: int | None = None):
        self.table = table
        self.max_moves = max_moves
        self.game: Game | None = None
        self._ids = {(p.group, p.variant): i for i, p in enumerate(all_pieces(table))}

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        self.game = Game(seed=seed, table=self.table)
        return self._obs(), self._info()

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        assert self.game is not None, "call reset() first"
        slot, pos = divmod(int(action), N * N)
        result = self.game.step(Move(slot, pos // N, pos % N))
        truncated = self.max_moves is not None and self.game.moves >= self.max_moves and not result.done
        info = self._info() | {"lines": result.lines, "points": result.points}
        return self._obs(), float(result.points), result.done, truncated, info

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(N_ACTIONS, dtype=bool)
        for m in self.game.legal_moves():
            mask[m.slot * N * N + m.row * N + m.col] = True
        return mask

    def _obs(self) -> dict:
        game = self.game
        pieces = np.zeros((3, 5, 5), dtype=np.int8)
        for i, piece in enumerate(game.pending):
            for r, c in piece.cells:
                pieces[i, r, c] = 1
        return {
            "board": np.array(game.board_rows(), dtype=np.int8),
            "pieces": pieces,
            "piece_ids": np.array([self._ids[(p.group, p.variant)] for p in game.pending]),
            "score": np.int64(game.score),
        }

    def _info(self) -> dict:
        return {"action_mask": self.action_mask(), "score": self.game.score, "moves": self.game.moves}
