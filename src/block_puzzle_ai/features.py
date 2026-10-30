"""After-state encoding shared by the heuristic baseline and the value network.

An after-state is what the player controls deterministically: the board after placing a piece and
clearing lines, plus the two pieces left in the tray. The third piece is spawned at random.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from block_puzzle_ai.batch_env import CELLS, BatchGame, Candidates, Tables
from block_puzzle_ai.game import N

# Spawns stop depending on the score once randTiles' level can exceed every threshold.
SCORE_PHASE_END = 12000
FIT_CHUNK = 16384


@dataclass
class Afterstates:
    index: torch.Tensor  # (K, 3) env, slot, pos of each legal move
    board: torch.Tensor  # (K, 100) bool
    planes: torch.Tensor  # (K, 3, 10, 10) board + fit maps of the two remaining pieces
    globals: torch.Tensor  # (K, n_globals)
    fit_any: torch.Tensor  # (K, P) bool: piece p fits somewhere
    points: torch.Tensor  # (K,) points of the move


def n_globals(t: Tables) -> int:
    return 3 * t.n_pieces + 5


def fit_maps(t: Tables, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(K, 100) bool boards -> fits (K, P, 100) bool (piece p fits with top-left at pos), counts (K, P)."""
    dtype = torch.float16 if boards.is_cuda else torch.float32
    masks = t.mask.flatten(0, 1).to(dtype).T  # (cell, P * pos)
    valid = t.valid.flatten()
    fits, counts = [], []
    for chunk in boards.split(FIT_CHUNK):
        fit = ((chunk.to(dtype) @ masks) == 0) & valid
        fits.append(fit)
        counts.append(fit.view(len(chunk), t.n_pieces, CELLS).to(dtype).sum(-1))  # bool.sum() would upcast to int64
    return torch.cat(fits).view(len(boards), t.n_pieces, CELLS), torch.cat(counts).float()


def isolated_empty(boards: torch.Tensor) -> torch.Tensor:
    """Number of empty cells whose four neighbours are all filled or outside the board."""
    empty = (~boards).float().view(-1, 1, N, N)
    kernel = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=empty.dtype, device=empty.device)
    open_neighbours = F.conv2d(empty, kernel.view(1, 1, 3, 3), padding=1)
    return ((empty > 0) & (open_neighbours == 0)).flatten(1).sum(1)


def encode(env: BatchGame, cand: Candidates, index: torch.Tensor | None = None) -> Afterstates:
    t = env.t
    if index is None:
        index = cand.legal.nonzero()
    b, slot, pos = index.unbind(1)
    board = cand.after[b, slot, pos]
    remaining = env.pending[b].gather(1, t.remaining[slot])  # (K, 2)

    fits, fit_count = fit_maps(t, board)
    fit_any = fit_count > 0
    fit_frac = fit_count / t.valid.sum(-1)
    k = torch.arange(len(index), device=board.device)
    remaining_maps = fits[k[:, None], remaining]  # (K, 2, 100)
    planes = torch.cat([board[:, None], remaining_maps], 1).view(-1, 3, N, N)

    remaining_hot = F.one_hot(remaining, t.n_pieces).sum(1)
    phase = (env.score[b].clamp(max=SCORE_PHASE_END) / SCORE_PHASE_END)[:, None]
    history = env.history[b]
    cooldown = t.hard[history.clamp(min=0)] & (history >= 0)
    filled = board.sum(1, keepdim=True) / CELLS
    globals_ = torch.cat(
        [fit_any.float(), fit_frac.float(), remaining_hot.float(), phase.float(), cooldown.float(), filled.float()],
        1,
    )
    return Afterstates(index, board, planes.float(), globals_, fit_any, cand.points[b, slot, pos])


def best_per_env(n_envs: int, index: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Argmax of ``values`` over each env's legal moves -> (slot, pos) per env (envs without moves get 0, 0)."""
    dense = torch.full((n_envs, 3 * CELLS), float("-inf"), device=values.device)
    flat = index[:, 1] * CELLS + index[:, 2]
    dense[index[:, 0], flat] = values.float()
    choice = dense.argmax(1)
    return choice // CELLS, choice % CELLS
