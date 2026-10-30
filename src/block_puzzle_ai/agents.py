"""Agents choose one move per game for a whole BatchGame at once."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import torch

from block_puzzle_ai.batch_env import BatchGame, Candidates, move_from_index
from block_puzzle_ai.features import Afterstates, best_per_env, encode, isolated_empty
from block_puzzle_ai.game import Game, Move
from block_puzzle_ai.model import ValueNet, load_checkpoint


class Agent(Protocol):
    name: str

    def act(self, env: BatchGame, cand: Candidates) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (slot, pos) per game. Only called when every game has a legal move."""
        ...


class RandomAgent:
    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.rng: torch.Generator | None = None

    def act(self, env: BatchGame, cand: Candidates) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rng is None or self.rng.device != env.device:
            self.rng = torch.Generator(env.device).manual_seed(self.seed)
        choice = torch.multinomial(cand.legal.flatten(1).float(), 1, generator=self.rng).squeeze(1)
        return choice // 100, choice % 100


class HeuristicAgent:
    """Greedy one-ply search: points + keep the board open + keep room for every piece shape."""

    name = "heuristic"

    def __init__(self, empty: float = 1.0, fit: float = 2.0, isolated: float = 5.0):
        self.weights = (empty, fit, isolated)

    def value(self, env: BatchGame, a: Afterstates) -> torch.Tensor:
        w_empty, w_fit, w_isolated = self.weights
        empty = 100 - a.board.sum(1)
        room = (a.fit_any.float() * env.t.cells.float()).sum(1)
        return a.points + w_empty * empty + w_fit * room - w_isolated * isolated_empty(a.board)

    def act(self, env: BatchGame, cand: Candidates) -> tuple[torch.Tensor, torch.Tensor]:
        a = encode(env, cand)
        return best_per_env(env.n, a.index, self.value(env, a))


class ValueAgent:
    """Picks argmax over legal moves of reward + gamma * V(after-state) with a trained ValueNet."""

    def __init__(self, net: ValueNet, gamma: float, reward_scale: float, name: str = "rl"):
        self.net = net
        self.gamma = gamma
        self.reward_scale = reward_scale
        self.name = name

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | torch.device = "cpu") -> ValueAgent:
        net, ckpt = load_checkpoint(path, device)
        return cls(net, ckpt["gamma"], ckpt["reward_scale"], name=Path(path).stem)

    @torch.no_grad()
    def q_values(self, env: BatchGame, cand: Candidates) -> tuple[Afterstates, torch.Tensor, torch.Tensor]:
        a = encode(env, cand)
        with torch.autocast(env.device.type, dtype=torch.bfloat16, enabled=env.device.type == "cuda"):
            v = self.net(a.planes, a.globals).float()
        return a, a.points * self.reward_scale + self.gamma * v, v

    def act(self, env: BatchGame, cand: Candidates) -> tuple[torch.Tensor, torch.Tensor]:
        a, q, _ = self.q_values(env, cand)
        return best_per_env(env.n, a.index, q)


def choose_move(agent: Agent, game: Game, device: str | torch.device = "cpu") -> Move:
    """Single-game helper for the reference simulator (used by the viewer)."""
    env = BatchGame.from_game(game, device)
    slot, pos = agent.act(env, env.candidates())
    return move_from_index(int(slot[0]), int(pos[0]))


def make_agent(spec: str, device: str | torch.device = "cpu") -> Agent:
    if spec == "random":
        return RandomAgent()
    if spec == "heuristic":
        return HeuristicAgent()
    return ValueAgent.from_checkpoint(spec, device)
