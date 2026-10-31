"""Agents choose one move per game for a whole BatchGame at once."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import torch

from block_puzzle_ai.batch_env import BatchGame, Candidates, move_from_index, spawn_distribution
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
    def from_checkpoint(cls, path: str | Path, device: str | torch.device = "cpu", **kwargs) -> ValueAgent:
        net, ckpt = load_checkpoint(path, device)
        return cls(net, ckpt["gamma"], ckpt["reward_scale"], name=Path(path).stem, **kwargs)

    @torch.no_grad()
    def q_values(self, env: BatchGame, cand: Candidates) -> tuple[Afterstates, torch.Tensor, torch.Tensor]:
        a = encode(env, cand)
        v = self.net.predict(a.planes, a.globals)
        return a, a.points * self.reward_scale + self.gamma * v, v

    def act(self, env: BatchGame, cand: Candidates) -> tuple[torch.Tensor, torch.Tensor]:
        a, q, _ = self.q_values(env, cand)
        return best_per_env(env.n, a.index, q)


class LookaheadAgent(ValueAgent):
    """ValueAgent that searches one move deeper wherever its own 1-ply estimate signals danger.

    In a state whose best 1-ply Q is below ``search_below``, each legal move m (or the ``top_k`` best)
    is re-scored as r(m) + gamma * E_x[max_m' q(m')]: the expectation over every piece x the game can
    spawn next, with the exact spawn probabilities, of the best 1-ply Q afterwards (0 = game over).
    """

    def __init__(
        self,
        net: ValueNet,
        gamma: float,
        reward_scale: float,
        name: str = "rl",
        search_below: float = 100.0,
        top_k: int | None = None,
        chunk: int = 512,
    ):
        super().__init__(net, gamma, reward_scale, f"{name}+search")
        self.search_below = search_below
        self.top_k = top_k
        self.chunk = chunk  # child states per network pass
        self.searched = 0  # states searched
        self.changed = 0  # of those, how often the search changed the move

    @torch.no_grad()
    def act(self, env: BatchGame, cand: Candidates) -> tuple[torch.Tensor, torch.Tensor]:
        a, q, _ = self.q_values(env, cand)
        slot, pos = best_per_env(env.n, a.index, q)
        best = torch.full((env.n,), float("-inf"), device=env.device).scatter_reduce_(0, a.index[:, 0], q, "amax")
        danger = best < self.search_below
        if not bool(danger.any()):
            return slot, pos
        in_danger = danger[a.index[:, 0]]
        rows = self._top_k(a, q, in_danger.nonzero().squeeze(1))
        q = q.masked_fill(in_danger, float("-inf"))
        q[rows] = a.points[rows] * self.reward_scale + self.gamma * self._expected_best(env, a, rows)
        new_slot, new_pos = best_per_env(env.n, a.index, q)
        self.searched += int(danger.sum())
        self.changed += int(((new_slot != slot) | (new_pos != pos))[danger].sum())
        return new_slot, new_pos

    def _top_k(self, a: Afterstates, q: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        if self.top_k is None:
            return rows
        rows = rows[torch.argsort(q[rows], descending=True)]
        rows = rows[torch.argsort(a.index[rows, 0], stable=True)]  # grouped by env, best first
        env = a.index[rows, 0]
        rank = torch.arange(len(rows), device=rows.device) - torch.searchsorted(env, env)
        return rows[rank < self.top_k]

    def _children(self, env: BatchGame, a: Afterstates, rows: torch.Tensor) -> tuple[BatchGame, torch.Tensor, torch.Tensor]:
        """Every state the after-states ``rows`` of ``a`` can lead to, one per piece the game may spawn.

        Returns those states, the position in ``rows`` each one comes from, and its probability.
        """
        t = env.t
        e = a.index[rows, 0]
        probs = spawn_distribution(t, env.score[e], env.history[e])  # the spawn uses the pre-move score
        parent, piece = probs.nonzero(as_tuple=True)
        e, row = e[parent], rows[parent]
        remaining = env.pending[e].gather(1, t.remaining[a.index[row, 1]])
        children = BatchGame.from_state(
            t,
            a.board[row],
            torch.cat([remaining, piece[:, None]], 1),
            env.score[e] + a.points[row],
            torch.cat([t.group[piece][:, None], env.history[e, :2]], 1),
        )
        return children, parent, probs[parent, piece]

    def _expected_best(self, env: BatchGame, a: Afterstates, rows: torch.Tensor) -> torch.Tensor:
        """E_x[max_m' q(m')] for the after-states ``rows`` of ``a``, over the piece x spawned next."""
        children, parent, prob = self._children(env, a, rows)
        expected = torch.zeros(len(rows), device=env.device)
        for chunk in torch.arange(children.n, device=env.device).split(self.chunk):
            child = BatchGame.from_state(
                env.t, children.board[chunk], children.pending[chunk], children.score[chunk], children.history[chunk]
            )
            ca, cq, _ = self.q_values(child, child.candidates())
            best = torch.zeros(child.n, device=env.device)  # no legal move: game over, value 0
            best.scatter_reduce_(0, ca.index[:, 0], cq, "amax", include_self=False)
            expected.index_add_(0, parent[chunk], prob[chunk] * best)
        return expected


def choose_move(agent: Agent, game: Game, device: str | torch.device = "cpu") -> Move:
    """Single-game helper for the reference simulator (used by the viewer)."""
    env = BatchGame.from_game(game, device)
    slot, pos = agent.act(env, env.candidates())
    return move_from_index(int(slot[0]), int(pos[0]))


def make_agent(
    spec: str, device: str | torch.device = "cpu", search_below: float | None = None, top_k: int | None = None
) -> Agent:
    """random | heuristic | checkpoint path; ``search_below`` turns a checkpoint into a LookaheadAgent."""
    if spec == "random":
        return RandomAgent()
    if spec == "heuristic":
        return HeuristicAgent()
    if search_below is not None:
        return LookaheadAgent.from_checkpoint(spec, device, search_below=search_below, top_k=top_k)
    return ValueAgent.from_checkpoint(spec, device)
