"""Record rl_v1's best Q per move and find out how well it flags the moves before a game over."""
import sys
import time

import numpy as np
import torch

from block_puzzle_ai.agents import ValueAgent
from block_puzzle_ai.batch_env import BatchGame
from block_puzzle_ai.features import best_per_env

dev = "cuda"
n_games, cap = int(sys.argv[1]), int(sys.argv[2])
agent = ValueAgent.from_checkpoint("models/rl_v1.pt", dev)
env = BatchGame(n_games, dev, seed=4242)
ids = torch.arange(n_games, device=dev)
traces = [[] for _ in range(n_games)]
died = [False] * n_games
t0 = time.perf_counter()
with torch.no_grad():
    while env.n:
        cand = env.candidates()
        finished = cand.terminal | (env.moves >= cap)
        if finished.any():
            for i in finished.nonzero().squeeze(1).tolist():
                died[int(ids[i])] = bool(cand.terminal[i])
            alive = (~finished).nonzero().squeeze(1)
            env.keep(alive)
            ids = ids[alive]
            if not env.n:
                break
            cand = env.candidates()
        a, q, v = agent.q_values(env, cand)
        best = torch.full((env.n,), float("-inf"), device=dev).scatter_reduce_(0, a.index[:, 0], q, "amax")
        n_legal = torch.bincount(a.index[:, 0], minlength=env.n)
        filled = env.board.sum(1)
        rows = torch.stack([best, n_legal.float(), filled.float()], 1).cpu().numpy()
        for j, gid in enumerate(ids.tolist()):
            traces[gid].append(rows[j])
        slot, pos = best_per_env(env.n, a.index, q)
        env.step(cand, slot, pos)
print(f"{time.perf_counter() - t0:.0f}s, {sum(died)} deaths in {n_games} games")
np.savez_compressed(sys.argv[3], **{f"g{i}": np.array(t) for i, t in enumerate(traces)}, died=np.array(died))
