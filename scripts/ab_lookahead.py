"""A/B: rl_v1 1-ply vs fp32 head vs gated lookahead, same seeds; one JSON line per arm.

Arm names: base | fp32 | search-<below>-<k|all>, with ":fp32" appended to run a search arm with the fp32 head.
"""
import json
import sys
import time

import torch

from block_puzzle_ai.agents import LookaheadAgent, ValueAgent
from block_puzzle_ai.evaluate import run_games, summarize
from block_puzzle_ai.model import ValueNet

GAMES, SEED, CAP, OUT = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
arms = sys.argv[5].split(",")
bf16_forward = ValueNet.forward


def fp32_head_forward(self, planes, globals_):
    """Same net, but the MLP head (and so the output) in fp32 instead of bf16."""
    planes = planes.contiguous(memory_format=torch.channels_last)
    x = self.blocks(torch.relu(self.stem(planes)))
    x = torch.relu(self.reduce(x)).flatten(1)
    with torch.autocast("cuda", enabled=False):
        return self.head(torch.cat([x.float(), globals_.float()], 1)).squeeze(1)


def make(arm):
    ValueNet.forward = fp32_head_forward if "fp32" in arm else bf16_forward
    if arm.startswith("search"):
        _, below, k = arm.split(":")[0].split("-")
        return LookaheadAgent.from_checkpoint(
            "models/rl_v1.pt", "cuda", search_below=float(below), top_k=None if k == "all" else int(k)
        )
    return ValueAgent.from_checkpoint("models/rl_v1.pt", "cuda")


for arm in arms:
    agent = make(arm)
    start = time.perf_counter()
    summary = summarize(run_games(agent, GAMES, "cuda", SEED, CAP))
    summary |= {"arm": arm, "seconds": round(time.perf_counter() - start)}
    if isinstance(agent, LookaheadAgent):
        summary |= {"searched": agent.searched, "changed": agent.changed}
    print(json.dumps(summary), flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(summary) + "\n")
