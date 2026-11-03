"""Time each phase of one training step (512 envs) with the trained net.

``--eager`` times the pre-#9 path: eager inference in 16k-row chunks and an uncompiled, unfused update.
"""
import sys
import time
from collections import defaultdict

import torch

from block_puzzle_ai.batch_env import BatchGame
from block_puzzle_ai.features import encode
from block_puzzle_ai.model import load_checkpoint
from block_puzzle_ai.train import Config, Learner, StateEval, choose

EAGER = "--eager" in sys.argv
dev = torch.device("cuda")
cfg = Config()
net, _ = load_checkpoint("models/rl_v1.pt", dev)
if EAGER:

    def eager_predict(planes, globals_):  # ValueNet.predict before #9
        out = []
        for p, g in zip(planes.split(16384), globals_.split(16384)):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(net(p, g).float())
        return torch.cat(out)

    net.predict = eager_predict
env = BatchGame(cfg.envs, dev, seed=0)
rng = torch.Generator(dev).manual_seed(1)
T = defaultdict(float)


def timed(name, fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    T[name] += time.perf_counter() - t0
    return out


def step(record):
    cand = timed("candidates", env.candidates) if record else env.candidates()
    done = cand.terminal
    if done.any():
        env.reset(done)
        cand = env.candidates()
    a = timed("encode", lambda: encode(env, cand)) if record else encode(env, cand)
    v = timed("net_infer", lambda: net.predict(a.planes, a.globals)) if record else net.predict(a.planes, a.globals)
    q = a.points * cfg.reward_scale + cfg.gamma * v
    best = torch.full((env.n,), float("-inf"), device=dev)
    best.scatter_reduce_(0, a.index[:, 0], q, "amax")
    ev = StateEval(cand, a, q, best, cand.terminal)
    rows, move, _ = choose(ev, env.n, 0.005, rng)
    f = lambda: env.step(cand, move // 100, move % 100)
    timed("env_step", f) if record else f()
    return len(a.index)


with torch.no_grad():
    for _ in range(300):  # warm up into mid-game states
        step(False)
K = 0
steps = 64
t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(steps):
        K += step(True)
torch.cuda.synchronize()
wall = time.perf_counter() - t0

# update phase: the real Learner on one rollout of random data (cfg.rollout x cfg.envs samples)
net.train()
learner = Learner(net, Config(lr=1e-9), rng)
if EAGER:
    learner.forward = net
    learner.opt = torch.optim.AdamW(net.parameters(), lr=1e-9, weight_decay=1e-5)
a_g = encode(env, env.candidates()).globals.shape[1]
planes = torch.rand(cfg.rollout * cfg.envs, 3, 10, 10, device=dev) < 0.4
globals_ = torch.rand(cfg.rollout * cfg.envs, a_g, device=dev)
target = torch.rand(cfg.rollout * cfg.envs, device=dev)


def update():
    learner.update(planes, globals_, target, lr=1e-9)


update()
n_upd = 4
for _ in range(n_upd):
    timed("update(per 16 steps)", update)

per_step = {k: v / steps for k, v in T.items() if k != "update(per 16 steps)"}
per_step["update"] = T["update(per 16 steps)"] / n_upd / cfg.rollout
total = sum(per_step.values())
print(f"after-states per step: {K / steps:,.0f} (avg {K / steps / env.n:.0f} per env)")
for k, v in sorted(per_step.items(), key=lambda kv: -kv[1]):
    print(f"{k:12s} {v * 1e3:7.2f} ms/step  {v / total:5.1%}")
print(f"total {total * 1e3:.2f} ms/step -> {env.n / total:,.0f} moves/s (rollout wall incl. glue: {env.n * steps / wall:,.0f} moves/s)")
