"""Time each phase of one training step (512 envs) with the trained net."""
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

from block_puzzle_ai.batch_env import BatchGame
from block_puzzle_ai.features import encode
from block_puzzle_ai.model import load_checkpoint
from block_puzzle_ai.train import Config, choose, StateEval

dev = torch.device("cuda")
cfg = Config()
net, _ = load_checkpoint("models/rl_v1.pt", dev)
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

# update phase: 2 epochs over rollout 16 x 512 samples, minibatch 2048
net.train()
opt = torch.optim.AdamW(net.parameters(), lr=1e-9)
x = torch.rand(16 * 512, 3, 10, 10, device=dev).round()
g = torch.rand(16 * 512, net_g := net.head[0].in_features - net.trunk_out if hasattr(net, "trunk_out") else 0, device=dev) if False else None


def update():
    planes = torch.rand(16 * 512, 3, 10, 10, device=dev).round()
    globals_ = torch.rand(16 * 512, a_g, device=dev)
    target = torch.rand(16 * 512, device=dev)
    for _ in range(cfg.epochs):
        for idx in torch.randperm(len(target), device=dev).split(cfg.minibatch):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = net(planes[idx], globals_[idx]).float()
            loss = F.mse_loss(pred, target[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            loss.item()


a_g = encode(env, env.candidates()).globals.shape[1]
update()
n_upd = 4
for _ in range(n_upd):
    timed("update(per 16 steps)", update)

per_step = {k: v / steps for k, v in T.items() if k != "update(per 16 steps)"}
per_step["update"] = T["update(per 16 steps)"] / n_upd / 16
total = sum(per_step.values())
print(f"after-states per step: {K / steps:,.0f} (avg {K / steps / env.n:.0f} per env)")
for k, v in sorted(per_step.items(), key=lambda kv: -kv[1]):
    print(f"{k:12s} {v * 1e3:7.2f} ms/step  {v / total:5.1%}")
print(f"total {total * 1e3:.2f} ms/step -> {env.n / total:,.0f} moves/s (rollout wall incl. glue: {env.n * steps / wall:,.0f} moves/s)")
