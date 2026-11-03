"""Train the after-state value network by self-play.

The policy is greedy over legal moves of  r + gamma * V(after-state)  (epsilon-greedy while training).
V(x_t) is regressed on Watkins Q(lambda) returns built from the greedy one-step targets
m_{t+1} = max_a [r_a + gamma * V(x_a)] seen while acting:

    G_t = 0                                                   if s_{t+1} is game over
    G_t = m_{t+1}                                             if the game was cut or a_{t+1} explored
    G_t = (1 - lam) * m_{t+1} + lam * (r_{t+1} + gamma * G_{t+1})   otherwise

The run lasts --minutes (or --total-moves); the learning rate warms up, then decays on a cosine to --lr-end.
best.pt is chosen on eval moves per game over, which the eval move cap does not saturate.

uv run python -m block_puzzle_ai.train --minutes 90 --out runs/main
uv run python -m block_puzzle_ai.train --resume runs/main      # continue an interrupted run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from block_puzzle_ai.agents import ValueAgent
from block_puzzle_ai.batch_env import CELLS, BatchGame, Candidates
from block_puzzle_ai.evaluate import run_games, summarize
from block_puzzle_ai.features import Afterstates, encode, n_globals
from block_puzzle_ai.model import ValueNet, load_checkpoint, save_checkpoint

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


@dataclass
class Config:
    envs: int = 512
    rollout: int = 16
    gamma: float = 0.99
    lam: float = 0.8
    reward_scale: float = 0.01  # keep reward_scale = 1 - gamma so that V stays near points per move (~108)
    lr: float = 3e-4
    lr_end: float = 3e-5
    warmup_moves: int = 1_000_000
    epochs: int = 2
    minibatch: int = 2048
    eps_start: float = 0.10
    eps_end: float = 0.001
    eps_decay_moves: int = 4_000_000
    max_moves: int = 50_000
    restart_frac: float = 0.0  # share of finished games restarted from a state 50-500 moves before a game over
    channels: int = 32
    blocks: int = 3
    hidden: int = 256
    fp32_head: bool = True  # also switched on for an --init net; see ValueNet.set_fp32_head
    eval_minutes: float = 30.0  # also writes a snapshot to ckpt/
    eval_games: int = 256
    eval_max_moves: int = 20_000
    eval_seed: int = 2024
    save_minutes: float = 10.0  # resume.pt
    minutes: float = 60.0
    total_moves: int = 0  # if set, the run lasts this many moves instead of `minutes`
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    init: str | None = None


@dataclass
class StateEval:
    cand: Candidates
    a: Afterstates
    q: torch.Tensor  # (K,) r + gamma V for every legal move
    best: torch.Tensor  # (B,) max q per game, 0 for game over
    terminal: torch.Tensor  # (B,)


@torch.no_grad()
def evaluate_state(env: BatchGame, net: ValueNet, cfg: Config, cand: Candidates | None = None) -> StateEval:
    cand = cand if cand is not None else env.candidates()
    a = encode(env, cand)
    q = a.points * cfg.reward_scale + cfg.gamma * net.predict(a.planes, a.globals)
    best = torch.full((env.n,), float("-inf"), device=env.device)
    best.scatter_reduce_(0, a.index[:, 0], q, "amax")
    terminal = cand.terminal
    best[terminal] = 0.0
    return StateEval(cand, a, q, best, terminal)


def choose(ev: StateEval, n: int, eps: float, rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Epsilon-greedy over legal moves. Returns (row into ev.a, flat move index, explored flag) per game."""
    device = ev.q.device
    flat = ev.a.index[:, 1] * CELLS + ev.a.index[:, 2]
    rows = torch.full((n, 3 * CELLS), -1, dtype=torch.long, device=device)
    rows[ev.a.index[:, 0], flat] = torch.arange(len(flat), device=device)
    q = torch.full((n, 3 * CELLS), float("-inf"), device=device)
    q[ev.a.index[:, 0], flat] = ev.q
    greedy = q.argmax(1)
    random_move = torch.multinomial((rows >= 0).float(), 1, generator=rng).squeeze(1)
    explore = torch.rand(n, generator=rng, device=device) < eps
    move = torch.where(explore, random_move, greedy)
    explored = explore & (move != greedy)
    return rows[torch.arange(n, device=device), move], move, explored


def lambda_returns(r, best, terminal, cut, explored, cfg: Config) -> torch.Tensor:
    """r, explored: (T, B) for actions 0..T-1; best, terminal, cut: (T + 1, B) for states 0..T."""
    steps = r.shape[0]
    g = torch.zeros_like(r)
    nxt = torch.where(terminal[steps], 0.0, best[steps])
    g[steps - 1] = nxt
    for t in range(steps - 2, -1, -1):
        mix = (1 - cfg.lam) * best[t + 1] + cfg.lam * (r[t + 1] + cfg.gamma * g[t + 1])
        stop = cut[t + 1] | explored[t + 1]
        g[t] = torch.where(terminal[t + 1], 0.0, torch.where(stop, best[t + 1], mix))
    return g


def learning_rate(cfg: Config, progress: float, moves: int) -> float:
    """Linear warm-up over the first ``warmup_moves``, then a cosine from lr to lr_end over the run (progress 0..1)."""
    cosine = cfg.lr_end + 0.5 * (cfg.lr - cfg.lr_end) * (1 + math.cos(math.pi * min(1.0, progress)))
    return cosine * min(1.0, (moves + 1) / cfg.warmup_moves) if cfg.warmup_moves else cosine


class RestartPool:
    """Restarts finished games from states shortly before a game over, where deaths are learned (#10, technique 3).

    Every ``every`` steps all games are copied into a ring of ``depth`` snapshots. When a game ends in a game over,
    its snapshots from ``every`` to ``every * depth`` moves earlier join a pool of ``capacity`` states, and each
    finished game restarts from a random pool state with probability ``frac`` (otherwise from an empty board).
    """

    def __init__(self, env: BatchGame, frac: float, every: int = 50, depth: int = 10, capacity: int = 1 << 16):
        self.frac, self.every, self.depth, self.capacity = frac, every, depth, capacity
        state = env.state()
        self.ring = {k: v.new_zeros(depth, *v.shape) for k, v in state.items()}
        self.ring_game = torch.full((depth, env.n), -1, dtype=torch.long, device=env.device)
        self.pool = {k: v.new_zeros(capacity, *v.shape[1:]) for k, v in state.items()}
        self.game = torch.zeros(env.n, dtype=torch.long, device=env.device)  # games each env has finished
        self.restarted = torch.zeros(env.n, dtype=torch.bool, device=env.device)  # current game came from the pool
        self.size = self.head = self.steps = self.restarts = 0

    def snapshot(self, env: BatchGame) -> None:
        """Call once per step, on states that are not game over."""
        if self.steps % self.every == 0:
            slot = self.steps // self.every % self.depth
            for k, v in env.state().items():
                self.ring[k][slot] = v
            self.ring_game[slot] = self.game
        self.steps += 1

    def reset(self, env: BatchGame, done: torch.Tensor, terminal: torch.Tensor, rng: torch.Generator) -> None:
        """env.reset(done), then move a share of those games onto pool states; ``terminal`` marks the game overs."""
        age = env.moves - self.ring["moves"]
        new = (self.ring_game == self.game) & terminal & (age >= self.every)
        n = int(new.sum())
        if n:
            idx = (self.head + torch.arange(n, device=env.device)) % self.capacity
            for k, v in self.ring.items():
                self.pool[k][idx] = v[new]
            self.head = (self.head + n) % self.capacity
            self.size = min(self.capacity, self.size + n)
        env.reset(done)
        self.game[done] += 1
        self.restarted[done] = False
        if not self.size:
            return
        idx = (done & (torch.rand(env.n, generator=rng, device=env.device) < self.frac)).nonzero().squeeze(1)
        src = torch.randint(self.size, (len(idx),), generator=rng, device=env.device)
        env.set_state(idx, {k: v[src] for k, v in self.pool.items()})
        self.restarted[idx] = True
        self.restarts += len(idx)

    def state_dict(self) -> dict:
        return dict(vars(self))

    def load_state_dict(self, state: dict) -> None:
        vars(self).update(state)


class Learner:
    """Regresses the net on a rollout's targets: ``cfg.epochs`` passes over shuffled minibatches."""

    def __init__(self, net: ValueNet, cfg: Config, rng: torch.Generator):
        self.net, self.cfg, self.rng = net, cfg, rng
        cuda = next(net.parameters()).is_cuda
        self.opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-5, fused=cuda)
        # CUDA graphs over forward + backward: each minibatch is launch-bound, and they all have one size
        self.forward = torch.compile(net, mode="reduce-overhead", dynamic=False) if cuda else net

    def update(self, planes: torch.Tensor, globals_: torch.Tensor, target: torch.Tensor, lr: float) -> float:
        """One update on (planes, globals_) -> target; returns the mean minibatch loss with a single host sync."""
        cuda = target.is_cuda
        for group in self.opt.param_groups:
            group["lr"] = lr
        losses = []
        for _ in range(self.cfg.epochs):
            perm = torch.randperm(len(target), device=target.device, generator=self.rng)
            for idx in perm.split(self.cfg.minibatch):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                    pred = self.forward(planes[idx].float(), globals_[idx]).float()
                loss = F.mse_loss(pred, target[idx])
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.opt.step()
                losses.append(loss.detach())
        return float(torch.stack(losses).mean())


def train(cfg: Config, out: Path, resume: bool = False) -> None:
    """Train into ``out``; with ``resume``, continue the run saved in out/resume.pt."""
    out.mkdir(parents=True, exist_ok=True)
    if not resume:
        (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    env = BatchGame(cfg.envs, device, seed=cfg.seed)
    if resume:
        net, saved = load_checkpoint(out / "resume.pt", device)
    elif cfg.init:
        net, _ = load_checkpoint(cfg.init, device)
        net.set_fp32_head(cfg.fp32_head)
    else:
        net = ValueNet(n_globals(env.t), cfg.channels, cfg.blocks, cfg.hidden, cfg.fp32_head)
        net = net.to(device, memory_format=torch.channels_last)
    rng = torch.Generator(device).manual_seed(cfg.seed + 1)
    learner = Learner(net, cfg, rng)
    pool = RestartPool(env, cfg.restart_frac) if cfg.restart_frac > 0 else None

    recent = deque(maxlen=500)
    total_moves = 0
    episodes = 0
    best_train = 0
    best_eval = -math.inf
    elapsed = 0.0
    if resume:
        state = saved["train"]
        learner.opt.load_state_dict(state["opt"])
        total_moves, episodes, best_train, best_eval, elapsed = (
            state[k] for k in ("moves", "episodes", "best_train", "best_eval", "elapsed")
        )
        recent.extend(map(tuple, state["recent"].tolist()))
        rng.set_state(state["rng"].cpu())
        env.rng.set_state(state["env_rng"].cpu())
        env.set_state(torch.arange(env.n, device=device), state["env"])
        if pool:
            pool.load_state_dict(state["pool"])
        # drop what was logged after the save
        records = (out / "metrics.jsonl").read_text().splitlines()
        kept = [r for r in records if json.loads(r)["moves"] <= total_moves]
        (out / "metrics.jsonl").write_text("".join(r + "\n" for r in kept))
    metrics_file = (out / "metrics.jsonl").open("a")
    start = time.perf_counter() - elapsed
    last_eval = last_log = last_save = time.perf_counter()
    log_moves = 0
    t_steps, n = cfg.rollout, cfg.envs

    def progress() -> float:
        if cfg.total_moves:
            return total_moves / cfg.total_moves
        return (time.perf_counter() - start) / (cfg.minutes * 60)

    def log(record: dict) -> None:
        record |= {"elapsed_min": round((time.perf_counter() - start) / 60, 2), "moves": total_moves}
        metrics_file.write(json.dumps(record) + "\n")
        metrics_file.flush()
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()), flush=True)

    def run_eval() -> None:
        nonlocal best_eval
        net.eval()
        agent = ValueAgent(net, cfg.gamma, cfg.reward_scale)
        summary = summarize(run_games(agent, cfg.eval_games, cfg.device, cfg.eval_seed, cfg.eval_max_moves))
        net.train()
        meta = {"gamma": cfg.gamma, "reward_scale": cfg.reward_scale, "eval": summary}
        save_checkpoint(out / "latest.pt", net, **meta)
        save_checkpoint(out / "ckpt" / f"{total_moves // 1_000_000:05d}M.pt", net, **meta)
        if summary["moves_per_death"] > best_eval:
            best_eval = summary["moves_per_death"]
            save_checkpoint(out / "best.pt", net, **meta)
        log({"type": "eval"} | {f"eval_{k}": v for k, v in summary.items()} | {"best_eval_moves_per_death": best_eval})

    def save() -> None:
        """Everything needed to continue the run; written to a temp file first so a crash cannot corrupt it."""
        state = {
            "opt": learner.opt.state_dict(),
            "moves": total_moves,
            "episodes": episodes,
            "best_train": best_train,
            "best_eval": best_eval,
            "elapsed": time.perf_counter() - start,
            "recent": torch.tensor(list(recent), dtype=torch.long).view(-1, 2),
            "rng": rng.get_state(),
            "env_rng": env.rng.get_state(),
            "env": env.state(),
            "pool": pool.state_dict() if pool else None,
        }
        save_checkpoint(out / "resume.tmp", net, gamma=cfg.gamma, reward_scale=cfg.reward_scale, train=state)
        (out / "resume.tmp").replace(out / "resume.pt")

    def observe() -> tuple[StateEval, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate the reached states, restarting finished games first so the net runs once per step.

        Returns the evaluation to act on plus (best, terminal, cut) of the states reached before restarting.
        """
        nonlocal episodes, best_train
        cand = env.candidates()
        terminal = cand.terminal
        cut = (env.moves >= cfg.max_moves) & ~terminal
        cut_best = evaluate_state(env, net, cfg, cand).best if cut.any() else None  # rare: bootstrap before reset
        done = terminal | cut
        if done.any():
            fresh = done if pool is None else done & ~pool.restarted  # games from the pool skew the score stats
            for i in fresh.nonzero().squeeze(1).tolist():
                score = int(env.score[i])
                recent.append((score, int(env.moves[i])))
                best_train = max(best_train, score)
            episodes += int(done.sum())
            if pool:
                pool.reset(env, done, terminal, rng)
            else:
                env.reset(done)
            cand = env.candidates()
        ev = evaluate_state(env, net, cfg, cand)
        best = torch.where(terminal, 0.0, ev.best)
        if cut_best is not None:
            best = torch.where(cut, cut_best, best)
        return ev, best, terminal, cut

    ev, *_ = observe()
    while progress() < 1:
        planes = torch.zeros(t_steps, n, 3, 10, 10, dtype=torch.bool, device=device)
        globals_ = torch.zeros(t_steps, n, ev.a.globals.shape[1], device=device)
        r = torch.zeros(t_steps, n, device=device)
        explored = torch.zeros(t_steps, n, dtype=torch.bool, device=device)
        best = torch.zeros(t_steps + 1, n, device=device)
        terminal = torch.zeros(t_steps + 1, n, dtype=torch.bool, device=device)
        cut = torch.zeros(t_steps + 1, n, dtype=torch.bool, device=device)
        eps = cfg.eps_end + (cfg.eps_start - cfg.eps_end) * max(0.0, 1 - total_moves / cfg.eps_decay_moves)

        for t in range(t_steps):
            rows, move, explored[t] = choose(ev, n, eps, rng)
            planes[t] = ev.a.planes[rows].bool()
            globals_[t] = ev.a.globals[rows]
            r[t] = ev.a.points[rows] * cfg.reward_scale
            env.step(ev.cand, move // CELLS, move % CELLS)
            ev, best[t + 1], terminal[t + 1], cut[t + 1] = observe()
            if pool:
                pool.snapshot(env)
        lr = learning_rate(cfg, progress(), total_moves)
        total_moves += t_steps * n
        log_moves += t_steps * n

        target = lambda_returns(r, best, terminal, cut, explored, cfg).flatten()
        loss = learner.update(planes.flatten(0, 1), globals_.flatten(0, 1), target, lr)

        now = time.perf_counter()
        if now - last_log > 30 and recent:
            scores = [s for s, _ in recent]
            log(
                {
                    "type": "train",
                    "moves_per_s": round(log_moves / (now - last_log)),
                    "episodes": episodes,
                    "eps": eps,
                    "lr": lr,
                    "loss": loss,
                    "recent_mean": statistics.fmean(scores),
                    "recent_median": statistics.median(scores),
                    "recent_moves": statistics.fmean(m for _, m in recent),
                    "best_train_score": best_train,
                    "value_mean": float(target.mean()),
                }
                | ({"pool": pool.size, "restarts": pool.restarts} if pool else {})
            )
            last_log, log_moves = now, 0
        if now - last_eval > cfg.eval_minutes * 60:
            run_eval()
            last_eval = time.perf_counter()
        if now - last_save > cfg.save_minutes * 60:
            save()
            last_save = time.perf_counter()
    run_eval()
    save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="runs/main")
    parser.add_argument(
        "--resume", metavar="RUN_DIR", help="continue RUN_DIR with its saved config (other options are ignored)"
    )
    for field, value in asdict(Config()).items():
        flag = f"--{field.replace('_', '-')}"
        if isinstance(value, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=value)
        else:
            parser.add_argument(flag, type=type(value) if value is not None else str, default=value)
    args = vars(parser.parse_args())
    out, resume = Path(args.pop("out")), args.pop("resume")
    if resume:
        out = Path(resume)
        train(Config(**json.loads((out / "config.json").read_text())), out, resume=True)
    else:
        train(Config(**args), out)


if __name__ == "__main__":
    main()
