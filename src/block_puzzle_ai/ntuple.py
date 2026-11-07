"""n-tuple after-state value function trained by TD(0) self-play on the C++ engine (#10 technique 7).

Design: docs/superpowers/specs/2026-11-03-ntuple-learner-design.md. This module picks the tuples (one
representative per orbit of a shape's translations under the 8 board symmetries) and holds the Python
twins of the C++ symmetries that the tests check the engine against.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import struct
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from block_puzzle_ai import _engine
from block_puzzle_ai.engine import cpp_tables
from block_puzzle_ai.evaluate import GameResult, summarize
from block_puzzle_ai.game import N
from block_puzzle_ai.pieces import all_pieces

_LOW64 = (1 << 64) - 1
MAGIC = b"BPANTW01"


def transform_cell(t: int, r: int, c: int) -> tuple[int, int]:
    """Symmetry t (0..7) of the 10x10 board: t & 4 transposes, then t & 2 flips rows, then t & 1 flips columns."""
    if t & 4:
        r, c = c, r
    if t & 2:
        r = N - 1 - r
    if t & 1:
        c = N - 1 - c
    return r, c


def transform_board(t: int, board: int) -> int:
    out = 0
    for i in range(N * N):
        if board >> i & 1:
            r, c = transform_cell(t, i // N, i % N)
            out |= 1 << (r * N + c)
    return out


def transform_piece(t: int, piece_id: int, table: str = "new") -> int:
    """Id of the image of a piece under symmetry t, normalized to its bounding-box top-left."""
    pieces = all_pieces(table)
    cells = [transform_cell(t, r, c) for r, c in pieces[piece_id].cells]
    r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
    image = sorted((r - r0, c - c0) for r, c in cells)
    for i, p in enumerate(pieces):
        if sorted(p.cells) == image:
            return i
    raise ValueError(f"piece {piece_id} has no image under symmetry {t} in table {table!r}")


def shape_cells(name: str) -> list[tuple[int, int]]:
    """Cells (row, col) of a named tuple shape: window4 (4x4), line (1x10) or rect<h>x<w>."""
    if name == "window4":
        h, w = 4, 4
    elif name == "line":
        h, w = 1, N
    elif m := re.fullmatch(r"rect(\d+)x(\d+)", name):
        h, w = int(m[1]), int(m[2])
        if not (1 <= h <= N and 1 <= w <= N and h * w <= 20):
            raise ValueError(f"shape {name}: needs 1 <= h, w <= 10 and h * w <= 20")
    else:
        raise ValueError(f"unknown tuple shape {name!r} (window4, line, rect<h>x<w>)")
    return [(r, c) for r in range(h) for c in range(w)]


def shapes_to_tuples(names: list[str]) -> list[list[int]]:
    """One tuple per orbit of each shape's translations under the 8 symmetries: its smallest image, cells sorted."""
    tuples = []
    for name in names:
        cells = shape_cells(name)
        h, w = 1 + max(r for r, _ in cells), 1 + max(c for _, c in cells)
        seen: set[frozenset[int]] = set()
        reps = []
        for r0 in range(N - h + 1):
            for c0 in range(N - w + 1):
                inst = [(r0 + r, c0 + c) for r, c in cells]
                images = [sorted(r * N + c for r, c in (transform_cell(t, r, c) for r, c in inst)) for t in range(8)]
                if frozenset(images[0]) in seen:
                    continue
                seen.update(frozenset(img) for img in images)
                reps.append(min(images))
        tuples += sorted(reps)
    return tuples


def fit_bucket(n: int) -> int:
    """Bucket of a placement count: 0, 1, 2, 3-4, 5-8, 9-16, 17-32, 33+ -> 0..7."""
    return n if n < 3 else min(7, (n - 1).bit_length() + 1)


def make_model(tuples: list[list[int]], table: str = "new") -> _engine.NTuple:
    """An n-tuple model with zero weights over the given tuples (absolute cell lists)."""
    return _engine.NTuple(cpp_tables(table), tuples)


def value(model: _engine.NTuple, board: int, remaining: Sequence[int], cooldown: bool) -> float:
    """V of an after-state: a 100-bit board, the 2 remaining piece ids and the cooldown flag."""
    return model.value(board & _LOW64, board >> 64, list(remaining), cooldown)


def evaluate(
    model: _engine.NTuple, games: int, seed: int, max_moves: int, threads: int, gamma: float = 1.0
) -> list[GameResult]:
    """Greedy 1-ply games on ``threads`` threads; results are in game order and independent of the thread count."""
    return [GameResult(*r) for r in _engine.evaluate_ntuple(model, games, seed, max_moves, threads, gamma)]


# ---- config and checkpoints ---------------------------------------------------------------------------


@dataclass
class NTupleConfig:
    shapes: list[str] = field(default_factory=lambda: ["window4", "line"])
    gamma: float = 1.0
    alpha: float = 0.1  # step-size cap; rollbacks halve it
    alpha_k: float = 12_500.0  # the step size follows min(alpha, alpha_k / mean V)
    restart_frac: float = 0.5
    threads: int = 24
    seed: int = 0
    tc: bool = False  # temporal-coherence step sizes per weight; alpha is then the meta step size
    init: str | None = None  # the checkpoint a run started from, if not from zero weights


def save_checkpoint(
    path: str | Path,
    cfg: NTupleConfig,
    tuples: list[list[int]],
    weights: np.ndarray,
    moves_trained: int,
    elapsed_min: float,
    tc: np.ndarray | None = None,
) -> None:
    """MAGIC, uint32 JSON length, JSON header, raw float32 weights, then the raw float64 (E, A) pairs of
    temporal coherence if given; written to path.tmp, then renamed."""
    path = Path(path)
    weights = np.ascontiguousarray(weights, dtype=np.float32)
    tc = None if tc is None else np.ascontiguousarray(tc, dtype=np.float64)
    if tc is not None and tc.shape != (len(weights), 2):
        raise ValueError(f"tc state has shape {tc.shape}, expected ({len(weights)}, 2)")
    header = json.dumps(
        {
            "config": asdict(cfg),
            "tuples": tuples,
            "moves_trained": moves_trained,
            "elapsed_min": elapsed_min,
            "n_weights": len(weights),
        }
        | ({} if tc is None else {"tc": True})
    ).encode()
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        f.write(MAGIC + struct.pack("<I", len(header)) + header)
        f.write(weights.tobytes())
        if tc is not None:
            f.write(tc.tobytes())
    tmp.replace(path)


def load_checkpoint(path: str | Path) -> tuple[NTupleConfig, list[list[int]], np.ndarray, dict]:
    raw = Path(path).read_bytes()
    if raw[:8] != MAGIC:
        raise ValueError(f"{path}: not an n-tuple checkpoint (bad magic)")
    (n,) = struct.unpack("<I", raw[8:12])
    header = json.loads(raw[12 : 12 + n])
    body, nw = raw[12 + n :], header["n_weights"]
    tc = None
    if header.get("tc"):
        if len(body) != 20 * nw:
            raise ValueError(f"{path}: {len(body)} bytes of weights and tc state, header says {nw} weights")
        tc = np.frombuffer(body[4 * nw :], dtype=np.float64).reshape(nw, 2)
        body = body[: 4 * nw]
    weights = np.frombuffer(body, dtype=np.float32) if len(body) % 4 == 0 else np.empty(0)
    if len(weights) != nw:
        raise ValueError(f"{path}: {len(weights)} weights, header says {nw}")
    meta = {"moves_trained": header["moves_trained"], "elapsed_min": header["elapsed_min"], "tc": tc}
    return NTupleConfig(**header["config"]), header["tuples"], weights, meta


def load_model(path: str | Path) -> tuple[_engine.NTuple, NTupleConfig, dict]:
    cfg, tuples, weights, meta = load_checkpoint(path)
    model = make_model(tuples)
    if model.n_weights != len(weights):
        raise ValueError(f"{path}: {len(weights)} weights, its tuples need {model.n_weights}")
    model.weights[:] = weights
    if meta["tc"] is not None:
        model.enable_tc()
        model.tc_state[:] = meta["tc"]
    return model, cfg, meta


# ---- training stability (docs/RESULTS.md, "n-tuple learner") ------------------------------------------


def scheduled_alpha(cap: float, k: float, value_mean: float) -> float:
    """Step size for the next chunk: min(cap, k / mean V).

    With gamma = 1 every game over moves its weights by about alpha * V / n_lookups. Once that is large next
    to the gaps between moves, a chance cluster of game overs reorders the policy, which causes more game
    overs: the trained net collapses to random play and does not recover. Scaling alpha with 1 / V keeps
    the per-game-over shift constant.
    """
    return cap if value_mean <= 0 else min(cap, k / value_mean)


class CollapseGuard:
    """Flags a chunk whose death rate or mean value breaks from the recent healthy chunks.

    A chunk is flagged when, with at least ``min_history`` healthy chunks on record, its game overs per move
    exceed ``death_factor`` times their median (and it had at least ``min_deaths`` game overs, so Poisson
    noise in long-game chunks does not count), or its mean V falls below ``value_drop`` times their maximum.
    Flagged chunks are not added to the history.
    """

    def __init__(
        self, window: int = 10, min_history: int = 5, death_factor: float = 4.0, value_drop: float = 0.8, min_deaths: int = 50
    ):
        self.history: deque[tuple[float, float]] = deque(maxlen=window)
        self.min_history, self.death_factor, self.value_drop, self.min_deaths = min_history, death_factor, value_drop, min_deaths

    def check(self, deaths: int, moves: int, value_mean: float) -> bool:
        rate = deaths / max(moves, 1)
        if len(self.history) >= self.min_history:
            median_rate = statistics.median(r for r, _ in self.history)
            max_value = max(v for _, v in self.history)
            if (deaths >= self.min_deaths and rate > self.death_factor * median_rate) or value_mean < self.value_drop * max_value:
                return True
        self.history.append((rate, value_mean))
        return False


# ---- CLI ----------------------------------------------------------------------------------------------


def _best_so_far(metrics: Path) -> float:
    if not metrics.exists():
        return 0.0
    return max((json.loads(line).get("best_eval_moves_per_death", 0.0) for line in metrics.read_text().splitlines()), default=0.0)


def train(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = ("alpha", "alpha_k", "restart_frac", "threads", "seed")
    overrides = {k: getattr(args, k) for k in names if getattr(args, k) is not None}
    if args.tc:
        overrides["tc"] = True
    if args.resume:
        model, cfg, meta = load_model(out / "latest.ntw")
        if args.shapes is not None and args.shapes.split(",") != cfg.shapes:
            raise SystemExit(f"--resume: the run uses shapes {cfg.shapes}; they cannot change")
        if args.gamma is not None and args.gamma != cfg.gamma:
            raise SystemExit(f"--resume: the run uses gamma {cfg.gamma}; it cannot change")
        cfg = NTupleConfig(**(asdict(cfg) | overrides))
        moves, elapsed0 = meta["moves_trained"], meta["elapsed_min"]
    elif args.init:
        model, cfg, _ = load_model(args.init)
        if args.shapes is not None and args.shapes.split(",") != cfg.shapes:
            raise SystemExit(f"--init: {args.init} uses shapes {cfg.shapes}; they cannot change")
        if args.gamma is not None and args.gamma != cfg.gamma:
            raise SystemExit(f"--init: {args.init} uses gamma {cfg.gamma}; it cannot change")
        cfg = NTupleConfig(**(asdict(cfg) | overrides | {"init": str(args.init)}))
        if args.alpha is None:  # the source run's cap may have been halved by rollbacks; start from the default
            cfg.alpha = min(0.1, 0.5 / cfg.threads)
        moves, elapsed0 = 0, 0.0
    else:
        base = {"shapes": args.shapes.split(",")} if args.shapes else {}
        if args.gamma is not None:
            base["gamma"] = args.gamma
        cfg = NTupleConfig(**(base | overrides))
        if args.alpha is None:  # from scratch, 24 threads at 0.1 inflated V 20x within 1M moves (docs/RESULTS.md)
            cfg.alpha = min(0.1, 0.5 / cfg.threads)
        model = make_model(shapes_to_tuples(cfg.shapes))
        moves, elapsed0 = 0, 0.0
    if cfg.tc:
        model.enable_tc()  # keeps a loaded state; from zero otherwise
    tuples = [list(t) for t in model.tuples]

    def tc_state() -> np.ndarray | None:
        return model.tc_state if model.tc_enabled else None

    def snapshot() -> tuple[np.ndarray, np.ndarray | None]:
        tc = tc_state()
        return model.weights.copy(), None if tc is None else tc.copy()
    # mixing in the moves so far gives a resumed run fresh random streams
    trainer = _engine.Trainer(model, cfg.gamma, 0.0, cfg.restart_frac, cfg.threads, cfg.seed + moves)
    probe = trainer.run(1000 * cfg.threads)  # alpha 0: measures V before the first learning chunk; not counted
    trainer.alpha = scheduled_alpha(cfg.alpha, cfg.alpha_k, probe["value_mean"])
    guard = CollapseGuard()
    snapshots: deque[tuple[np.ndarray, np.ndarray | None]] = deque([snapshot()], maxlen=3)
    metrics = (out / "metrics.jsonl").open("a")
    best = _best_so_far(out / "metrics.jsonl")
    start = time.perf_counter()

    def elapsed() -> float:
        return elapsed0 + (time.perf_counter() - start) / 60

    def done() -> bool:
        return moves >= args.total_moves if args.total_moves is not None else elapsed() >= 60 * args.hours

    def log(record: dict) -> None:
        metrics.write(json.dumps(record | {"elapsed_min": round(elapsed(), 2), "moves": moves}) + "\n")
        metrics.flush()

    def evaluate_and_save() -> None:
        nonlocal best
        t0 = time.perf_counter()
        results = evaluate(model, args.eval_games, args.eval_seed, args.eval_max_moves, cfg.threads, cfg.gamma)
        summary = summarize(results)
        improved = summary["moves_per_death"] > best
        best = max(best, summary["moves_per_death"])
        save_checkpoint(out / "latest.ntw", cfg, tuples, model.weights, moves, elapsed(), tc_state())
        if improved:
            save_checkpoint(out / "best.ntw", cfg, tuples, model.weights, moves, elapsed(), tc_state())
        log({"type": "eval"} | {f"eval_{k}": v for k, v in summary.items()} | {"best_eval_moves_per_death": best})
        print(
            f"[eval] {moves:,} moves, {elapsed():.1f} min: mean {summary['mean']:,.0f}, "
            f"{summary['moves_per_death']:,.0f} moves/game over{' (best)' if improved else ''}, "
            f"{time.perf_counter() - t0:.0f}s",
            flush=True,
        )

    chunk = 50_000 * cfg.threads
    last_eval = elapsed()
    try:
        while not done():
            if args.total_moves is not None:
                chunk = min(chunk, args.total_moves - moves)
            st = trainer.run(chunk)
            moves += st["moves"]
            rate = st["moves"] / max(st["seconds"], 1e-9)
            chunk = max(1000, int(rate * args.chunk_seconds))
            keys = ("games", "deaths", "mean_score", "mean_moves", "restarts", "value_mean")
            log({"type": "train", "moves_per_s": round(rate)} | {k: st[k] for k in keys} | {"alpha": trainer.alpha})
            print(f"[train] {moves:,} moves, {rate:,.0f}/s, games {st['games']}, mean {st['mean_score']:,.0f}", flush=True)
            if guard.check(st["deaths"], st["moves"], st["value_mean"]):
                weights, tc = snapshots[0]
                model.weights[:] = weights
                if tc is not None:
                    model.tc_state[:] = tc
                snapshots = deque([snapshots[0]], maxlen=3)
                cfg.alpha /= 2
                log({"type": "rollback", "deaths": st["deaths"], "value_mean": st["value_mean"], "alpha_cap": cfg.alpha})
                print(f"[rollback] {moves:,} moves: restored weights from up to 3 chunks ago, alpha cap {cfg.alpha:.4g}", flush=True)
                healthy_v = max((v for _, v in guard.history), default=st["value_mean"])
                trainer.alpha = scheduled_alpha(cfg.alpha, cfg.alpha_k, healthy_v)
            else:
                snapshots.append(snapshot())
                trainer.alpha = scheduled_alpha(cfg.alpha, cfg.alpha_k, st["value_mean"])
            if elapsed() - last_eval >= args.eval_minutes:
                evaluate_and_save()
                last_eval = elapsed()
        evaluate_and_save()
    except KeyboardInterrupt:
        save_checkpoint(out / "latest.ntw", cfg, tuples, model.weights, moves, elapsed(), tc_state())
        print(f"interrupted: saved {out / 'latest.ntw'} at {moves:,} moves", flush=True)
    finally:
        metrics.close()


def evaluate_cmd(args: argparse.Namespace) -> None:
    from block_puzzle_ai import ntuple_search

    if ntuple_search.search_requested(args):
        tiers = ntuple_search.tiers_from_args(args.search_k, args.search_below, args.deep_below, args.deep_k)
        out = ntuple_search.run_search_eval(
            args.weights, args.games, args.seed, args.max_moves, args.threads, tiers, args.games_file
        )
    else:
        model, cfg, _ = load_model(args.weights)
        start = time.perf_counter()
        results = evaluate(model, args.games, args.seed, args.max_moves, args.threads, cfg.gamma)
        out = summarize(results) | {"agent": f"ntuple:{args.weights}", "seconds": round(time.perf_counter() - start, 1)}
        out["results"] = [asdict(r) for r in results]
    if args.json:
        print(json.dumps(out))
    else:
        for k, v in out.items():
            if k != "results":
                print(f"{k:>20}: {v:,.1f}" if isinstance(v, float) else f"{k:>20}: {v}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m block_puzzle_ai.ntuple", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="TD(0) self-play training on all CPU threads")
    t.add_argument("--out", required=True)
    stop = t.add_mutually_exclusive_group(required=True)
    stop.add_argument("--hours", type=float, help="stop at this much training time, counting resumed runs")
    stop.add_argument("--total-moves", type=int, help="stop at this many training moves, counting resumed runs")
    start = t.add_mutually_exclusive_group()
    start.add_argument("--resume", action="store_true", help="continue from OUT/latest.ntw")
    start.add_argument("--init", help="start a new run from this checkpoint's weights, shapes and gamma")
    t.add_argument("--tc", action="store_true", help="temporal-coherence step sizes per weight (ALPHA is the meta step size)")
    t.add_argument("--shapes", help="comma-separated tuple shapes (default window4,line)")
    t.add_argument("--gamma", type=float)
    t.add_argument("--alpha", type=float, help="step-size cap (default min(0.1, 0.5 / threads))")
    t.add_argument("--alpha-k", type=float, help="the step size follows min(cap, ALPHA_K / mean V) (default 12500)")
    t.add_argument("--restart-frac", type=float)
    t.add_argument("--threads", type=int)
    t.add_argument("--seed", type=int)
    t.add_argument("--eval-minutes", type=float, default=15)
    t.add_argument("--eval-games", type=int, default=256)
    t.add_argument("--eval-seed", type=int, default=2024)
    t.add_argument("--eval-max-moves", type=int, default=200_000)
    t.add_argument("--chunk-seconds", type=float, default=10, help="training chunk length; the collapse guard checks each chunk")
    e = sub.add_parser("evaluate", help="games with a checkpoint: greedy 1-ply, or searched with the search options")
    e.add_argument("--weights", required=True)
    e.add_argument("--games", type=int, default=64)
    e.add_argument("--seed", type=int, default=777)
    e.add_argument("--max-moves", type=int, default=200_000)
    e.add_argument("--threads", type=int, default=24)
    e.add_argument("--json", action="store_true", help="print evaluate.py-style JSON, with per-game results")
    s = e.add_argument_group("search (docs/superpowers/specs/2026-11-05-ntuple-search-design.md)")
    s.add_argument("--search-k", type=int, help="depth-2 search on the top K moves, on every move")
    s.add_argument("--search-below", type=float, help="search only when the best 1-ply value is below this (K default 8)")
    s.add_argument("--deep-below", type=float, help="depth-3 search when the best 1-ply value is below this")
    s.add_argument("--deep-k", type=lambda v: [int(x) for x in v.split(",")], help="depth-3 widths K1,K2")
    s.add_argument("--games-file", help="per-game JSONL; rerunning with the same arguments plays only missing games")
    args = parser.parse_args(argv)
    train(args) if args.command == "train" else evaluate_cmd(args)


if __name__ == "__main__":
    main()
