"""n-tuple after-state value function trained by TD(0) self-play on the C++ engine (#10 technique 7).

Design: docs/superpowers/specs/2026-11-03-ntuple-learner-design.md. This module picks the tuples (one
representative per orbit of a shape's translations under the 8 board symmetries) and holds the Python
twins of the C++ symmetries that the tests check the engine against.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import time
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
    alpha: float = 0.1
    restart_frac: float = 0.5
    threads: int = 24
    seed: int = 0


def save_checkpoint(
    path: str | Path, cfg: NTupleConfig, tuples: list[list[int]], weights: np.ndarray, moves_trained: int, elapsed_min: float
) -> None:
    """MAGIC, uint32 JSON length, JSON header, raw float32 weights; written to path.tmp, then renamed."""
    path = Path(path)
    weights = np.ascontiguousarray(weights, dtype=np.float32)
    header = json.dumps(
        {
            "config": asdict(cfg),
            "tuples": tuples,
            "moves_trained": moves_trained,
            "elapsed_min": elapsed_min,
            "n_weights": len(weights),
        }
    ).encode()
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        f.write(MAGIC + struct.pack("<I", len(header)) + header)
        f.write(weights.tobytes())
    tmp.replace(path)


def load_checkpoint(path: str | Path) -> tuple[NTupleConfig, list[list[int]], np.ndarray, dict]:
    raw = Path(path).read_bytes()
    if raw[:8] != MAGIC:
        raise ValueError(f"{path}: not an n-tuple checkpoint (bad magic)")
    (n,) = struct.unpack("<I", raw[8:12])
    header = json.loads(raw[12 : 12 + n])
    weights = np.frombuffer(raw[12 + n :], dtype=np.float32) if (len(raw) - 12 - n) % 4 == 0 else np.empty(0)
    if len(weights) != header["n_weights"]:
        raise ValueError(f"{path}: {len(weights)} weights, header says {header['n_weights']}")
    meta = {"moves_trained": header["moves_trained"], "elapsed_min": header["elapsed_min"]}
    return NTupleConfig(**header["config"]), header["tuples"], weights, meta


def load_model(path: str | Path) -> tuple[_engine.NTuple, NTupleConfig, dict]:
    cfg, tuples, weights, meta = load_checkpoint(path)
    model = make_model(tuples)
    if model.n_weights != len(weights):
        raise ValueError(f"{path}: {len(weights)} weights, its tuples need {model.n_weights}")
    model.weights[:] = weights
    return model, cfg, meta


# ---- CLI ----------------------------------------------------------------------------------------------


def _best_so_far(metrics: Path) -> float:
    if not metrics.exists():
        return 0.0
    return max((json.loads(line).get("best_eval_moves_per_death", 0.0) for line in metrics.read_text().splitlines()), default=0.0)


def train(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    overrides = {k: getattr(args, k) for k in ("alpha", "restart_frac", "threads", "seed") if getattr(args, k) is not None}
    if args.resume:
        model, cfg, meta = load_model(out / "latest.ntw")
        if args.shapes is not None and args.shapes.split(",") != cfg.shapes:
            raise SystemExit(f"--resume: the run uses shapes {cfg.shapes}; they cannot change")
        if args.gamma is not None and args.gamma != cfg.gamma:
            raise SystemExit(f"--resume: the run uses gamma {cfg.gamma}; it cannot change")
        cfg = NTupleConfig(**(asdict(cfg) | overrides))
        moves, elapsed0 = meta["moves_trained"], meta["elapsed_min"]
    else:
        base = {"shapes": args.shapes.split(",")} if args.shapes else {}
        if args.gamma is not None:
            base["gamma"] = args.gamma
        cfg = NTupleConfig(**(base | overrides))
        if args.alpha is None:  # lock-free updates diverge once alpha * threads exceeds ~1 (docs/RESULTS.md)
            cfg.alpha = min(0.1, 0.5 / cfg.threads)
        model = make_model(shapes_to_tuples(cfg.shapes))
        moves, elapsed0 = 0, 0.0
    tuples = [list(t) for t in model.tuples]
    # mixing in the moves so far gives a resumed run fresh random streams
    trainer = _engine.Trainer(model, cfg.gamma, cfg.alpha, cfg.restart_frac, cfg.threads, cfg.seed + moves)
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
        save_checkpoint(out / "latest.ntw", cfg, tuples, model.weights, moves, elapsed())
        if improved:
            save_checkpoint(out / "best.ntw", cfg, tuples, model.weights, moves, elapsed())
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
            if elapsed() - last_eval >= args.eval_minutes:
                evaluate_and_save()
                last_eval = elapsed()
        evaluate_and_save()
    except KeyboardInterrupt:
        save_checkpoint(out / "latest.ntw", cfg, tuples, model.weights, moves, elapsed())
        print(f"interrupted: saved {out / 'latest.ntw'} at {moves:,} moves", flush=True)
    finally:
        metrics.close()


def evaluate_cmd(args: argparse.Namespace) -> None:
    model, cfg, _ = load_model(args.weights)
    start = time.perf_counter()
    results = evaluate(model, args.games, args.seed, args.max_moves, args.threads, cfg.gamma)
    summary = summarize(results) | {"agent": f"ntuple:{args.weights}", "seconds": round(time.perf_counter() - start, 1)}
    if args.json:
        print(json.dumps(summary | {"results": [asdict(r) for r in results]}))
    else:
        for k, v in summary.items():
            print(f"{k:>20}: {v:,.1f}" if isinstance(v, float) else f"{k:>20}: {v}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m block_puzzle_ai.ntuple", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="TD(0) self-play training on all CPU threads")
    t.add_argument("--out", required=True)
    stop = t.add_mutually_exclusive_group(required=True)
    stop.add_argument("--hours", type=float, help="stop at this much training time, counting resumed runs")
    stop.add_argument("--total-moves", type=int, help="stop at this many training moves, counting resumed runs")
    t.add_argument("--resume", action="store_true", help="continue from OUT/latest.ntw")
    t.add_argument("--shapes", help="comma-separated tuple shapes (default window4,line)")
    t.add_argument("--gamma", type=float)
    t.add_argument("--alpha", type=float, help="TD step size (default min(0.1, 0.5 / threads))")
    t.add_argument("--restart-frac", type=float)
    t.add_argument("--threads", type=int)
    t.add_argument("--seed", type=int)
    t.add_argument("--eval-minutes", type=float, default=15)
    t.add_argument("--eval-games", type=int, default=256)
    t.add_argument("--eval-seed", type=int, default=2024)
    t.add_argument("--eval-max-moves", type=int, default=200_000)
    t.add_argument("--chunk-seconds", type=float, default=60)
    e = sub.add_parser("evaluate", help="greedy 1-ply games with a checkpoint")
    e.add_argument("--weights", required=True)
    e.add_argument("--games", type=int, default=64)
    e.add_argument("--seed", type=int, default=777)
    e.add_argument("--max-moves", type=int, default=200_000)
    e.add_argument("--threads", type=int, default=24)
    e.add_argument("--json", action="store_true", help="print evaluate.py-style JSON, with per-game results")
    args = parser.parse_args(argv)
    train(args) if args.command == "train" else evaluate_cmd(args)


if __name__ == "__main__":
    main()
