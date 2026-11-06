"""Expectimax search over the n-tuple value: tiers, resumable evaluations and their summary (#10).

Design: docs/superpowers/specs/2026-11-05-ntuple-search-design.md; the search itself is cpp/search.hpp.
``python -m block_puzzle_ai.ntuple evaluate`` calls run_search_eval when given any search option.

A tier (below, widths) applies when the root's best 1-ply value Q1 is below ``below``; the last such tier
is used. widths [k] is depth 2 on the top k moves, [k1, k2] depth 3. Thresholds are on the weights' value
scale, so new weights need new thresholds.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from block_puzzle_ai import _engine
from block_puzzle_ai.evaluate import GameResult, summarize

Tier = tuple[float, list[int]]
INF = float("inf")
GAME_KEYS = ("game", "score", "moves", "lines", "truncated", "searched", "changed", "seconds", "last_q1")


def tiers_from_args(
    search_k: int | None, search_below: float | None, deep_below: float | None, deep_k: list[int] | None
) -> list[Tier]:
    """The tiers of the CLI options: --search-k K is (inf, [K]); --search-below G gates it at G (K defaults to 8);
    --deep-below G2 --deep-k K1,K2 adds (G2, [K1, K2])."""
    tiers: list[Tier] = []
    if search_k is not None or search_below is not None:
        tiers.append((INF if search_below is None else search_below, [8 if search_k is None else search_k]))
    if (deep_below is None) != (deep_k is None):
        raise SystemExit("--deep-below and --deep-k must be given together")
    if deep_k is not None:
        if len(deep_k) != 2:
            raise SystemExit(f"--deep-k takes two widths K1,K2, got {deep_k}")
        tiers.append((deep_below, list(deep_k)))
    try:
        _engine.check_search_tiers(tiers)
    except ValueError as e:
        raise SystemExit(f"search options: {e}") from None
    return tiers


def _header(weights: str | Path, games: int, seed: int, max_moves: int, gamma: float, tiers: list[Tier]) -> dict:
    return {
        "weights": str(weights),
        "games": games,
        "seed": seed,
        "max_moves": max_moves,
        "gamma": gamma,
        "tiers": [[None if below == INF else below, list(widths)] for below, widths in tiers],
    }


def _read_games_file(path: Path, header: dict) -> dict[int, dict]:
    """Finished games of an earlier run with the same header. A partial last line (the process was killed while
    writing it) is dropped and the file rewritten without it."""
    lines = path.read_text().splitlines()
    try:
        old = json.loads(lines[0])
    except (IndexError, json.JSONDecodeError):
        raise SystemExit(f"{path}: the games file has no readable header; use another --games-file") from None
    if old != header:
        diff = ", ".join(f"{k} {old.get(k)!r} (now {header[k]!r})" for k in header if old.get(k) != header[k])
        raise SystemExit(f"{path}: the games file was written with other settings: {diff}; use another --games-file")
    done: dict[int, dict] = {}
    for i, line in enumerate(lines[1:], start=2):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines):
                break
            raise SystemExit(f"{path}:{i}: unreadable game line") from None
        if rec["game"] in done or not 0 <= rec["game"] < header["games"]:
            raise SystemExit(f"{path}:{i}: game {rec['game']} is repeated or out of range")
        done[rec["game"]] = rec
    path.write_text("".join(json.dumps(x) + "\n" for x in [old, *done.values()]))
    return done


def run_search_eval(
    weights: str | Path,
    games: int,
    seed: int,
    max_moves: int,
    threads: int,
    tiers: list[Tier],
    games_file: str | Path | None = None,
) -> dict:
    """Searched games with a checkpoint (its gamma included), resumable through ``games_file``.

    The games file holds a header line, then one JSON line per finished game, written as each game finishes.
    Rerunning with the same arguments plays only the missing games; other settings stop with SystemExit.
    Returns evaluate.summarize over every game, plus agent, seconds (this call), search and results.
    """
    from block_puzzle_ai.ntuple import load_model  # ntuple imports this module

    model, cfg, _ = load_model(weights)
    header = _header(weights, games, seed, max_moves, cfg.gamma, tiers)
    start = time.perf_counter()
    done: dict[int, dict] = {}
    out = None
    if games_file is not None:
        path = Path(games_file)
        if path.exists() and path.stat().st_size:
            done = _read_games_file(path, header)
            out = path.open("a")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            out = path.open("w")
            out.write(json.dumps(header) + "\n")
            out.flush()

    def on_game(*row) -> None:
        out.write(json.dumps(dict(zip(GAME_KEYS, row))) + "\n")
        out.flush()

    try:
        rows = _engine.evaluate_ntuple_search(
            model, games, seed, max_moves, threads, cfg.gamma, tiers, sorted(done), on_game if out else None
        )
    finally:
        if out:
            out.close()
    records = done | {row[0]: dict(zip(GAME_KEYS, row)) for row in rows}
    records = [records[g] for g in sorted(records)]
    results = [GameResult(r["score"], r["moves"], r["lines"], r["truncated"]) for r in records]
    total_moves = sum(r["moves"] for r in records)
    search = {
        "tiers": header["tiers"],
        "searched": [sum(r["searched"][i] for r in records) for i in range(len(tiers))],
        "changed": [sum(r["changed"][i] for r in records) for i in range(len(tiers))],
        "us_per_move": sum(r["seconds"] for r in records) * 1e6 / max(1, total_moves),
    }
    return summarize(results) | {
        "agent": f"ntuple:{weights}",
        "seconds": round(time.perf_counter() - start, 1),
        "search": search,
        "results": [asdict(r) for r in results],
    }


def search_requested(args) -> bool:
    return any(getattr(args, k) is not None for k in ("search_k", "search_below", "deep_below", "deep_k", "games_file"))
