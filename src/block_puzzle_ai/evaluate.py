"""Evaluation harness: play N seeded games with an agent and report score statistics.

uv run python -m block_puzzle_ai.evaluate --agent heuristic --games 256
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass

import torch

from block_puzzle_ai.agents import Agent, LookaheadAgent, make_agent
from block_puzzle_ai.batch_env import BatchGame


@dataclass
class GameResult:
    score: int
    moves: int
    lines: int
    truncated: bool


@torch.no_grad()
def run_games(agent: Agent, n_games: int, device: str, seed: int, max_moves: int) -> list[GameResult]:
    env = BatchGame(n_games, device, seed=seed)
    ids = torch.arange(n_games, device=device)
    results: list[GameResult | None] = [None] * n_games
    while env.n:
        cand = env.candidates()
        finished = cand.terminal | (env.moves >= max_moves)
        if finished.any():
            for i in finished.nonzero().squeeze(1).tolist():
                results[int(ids[i])] = GameResult(
                    int(env.score[i]), int(env.moves[i]), int(env.lines_cleared[i]), not bool(cand.terminal[i])
                )
            alive = (~finished).nonzero().squeeze(1)
            env.keep(alive)
            ids = ids[alive]
            if not env.n:
                break
            cand = env.candidates()
        slot, pos = agent.act(env, cand)
        env.step(cand, slot, pos)
    return results  # type: ignore[return-value]


def summarize(results: list[GameResult]) -> dict:
    """Score statistics plus the death rate, which stays meaningful when games hit the move cap.

    The risk of a game over is about constant per move, so moves_per_death estimates the expected moves until death.
    """
    scores = [r.score for r in results]
    moves = sum(r.moves for r in results)
    deaths = sum(not r.truncated for r in results)
    return {
        "games": len(results),
        "mean": statistics.fmean(scores),
        "median": statistics.median(scores),
        "min": min(scores),
        "max": max(scores),
        "stdev": statistics.pstdev(scores),
        "mean_moves": statistics.fmean(r.moves for r in results),
        "points_per_move": sum(scores) / max(1, moves),
        "truncated": len(results) - deaths,
        "deaths": deaths,
        "moves_per_death": moves / max(1, deaths),
        "deaths_per_m_moves": deaths / max(1, moves) * 1e6,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", default="heuristic", help="random | heuristic | path/to/checkpoint.pt")
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-moves", type=int, default=100_000)
    parser.add_argument("--search", action="store_true", help="lookahead search with the checkpoint's own gate")
    parser.add_argument("--search-below", type=float, help="lookahead search where the best 1-ply Q is below this")
    parser.add_argument("--top-k", type=int, default=8, help="search only the k best 1-ply moves (0 = all)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = parser.parse_args()

    agent = make_agent(args.agent, args.device, args.search_below, args.top_k or None, args.search)
    start = time.perf_counter()
    results = run_games(agent, args.games, args.device, args.seed, args.max_moves)
    summary = summarize(results) | {"agent": agent.name, "seconds": round(time.perf_counter() - start, 1)}
    if isinstance(agent, LookaheadAgent):
        summary |= {
            "search_below": agent.search_below,
            "top_k": agent.top_k,
            "searched": agent.searched,
            "search_changed_move": agent.changed,
        }
    if args.json:
        print(json.dumps(summary | {"results": [asdict(r) for r in results]}))
    else:
        print(" ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in summary.items()))


if __name__ == "__main__":
    main()
