"""Random-play throughput of the C++ engine (#11) against the Python and GPU simulators.

bench_engine.py [MOVES_PER_THREAD (default 2_000_000)]
"""

import random
import sys
import time

import torch

from block_puzzle_ai.agents import RandomAgent
from block_puzzle_ai.batch_env import BatchGame
from block_puzzle_ai.engine import afterstate_play, random_play
from block_puzzle_ai.game import Game


def game_py(moves: int) -> float:
    game, pick = Game(0), random.Random(0)
    start = time.perf_counter()
    for _ in range(moves):
        if game.done:
            game.reset()
        game.step(pick.choice(game.legal_moves()))
    return moves / (time.perf_counter() - start)


@torch.no_grad()
def batch_game(n_games: int = 4096, steps: int = 500) -> float:
    env, agent = BatchGame(n_games, "cuda", seed=0), RandomAgent()

    def one_step() -> None:
        cand = env.candidates()
        if bool(cand.terminal.any()):
            env.reset(cand.terminal)
            cand = env.candidates()
        env.step(cand, *agent.act(env, cand))

    one_step()  # warm-up
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        one_step()
    torch.cuda.synchronize()
    return n_games * steps / (time.perf_counter() - start)


def main() -> None:
    per_thread = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
    rows = []
    threads_label = {1: "1 thread", 12: "12 threads", 24: "24 threads"}
    for threads, label in threads_label.items():
        moves, games, seconds = random_play(threads, per_thread, seed=1)
        rows.append((f"C++ random_play, {label}", moves / seconds, f"{moves / games:.1f} moves/game"))
    for threads, label in threads_label.items():
        moves, _, after, seconds = afterstate_play(threads, per_thread // 4, seed=1)
        rows.append((f"C++ afterstate_play, {label}", moves / seconds, f"{after / seconds / 1e6:.0f}M after-states/s"))
    rows.append(("game.py, 1 thread", game_py(20_000), ""))
    if torch.cuda.is_available():
        rows.append(("BatchGame on CUDA, 4,096 games", batch_game(), ""))
    print(f"{'simulator':<34}{'moves/s':>14}  note")
    for name, rate, note in rows:
        print(f"{name:<34}{rate:>14,.0f}  {note}")


if __name__ == "__main__":
    main()
