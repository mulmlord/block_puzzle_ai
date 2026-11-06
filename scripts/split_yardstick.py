"""Split yardstick (ticket #10, goal: mean 1B): the death rate of a searched n-tuple agent from danger excursions.

Every game over seen so far had a best Q1 below 85.2k on its last move, so every game over ends an excursion below
92k (the deep tier). Then

    death rate = e1 x q1 x p2
    e1 = entries into Q1 < 92k per move                                  (free play)
    q1 = P(reach Q1 < 85k before Q1 >= 92k again | entry)                (restarts from entry states, --capture)
    p2 = P(game over before Q1 >= 92k again | first decision below 85k)  (restarts from the captured states)

Restarts start at saved states with fresh spawns; a state (board, tray, score, history) is all the future depends on.
Spawns depend only on the RNG and the spawned groups, not on the moves, so restarts with the same seed face the same
pieces under every arm (paired comparisons). Without --capture a restart runs to Q1 >= 92k or a game over, which
estimates q1 x p2 in one stage. Results and validation: docs/RESULTS.md, "Yardstick".

    uv run python scripts/split_yardstick.py free DIR [--arm A] --workers 24 --burn 150000 --moves 400000
    uv run python scripts/split_yardstick.py roll DIR [--arm A] --reps 20 --capture --seed 32
    uv run python scripts/split_yardstick.py roll DIR --arm A --reps 10 --seed 41 --states DIR/roll_32.json --tag _s2_A
    uv run python scripts/split_estimate.py DIR DIR/roll_32.json DIR/roll_41_s2_A.json [DIR/roll_41_s2_B.json ...]

Arms are the acceptance config (k = 2 on every move, [4, 2] below 92k) with other widths on every move (k3, k4) or
below 85k. The 75k and 65k tiers only label the Q1 band.
"""

import argparse
import json
import math
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

from block_puzzle_ai import _engine
from block_puzzle_ai.engine import cpp_tables
from block_puzzle_ai.ntuple import load_model

WEIGHTS = "models/ntuple_v1.ntw"
INF = math.inf
def arm(widths, k=2):
    """k on every move, [4, 2] below 92k and ``widths`` below 85k."""
    return [(INF, [k]), (92000.0, [4, 2]), (85000.0, widths), (75000.0, widths), (65000.0, widths)]


ARMS = {"accept": arm([4, 2]), "d4": arm([4, 2, 2]), "w8": arm([8, 4]), "w8d4": arm([8, 4, 2]), "w6d4": arm([6, 3, 2]),
        "k3": arm([4, 2], 3), "k4": arm([4, 2], 4)}
TIERS = ARMS["accept"]
_M = {}


def model():
    if not _M:
        m, cfg, _ = load_model(WEIGHTS)
        _M.update(model=m, gamma=cfg.gamma, tables=cpp_tables())
    return _M["model"], _M["gamma"], _M["tables"]


def decide(m, gamma, g, tiers=TIERS):
    return _engine.search_move(m, g.board_lo, g.board_hi, list(g.pending), g.score, list(g.history), gamma, tiers)


def state_of(g):
    return [g.board_lo, g.board_hi, list(g.pending), g.score, list(g.history)]


def free(args):
    """One worker's free play: entries into Q1 < 92k counted (and their states kept) after ``burn`` moves."""
    seed, burn, total, arm_name = args
    tiers = ARMS[arm_name]
    m, gamma, T = model()
    rng = random.Random(seed)
    g = _engine.Game(T, rng.getrandbits(64), True)
    counted = deaths = games = points = 0
    bands = [0] * len(TIERS)
    secs = [0.0] * len(TIERS)
    entries = []
    prev_in = False
    while counted < total:
        live = g.moves >= burn
        t0 = time.perf_counter()
        slot, pos, _, tier, _ = decide(m, gamma, g, tiers)
        dt = time.perf_counter() - t0
        inside = tier >= 1
        if live:
            counted += 1
            bands[tier] += 1
            secs[tier] += dt
            if inside and not prev_in:
                entries.append(state_of(g))
        prev_in = inside
        gained, _, _ = g.step(slot, pos // 10, pos % 10)
        points += gained if live else 0
        if g.done:
            deaths += live
            games += 1
            g = _engine.Game(T, rng.getrandbits(64), True)
            prev_in = False
    return {"seed": seed, "arm": arm_name, "counted": counted, "points": points, "deaths": deaths, "restarts": games,
            "bands": bands, "secs": secs, "entries": entries}


def roll(args):
    """Restarts from ``state`` until Q1 >= 92k again (0) or game over (1). With ``capture``, a restart stops at its
    first decision below 85k instead and returns that state (2, ..., state)."""
    idx, state, seeds, arm_name, capture = args
    m, gamma, T = model()
    tiers = ARMS[arm_name]
    out = []
    for s in seeds:
        g = _engine.Game(T, s, True)
        g.set_state(*state)
        moves, deepest, t0 = 0, 0, time.perf_counter()
        while True:
            slot, pos, _, tier, _ = decide(m, gamma, g, tiers)
            if moves and tier == 0:
                out.append((0, moves, deepest, time.perf_counter() - t0))  # left the danger set
                break
            if capture and tier >= 2:
                out.append((2, moves, tier, time.perf_counter() - t0, state_of(g)))
                break
            deepest = max(deepest, tier)
            g.step(slot, pos // 10, pos % 10)
            moves += 1
            if g.done:
                out.append((1, moves, deepest, time.perf_counter() - t0))
                break
    return idx, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["free", "roll"])
    p.add_argument("out")
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--burn", type=int, default=150_000)
    p.add_argument("--moves", type=int, default=400_000)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--arm", choices=sorted(ARMS), default="accept")
    p.add_argument("--capture", action="store_true", help="stop restarts at their first decision below 85k")
    p.add_argument("--states", help="restart from the states captured in this roll file instead of free.json")
    p.add_argument("--limit", type=int, help="only the first N states")
    p.add_argument("--tag", default="", help="suffix of the output file")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.mode == "free":
        jobs = [(a.seed * 1000 + w, a.burn, a.moves, a.arm) for w in range(a.workers)]
        with Pool(a.workers) as pool:
            res = pool.map(free, jobs)
        (out / "free.json").write_text(json.dumps(res))
        n = sum(r["counted"] for r in res)
        e = sum(len(r["entries"]) for r in res)
        bands = [sum(r["bands"][i] for r in res) for i in range(len(TIERS))]
        secs = [sum(r["secs"][i] for r in res) for i in range(len(TIERS))]
        pts = sum(r["points"] for r in res) / n
        print(f"{a.arm}: {n} moves counted, {pts:.2f} points/move, {e} entries ({e / n:.3e}/move), "
              f"{sum(r['deaths'] for r in res)} deaths, bands {bands}, "
              f"ms/decision {[round(1e3 * s / max(1, b), 2) for s, b in zip(secs, bands)]}, {time.time() - t0:.0f}s")
    else:
        if a.states:
            states = [x[4] for r in json.loads(Path(a.states).read_text()) for x in r if x[0] == 2]
        else:
            states = [s for r in json.loads((out / "free.json").read_text()) for s in r["entries"]]
        states = states[: a.limit]
        rng = random.Random(a.seed)
        jobs = [(i, s, [rng.getrandbits(64) for _ in range(a.reps)], a.arm, a.capture) for i, s in enumerate(states)]
        rows = [None] * len(jobs)
        with Pool(a.workers) as pool:
            for k, (i, o) in enumerate(pool.imap_unordered(roll, jobs, chunksize=4)):
                rows[i] = o
                if k % 500 == 0:
                    print(f"{k}/{len(jobs)} states {time.time() - t0:.0f}s", flush=True)
        (out / f"roll_{a.seed}{a.tag}.json").write_text(json.dumps(rows))
        n = sum(len(r) for r in rows)
        d = sum(x[0] == 1 for r in rows for x in r)
        h = sum(x[0] == 2 for r in rows for x in r)
        mv = sum(x[1] for r in rows for x in r)
        sec = sum(x[3] for r in rows for x in r)
        print(f"{a.arm}: {n} rollouts from {len(rows)} states: {d} game overs (p = {d / n:.3e}), {h} captured below "
              f"85k, {mv / n:.2f} moves and {sec / n:.3f} s each, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
