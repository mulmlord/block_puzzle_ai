"""Projected mean score from the death rate: the yardstick for long-surviving agents (ticket #10, goal: mean 1B).

Why it works: once the score passes about 1M, the spawn distribution stops changing and the n-tuple value ignores
the score. A fixed policy is then a time-homogeneous process, so the risk of a game over per move is about
constant and game lengths are exponential with mean m = moves per game over. Then

    mean score ~ points per move * m                          (no cap)
    mean score ~ points per move * m * (1 - exp(-cap / m))    (games stopped at ``cap`` moves)

Many short capped games estimate m as well as a few long games with the same number of game overs, and no single
game sets the wall time. Precision is set by the game overs D alone: the 95% interval on m is roughly ×/÷ e^(2/√D).

The game stores the score as a 32-bit int (docs/RULES.md), so the real game cannot show more than INT32_MAX.
``mean_int32`` is the projected mean with every game's score clamped there.

    uv run python -m block_puzzle_ai.yardstick runs/a.json [runs/b.json ...] [--cap 10000000]

Inputs are ``evaluate --json`` outputs (a ``results`` list) or games files (``--games-file`` JSONL).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

INT32_MAX = 2**31 - 1
AGE_EDGES = (0, 100_000, 300_000, 1_000_000, 3_000_000, 10_000_000, 30_000_000, 100_000_000)


def _poisson_cdf(k: int, mu: float) -> float:
    """P(X <= k) for X ~ Poisson(mu), summed in log space."""
    if k < 0:
        return 0.0
    if mu == 0:
        return 1.0
    logs = [i * math.log(mu) - mu - math.lgamma(i + 1) for i in range(k + 1)]
    top = max(logs)
    return min(1.0, math.exp(top) * sum(math.exp(x - top) for x in logs))


def _solve(f, target: float, lo: float, hi: float) -> float:
    """mu in [lo, hi] with f(mu) = target, for f decreasing in mu."""
    while f(hi) > target:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(mid) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def poisson_ci(d: int, level: float = 0.95) -> tuple[float, float]:
    """Exact (Garwood) interval for the mean of a Poisson count from ``d`` observed events."""
    a = (1 - level) / 2
    upper = _solve(lambda mu: _poisson_cdf(d, mu), a, float(d), d + 10 * math.sqrt(d) + 10)
    lower = 0.0 if d == 0 else _solve(lambda mu: _poisson_cdf(d - 1, mu), 1 - a, 0.0, float(d))
    return lower, upper


def expected_capped(m: float, cap: float) -> float:
    """E[min(L, cap)] for L exponential with mean m."""
    if math.isinf(m):
        return cap
    return m * -math.expm1(-cap / m)


def load_results(path: str | Path) -> list[dict]:
    """Per-game records (score, moves, truncated) from an ``evaluate --json`` output or a games file."""
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines()[1:] if line.strip()]
    return json.loads(path.read_text())["results"]


def project(results: list[dict], cap: float | None = None, level: float = 0.95) -> dict:
    """Death rate, points per move and the projected mean, each with its interval from the game-over count."""
    deaths = sum(not r["truncated"] for r in results)
    moves = sum(r["moves"] for r in results)
    points = sum(r["score"] for r in results)
    ppm = points / max(1, moves)
    d_lo, d_hi = poisson_ci(deaths, level)
    m = moves / deaths if deaths else math.inf
    m_lo, m_hi = moves / d_hi, (moves / d_lo if d_lo else math.inf)

    def mean(mm: float, c: float | None) -> float:
        return ppm * (mm if c is None else expected_capped(mm, c))

    def mean_int32(mm: float) -> float:  # score ~ ppm * moves, clamped at INT32_MAX
        return INT32_MAX if math.isinf(mm) else ppm * mm * -math.expm1(-INT32_MAX / (ppm * mm)) if ppm else 0.0

    out = {
        "games": len(results),
        "deaths": deaths,
        "moves": moves,
        "points_per_move": ppm,
        "moves_per_death": m,
        "moves_per_death_ci": [m_lo, m_hi],
        "mean_uncapped": mean(m, None),
        "mean_uncapped_ci": [mean(m_lo, None), mean(m_hi, None)],
        "mean_int32": mean_int32(m),
        "mean_int32_ci": [mean_int32(m_lo), mean_int32(m_hi)],
        "observed_mean": points / max(1, len(results)),
        "observed_cap": max((r["moves"] for r in results if r["truncated"]), default=None),
    }
    if cap is not None:
        out |= {"cap": cap, "mean_at_cap": mean(m, cap), "mean_at_cap_ci": [mean(m_lo, cap), mean(m_hi, cap)]}
    return out


def hazard_by_age(results: list[dict], edges: tuple[int, ...] = AGE_EDGES) -> list[tuple[int, int, int, int]]:
    """(age from, age to, moves played at that age, game overs at that age) for each age bin with exposure."""
    out = []
    for a, b in zip(edges, edges[1:]):
        exposure = sum(max(0, min(r["moves"], b) - a) for r in results)
        deaths = sum(1 for r in results if not r["truncated"] and a <= r["moves"] < b)
        if exposure:
            out.append((a, b, exposure, deaths))
    return out


def age_drift(results: list[dict]) -> tuple[float, float, float] | None:
    """Death-rate ratio late / early, split where half the moves have been played, with a 95% CI; None when either
    half has no game over. A ratio well above 1 means the risk grows with game age and projections from short
    games are optimistic."""
    ages = sorted(r["moves"] for r in results)
    total = sum(ages)
    if not total:
        return None
    lo, hi = 0, max(ages)  # the age s where sum(min(moves, s)) = total / 2
    for _ in range(100):
        s = (lo + hi) / 2
        if sum(min(m, s) for m in ages) < total / 2:
            lo = s
        else:
            hi = s
    split = (lo + hi) / 2
    early_d = sum(1 for r in results if not r["truncated"] and r["moves"] < split)
    late_d = sum(1 for r in results if not r["truncated"] and r["moves"] >= split)
    if not early_d or not late_d:
        return None
    ratio = late_d / early_d  # equal exposure in both halves
    se = math.sqrt(1 / early_d + 1 / late_d)
    return ratio, ratio * math.exp(-1.96 * se), ratio * math.exp(1.96 * se)


def _m(x: float) -> str:
    return "inf" if math.isinf(x) else f"{x / 1e6:,.1f}M"


def report(path: str | Path, cap: float | None) -> dict:
    results = load_results(path)
    p = project(results, cap)
    print(f"{path}: {p['games']} games, {p['deaths']} game overs, {p['moves'] / 1e6:,.1f}M moves")
    lo, hi = p["moves_per_death_ci"]
    print(f"  moves per game over {_m(p['moves_per_death'])} [{_m(lo)}, {_m(hi)}], points per move {p['points_per_move']:.1f}")
    lo, hi = p["mean_uncapped_ci"]
    print(f"  projected mean, no cap: {_m(p['mean_uncapped'])} [{_m(lo)}, {_m(hi)}]")
    lo, hi = p["mean_int32_ci"]
    print(f"  projected mean, scores clamped at 2^31: {_m(p['mean_int32'])} [{_m(lo)}, {_m(hi)}]")
    if cap is not None:
        lo, hi = p["mean_at_cap_ci"]
        print(f"  projected mean at a {cap:,.0f}-move cap: {_m(p['mean_at_cap'])} [{_m(lo)}, {_m(hi)}]")
    print(f"  observed mean {_m(p['observed_mean'])} (games stopped at {p['observed_cap'] or 'no cap'})")
    for a, b, exposure, deaths in hazard_by_age(results):
        rate = f"{_m(exposure / deaths)} moves per game over" if deaths else "no game over"
        print(f"    age {a / 1e6:5.1f}-{b / 1e6:5.1f}M: {exposure / 1e6:8.1f}M moves, {deaths:3d} game overs, {rate}")
    drift = age_drift(results)
    if drift:
        print(f"  death rate late / early half of the moves: {drift[0]:.2f} [{drift[1]:.2f}, {drift[2]:.2f}]")
    return p


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m block_puzzle_ai.yardstick", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    parser.add_argument("runs", nargs="+", help="evaluate --json outputs or games files")
    parser.add_argument("--cap", type=float, help="also project the mean with games stopped at this many moves")
    args = parser.parse_args(argv)
    projections = [report(path, args.cap) for path in args.runs]
    base = projections[0]
    for path, p in zip(args.runs[1:], projections[1:]):
        if not (base["deaths"] and p["deaths"]):
            continue
        ratio = (p["deaths"] / p["moves"]) / (base["deaths"] / base["moves"])
        se = math.sqrt(1 / p["deaths"] + 1 / base["deaths"])
        print(f"{path} vs {args.runs[0]}: death rate {ratio:.2f} [{ratio * math.exp(-1.96 * se):.2f}, "
              f"{ratio * math.exp(1.96 * se):.2f}], points per move {p['points_per_move'] - base['points_per_move']:+.1f}")  # fmt: skip


if __name__ == "__main__":
    main()
