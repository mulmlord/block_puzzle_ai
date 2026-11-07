"""Tests of the death-rate yardstick (block_puzzle_ai.yardstick)."""

import json
import math
import random

import pytest

from block_puzzle_ai.yardstick import (
    INT32_MAX,
    age_drift,
    expected_capped,
    hazard_by_age,
    load_results,
    main,
    poisson_ci,
    project,
)


@pytest.mark.parametrize(
    "d, lo, hi",  # Garwood intervals, from chi-square quantiles
    [(0, 0.0, 3.6889), (1, 0.02532, 5.5716), (2, 0.24221, 7.2247), (6, 2.2019, 13.0594), (10, 4.7954, 18.3904),
     (63, 48.412, 80.598)],  # fmt: skip
)
def test_poisson_ci_matches_tables(d, lo, hi):
    got = poisson_ci(d)
    assert got[0] == pytest.approx(lo, rel=1e-3, abs=1e-9) and got[1] == pytest.approx(hi, rel=1e-3)


def games(rng, n, m, cap, ppm=100.0):
    """n games with exponential lengths of mean m, stopped at cap, scoring ppm points per move."""
    out = []
    for _ in range(n):
        length = rng.expovariate(1 / m)
        moves = int(min(length, cap))
        out.append({"score": int(ppm * moves), "moves": moves, "truncated": length >= cap})
    return out


def test_short_capped_games_recover_the_mean():
    """96 games stopped at m / 3 project the uncapped mean of exponential games."""
    p = project(games(random.Random(1), 96, 3e6, 1e6), cap=1e7)  # seed 0 draws 16 game overs for an expected 27
    assert p["deaths"] > 15
    lo, hi = p["mean_uncapped_ci"]
    assert lo < 3e8 < hi and p["mean_uncapped"] == pytest.approx(3e8, rel=0.35)
    assert p["mean_at_cap"] == pytest.approx(100 * expected_capped(p["moves_per_death"], 1e7))
    assert p["mean_at_cap"] < p["mean_uncapped"] and p["points_per_move"] == pytest.approx(100, abs=0.01)


def test_interval_coverage():
    """The 95% interval on moves per game over covers the true value in about 95% of runs."""
    rng = random.Random(1)
    covered = 0
    for _ in range(400):
        lo, hi = project(games(rng, 48, 2e6, 5e5))["moves_per_death_ci"]
        covered += lo <= 2e6 <= hi
    assert 0.91 <= covered / 400 <= 0.99


def test_no_game_over():
    p = project([{"score": 1000, "moves": 10, "truncated": True}] * 4, cap=100)
    assert p["deaths"] == 0 and math.isinf(p["moves_per_death"]) and math.isinf(p["mean_uncapped_ci"][1])
    assert p["mean_at_cap_ci"][1] == pytest.approx(100 * 100)  # every game reaches the cap
    assert p["mean_int32_ci"][1] == INT32_MAX
    assert age_drift(p and [{"score": 0, "moves": 10, "truncated": True}]) is None


def test_expected_capped_and_int32_clamp():
    assert expected_capped(1e6, 1e12) == pytest.approx(1e6)
    assert expected_capped(1e12, 1e6) == pytest.approx(1e6, rel=1e-5)
    assert expected_capped(math.inf, 5.0) == 5.0
    p = project([{"score": 100 * 10**7, "moves": 10**7, "truncated": False}], cap=1e7)  # m = 1e7 moves, mean 1e9
    assert p["mean_int32"] == pytest.approx(1e9 * (1 - math.exp(-INT32_MAX / 1e9)))  # 0.88B
    assert p["mean_int32_at_cap"] == pytest.approx(p["mean_at_cap"])  # a 1e7-move cap stops scores at 1e9 < 2^31
    p = project([{"score": 100 * 10**7, "moves": 10**7, "truncated": False}], cap=1e8)
    assert p["mean_int32_at_cap"] == pytest.approx(p["mean_int32"]) and p["mean_int32"] < p["mean_at_cap"]
    two = [{"score": 3 * 10**9, "moves": 3 * 10**7, "truncated": False}, {"score": 10**9, "moves": 10**7, "truncated": True}]
    p = project(two)
    assert p["observed_mean"] == 2e9 and p["observed_mean_int32"] == (INT32_MAX + 1e9) / 2 and p["games_past_int32"] == 1


def test_hazard_by_age_and_drift():
    results = [{"score": 0, "moves": m, "truncated": t} for m, t in ((50_000, False), (2_000_000, False), (5_000_000, True))]
    bins = {(a, b): (e, d) for a, b, e, d in hazard_by_age(results)}
    assert bins[0, 100_000] == (250_000, 1)
    assert bins[1_000_000, 3_000_000] == (3_000_000, 1)
    assert bins[3_000_000, 10_000_000] == (2_000_000, 0)
    assert sum(e for e, _ in bins.values()) == 7_050_000
    rng = random.Random(2)  # the risk doubles after 1M moves: late / early must come out above 1
    rising = []
    for _ in range(400):
        first = rng.expovariate(1 / 2e6)
        length = first if first < 1e6 else 1e6 + rng.expovariate(1 / 1e6)
        rising.append({"score": 0, "moves": int(min(length, 4e6)), "truncated": length >= 4e6})
    ratio, lo, _ = age_drift(rising)
    assert ratio > 1.4 and lo > 1
    ratio, lo, hi = age_drift(games(random.Random(3), 400, 2e6, 4e6))
    assert lo < 1 < hi


def test_cli_reads_json_and_games_files(tmp_path, capsys):
    rs = games(random.Random(4), 20, 1e5, 5e4)
    (tmp_path / "a.json").write_text(json.dumps({"results": rs}))
    (tmp_path / "b.jsonl").write_text("\n".join(json.dumps(x) for x in [{"header": 1}, *rs]) + "\n")
    assert load_results(tmp_path / "a.json") == load_results(tmp_path / "b.jsonl") == rs
    main([str(tmp_path / "a.json"), str(tmp_path / "b.jsonl"), "--cap", "1e7"])
    out = capsys.readouterr().out
    assert "projected mean, no cap" in out and "death rate 1.00 [" in out
