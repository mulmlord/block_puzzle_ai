"""Tests of the expectimax search over the n-tuple value (cpp/search.hpp, block_puzzle_ai.ntuple_search).

Spec: docs/superpowers/specs/2026-11-05-ntuple-search-design.md, section "Search".
"""

import json
import math
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from block_puzzle_ai import _engine
from block_puzzle_ai.engine import CppGame
from block_puzzle_ai.game import Move
from block_puzzle_ai.ntuple import NTupleConfig, load_model, main, make_model, save_checkpoint, shapes_to_tuples, value
from block_puzzle_ai.ntuple_search import run_search_eval, tiers_from_args
from block_puzzle_ai.pieces import HARD_GROUPS, all_pieces

DATA = Path(__file__).parent / "data"
MODELS = Path(__file__).parent.parent / "models"


def random_model():
    """The random_model fixture of test_ntuple.py: window4 + line tuples, weights N(0, 1) from seed 0."""
    tuples = shapes_to_tuples(["window4", "line"])
    model = make_model(tuples)
    model.weights[:] = np.random.default_rng(0).normal(size=model.n_weights).astype(np.float32)
    return tuples, model


def test_split_value_matches_golden():
    """Values recorded before the split (scripts/make_ntuple_golden.py) must come back bit for bit."""
    _, model = random_model()
    entries = json.loads((DATA / "ntuple_values_golden.json").read_text())
    assert len(entries) == 600
    for e in entries:
        assert value(model, e["board"], e["remaining"], e["cooldown"]) == float.fromhex(e["value"])


# ---- a slow reference of the search, transcribed from the spec on CppGame and ntuple.value -------------

G = float(np.float32(0.999))  # the engine takes gamma as float32
INF = float("inf")
LOW64 = (1 << 64) - 1
HARD = set(HARD_GROUPS)
ref_dead = Counter()  # ref_B calls on a state with no legal move


def cooldown(history):
    return any(h >= 0 and h in HARD for h in history[:3])


def ref_q1(model, gamma, board, pending, score, history):
    """[(slot, pos, points, after_board, remaining, q1)] in engine order."""
    g = CppGame(0)
    g.set_state(board, [all_pieces()[p] for p in pending], score, history)
    out = []
    for move, after, points, _ in g.afterstates():
        rem = [pending[i] for i in range(3) if i != move.slot]
        q1 = points + gamma * value(model, after, rem, cooldown(history))
        out.append((move.slot, move.row * 10 + move.col, points, after, rem, q1))
    return out


def ref_E(model, gamma, parent_score, parent_history, after, rem, points, widths):
    g = CppGame(0)
    g.set_state(0, [all_pieces()[p] for p in (*rem, 0)], parent_score, parent_history)
    total = 0.0
    for x, px in enumerate(g.spawn_distribution()):
        if px == 0:
            continue
        hist = [all_pieces()[x].group, *parent_history[:2]]
        total += px * ref_B(model, gamma, after, [*rem, x], parent_score + points, hist, widths)
    return points + gamma * total


def ref_B(model, gamma, board, pending, score, history, widths):
    q = ref_q1(model, gamma, board, pending, score, history)
    if not q:
        ref_dead["B"] += 1
        return 0.0
    if not widths:
        return max(m[5] for m in q)
    kept = sorted(q, key=lambda m: -m[5])[: widths[0] or None]  # stable: ties keep engine order
    return max(ref_E(model, gamma, score, history, m[3], m[4], m[2], widths[1:]) for m in kept)


# ---- states and engine calls ---------------------------------------------------------------------------


def snapshot(game, score=None):
    """(board, pending ids, score, history) of a CppGame."""
    ids = [all_pieces().index(p) for p in game.pending]
    return game.board, ids, game.score if score is None else score, list(game.history)


MID_SEEDS = (1, 2, 3, 4, 5, 6, 7, 8, 12, 21)  # seeds 12 and 21 have a hard group in history[:3]; 1-10 have none


def mid_game_states(seeds=MID_SEEDS):
    """One state per seed after 30-300 random moves (restarting at game overs), with a random score in
    [0, 30000): random play scores too little to reach the spawn thresholds, and the spawn distribution
    depends on the score."""
    out = []
    for seed in seeds:
        rng, game = random.Random(seed), CppGame(seed)
        for _ in range(rng.randint(30, 300)):
            game.step(rng.choice(game.legal_moves()))
            if game.done:
                game.reset()
        out.append(snapshot(game, rng.randrange(30_000)))
    return out


def near_death_states(seed):
    """The last 3 states with a legal move of one random game, and its final state (no legal move)."""
    rng, game, trail = random.Random(seed), CppGame(seed), []
    while not game.done:
        trail.append(snapshot(game))
        game.step(rng.choice(game.legal_moves()))
    return trail[-3:], snapshot(game)


def scores(model, state, widths):
    board, pending, score, history = state
    return _engine.search_scores(model, board & LOW64, board >> 64, pending, score, history, G, widths)


def move(model, state, tiers):
    board, pending, score, history = state
    return _engine.search_move(model, board & LOW64, board >> 64, pending, score, history, G, tiers)


def check_against_reference(model, state, widths):
    """search_scores keeps the reference's top widths[0] moves, in engine order, with the same Q1 and score."""
    board, pending, score, history = state
    q = ref_q1(model, G, board, pending, score, history)
    kept = sorted(q, key=lambda m: -m[5])[: widths[0] or None]
    got = scores(model, state, widths)
    assert [(s, p) for s, p, *_ in got] == sorted((m[0], m[1]) for m in kept)
    ref = {(m[0], m[1]): m for m in kept}
    for slot, pos, points, q1, score_ in got:
        m = ref[slot, pos]
        assert (points, q1) == (m[2], m[5])
        assert score_ == pytest.approx(ref_E(model, G, score, history, m[3], m[4], m[2], widths[1:]), rel=1e-9)
    return got


def test_depth2_matches_reference():
    _, model = random_model()
    states = mid_game_states()
    assert sum(cooldown(s[3]) for s in states) >= 2
    assert sum(s[2] > 3000 for s in states) >= 2
    for k, sub in ((1, states), (4, states), (0, states)):
        for state in sub:
            check_against_reference(model, state, [k])


def test_depth3_matches_reference():
    _, model = random_model()
    check_against_reference(model, mid_game_states([11])[0], [2, 2])


def test_search_move_picks_max_and_tiers():
    _, model = random_model()
    for state in mid_game_states()[:5]:
        got = scores(model, state, [4])
        best = max(sc for *_, sc in got)
        expect = next((s, p) for s, p, _, _, sc in got if sc == best)  # first maximum in engine order
        q = ref_q1(model, G, *state)
        best_q1 = max(m[5] for m in q)
        one = next((m[0], m[1]) for m in q if m[5] == best_q1)
        slot, pos, score_, tier, changed = move(model, state, [(INF, [4])])
        assert ((slot, pos), score_, tier, changed) == (expect, best, 0, expect != one)
        assert move(model, state, [(INF, [4]), (best_q1 + 1, [2, 2])])[3] == 1
        assert move(model, state, [(best_q1 - 1, [4])]) == (*one, best_q1, -1, False)


def test_search_edge_cases():
    _, model = random_model()
    near, final = [], []
    for seed in range(1, 21):
        n, f = near_death_states(seed)
        near += n
        final.append(f)
    assert scores(model, near[0], [100]) == scores(model, near[0], [0])
    single = [s for s in near if len(ref_q1(model, G, *s)) == 1]
    assert single
    assert len(check_against_reference(model, single[0], [4])) == 1
    ref_dead.clear()
    for state in near:
        check_against_reference(model, state, [4])
    assert ref_dead["B"] > 0  # some children have no legal move
    for state in final:
        assert scores(model, state, [4]) == [] and move(model, state, [(INF, [4])]) is None


@pytest.mark.parametrize(
    "tiers",
    [
        [(1.0, [4]), (2.0, [4])],  # increasing below
        [(1.0, [4]), (1.0, [4])],  # equal below
        [(INF, [4, 4]), (1.0, [4])],  # decreasing depth
        [(INF, [])],  # empty widths
        [(INF, [-1])],  # negative width
        [(float("nan"), [4])],
    ],
)
def test_tiers_rejected(tiers):
    _, model = random_model()
    state = snapshot(CppGame(0))
    with pytest.raises(ValueError):
        move(model, state, tiers)
    if len(tiers) == 1 and tiers[0][1] != [4]:
        with pytest.raises(ValueError):
            scores(model, state, tiers[0][1])


# ---- evaluate_search -------------------------------------------------------------------------------------


def run_search(model, games, seed, max_moves, threads, tiers, skip=(), on_game=None, gamma=G):
    return _engine.evaluate_ntuple_search(model, games, seed, max_moves, threads, gamma, tiers, list(skip), on_game)


def test_no_tiers_matches_evaluate():
    _, model = random_model()
    got = run_search(model, 24, 5, 50, 8, [])
    expected = _engine.evaluate_ntuple(model, 24, 5, 50, 8, G)
    assert [r[0] for r in got] == list(range(24))
    assert [r[1:5] for r in got] == expected
    assert any(r[3] for r in expected) and not all(r[3] for r in expected)  # some games hit the cap, some die first
    assert all(r[5] == [] and r[6] == [] for r in got)


def test_search_threads_independent():
    _, model = random_model()
    one = run_search(model, 8, 3, 200, 1, [(INF, [2])])
    eight = run_search(model, 8, 3, 200, 8, [(INF, [2])])
    strip = [[*r[:7], r[8]] for r in one]
    assert strip == [[*r[:7], r[8]] for r in eight]
    assert all(r[5][0] == r[2] for r in one)  # every move searched
    assert sum(r[6][0] for r in one) > 0


def test_skip_and_callback():
    _, model = random_model()
    tiers = [(INF, [2]), (50.0, [4])]
    full = run_search(model, 6, 9, 100, 3, tiers)
    calls = []
    part = run_search(model, 6, 9, 100, 3, tiers, skip=[1, 4], on_game=lambda *row: calls.append(row))
    assert [r[0] for r in part] == [0, 2, 3, 5]
    assert [r[:7] + r[8:] for r in part] == [r[:7] + r[8:] for r in full if r[0] in (0, 2, 3, 5)]
    assert sorted(calls) == sorted(part)


def test_callback_exception_stops():
    """The first finished game raises in the callback; the other games, which may run for millions of moves,
    stop at their next move, and the callback is not called again."""
    model = load_model(MODELS / "ntuple_v1.ntw")[0]
    calls = []

    def boom(*row):
        calls.append(row)
        raise RuntimeError("boom")

    t0 = time.perf_counter()
    with pytest.raises(RuntimeError, match="boom"):
        run_search(model, 16, 0, 10**9, 8, [], on_game=boom)
    assert len(calls) == 1 and time.perf_counter() - t0 < 5  # 0.3 s; without the per-move stop check, 9 s


def test_divergence_raises():
    _, model = random_model()
    model.weights[:] = np.nan
    with pytest.raises(RuntimeError, match="diverged"):
        run_search(model, 4, 0, 100, 2, [(INF, [2])])


def test_last_q1():
    _, model = random_model()
    for r in run_search(model, 6, 1, 5, 2, [(INF, [2])]):
        assert len(r[8]) == 5 and r[2] == 5
    games = run_search(model, 6, 1, 300, 2, [(INF, [2])])
    assert all(r[2] >= 20 for r in games)
    for r in games:
        assert len(r[8]) == 20 and all(math.isfinite(q) for q in r[8])


def stream_seed(seed, g):
    """stream_seeds(seed, n)[g]: the (g + 1)-th splitmix64 output from ``seed``."""
    mask = (1 << 64) - 1
    for _ in range(g + 1):
        seed = (seed + 0x9E3779B97F4A7C15) & mask
        z = seed
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        out = z ^ (z >> 31)
    return out


def test_game_replays_with_search_move():
    """Game g of evaluate_search is search_move played on CppGame(stream_seed(seed, g), fast_rng=True); last_q1 holds
    the best Q1 of the last 20 moves played."""
    _, model = random_model()
    tiers = [(INF, [2]), (60.0, [3])]
    for g, score, moves, lines, truncated, searched, changed, _, last_q1 in run_search(model, 3, 11, 10**6, 3, tiers):
        game, best_q1, used = CppGame(stream_seed(11, g), fast_rng=True), [], [0, 0]
        while not game.done:
            state = snapshot(game)
            best_q1.append(max(m[5] for m in ref_q1(model, G, *state)))
            slot, pos, _, tier, _ = move(model, state, tiers)
            used[tier] += 1
            game.step(Move(slot, pos // 10, pos % 10))
        assert (game.score, game.moves, game.lines_cleared, False) == (score, moves, lines, truncated)
        assert used == searched and last_q1 == best_q1[-20:]


# ---- ntuple_search: tiers, games file, CLI ---------------------------------------------------------------


def test_tiers_from_args():
    assert tiers_from_args(4, None, None, None) == [(INF, [4])]
    assert tiers_from_args(None, 7e4, None, None) == [(7e4, [8])]
    assert tiers_from_args(2, 7e4, None, None) == [(7e4, [2])]
    assert tiers_from_args(4, None, 6e4, [8, 4]) == [(INF, [4]), (6e4, [8, 4])]
    assert tiers_from_args(None, None, None, None) == []
    for bad in ((4, None, None, [8, 4]), (4, None, 6e4, None), (4, None, 6e4, [8, 4, 2]), (None, 5e4, 6e4, [8, 4])):
        with pytest.raises(SystemExit):
            tiers_from_args(*bad)


@pytest.fixture
def checkpoint(tmp_path):
    tuples, model = random_model()
    path = tmp_path / "m.ntw"
    save_checkpoint(path, NTupleConfig(gamma=0.999), tuples, model.weights, 0, 0.0)
    return path


def test_run_search_eval_uses_checkpoint_gamma(tmp_path):
    """The checkpoint's gamma reaches the search. 0.5, not the real 0.999: scaling V by 0.999 seldom reorders moves,
    so a dropped gamma would not show."""
    tuples, model = random_model()
    model.weights[:] *= 10  # value differences comparable to the rewards
    path = tmp_path / "g05.ntw"
    save_checkpoint(path, NTupleConfig(gamma=0.5), tuples, model.weights, 0, 0.0)
    out = run_search_eval(path, 4, 2, 200, 2, [(INF, [2])], tmp_path / "g.jsonl")
    scores_ = [r["score"] for r in out["results"]]
    assert scores_ == [r[1] for r in run_search(model, 4, 2, 200, 2, [(INF, [2])], gamma=0.5)]
    assert scores_ != [r[1] for r in run_search(model, 4, 2, 200, 2, [(INF, [2])], gamma=1.0)]
    assert json.loads((tmp_path / "g.jsonl").read_text().splitlines()[0])["gamma"] == 0.5


def eval_args(checkpoint, games_file, seed=4, max_moves=150, tiers=None):
    return dict(weights=checkpoint, games=6, seed=seed, max_moves=max_moves, threads=3,
                tiers=tiers or [(INF, [2]), (40.0, [3])], games_file=games_file)  # fmt: skip


def test_resume_matches_full_run(checkpoint, tmp_path):
    a = run_search_eval(**eval_args(checkpoint, tmp_path / "a.jsonl"))
    lines = (tmp_path / "a.jsonl").read_text().splitlines()
    assert len(lines) == 7 and a["games"] == 6
    (tmp_path / "b.jsonl").write_text("\n".join(lines[:4]) + "\n")
    b = run_search_eval(**eval_args(checkpoint, tmp_path / "b.jsonl"))
    assert b["results"] == a["results"] and b["search"]["searched"] == a["search"]["searched"]
    assert len((tmp_path / "b.jsonl").read_text().splitlines()) == 7


def test_resume_drops_partial_last_line(checkpoint, tmp_path):
    a = run_search_eval(**eval_args(checkpoint, tmp_path / "a.jsonl"))
    lines = (tmp_path / "a.jsonl").read_text().splitlines()
    (tmp_path / "b.jsonl").write_text("\n".join(lines[:4]) + '\n{"game": 3, "sco')
    b = run_search_eval(**eval_args(checkpoint, tmp_path / "b.jsonl"))
    assert b["results"] == a["results"]
    games = [json.loads(line)["game"] for line in (tmp_path / "b.jsonl").read_text().splitlines()[1:]]
    assert sorted(games) == list(range(6))


def test_resume_rejects_other_settings(checkpoint, tmp_path):
    run_search_eval(**eval_args(checkpoint, tmp_path / "a.jsonl"))
    other = tmp_path / "other.ntw"
    other.write_bytes(checkpoint.read_bytes())
    for change in ({"seed": 5}, {"max_moves": 151}, {"tiers": [(INF, [3])]}, {"checkpoint": other}):
        kwargs = {"checkpoint": checkpoint} | change
        with pytest.raises(SystemExit, match="other settings"):
            run_search_eval(**eval_args(games_file=tmp_path / "a.jsonl", **kwargs))


def test_summary_works_with_scripts(checkpoint, tmp_path, capsys):
    main(["evaluate", "--weights", str(checkpoint), "--games", "4", "--seed", "1", "--max-moves", "100",
          "--threads", "2", "--search-k", "2", "--games-file", str(tmp_path / "g.jsonl"), "--json"])  # fmt: skip
    out = json.loads(capsys.readouterr().out)
    assert out["games"] == 4 and len(out["results"]) == 4
    assert out["search"]["tiers"] == [[None, [2]]] and out["search"]["us_per_move"] > 0
    (tmp_path / "s.json").write_text(json.dumps(out))
    root = Path(__file__).parent.parent
    for script, arg in (("bootstrap_mean.py", [tmp_path / "s.json"] * 2), ("ab_compare.py", [tmp_path / "c.jsonl"])):
        (tmp_path / "c.jsonl").write_text(json.dumps({k: v for k, v in out.items() if k != "results"} | {"arm": "x"}) + "\n")
        subprocess.run([sys.executable, root / "scripts" / script, *arg], check=True, capture_output=True)


def test_evaluate_cli_without_search_unchanged(checkpoint, capsys):
    main(["evaluate", "--weights", str(checkpoint), "--games", "4", "--seed", "1", "--max-moves", "100", "--threads", "2",
          "--json"])  # fmt: skip
    out = json.loads(capsys.readouterr().out)
    model = load_model(checkpoint)[0]
    expected = _engine.evaluate_ntuple(model, 4, 1, 100, 2, 0.999)
    assert [(r["score"], r["moves"]) for r in out["results"]] == [(r[0], r[1]) for r in expected]
    assert "search" not in out
