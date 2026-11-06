"""Tests of the expectimax search over the n-tuple value (cpp/search.hpp, block_puzzle_ai.ntuple_search).

Spec: docs/superpowers/specs/2026-11-05-ntuple-search-design.md, section "Search".
"""

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from block_puzzle_ai import _engine
from block_puzzle_ai.engine import CppGame
from block_puzzle_ai.ntuple import make_model, shapes_to_tuples, value
from block_puzzle_ai.pieces import HARD_GROUPS, all_pieces

DATA = Path(__file__).parent / "data"


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
