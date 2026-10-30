import random

import torch

from block_puzzle_ai.batch_env import BatchGame, move_from_index, spawn_from_uniforms, tables_for
from block_puzzle_ai.game import N, Game, PieceGenerator
from block_puzzle_ai.pieces import HARD_GROUPS, all_pieces


class UniformRng:
    """randrange(n) = int(u * n) for a scripted list of uniforms (same mapping as spawn_from_uniforms)."""

    def __init__(self, uniforms):
        self.uniforms = list(uniforms)

    def randrange(self, n):
        return int(self.uniforms.pop(0) * n)


def board_int(row: torch.Tensor) -> int:
    return sum(1 << i for i in range(N * N) if row[i])


def test_spawn_matches_reference_generator():
    t = tables_for("cpu")
    rng = random.Random(3)
    pieces = all_pieces()
    for _ in range(3000):
        score = rng.choice([0, rng.randrange(20000), 3003, 3004, rng.randrange(10**6)])
        history = [rng.randrange(12) for _ in range(rng.randrange(4))]
        u = [rng.random() for _ in range(4)]
        piece, new_history = spawn_from_uniforms(
            t,
            torch.tensor([score]),
            torch.tensor([(history + [-1, -1, -1])[:3]]),
            torch.tensor([u], dtype=torch.float64),
        )
        cooldown = any(h in HARD_GROUPS for h in history)
        draws = [u[0], u[1]] + ([u[2]] if cooldown else []) + [u[3]]
        gen = PieceGenerator(UniformRng(draws), history=list(history))
        expected = gen.spawn(score)
        assert pieces[int(piece)] == expected
        assert new_history[0].tolist() == (gen.history + [-1, -1, -1])[:3]


def test_candidates_and_steps_match_reference_game():
    rng = random.Random(11)
    for seed in range(4):
        game = Game(seed=seed)
        while not game.done:
            env = BatchGame.from_game(game)
            cand = env.candidates()
            legal = {move_from_index(s, p) for s, p in cand.legal[0].nonzero().tolist()}
            assert legal == set(game.legal_moves())

            move = rng.choice(sorted(legal))
            pos = move.row * N + move.col
            result = game.step(move)
            assert int(cand.points[0, move.slot, pos]) == result.points
            assert int(cand.lines[0, move.slot, pos]) == result.lines
            assert board_int(cand.after[0, move.slot, pos]) == game.board
            assert bool(BatchGame.from_game(game).candidates().terminal[0]) == game.done


def test_batch_step_keeps_slot_order_and_tracks_score():
    env = BatchGame(64, seed=5)
    total = torch.zeros(64, dtype=torch.long)
    for _ in range(30):
        cand = env.candidates()
        alive = ~cand.terminal
        if not alive.all():
            env.reset(~alive)
            total[~alive] = 0
            cand = env.candidates()
        flat = cand.legal.flatten(1).float()
        choice = torch.multinomial(flat, 1, generator=torch.Generator().manual_seed(0)).squeeze(1)
        slot, pos = choice // (N * N), choice % (N * N)
        before = env.pending.clone()
        total += env.step(cand, slot, pos)
        for b in range(64):
            kept = [before[b, s] for s in range(3) if s != slot[b]]
            assert env.pending[b, :2].tolist() == [int(k) for k in kept]
    assert torch.equal(total, env.score)
