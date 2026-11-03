"""Tests of the n-tuple value function, trainer and CLI (block_puzzle_ai.ntuple, cpp/ntuple.hpp)."""

import json
import random
import signal
import subprocess
import sys
import time

import numpy as np
import pytest

from block_puzzle_ai import _engine
from block_puzzle_ai.engine import CppGame, cpp_tables
from block_puzzle_ai.game import placements
from block_puzzle_ai.ntuple import evaluate as nt_evaluate
from block_puzzle_ai.ntuple import (
    NTupleConfig,
    fit_bucket,
    load_checkpoint,
    load_model,
    main,
    make_model,
    save_checkpoint,
    shape_cells,
    shapes_to_tuples,
    transform_board,
    transform_cell,
    transform_piece,
    value,
)
from block_puzzle_ai.pieces import all_pieces


def orbit(cells):
    return {frozenset(r * 10 + c for r, c in (transform_cell(t, i // 10, i % 10) for i in cells)) for t in range(8)}


@pytest.mark.parametrize("name, count", [("window4", 10), ("line", 5)])
def test_orbits(name, count):
    reps = shapes_to_tuples([name])
    assert len(reps) == count and all(r == sorted(r) for r in reps)
    cells = shape_cells(name)
    h, w = 1 + max(r for r, _ in cells), 1 + max(c for _, c in cells)
    for r0 in range(11 - h):
        for c0 in range(11 - w):
            inst = frozenset((r0 + r) * 10 + c0 + c for r, c in cells)
            assert sum(inst in orbit(rep) for rep in reps) == 1
    assert len(shapes_to_tuples(["window4", "line"])) == 15


def test_transforms_form_a_group_and_map_pieces():
    rng = random.Random(0)
    board = rng.getrandbits(100)
    assert len({transform_board(t, board) for t in range(8)}) == 8
    for p in range(len(all_pieces())):
        images = {transform_piece(t, p) for t in range(8)}
        assert {all_pieces()[i].group for i in images} == {all_pieces()[p].group}


def test_fit_bucket_and_bad_shape():
    assert [fit_bucket(n) for n in (0, 1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33, 99)] == [0, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
    with pytest.raises(ValueError):
        shape_cells("blob")


def n_placements(piece, board):
    return sum(not p.mask & board for p in placements(piece))


def brute_value(tuples, w, board, remaining, cooldown):
    pieces, v, off = all_pieces(), 0.0, 0
    for cells in tuples:
        for t in range(8):
            b = transform_board(t, board)
            v += w[off + sum((b >> c & 1) << j for j, c in enumerate(cells))]
        off += 1 << len(cells)
    for g in range(12):
        count = sum(n_placements(p, board) for p in pieces if p.group == g)
        v += w[off + (cooldown * 12 + g) * 8 + fit_bucket(count)]
    for p in remaining:
        v += w[off + 192 + pieces[p].group * 8 + fit_bucket(n_placements(pieces[p], board))]
    return v


def sample_afterstates(n, seed):
    """n/2 after-states from random play, n/2 random boards; (board, remaining ids, cooldown)."""
    rng, out = random.Random(seed), []
    game = CppGame(seed)
    while len(out) < n // 2:
        if game.done:
            game.reset()
        move, after, _, _ = rng.choice(game.afterstates())
        ids = [all_pieces().index(p) for p in game.pending]
        out.append((after, [ids[i] for i in range(3) if i != move.slot], rng.random() < 0.3))
        game.step(move)
    while len(out) < n:
        density = rng.uniform(0.2, 0.7)
        board = sum(1 << c for c in range(100) if rng.random() < density)
        out.append((board, [rng.randrange(28), rng.randrange(28)], rng.random() < 0.3))
    return out


@pytest.fixture
def random_model():
    tuples = shapes_to_tuples(["window4", "line"])
    model = make_model(tuples)
    model.weights[:] = np.random.default_rng(0).normal(size=model.n_weights).astype(np.float32)
    return tuples, model


def test_value_matches_brute_force(random_model):
    tuples, model = random_model
    assert model.n_weights == 10 * 65536 + 5 * 1024 + 288 and model.n_lookups == 8 * 15 + 14
    for board, rem, cd in sample_afterstates(60, 1):
        expected = brute_value(tuples, model.weights, board, rem, cd)
        assert value(model, board, rem, cd) == pytest.approx(expected, rel=1e-5, abs=1e-3)


def test_value_is_symmetric(random_model):
    _, model = random_model
    for board, rem, cd in sample_afterstates(200, 2):
        v = value(model, board, rem, cd)
        for t in range(8):
            image = value(model, transform_board(t, board), [transform_piece(t, p) for p in rem], cd)
            assert image == pytest.approx(v, rel=1e-5, abs=1e-3)


@pytest.mark.parametrize("tuples", [[list(range(21))], [[100]], [[3, 3]], [[]]])
def test_ntuple_rejects_bad_tuples(tuples):
    with pytest.raises(ValueError):
        _engine.NTuple(cpp_tables(), tuples)


def test_value_rejects_bad_remaining(random_model):
    _, model = random_model
    for rem in ([0], [0, 28], [-1, 0]):
        with pytest.raises(ValueError):
            model.value(0, 0, rem, False)


def test_update_moves_toward_target(random_model):
    _, model = random_model
    for board, rem, cd in sample_afterstates(20, 3):
        before = value(model, board, rem, cd)
        model.update(board & (2**64 - 1), board >> 64, rem, cd, before + 50.0, 0.1)
        after = value(model, board, rem, cd)
        assert before < after < before + 50.0


def test_update_alpha_one_lands_on_target():
    tuples = [[0, 1, 2, 3, 10, 11, 12, 13]]
    model, rng = make_model(tuples), random.Random(4)
    while True:  # a board whose 8 symmetric indices are distinct, with remaining pieces from different groups
        board = rng.getrandbits(100)
        idx = {sum((transform_board(t, board) >> c & 1) << j for j, c in enumerate(tuples[0])) for t in range(8)}
        if len(idx) == 8:
            break
    rem = [0, 27]
    assert all_pieces()[0].group != all_pieces()[27].group
    model.update(board & (2**64 - 1), board >> 64, rem, False, 123.5, 1.0)
    assert value(model, board, rem, False) == pytest.approx(123.5, abs=1e-3)


def test_update_precision_at_large_values():
    model = make_model(shapes_to_tuples(["window4", "line"]))
    model.weights[:] = 1e8 / model.n_lookups
    board, rem = 0, [0, 1]
    before = value(model, board, rem, False)
    model.update(0, 0, rem, False, before + 1000.0, 0.1)
    assert value(model, board, rem, False) - before >= 50.0  # nominal step: 0.1 * 1000 = 100


def test_evaluate_deterministic_across_threads(random_model):
    _, model = random_model
    model.weights[:] *= 0.01
    one = nt_evaluate(model, games=16, seed=3, max_moves=2000, threads=1)
    assert nt_evaluate(model, games=16, seed=3, max_moves=2000, threads=4) == one
    assert len(one) == 16 and all(r.moves > 0 for r in one)


def test_evaluate_zero_cap(random_model):
    _, model = random_model
    assert all(r.truncated and r.moves == 0 and r.score == 0 for r in nt_evaluate(model, 8, 1, 0, 2))


def test_trainer_chunks_equal_one_run():
    tuples = shapes_to_tuples(["line"])
    a, b = make_model(tuples), make_model(tuples)
    ta, tb = _engine.Trainer(a, 1.0, 0.1, 0.5, 1, 7), _engine.Trainer(b, 1.0, 0.1, 0.5, 1, 7)
    ta.run(3000)
    ta.run(3000)
    tb.run(6000)
    assert np.array_equal(a.weights, b.weights)


def test_training_improves():
    model = make_model(shapes_to_tuples(["window4", "line"]))
    stats = _engine.Trainer(model, 1.0, 0.1, 0.5, 4, 0).run(3_000_000)
    assert stats["moves"] == 3_000_000 and stats["games"] > 0
    results = nt_evaluate(model, games=64, seed=777, max_moves=100_000, threads=4)
    assert sum(r.moves for r in results) / 64 > 150  # random play: 24, tuned heuristic: 279


def test_trainer_and_evaluate_reject_zero_threads(random_model):
    _, model = random_model
    with pytest.raises(ValueError):
        _engine.Trainer(model, 1.0, 0.1, 0.5, 0, 0)
    with pytest.raises(ValueError):
        nt_evaluate(model, 4, 0, 100, 0)


def test_trainer_restarts_from_pool():
    """restart_frac 1: every game after the first restarts from a pool state, which must be playable."""
    for seed in range(10):  # small early pools: a game over snapshotted into the pool gets drawn quickly
        model = make_model(shapes_to_tuples(["line"]))
        stats = _engine.Trainer(model, 1.0, 0.1, 1.0, 1, seed).run(5_000)
        assert stats["games"] > 10 and stats["restarts"] >= stats["games"] - 1


def test_checkpoint_roundtrip(tmp_path, random_model):
    tuples, model = random_model
    cfg = NTupleConfig(alpha=0.05)
    save_checkpoint(tmp_path / "m.ntw", cfg, tuples, model.weights, 1234, 5.5)
    loaded, cfg2, meta = load_model(tmp_path / "m.ntw")
    assert cfg2 == cfg and meta == {"moves_trained": 1234, "elapsed_min": 5.5}
    assert np.array_equal(loaded.weights, model.weights)
    for board, rem, cd in sample_afterstates(10, 5):
        assert value(loaded, board, rem, cd) == value(model, board, rem, cd)


def test_checkpoint_rejects_corruption(tmp_path, random_model):
    tuples, model = random_model
    path = tmp_path / "m.ntw"
    save_checkpoint(path, NTupleConfig(), tuples, model.weights, 0, 0.0)
    raw = path.read_bytes()
    (tmp_path / "magic.ntw").write_bytes(b"XXXXXXXX" + raw[8:])
    (tmp_path / "short.ntw").write_bytes(raw[:-4])
    for bad in ("magic.ntw", "short.ntw"):
        with pytest.raises(ValueError):
            load_checkpoint(tmp_path / bad)


def train_args(out, moves):
    return ["train", "--out", str(out), "--total-moves", str(moves), "--threads", "2", "--shapes", "line",
            "--eval-minutes", "0", "--eval-games", "8", "--eval-max-moves", "2000", "--chunk-seconds", "1"]  # fmt: skip


def test_train_resume_and_evaluate_cli(tmp_path, capsys):
    main(train_args(tmp_path, 100_000))
    main(train_args(tmp_path, 200_000) + ["--resume", "--alpha", "0.05"])
    cfg, _, _, meta = load_checkpoint(tmp_path / "latest.ntw")
    assert meta["moves_trained"] >= 200_000 and cfg.alpha == 0.05
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    evals = [r for r in records if r["type"] == "eval"]
    keys = {"eval_mean", "eval_median", "eval_max", "eval_moves_per_death", "elapsed_min", "moves"}
    assert evals and keys <= evals[0].keys()
    assert any(r["type"] == "train" and r["moves_per_s"] > 0 for r in records)
    capsys.readouterr()
    main(["evaluate", "--weights", str(tmp_path / "best.ntw"), "--games", "4", "--seed", "1", "--max-moves", "500",
          "--threads", "2", "--json"])  # fmt: skip
    out = json.loads(capsys.readouterr().out)
    assert out["games"] == 4 and len(out["results"]) == 4


def test_resume_keeps_best(tmp_path):
    main(train_args(tmp_path, 100_000))
    lines = (tmp_path / "metrics.jsonl").read_text().splitlines()
    best = max(json.loads(line).get("best_eval_moves_per_death", 0) for line in lines)
    best_bytes = (tmp_path / "best.ntw").read_bytes()
    record = json.dumps({"type": "eval", "best_eval_moves_per_death": best * 1e6})
    (tmp_path / "metrics.jsonl").write_text("\n".join(lines) + "\n" + record + "\n")
    main(train_args(tmp_path, 200_000) + ["--resume"])
    assert (tmp_path / "best.ntw").read_bytes() == best_bytes  # nothing beat the recorded best


def test_interrupt_saves_latest(tmp_path):
    proc = subprocess.Popen([sys.executable, "-m", "block_puzzle_ai.ntuple", "train", "--out", str(tmp_path), "--hours",
                             "1", "--threads", "1", "--shapes", "line", "--eval-minutes", "60", "--chunk-seconds", "1"])  # fmt: skip
    time.sleep(8)
    proc.send_signal(signal.SIGINT)
    assert proc.wait(timeout=30) == 0
    assert load_checkpoint(tmp_path / "latest.ntw")[3]["moves_trained"] > 0


def test_divergence_raises_instead_of_crashing():
    """Non-finite values must stop training and evaluation with an error, not abort the process."""
    model = make_model(shapes_to_tuples(["line"]))
    model.weights[:] = np.nan
    with pytest.raises(RuntimeError, match="diverged"):
        _engine.Trainer(model, 1.0, 0.1, 0.5, 2, 0).run(1000)
    with pytest.raises(RuntimeError, match="diverged"):
        nt_evaluate(model, 4, 0, 100, 2)


def test_default_alpha_scales_with_threads(tmp_path):
    """Lock-free updates stay stable while alpha * threads <= 0.5, so the default alpha is min(0.1, 0.5 / threads)."""
    for threads, alpha in ((2, 0.1), (10, 0.05)):
        out = tmp_path / str(threads)
        main(["train", "--out", str(out), "--total-moves", "20000", "--threads", str(threads), "--shapes", "line",
              "--eval-games", "2", "--eval-max-moves", "100"])  # fmt: skip
        assert load_checkpoint(out / "latest.ntw")[0].alpha == pytest.approx(alpha)
