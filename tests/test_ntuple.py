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
    CollapseGuard,
    NTupleConfig,
    fit_bucket,
    load_checkpoint,
    load_model,
    main,
    make_model,
    save_checkpoint,
    scheduled_alpha,
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


def test_update_tc_first_step_is_a_plain_step(random_model):
    """Before a weight's first step its rate is 1, so update_tc moves V like update."""
    _, model = random_model
    plain = make_model(model.tuples)
    plain.weights[:] = model.weights
    model.enable_tc()
    for board, rem, cd in sample_afterstates(6, 8):
        fresh = make_model(model.tuples)  # every weight unvisited
        fresh.weights[:] = model.weights
        fresh.enable_tc()
        target = value(fresh, board, rem, cd) + 500.0
        fresh.update_tc(board & (2**64 - 1), board >> 64, rem, cd, target, 0.1)
        plain.update(board & (2**64 - 1), board >> 64, rem, cd, target, 0.1)
        assert value(fresh, board, rem, cd) == pytest.approx(value(plain, board, rem, cd), abs=1e-3)
        plain.weights[:] = model.weights


def test_update_tc_slows_weights_whose_steps_cancel():
    tuples = [[0, 1, 2, 3, 10, 11, 12, 13]]
    model = make_model(tuples)
    model.enable_tc()
    board, rem = 0b111, [0, 27]
    lo, hi = board & (2**64 - 1), board >> 64
    for k in range(50):  # targets alternately above and below V: the steps cancel, so |E| / A falls
        model.update_tc(lo, hi, rem, False, value(model, board, rem, False) + (100.0 if k % 2 else -100.0), 0.5)
    e, a = model.tc_state[:, 0], model.tc_state[:, 1]
    touched = a > 0
    assert touched.sum() >= 3 and np.all(np.abs(e[touched]) / a[touched] < 0.1)
    before = value(model, board, rem, False)
    model.update_tc(lo, hi, rem, False, before + 100.0, 0.5)
    assert 0 < value(model, board, rem, False) - before < 0.1 * 0.5 * 100.0
    agreeing = make_model(tuples)
    agreeing.enable_tc()
    for _ in range(50):  # targets always above V: the rate stays 1
        agreeing.update_tc(lo, hi, rem, False, value(agreeing, board, rem, False) + 100.0, 0.5)
    e, a = agreeing.tc_state[:, 0], agreeing.tc_state[:, 1]
    assert np.allclose(np.abs(e[a > 0]), a[a > 0])


def test_tc_state_needs_enable(random_model):
    _, model = random_model
    assert not model.tc_enabled
    with pytest.raises(RuntimeError):
        model.tc_state
    with pytest.raises(RuntimeError):
        model.update_tc(0, 0, [0, 1], False, 1.0, 0.1)
    model.enable_tc()
    assert model.tc_enabled and model.tc_state.shape == (model.n_weights, 2) and not model.tc_state.any()


def test_trainer_uses_tc_once_enabled():
    tuples = shapes_to_tuples(["line"])
    a, b = make_model(tuples), make_model(tuples)
    b.enable_tc()
    _engine.Trainer(a, 1.0, 0.1, 0.5, 1, 7).run(5000)
    _engine.Trainer(b, 1.0, 0.1, 0.5, 1, 7).run(5000)
    assert not np.array_equal(a.weights, b.weights)
    assert np.isfinite(b.weights).all() and (b.tc_state[:, 1] > 0).sum() > 100
    rates = np.abs(b.tc_state[:, 0]) / np.maximum(b.tc_state[:, 1], 1e-300)
    assert rates[b.tc_state[:, 1] > 0].min() < 0.5


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
    assert cfg2 == cfg and meta == {"moves_trained": 1234, "elapsed_min": 5.5, "tc": None}
    assert not loaded.tc_enabled
    assert np.array_equal(loaded.weights, model.weights)
    for board, rem, cd in sample_afterstates(10, 5):
        assert value(loaded, board, rem, cd) == value(model, board, rem, cd)


def test_checkpoint_roundtrip_with_tc(tmp_path, random_model):
    tuples, model = random_model
    model.enable_tc()
    model.tc_state[:] = np.random.default_rng(1).random((model.n_weights, 2))
    save_checkpoint(tmp_path / "m.ntw", NTupleConfig(tc=True), tuples, model.weights, 7, 1.0, model.tc_state)
    loaded, cfg, meta = load_model(tmp_path / "m.ntw")
    assert cfg.tc and loaded.tc_enabled
    assert np.array_equal(loaded.tc_state, model.tc_state) and np.array_equal(loaded.weights, model.weights)
    raw = (tmp_path / "m.ntw").read_bytes()
    (tmp_path / "short.ntw").write_bytes(raw[:-8])
    with pytest.raises(ValueError):
        load_checkpoint(tmp_path / "short.ntw")
    with pytest.raises(ValueError):
        save_checkpoint(tmp_path / "x.ntw", NTupleConfig(), tuples, model.weights, 0, 0.0, model.tc_state[1:])


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


def test_train_tc_and_init_cli(tmp_path):
    main(train_args(tmp_path / "a", 100_000) + ["--tc"])
    cfg, _, _, meta = load_checkpoint(tmp_path / "a" / "latest.ntw")
    assert cfg.tc and cfg.init is None and meta["tc"] is not None and (meta["tc"][:, 1] > 0).any()
    main(train_args(tmp_path / "a", 150_000) + ["--resume"])  # keeps its tc state and goes on with it
    resumed = load_checkpoint(tmp_path / "a" / "latest.ntw")[3]["tc"]
    assert resumed[:, 1].sum() > meta["tc"][:, 1].sum()
    init = str(tmp_path / "a" / "latest.ntw")
    main(train_args(tmp_path / "b", 50_000) + ["--init", init])
    cfg_b, _, _, meta_b = load_checkpoint(tmp_path / "b" / "latest.ntw")
    assert cfg_b.init == init and cfg_b.tc and meta_b["moves_trained"] >= 50_000
    assert cfg_b.alpha == pytest.approx(0.1)  # the default cap for 2 threads, not the source run's
    model, _, _ = load_model(init)  # the same weights without tc state: --init then starts plain TD
    save_checkpoint(tmp_path / "plain.ntw", NTupleConfig(shapes=["line"]), [list(t) for t in model.tuples], model.weights, 0, 0.0)
    main(train_args(tmp_path / "c", 20_000) + ["--init", str(tmp_path / "plain.ntw")])
    cfg_c, _, _, meta_c = load_checkpoint(tmp_path / "c" / "latest.ntw")
    assert not cfg_c.tc and meta_c["tc"] is None
    with pytest.raises(SystemExit):
        main(train_args(tmp_path / "d", 20_000) + ["--init", init, "--shapes", "window4"])


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


def test_scheduled_alpha_follows_value_scale():
    assert scheduled_alpha(0.0208, 12_500, 2_400_000) == pytest.approx(12_500 / 2_400_000)
    assert scheduled_alpha(0.0208, 12_500, 100_000) == pytest.approx(0.0208)  # capped while V is small
    assert scheduled_alpha(0.0208, 12_500, 0.0) == pytest.approx(0.0208)


def test_collapse_guard():
    guard = CollapseGuard()
    for _ in range(4):
        assert not guard.check(deaths=20, moves=2_000_000, value_mean=2.4e6)
    assert not guard.check(2000, 2_000_000, 2.4e6)  # too little history to judge
    guard = CollapseGuard()
    for _ in range(10):
        assert not guard.check(20, 2_000_000, 2.4e6)
    assert not guard.check(78, 2_000_000, 2.0e6)  # within 4x deaths and 0.8x value
    assert guard.check(100, 2_000_000, 2.4e6)  # death spike
    assert guard.check(20, 2_000_000, 1.8e6)  # value drop
    assert not guard.check(20, 2_000_000, 2.5e6)  # flagged chunks stay out of the history
    guard = CollapseGuard()
    for _ in range(10):
        assert not guard.check(2, 2_000_000, 9e7)
    assert not guard.check(12, 2_000_000, 9e7)  # 6x, but too few deaths to be more than noise


def test_value_mean_reports_v_for_gamma_below_one():
    model = make_model(shapes_to_tuples(["line"]))
    model.weights[:] = 1000.0 / model.n_lookups  # V = 1000 everywhere
    stats = _engine.Trainer(model, 0.5, 0.0, 0.5, 1, 0).run(200)  # alpha 0: weights stay put
    assert stats["value_mean"] == pytest.approx(1000.0, rel=1e-4)


def test_train_schedules_alpha_and_rolls_back(tmp_path, monkeypatch):
    calls = {"n": 0}

    def flag_third(self, deaths, moves, value_mean):
        calls["n"] += 1
        return calls["n"] == 3

    monkeypatch.setattr(CollapseGuard, "check", flag_third)
    main(["train", "--out", str(tmp_path), "--total-moves", "300000", "--threads", "2", "--shapes", "line",
          "--alpha", "0.02", "--alpha-k", "10", "--eval-minutes", "60", "--eval-games", "2", "--eval-max-moves", "100",
          "--chunk-seconds", "0.05"])  # fmt: skip
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    rollbacks = [r for r in records if r["type"] == "rollback"]
    assert len(rollbacks) == 1 and rollbacks[0]["alpha_cap"] == pytest.approx(0.01)
    trains = [r for r in records if r["type"] == "train"]
    later = trains[trains.index(next(r for r in trains if r["moves"] > rollbacks[0]["moves"])) :]
    assert all(r["alpha"] <= 0.01 + 1e-9 for r in later)
    assert any(r["alpha"] < 0.01 - 1e-6 for r in later)  # the value schedule (K / V) is below the cap once V grows
    assert load_checkpoint(tmp_path / "latest.ntw")[0].alpha == pytest.approx(0.01)
