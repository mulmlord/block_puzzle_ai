import dataclasses
import json

import pytest
import torch

from block_puzzle_ai.batch_env import BatchGame
from block_puzzle_ai.model import ValueNet
from block_puzzle_ai.train import Config, Learner, RestartPool, lambda_returns, learning_rate, train


def col(*values, dtype=torch.float32):
    return torch.tensor(values, dtype=dtype)[:, None]


def test_lambda_returns_hand_computed():
    cfg = Config(gamma=0.5, lam=0.5)
    r = col(1.0, 2.0, 4.0)
    best = col(0.0, 10.0, 20.0, 40.0)
    no = col(False, False, False, False, dtype=torch.bool)
    g = lambda_returns(r, best, no, no, no[:3], cfg)
    g2 = 40.0
    g1 = 0.5 * 20.0 + 0.5 * (4.0 + 0.5 * g2)
    g0 = 0.5 * 10.0 + 0.5 * (2.0 + 0.5 * g1)
    assert g.squeeze(1).tolist() == pytest.approx([g0, g1, g2])


def test_lambda_returns_terminal_cut_and_exploration():
    cfg = Config(gamma=0.5, lam=0.5)
    r = col(1.0, 2.0, 4.0)
    best = col(0.0, 10.0, 20.0, 40.0)
    terminal = col(False, True, False, False, dtype=torch.bool)
    no = col(False, False, False, False, dtype=torch.bool)
    g = lambda_returns(r, best, terminal, no, no[:3], cfg)
    assert g[0, 0] == 0.0  # x_0 led to game over

    explored = col(False, False, True, dtype=torch.bool)
    g = lambda_returns(r, best, no, no, explored, cfg)
    assert g[1, 0] == 20.0  # exploratory a_2 cuts the trace: bootstrap from m_2

    cut = col(False, False, False, True, dtype=torch.bool)
    g = lambda_returns(r, best, no, cut, no[:3], cfg)
    assert g[2, 0] == 40.0  # truncated game still bootstraps


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU"))]
)
def test_learner_fits_a_fixed_batch(device):
    torch.manual_seed(0)
    cfg = Config(lr=3e-3, epochs=2, minibatch=256, device=device)
    net = ValueNet(n_globals=5, channels=8, blocks=1, hidden=32).to(device, memory_format=torch.channels_last)
    learner = Learner(net, cfg, torch.Generator(device).manual_seed(1))
    planes = torch.rand(1024, 3, 10, 10, device=device) < 0.4
    globals_ = torch.rand(1024, 5, device=device)
    target = planes.flatten(1).float().mean(1) * 3 + globals_[:, 0]  # learnable from both inputs
    losses = [learner.update(planes, globals_, target, cfg.lr) for _ in range(30)]
    assert all(isinstance(loss, float) for loss in losses)
    assert losses[-1] < 0.1 * losses[0]


def test_learning_rate_warms_up_then_decays_to_lr_end():
    cfg = Config(lr=1e-3, lr_end=1e-4, warmup_moves=100)
    assert learning_rate(cfg, 0.0, 49) == pytest.approx(0.5e-3)
    assert learning_rate(cfg, 0.0, 1000) == pytest.approx(1e-3)
    assert learning_rate(cfg, 0.5, 1000) == pytest.approx(0.55e-3)
    assert learning_rate(cfg, 1.0, 1000) == pytest.approx(1e-4)
    assert learning_rate(cfg, 1.5, 1000) == pytest.approx(1e-4)


def test_restart_pool_keeps_states_from_before_a_game_over():
    env = BatchGame(2, "cpu", seed=0)
    pool = RestartPool(env, frac=1.0, every=2, depth=3)
    for step in range(6):  # snapshots at steps 0, 2, 4 of moves 10, 12, 14
        env.moves[:] = 10 + step
        env.score[:] = 100 * (10 + step)
        pool.snapshot(env)

    env.moves[:] = 15
    over = torch.tensor([True, False])
    pool.reset(env, over, over, torch.Generator().manual_seed(0))
    assert pool.size == 2  # 5 and 3 moves before the game over; 1 move before is too close
    assert sorted(pool.pool["score"][:2].tolist()) == [1000, 1200]
    assert pool.restarted.tolist() == [True, False]
    assert int(env.moves[0]) in (10, 12) and int(env.score[0]) == 100 * int(env.moves[0])
    assert int(env.moves[1]) == 15  # untouched

    pool.reset(env, over, over, torch.Generator().manual_seed(0))
    assert pool.size == 2  # the ring holds snapshots of env 0's previous game only


def test_resume_continues_the_run(tmp_path):
    cfg = Config(
        envs=4,
        rollout=2,
        total_moves=32,
        channels=4,
        blocks=1,
        hidden=8,
        eval_games=2,
        eval_max_moves=5,
        eval_minutes=1e9,
        save_minutes=1e9,
        restart_frac=0.5,
        device="cpu",
    )
    train(cfg, tmp_path)
    train(dataclasses.replace(cfg, total_moves=64), tmp_path, resume=True)

    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [r["moves"] for r in records if r["type"] == "eval"] == [32, 64]
    state = torch.load(tmp_path / "resume.pt", weights_only=True)["train"]
    assert state["moves"] == 64
    steps_per_rollout = cfg.epochs  # one minibatch per epoch
    assert float(state["opt"]["state"][0]["step"]) == 64 // 8 * steps_per_rollout  # optimizer state carried over
    assert sorted(p.name for p in (tmp_path / "ckpt").iterdir()) == ["00000M.pt"]
