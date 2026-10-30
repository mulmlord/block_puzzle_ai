import pytest
import torch

from block_puzzle_ai.train import Config, lambda_returns


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
