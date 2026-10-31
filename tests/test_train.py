import pytest
import torch

from block_puzzle_ai.model import ValueNet
from block_puzzle_ai.train import Config, Learner, lambda_returns


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
    losses = [learner.update(planes, globals_, target) for _ in range(30)]
    assert all(isinstance(loss, float) for loss in losses)
    assert losses[-1] < 0.1 * losses[0]
