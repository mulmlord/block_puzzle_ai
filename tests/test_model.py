import pytest
import torch

from block_puzzle_ai.batch_env import tables_for
from block_puzzle_ai.features import FIT_CHUNK, fit_maps
from block_puzzle_ai.model import ValueNet, load_checkpoint

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.fixture(scope="module")
def net() -> ValueNet:
    return load_checkpoint("models/rl_v1.pt", "cuda")[0]


def inputs(net: ValueNet, k: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator("cuda").manual_seed(seed)
    planes = (torch.rand(k, 3, 10, 10, device="cuda", generator=gen) < 0.4).float()
    return planes, torch.rand(k, net.config["n_globals"], device="cuda", generator=gen)


def eager(net: ValueNet, planes: torch.Tensor, globals_: torch.Tensor) -> torch.Tensor:
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return net(planes, globals_).float()


def assert_close_bf16(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Different kernels may round a bf16 output to a neighbouring value (steps of 0.5 around 100)."""
    assert out.shape == ref.shape
    assert (out - ref).abs().max() <= 0.01 * ref.abs().max() + 1e-3


@cuda
@pytest.mark.parametrize("k", [1, 2047, 2048, 2049, 5000])
def test_cuda_predict_matches_eager_forward_for_any_batch_size(net, k):
    planes, globals_ = inputs(net, k, seed=k)
    assert_close_bf16(net.predict(planes, globals_), eager(net, planes, globals_))


@cuda
def test_cuda_predict_results_survive_later_calls(net):
    first_in, second_in = inputs(net, 3000, seed=1), inputs(net, 3000, seed=2)
    first = net.predict(*first_in)
    kept = first.clone()
    net.predict(*second_in)
    assert torch.equal(first, kept)
    assert_close_bf16(first, eager(net, *first_in))


@cuda
def test_cuda_predict_sees_in_place_weight_updates():
    net = load_checkpoint("models/rl_v1.pt", "cuda")[0]
    planes, globals_ = inputs(net, 100, seed=3)
    before = net.predict(planes, globals_)
    with torch.no_grad():
        net.head[-1].bias.add_(10.0)  # what an optimizer step does: same tensor, new values
    after = net.predict(planes, globals_)
    assert_close_bf16(after, before + 10.0)


@cuda
def test_cuda_predict_leaves_the_state_dict_alone(net):
    keys = list(net.state_dict())
    net.predict(*inputs(net, 10, seed=4))
    assert list(net.state_dict()) == keys


@cuda
@pytest.mark.parametrize("k", [0, 1, 5000, FIT_CHUNK + 3])
def test_cuda_fit_maps_match_cpu(k):
    gen = torch.Generator().manual_seed(k)
    boards = torch.rand(k, 100, generator=gen) < torch.rand(k, 1, generator=gen)  # fill levels from empty to full
    fits, counts = fit_maps(tables_for("cpu"), boards)
    cuda_fits, cuda_counts = fit_maps(tables_for("cuda"), boards.cuda())
    assert torch.equal(cuda_fits.cpu(), fits) and torch.equal(cuda_counts.cpu(), counts)


@cuda
def test_fp32_head_removes_most_bf16_output_rounding():
    bf16 = load_checkpoint("models/rl_v1.pt", "cuda")[0]
    fp32 = load_checkpoint("models/rl_v1.pt", "cuda")[0]
    fp32.set_fp32_head(True)
    planes, globals_ = inputs(bf16, 4096, seed=4)
    with torch.no_grad():
        ref = bf16(planes, globals_)  # no autocast: all fp32
    err_bf16 = (bf16.predict(planes, globals_) - ref).abs().mean()
    err_fp32 = (fp32.predict(planes, globals_) - ref).abs().mean()
    assert err_fp32 < 0.5 * err_bf16


def test_fp32_head_is_saved_and_old_checkpoints_load_without_it(tmp_path):
    from block_puzzle_ai.model import save_checkpoint

    assert not load_checkpoint("models/rl_v1.pt")[0].fp32_head
    net = ValueNet(n_globals=5, channels=4, blocks=1, hidden=8, fp32_head=True)
    save_checkpoint(tmp_path / "net.pt", net, gamma=0.99, reward_scale=0.01)
    assert load_checkpoint(tmp_path / "net.pt")[0].fp32_head
