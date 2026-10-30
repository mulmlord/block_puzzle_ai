import numpy as np
import torch

from block_puzzle_ai.agents import HeuristicAgent, RandomAgent, ValueAgent, choose_move
from block_puzzle_ai.batch_env import BatchGame
from block_puzzle_ai.env import BlockPuzzleEnv
from block_puzzle_ai.evaluate import run_games, summarize
from block_puzzle_ai.features import encode, n_globals
from block_puzzle_ai.game import Game
from block_puzzle_ai.model import ValueNet, load_checkpoint, save_checkpoint


def test_gym_style_env_episode():
    env = BlockPuzzleEnv()
    obs, info = env.reset(seed=1)
    assert obs["board"].shape == (10, 10) and obs["pieces"].shape == (3, 5, 5)
    rng = np.random.default_rng(0)
    total, done = 0.0, False
    while not done:
        action = rng.choice(np.flatnonzero(info["action_mask"]))
        obs, reward, done, truncated, info = env.step(action)
        total += reward
    assert total == info["score"] > 0
    assert not info["action_mask"].any()


def test_features_shapes_and_fit_maps():
    env = BatchGame(4, seed=0)
    cand = env.candidates()
    a = encode(env, cand)
    assert a.planes.shape == (len(a.index), 3, 10, 10)
    assert a.globals.shape == (len(a.index), n_globals(env.t))
    # on a nearly empty board every piece fits somewhere
    assert a.fit_any.all()


def test_agents_play_valid_games():
    for agent in (RandomAgent(), HeuristicAgent()):
        results = run_games(agent, 8, "cpu", seed=3, max_moves=50)
        assert len(results) == 8 and all(r.moves > 0 for r in results)
        assert summarize(results)["mean"] > 0


def test_value_agent_checkpoint_roundtrip(tmp_path):
    t = BatchGame(1).t
    net = ValueNet(n_globals(t), channels=8, blocks=1, hidden=16)
    save_checkpoint(tmp_path / "net.pt", net, gamma=0.99, reward_scale=0.01)
    loaded, ckpt = load_checkpoint(tmp_path / "net.pt")
    for p, q in zip(net.parameters(), loaded.parameters()):
        assert torch.equal(p, q)
    agent = ValueAgent.from_checkpoint(tmp_path / "net.pt")
    game = Game(seed=2)
    move = choose_move(agent, game)
    assert move in game.legal_moves()
