import numpy as np
import torch

from block_puzzle_ai.agents import HeuristicAgent, LookaheadAgent, RandomAgent, ValueAgent, choose_move, make_agent
from block_puzzle_ai.batch_env import BatchGame, spawn_distribution
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


class LinearValue:
    """Stub value net that reacts to the order of the remaining pieces and to every global feature."""

    def __init__(self, n_globals: int):
        self.w = torch.rand(n_globals, generator=torch.Generator().manual_seed(0))

    def predict(self, planes: torch.Tensor, globals_: torch.Tensor) -> torch.Tensor:
        return 50 + 0.3 * planes[:, 1].flatten(1).sum(1) + 0.1 * planes[:, 2].flatten(1).sum(1) + globals_ @ self.w


def lookahead_test_states(t):
    midgame = BatchGame(1, seed=5)
    heuristic = HeuristicAgent()
    for _ in range(60):
        cand = midgame.candidates()
        midgame.step(cand, *heuristic.act(midgame, cand))
    midgame.pending[0] = torch.tensor([0, 9, 27])  # three different pieces, so their order matters
    midgame.history[0] = torch.tensor([8, 1, 0])  # hard-piece cooldown
    board = torch.ones(1, 100, dtype=torch.bool)
    board[0, [r * 10 + 3 * r % 10 for r in range(10)]] = False  # one hole per row and column
    # the dot clears a row and a column; then a spawned square, 3x3 or U piece ends the game
    crowded = BatchGame.from_state(t, board, torch.tensor([[0, 13, 18]]), torch.tensor([20000]), torch.full((1, 3), -1))
    return midgame, crowded


def stepped_with_every_spawn(env: BatchGame, slot: int, pos: int) -> tuple[BatchGame, torch.Tensor, torch.Tensor]:
    """Copies of a one-game env after env.step(slot, pos), copy i spawning piece i of the possible ones."""
    t = env.t
    probs = spawn_distribution(t, env.score, env.history)[0]
    pieces = probs.nonzero().squeeze(1)
    n = len(pieces)
    copies = BatchGame.from_state(
        t, env.board.repeat(n, 1), env.pending.repeat(n, 1), env.score.repeat(n), env.history.repeat(n, 1)
    )

    def forced_spawn(idx):  # what randTiles does when it draws piece i
        copies.history[idx] = torch.cat([t.group[pieces[idx]][:, None], copies.history[idx, :2]], 1)
        return pieces[idx]

    copies._spawn = forced_spawn
    copies.step(copies.candidates(), torch.full((n,), slot), torch.full((n,), pos))
    return copies, pieces, probs[pieces]


def test_lookahead_children_match_real_steps():
    t = BatchGame(1, seed=0).t
    agent = LookaheadAgent(LinearValue(n_globals(t)), gamma=0.99, reward_scale=0.01, search_below=float("inf"))
    for env in lookahead_test_states(t):
        a, _, _ = agent.q_values(env, env.candidates())
        for slot in a.index[:, 1].unique().tolist():
            row = int((a.index[:, 1] == slot).nonzero()[0])
            children, parent, prob = agent._children(env, a, torch.tensor([row]))
            copies, _, copy_prob = stepped_with_every_spawn(env, slot, int(a.index[row, 2]))
            assert (parent == 0).all() and torch.allclose(prob, copy_prob) and abs(float(prob.sum()) - 1) < 1e-5
            for name in ("board", "pending", "score", "history"):
                assert torch.equal(getattr(children, name), getattr(copies, name)), name


def test_lookahead_expectation_matches_real_steps():
    """The search's expectation equals the spawn-weighted values of the states env.step() reaches."""
    t = BatchGame(1, seed=0).t
    agent = LookaheadAgent(LinearValue(n_globals(t)), gamma=0.99, reward_scale=0.01, search_below=float("inf"))
    midgame, crowded = lookahead_test_states(t)
    for env in (midgame, crowded):
        a, q, _ = agent.q_values(env, env.candidates())
        row = int(q.argmax())
        exact = float(agent._expected_best(env, a, torch.tensor([row])))
        copies, _, prob = stepped_with_every_spawn(env, int(a.index[row, 1]), int(a.index[row, 2]))
        after = copies.candidates()
        ca, cq, _ = agent.q_values(copies, after)
        best = torch.zeros(copies.n).scatter_reduce_(0, ca.index[:, 0], cq, "amax", include_self=False)
        if env is crowded:
            assert 0.1 < float((prob * after.terminal).sum()) < 0.5
        assert abs(float((prob * best).sum()) - exact) < 1e-3


def test_lookahead_without_danger_plays_like_value_agent():
    torch.manual_seed(0)
    net = ValueNet(n_globals(BatchGame(1, seed=0).t), channels=8, blocks=1, hidden=32).eval()
    plain = ValueAgent(net, 0.99, 0.01)
    never = LookaheadAgent(net, 0.99, 0.01, search_below=float("-inf"))
    always = LookaheadAgent(net, 0.99, 0.01, search_below=float("inf"), top_k=3)
    env = BatchGame(3, seed=2)
    cand = env.candidates()
    assert all(torch.equal(x, y) for x, y in zip(plain.act(env, cand), never.act(env, cand)))
    slot, pos = always.act(env, cand)
    assert cand.legal[torch.arange(3), slot, pos].all() and always.searched == 3


def test_lookahead_gate_comes_from_the_checkpoint(tmp_path):
    net = ValueNet(n_globals(BatchGame(1, seed=0).t), channels=4, blocks=1, hidden=8)
    save_checkpoint(tmp_path / "old.pt", net, gamma=0.99, reward_scale=0.01)
    save_checkpoint(tmp_path / "new.pt", net, gamma=0.997, reward_scale=0.003, search_below=115.0)
    assert LookaheadAgent.from_checkpoint(tmp_path / "old.pt").search_below == LookaheadAgent.DEFAULT_GATE
    assert LookaheadAgent.from_checkpoint(tmp_path / "new.pt").search_below == 115.0
    assert LookaheadAgent.from_checkpoint(tmp_path / "new.pt", search_below=90.0).search_below == 90.0
    assert make_agent(str(tmp_path / "new.pt"), search=True).search_below == 115.0
