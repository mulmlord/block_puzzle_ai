"""Live viewer: the browser asks for one move at a time and animates what the agent did.

uv run python -m block_puzzle_ai.viewer            # http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from block_puzzle_ai.agents import Agent, HeuristicAgent, RandomAgent, ValueAgent
from block_puzzle_ai.batch_env import BatchGame, move_from_index
from block_puzzle_ai.features import best_per_env
from block_puzzle_ai.game import N, Game, Move

STATIC = Path(__file__).parent / "static"
ROOT = Path.cwd()
DEVICE = "cpu"  # the GPU is usually busy training; one game is cheap on CPU

app = FastAPI(title="Block Puzzle AI viewer")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
_agents: dict[tuple[str, float], Agent] = {}


def checkpoints() -> list[Path]:
    found = list(ROOT.glob("models/*.pt")) + list(ROOT.glob("runs/*/best.pt")) + list(ROOT.glob("runs/*/latest.pt"))
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def get_agent(agent_id: str) -> Agent:
    if agent_id == "random":
        return RandomAgent(seed=random.randrange(2**31))
    if agent_id == "heuristic":
        return HeuristicAgent()
    path = (ROOT / agent_id).resolve()
    if path not in [p.resolve() for p in checkpoints()]:
        raise HTTPException(404, f"unknown agent {agent_id}")
    key = (str(path), path.stat().st_mtime)
    if key not in _agents:
        _agents[key] = ValueAgent.from_checkpoint(path, DEVICE)
    return _agents[key]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/agents")
def list_agents() -> list[dict]:
    items = []
    for path in checkpoints():
        meta = torch.load(path, map_location="cpu", weights_only=True).get("eval") or {}
        rel = path.relative_to(ROOT).as_posix()
        label = f"RL · {rel}"
        if meta:
            label += f" (eval mean {meta['mean']:,.0f})"
        items.append({"id": rel, "label": label, "eval_mean": meta.get("mean")})
    items.sort(key=lambda a: a["eval_mean"] or 0, reverse=True)
    return items + [
        {"id": "heuristic", "label": "Heuristic baseline", "eval_mean": None},
        {"id": "random", "label": "Random moves", "eval_mean": None},
    ]


@app.get("/api/training")
def training() -> dict:
    runs = {}
    for metrics in sorted(ROOT.glob("runs/*/metrics.jsonl")):
        evals = []
        for line in metrics.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("type") == "eval":
                evals.append(
                    {
                        "minutes": rec["elapsed_min"],
                        "moves": rec["moves"],
                        "mean": rec["eval_mean"],
                        "median": rec["eval_median"],
                        "max": rec["eval_max"],
                    }
                )
        runs[metrics.parent.name] = evals
    baselines_file = ROOT / "models/baselines.json"
    baselines = json.loads(baselines_file.read_text()) if baselines_file.exists() else {}
    return {"runs": runs, "baselines": baselines}


def snapshot(game: Game) -> dict:
    return {
        "board": game.colors,
        "pending": [{"cells": [list(c) for c in p.cells], "group": p.group} for p in game.pending],
        "score": game.score,
        "moves": game.moves,
        "lines": game.lines_cleared,
        "done": game.done,
    }


def bits(mask: int) -> list[int]:
    return [i for i in range(N * N) if mask >> i & 1]


class Session:
    def __init__(self) -> None:
        self.game: Game | None = None
        self.agent: Agent | None = None
        self.seed = 0
        self.best = 0
        self.history: list[dict] = []

    def new_game(self, agent_id: str, seed: int | None) -> dict:
        self.agent = get_agent(agent_id)
        self.seed = seed if seed is not None else random.randrange(1_000_000)
        self.game = Game(seed=self.seed)
        return {"type": "state", "seed": self.seed, "state": snapshot(self.game), "best": self.best}

    def choose(self) -> tuple[Move, int, float | None]:
        env = BatchGame.from_game(self.game, DEVICE)
        cand = env.candidates()
        legal = int(cand.legal.sum())
        if isinstance(self.agent, ValueAgent):
            a, q, v = self.agent.q_values(env, cand)
            slot, pos = best_per_env(1, a.index, q)
            chosen = (a.index[:, 1] == slot) & (a.index[:, 2] == pos)
            expected = float(q[chosen][0]) / self.agent.reward_scale  # points now + discounted future
            return move_from_index(int(slot), int(pos)), legal, expected
        slot, pos = self.agent.act(env, cand)
        return move_from_index(int(slot[0]), int(pos[0])), legal, None

    def step(self) -> dict:
        game = self.game
        move, legal, expected = self.choose()
        group = game.pending[move.slot].group
        result = game.step(move)
        if result.done:
            self.best = max(self.best, game.score)
            self.history.append({"seed": self.seed, "score": game.score, "moves": game.moves})
        return {
            "type": "step",
            "move": move._asdict(),
            "group": group,
            "placed": bits(result.placed),
            "cleared": bits(result.cleared),
            "points": result.points,
            "lines": result.lines,
            "legal_moves": legal,
            "expected": expected,
            "state": snapshot(game),
            "best": max(self.best, game.score),
            "history": self.history[-40:],
        }


@app.websocket("/ws")
async def play(ws: WebSocket) -> None:
    await ws.accept()
    session = Session()
    try:
        while True:
            msg = await ws.receive_json()
            try:
                if msg["cmd"] == "new":
                    reply = await run_in_threadpool(session.new_game, msg.get("agent", "heuristic"), msg.get("seed"))
                elif msg["cmd"] == "step" and session.game and not session.game.done:
                    reply = await run_in_threadpool(session.step)
                else:
                    continue
            except HTTPException as exc:
                reply = {"type": "error", "message": exc.detail}
            await ws.send_json(reply)
    except WebSocketDisconnect:
        pass
