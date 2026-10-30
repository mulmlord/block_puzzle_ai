"""After-state value network V(after-state) ~ expected discounted future points."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.conv2(F.relu(self.conv1(x))))


class ValueNet(nn.Module):
    def __init__(self, n_globals: int, channels: int = 32, blocks: int = 3, hidden: int = 256):
        super().__init__()
        self.config = {"n_globals": n_globals, "channels": channels, "blocks": blocks, "hidden": hidden}
        self.stem = nn.Conv2d(3, channels, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.reduce = nn.Conv2d(channels, 8, 1)
        self.head = nn.Sequential(
            nn.Linear(8 * 100 + n_globals, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, planes: torch.Tensor, globals_: torch.Tensor) -> torch.Tensor:
        planes = planes.contiguous(memory_format=torch.channels_last)  # ~1.5x faster convs on GPU
        x = self.blocks(F.relu(self.stem(planes)))
        x = F.relu(self.reduce(x)).flatten(1)
        return self.head(torch.cat([x, globals_], 1)).squeeze(1)

    @torch.no_grad()
    def predict(self, planes: torch.Tensor, globals_: torch.Tensor, chunk: int = 16384) -> torch.Tensor:
        """Inference over many after-states in bounded-memory chunks (bf16 on GPU)."""
        cuda = planes.is_cuda
        out = []
        for p, g in zip(planes.split(chunk), globals_.split(chunk)):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                out.append(self(p, g).float())
        return torch.cat(out) if out else planes.new_zeros(0)


def save_checkpoint(path: str | Path, net: ValueNet, **extra) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": net.config, "state_dict": net.state_dict(), **extra}, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[ValueNet, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    net = ValueNet(**ckpt["config"]).to(device, memory_format=torch.channels_last)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt
