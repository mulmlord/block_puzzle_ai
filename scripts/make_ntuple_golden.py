"""Record n-tuple values from the current build into tests/data/ntuple_values_golden.json.

The split value of the search (docs/superpowers/specs/2026-11-05-ntuple-search-design.md) must stay
bit-identical to the value before the split, so this ran once on the old build. Values are stored as
float.hex so the test compares exact doubles.

    uv run python scripts/make_ntuple_golden.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

from test_ntuple import sample_afterstates  # noqa: E402

from block_puzzle_ai.ntuple import make_model, shapes_to_tuples, value  # noqa: E402


def main() -> None:
    model = make_model(shapes_to_tuples(["window4", "line"]))
    model.weights[:] = np.random.default_rng(0).normal(size=model.n_weights).astype(np.float32)
    out = [
        {"board": board, "remaining": rem, "cooldown": cd, "value": float.hex(value(model, board, rem, cd))}
        for board, rem, _ in sample_afterstates(300, 7)
        for cd in (False, True)
    ]
    path = ROOT / "tests" / "data" / "ntuple_values_golden.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out) + "\n")
    print(f"{len(out)} entries -> {path}")


if __name__ == "__main__":
    main()
