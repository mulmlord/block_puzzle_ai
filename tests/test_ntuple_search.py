"""Tests of the expectimax search over the n-tuple value (cpp/search.hpp, block_puzzle_ai.ntuple_search).

Spec: docs/superpowers/specs/2026-11-05-ntuple-search-design.md, section "Search".
"""

import json
from pathlib import Path

import numpy as np

from block_puzzle_ai.ntuple import make_model, shapes_to_tuples, value

DATA = Path(__file__).parent / "data"


def random_model():
    """The random_model fixture of test_ntuple.py: window4 + line tuples, weights N(0, 1) from seed 0."""
    tuples = shapes_to_tuples(["window4", "line"])
    model = make_model(tuples)
    model.weights[:] = np.random.default_rng(0).normal(size=model.n_weights).astype(np.float32)
    return tuples, model


def test_split_value_matches_golden():
    """Values recorded before the split (scripts/make_ntuple_golden.py) must come back bit for bit."""
    _, model = random_model()
    entries = json.loads((DATA / "ntuple_values_golden.json").read_text())
    assert len(entries) == 600
    for e in entries:
        assert value(model, e["board"], e["remaining"], e["cooldown"]) == float.fromhex(e["value"])
