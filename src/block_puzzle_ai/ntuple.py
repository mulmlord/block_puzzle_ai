"""n-tuple after-state value function trained by TD(0) self-play on the C++ engine (#10 technique 7).

Design: docs/superpowers/specs/2026-11-03-ntuple-learner-design.md. This module picks the tuples (one
representative per orbit of a shape's translations under the 8 board symmetries) and holds the Python
twins of the C++ symmetries that the tests check the engine against.
"""

from __future__ import annotations

import re

from block_puzzle_ai.game import N
from block_puzzle_ai.pieces import all_pieces


def transform_cell(t: int, r: int, c: int) -> tuple[int, int]:
    """Symmetry t (0..7) of the 10x10 board: t & 4 transposes, then t & 2 flips rows, then t & 1 flips columns."""
    if t & 4:
        r, c = c, r
    if t & 2:
        r = N - 1 - r
    if t & 1:
        c = N - 1 - c
    return r, c


def transform_board(t: int, board: int) -> int:
    out = 0
    for i in range(N * N):
        if board >> i & 1:
            r, c = transform_cell(t, i // N, i % N)
            out |= 1 << (r * N + c)
    return out


def transform_piece(t: int, piece_id: int, table: str = "new") -> int:
    """Id of the image of a piece under symmetry t, normalized to its bounding-box top-left."""
    pieces = all_pieces(table)
    cells = [transform_cell(t, r, c) for r, c in pieces[piece_id].cells]
    r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
    image = sorted((r - r0, c - c0) for r, c in cells)
    for i, p in enumerate(pieces):
        if sorted(p.cells) == image:
            return i
    raise ValueError(f"piece {piece_id} has no image under symmetry {t} in table {table!r}")


def shape_cells(name: str) -> list[tuple[int, int]]:
    """Cells (row, col) of a named tuple shape: window4 (4x4), line (1x10) or rect<h>x<w>."""
    if name == "window4":
        h, w = 4, 4
    elif name == "line":
        h, w = 1, N
    elif m := re.fullmatch(r"rect(\d+)x(\d+)", name):
        h, w = int(m[1]), int(m[2])
        if not (1 <= h <= N and 1 <= w <= N and h * w <= 20):
            raise ValueError(f"shape {name}: needs 1 <= h, w <= 10 and h * w <= 20")
    else:
        raise ValueError(f"unknown tuple shape {name!r} (window4, line, rect<h>x<w>)")
    return [(r, c) for r in range(h) for c in range(w)]


def shapes_to_tuples(names: list[str]) -> list[list[int]]:
    """One tuple per orbit of each shape's translations under the 8 symmetries: its smallest image, cells sorted."""
    tuples = []
    for name in names:
        cells = shape_cells(name)
        h, w = 1 + max(r for r, _ in cells), 1 + max(c for _, c in cells)
        seen: set[frozenset[int]] = set()
        reps = []
        for r0 in range(N - h + 1):
            for c0 in range(N - w + 1):
                inst = [(r0 + r, c0 + c) for r, c in cells]
                images = [sorted(r * N + c for r, c in (transform_cell(t, r, c) for r, c in inst)) for t in range(8)]
                if frozenset(images[0]) in seen:
                    continue
                seen.update(frozenset(img) for img in images)
                reps.append(min(images))
        tuples += sorted(reps)
    return tuples


def fit_bucket(n: int) -> int:
    """Bucket of a placement count: 0, 1, 2, 3-4, 5-8, 9-16, 17-32, 33+ -> 0..7."""
    return n if n < 3 else min(7, (n - 1).bit_length() + 1)
