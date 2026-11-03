# C++ game engine (ticket #11, scope A)

Date: 2026-11-03. Status: approved in chat, awaiting review of this spec.

## Goal

Build a fast, single-game, allocation-free C++ engine for the classic-mode rules. It is the foundation for CPU-only n-tuple TD training (#10 technique 7), which needs millions of moves per second across the 24 hardware threads.

Speed is the point. Every rule must still match `game.py` exactly, because `game.py` is verified against the shipped binary (`re/verify_rules.py`).

## Scope

In scope:
- the C++ engine core
- an exact port of the spawn RNG, and a fast RNG
- thin Python bindings
- build integration with `uv sync`
- parity tests against `game.py` and `batch_env.py`
- a benchmark script, with results in `docs/RESULTS.md`

Out of scope. These items were dropped from #11 when scope A was chosen; the ticket gets a comment saying so:
- the batched torch API and DLPack feature buffers
- `--backend cpp` for `evaluate` and the viewer
- the acceptance criterion "`rl_v1` with the C++ backend gives the same scores"
- the n-tuple learner itself, which is a separate spec that builds on this engine

## Build

- `pyproject.toml`: the build backend changes from `uv_build` to `scikit-build-core`, with `nanobind` as a build requirement.
  - `[tool.scikit-build] wheel.packages = ["src/block_puzzle_ai"]` keeps the package layout and ships `data/pieces.json`.
  - `[tool.uv] cache-keys` lists `pyproject.toml`, `CMakeLists.txt` and `cpp/**`, so `uv sync` rebuilds when C++ changes.
- `CMakeLists.txt` at the root builds a single nanobind module, `block_puzzle_ai._engine`.
  - Flags: `-O3`, C++20.
  - `-march=native` is behind the option `BPA_NATIVE`, default ON, because this is a local project.
  - `-ffast-math` is never used: the spawn level depends on exact float32 rounding.
- Toolchain: GCC 16 and CMake 4.4, both present.

## Files

| File | Purpose |
|---|---|
| `cpp/rng.hpp` | `PyRandom` (bit-exact `random.Random`) and `Xoshiro256` |
| `cpp/engine.hpp` | piece tables, state, legal moves, placement + clears, spawn, spawn distribution, random play |
| `cpp/bindings.cpp` | nanobind module `_engine` |
| `CMakeLists.txt` | build |
| `src/block_puzzle_ai/engine.py` | Python wrapper: builds tables from `pieces.py`, and offers `CppGame` with a `game.Game`-like interface |
| `tests/test_engine.py` | parity and edge-case tests |
| `scripts/bench_engine.py` | moves/s benchmark |

## Engine core (`cpp/engine.hpp`, header-only)

### Board and pieces

- **Board:** `unsigned __int128` with bit `r·10 + c`, row 0 at the top. This is the layout of `game.py`. Bits 100–127 are always 0.
- **`Tables`:** built once from data passed in by Python (`all_pieces()` and `HARD_GROUPS`), so `pieces.json` stays the only source of the piece table. Fixed-capacity arrays (at most 32 pieces, 16 groups and 9 cells per piece) hold:
  - for each piece: its cell offsets (`r·10 + c`), size, group, and its shape mask at position 0
  - `valid[p]`: a 100-bit mask of the legal top-left positions, meaning the bounding box fits on the board
  - for each group: its offset and size, and whether it is hard
- **Legal positions of a piece:** `valid[p] & ~(board >> off_0 | board >> off_1 | …)`.
  - Position `q` is blocked exactly when some cell `q + off_k` is filled.
  - Shifts cannot wrap a row, because `valid` only contains positions where the bounding box fits.
- **Placing and clearing**, in `apply(state, slot, pos)`:
  1. `b = board | (shape[p] << pos)`.
  2. A row is full when its 10-bit field is `0x3FF`. Full columns are the AND of the 10 row fields.
  3. Clear the union of all full lines.
  4. Score `10·size + 50·L·(L+1)`, where L counts rows and columns.

  Rows and columns are detected on the same board and then removed together, as in `clear_full_lines`.

### State

A plain copyable struct:
- `board`
- `pending[3]`, the piece ids
- `score`, `moves`, `lines`
- `history[3]`, the most recent group first, with −1 for an empty entry

Copying the struct is how search clones a state.

### Moves and spawning

**`step(state, slot, pos, rng)`:**
1. Check that the move is legal.
2. Remove `pending[slot]`, keeping the other two pieces in order.
3. Spawn a new piece with the score *before* this move, and append it as the third piece.
4. Apply the placement, add the points, and increment `moves` and `lines`.

This is the same order as `Game.step`.

`reset` empties the board, sets score, moves and lines to 0, and spawns 3 pieces at score 0. The history persists across resets, as in the app.

**`spawn(score, history, rng)`** is an exact port of `PieceGenerator.spawn`, including every draw it makes:
1. `r = randrange(1000)`.
2. `level = (int64)((float)r / 1000.0f * (float)score)`, in float32, truncated.
3. The allowed group count is `n_groups − 3`, plus 1 for each threshold the level exceeds (2999, 5999, 11999).
4. `group = randrange(allowed)`.
5. If any history entry is a hard group, `group = randrange(8)`. The first group draw is still consumed.
6. Push `group` onto the history, then `variant = randrange(group_size)`. This draw is consumed even when the group has only one variant.

**`spawn_distribution(score, history) → double[P]`** gives the exact probabilities, as in `batch_env.spawn_distribution`. It averages the uniform group choice over all 1000 values of `r`, then applies the cooldown override and splits each group's probability evenly over its variants.

**Legal-move order:** slot-major, then position ascending. This is the order of `Game.legal_moves()`, so parity tests can compare lists directly.

### RNGs (`cpp/rng.hpp`)

Both expose `randrange(n)` and are template parameters of `step`, `reset` and `spawn`.

- **`PyRandom`:** MT19937 seeded like CPython's `random.seed(int)`.
  - Seeding: `init_by_array` over the absolute value of the seed, split into 32-bit words, least significant first, with `[0]` for seed 0. Seeds are limited to `uint64`.
  - `getrandbits(k ≤ 32)` is `genrand_uint32() >> (32 − k)`.
  - `randrange(n)` draws `getrandbits(n.bit_length())` and repeats while the result is ≥ n.
  - This reproduces `game.Game(seed)` exactly.
- **`Xoshiro256`:** `xoshiro256**`, seeded through splitmix64.
  - `randrange(n)` uses Lemire's multiply-shift method with rejection.
  - It is used for speed runs and training.

### Random play (benchmark and smoke test)

`random_play(tables, threads, moves_per_thread, seed) → (moves, games, seconds)`:
- Each thread owns a `Xoshiro256`, seeded with the (thread + 1)-th output of a splitmix64 stream started at `seed`. (`splitmix64(seed + thread)` was the original plan; the final review found that it makes runs with neighbouring seeds share all but one thread stream.) Each thread plays games one after another.
- Each move picks uniformly among all legal `(slot, pos)` moves: count the bits of the three legal masks, draw an index, then select the matching bit.
- A game that has no legal move is reset.
- There is no shared state and no locking. The GIL is released.

`afterstate_play(...)` has the same signature. Before each move it also applies every legal move to a copy of the state, which is the per-move work a 1-ply learner does. It additionally returns the number of after-states.

## Python surface

`_engine` exports:
- `Tables(cells: list[list[int]], groups: list[int], hard_groups: list[int])`.
- `Game(tables, seed: int, fast_rng: bool = False)`:
  - `reset()`
  - `legal_moves() → list[(slot, row, col)]`
  - `step(slot, row, col) → (points, lines, done)`
  - properties: `board_lo` and `board_hi` (two `uint64`), `pending`, `score`, `moves`, `lines_cleared`, `history`, `done`
  - `set_state(board_lo, board_hi, pending, score, history)`
  - `spawn_distribution() → list[float]`
- `random_play(...)` and `afterstate_play(...)`.

Errors:
- An illegal move or a move after game over raises `ValueError`, through `std::invalid_argument`.
- Tables that exceed the fixed capacities raise `ValueError` at construction.

`block_puzzle_ai/engine.py` wraps these:
- `cpp_tables(table="new")`, cached.
- `CppGame(seed)`, with `board` as a Python int (`lo | hi << 64`), `pending` as `Piece` objects, and `legal_moves()` and `step(Move)` mirroring `game.Game`.

## Tests (`tests/test_engine.py`, in `uv run pytest`)

1. **Trajectory parity:** 1,000 seeded games, with `CppGame(seed)` stepped in lock-step with `game.Game(seed)`. Moves are drawn by a separate `random.Random` from the identical legal-move lists. At every step, the board, pending pieces, score, points, lines and `done` must be equal.
2. **After-state parity on hard positions:** 1,000 random boards, filled at densities from 30 to 90%, with random pending pieces. For every legal move of every piece, `apply` must match `game.py`'s `clear_full_lines` + `points_for` on the board and the points. These boards hit simultaneous row and column clears, which random play rarely reaches.
3. **Spawn parity across the thresholds:** random play keeps the score under about 2k, so the thresholds are tested directly.
   - Scores: 0, 2999–3001, 5999–6001, 11999–12001, 10⁵ and 4·10⁷, each with histories containing and not containing hard groups.
   - At each one, 500 spawns from `PyRandom(seed)` must equal `PieceGenerator(random.Random(seed))`.
4. **`spawn_distribution`** on the same grid must match `batch_env.spawn_distribution` to 1e-6.
5. **Errors:** an illegal move, a move after game over, and oversized tables each raise `ValueError`.
6. **Random play** with 2 threads runs, and returns positive moves with `games ≥ threads`.

## Benchmark (`scripts/bench_engine.py`)

The script reports `random_play` and `afterstate_play` at 1, 12 and 24 threads, plus `game.py` and `BatchGame` random play as reference points.

Initial targets:
- `random_play`: ≥ 1M moves/s on one thread, ≥ 10M moves/s on 24 threads
- `afterstate_play`: reported only

The targets get revised after the first measurement, and the results go into `docs/RESULTS.md` under ticket #11.

## Acceptance

- `uv sync` builds the extension from a clean checkout, and `uv run pytest` passes, including all engine parity tests.
- The benchmark has been run, its results are in `docs/RESULTS.md`, and the random-play targets are met or revised with the reason.
- The #11 ticket is updated: its checklist covers the scope-A items, and the dropped items are listed.
