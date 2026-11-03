# C++ Game Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a header-only C++ bitboard engine for the classic-mode rules. It must match `game.py` exactly (including the spawn RNG), have nanobind bindings that `uv sync` builds, and reach ≥ 1M random-play moves/s per thread.

**Architecture:**
- `cpp/rng.hpp` and `cpp/engine.hpp` hold the engine. It is a single game, and every function takes a `Tables` and a copyable `State`. Functions that draw random numbers are templated on the RNG.
- `cpp/bindings.cpp` exposes them as `block_puzzle_ai._engine`.
- `src/block_puzzle_ai/engine.py` wraps that module in a `game.Game`-like `CppGame`, built from the Python piece table.
- Parity tests drive `CppGame` and `game.Game` in lock-step.

**Tech Stack:** C++20 (GCC 16), CMake 4.4, nanobind, scikit-build-core, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-11-03-cpp-engine-design.md`. Read it first; this plan does not repeat its rules.

## Global Constraints

- The board is `unsigned __int128` with bit `r·10 + c`, row 0 at the top. Bits 100–127 are always 0.
- The spawn level is `(int64)((float)r / 1000.0f * (float)score)` in float32. Never use `-ffast-math`. Compile with `-ffp-contract=off`.
- Thresholds are 2999, 5999 and 11999, and the allowed group count is `n_groups − 3 + (#thresholds exceeded)`. The hard-group cooldown checks the last 3 spawned groups and then uses `randrange(8)`.
- `step` order is: check legality, remove the slot (the others keep their order), spawn with the **pre-move** score and append, place and clear, then score `10·size + 50·L·(L+1)`.
- Every random draw that `PieceGenerator.spawn` makes is also made in C++. That includes the overridden group draw and `randrange(1)` for single-variant groups.
- Piece ids are indices into `pieces.all_pieces(table)`. The piece data comes from Python; C++ has no copy of `pieces.json`.
- Fixed capacities: 32 pieces, 16 groups, 9 cells per piece. Larger tables raise `ValueError`.
- `-march=native` sits behind the CMake option `BPA_NATIVE`, default ON. `-O3`, Release.
- Legal moves are listed slot-major, then by position ascending, which is `Game.legal_moves()` order.
- Errors come from C++ as `std::invalid_argument`, which nanobind turns into Python `ValueError`.

## Review Focus

1. **Bad ids passed to `set_state`:** pending ids ≥ `n_pieces`, history entries outside [−1, `n_groups`), or board bits ≥ 100 must raise `ValueError`, never read out of bounds. Tested in Task 3, `test_set_state_rejects_bad_input`.
2. **Seeds ≥ 2³²:** CPython seeds them through several 32-bit key words, and 2⁶⁴−1 is the largest seed allowed. `PyRandom` must match `random.Random` for these seeds too. Tested in Task 1, `test_pyrandom_matches_cpython`.
3. **A second game after `reset()`:** it must continue with the spawn history of the first game, as the app does. Tested in Task 4: the trajectory test plays two games per seed.
4. **A stale extension:** editing `cpp/` must rebuild `_engine` on the next `uv run`. Otherwise the parity tests run old code. Checked in Task 1, Step 6.
5. **Odd thread counts:** `threads < 1` raises `ValueError`. With `moves_per_thread = 0` the result is 0 moves and `games == threads`. Tested in Task 5, `test_random_play_edge_counts`.

---

### Task 1: Build integration and RNGs

**Files:**
- Modify: `pyproject.toml` (the `[build-system]` section; add `[tool.scikit-build]` and `[tool.uv]`)
- Modify: `.gitignore` (add `build/`)
- Create: `CMakeLists.txt`, `cpp/rng.hpp`, `cpp/bindings.cpp`
- Test: `tests/test_engine.py`

**Interfaces:**
- Produces in C++ (`namespace bpa`, `cpp/rng.hpp`):
  - `class PyRandom { explicit PyRandom(uint64_t seed); uint32_t getrandbits(int k); uint32_t randrange(uint32_t n); }`
  - `class Xoshiro256 { explicit Xoshiro256(uint64_t seed); uint64_t next(); uint32_t randrange(uint32_t n); }`
  - `uint64_t splitmix64(uint64_t& state)`
- Produces in Python: `block_puzzle_ai._engine.PyRandom(seed)` with `.randrange(n)` and `.getrandbits(k)`, and `_engine.Xoshiro256(seed)` with `.randrange(n)`.

- [ ] **Step 1: Write the failing tests** in `tests/test_engine.py`

```python
import random
import pytest
from block_puzzle_ai import _engine

SEEDS = [0, 1, 42, 999, 2**32 - 1, 2**32 + 5, 2**40 + 123, 2**64 - 1]

@pytest.mark.parametrize("seed", SEEDS)
def test_pyrandom_matches_cpython(seed):
    ours, ref = _engine.PyRandom(seed), random.Random(seed)
    for n in [1000, 12, 1, 8, 2, 4, 9, 7, 3] * 200:  # interleaved like randTiles
        assert ours.randrange(n) == ref.randrange(n)
    assert [ours.getrandbits(k) for k in (1, 5, 31, 32)] == [ref.getrandbits(k) for k in (1, 5, 31, 32)]

def test_xoshiro_is_deterministic_and_in_range():
    a, b = _engine.Xoshiro256(7), _engine.Xoshiro256(7)
    draws = [a.randrange(10) for _ in range(100_000)]
    assert draws[:1000] == [b.randrange(10) for _ in range(1000)]
    counts = [draws.count(i) for i in range(10)]
    assert min(counts) > 9_500 and max(counts) < 10_500
```

- [ ] **Step 2: Run and see it fail**

Run: `uv run pytest tests/test_engine.py -v`
Expected: FAIL with `ImportError: cannot import name '_engine'`.

- [ ] **Step 3: Build integration**
- `pyproject.toml`:
  - `[build-system]`: `requires = ["scikit-build-core>=0.11", "nanobind>=2.4"]`, `build-backend = "scikit_build_core.build"`.
  - `[tool.scikit-build]`: `wheel.packages = ["src/block_puzzle_ai"]`, `cmake.build-type = "Release"`.
  - `[tool.uv]`: `cache-keys = [{ file = "pyproject.toml" }, { file = "CMakeLists.txt" }, { file = "cpp/**" }]`.
- `CMakeLists.txt`:
  - `find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)` and `find_package(nanobind CONFIG REQUIRED)`.
  - `nanobind_add_module(_engine NB_STATIC cpp/bindings.cpp)` and `install(TARGETS _engine LIBRARY DESTINATION block_puzzle_ai)`.
  - C++20, `-O3 -ffp-contract=off`, and `option(BPA_NATIVE "build with -march=native" ON)`.

- [ ] **Step 4: Implement `cpp/rng.hpp` and bind both classes in `cpp/bindings.cpp`**

`PyRandom` is MT19937 with CPython's seeding:
1. `init_genrand(19650218)`, then `init_by_array(key)`.
2. `key` is the seed's 32-bit words, least significant first. Seed 0 gives `key = [0]`.

Draws:
- `getrandbits(k)` for 1 ≤ k ≤ 32 is `genrand_uint32() >> (32 − k)`.
- `randrange(n)` sets `k = bit_width(n)`, then repeats `r = getrandbits(k)` while `r >= n`.

`Xoshiro256` is `xoshiro256**`. Its four state words come from `splitmix64`, and `randrange` uses Lemire's multiply-shift method with rejection.

- [ ] **Step 5: Run the tests and see them pass**

Run: `uv sync && uv run pytest tests/test_engine.py -v`
Expected: 9 passed.

- [ ] **Step 6: Check rebuilds and the existing suite**
1. Append a comment line to `cpp/rng.hpp`, then run `uv run python -c "import block_puzzle_ai._engine"`. Expected: uv prints that it is rebuilding `block-puzzle-ai`.
2. Remove the comment.
3. Run `uv run python -c "from block_puzzle_ai.pieces import all_pieces; print(len(all_pieces()))"`. Expected: `28`, which shows `data/pieces.json` is shipped.
4. Run `uv run pytest -q`. Expected: all 50 existing tests pass, plus the new ones.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock .gitignore CMakeLists.txt cpp/ tests/test_engine.py
git commit -m "Engine (#11): nanobind build and CPython-exact RNG"
```

---

### Task 2: Piece tables and spawning

**Files:**
- Create: `cpp/engine.hpp`, `src/block_puzzle_ai/engine.py`
- Modify: `cpp/bindings.cpp`, `tests/test_engine.py`

**Interfaces:**
- Consumes: `PyRandom` and `Xoshiro256` from Task 1.
- Produces in C++ (`cpp/engine.hpp`):
  - `using Board = unsigned __int128;`
  - `struct Tables { int n_pieces, n_groups; int size[32], group[32], offsets[32][9]; Board shape[32], valid[32]; int group_offset[16], group_size[16]; bool hard[16]; };`
    - `shape[p]` is the piece at position 0.
    - `valid[p]` holds the top-left positions where the bounding box fits.
  - `Tables make_tables(const std::vector<std::vector<int>>& cells, const std::vector<int>& groups, const std::vector<int>& hard_groups);`
    - `cells[p]` holds the offsets `r·10 + c`.
    - Groups must be non-decreasing and contiguous from 0.
    - Invalid input throws `std::invalid_argument`.
  - `template <class Rng> int spawn(const Tables&, int64_t score, int history[3], Rng&);` returns a piece id and updates `history`.
  - `void spawn_distribution(const Tables&, int64_t score, const int history[3], double* out);` fills `out[n_pieces]`.
- Produces in Python:
  - `_engine.Tables(cells, groups, hard_groups)` with property `n_pieces`.
  - `_engine.spawn_sequence(tables, seed: int, score: int, history: list[int], n: int) -> list[int]`. It uses `PyRandom(seed)`; the score stays fixed and the history evolves.
  - `_engine.spawn_distribution(tables, score: int, history: list[int]) -> list[float]`.
  - In `engine.py`:
    - `cpp_tables(table="new") -> _engine.Tables` (cached)
    - `spawn_sequence(seed, score, history, n, table="new") -> list[int]`, which pads `history` with −1 to 3 entries

- [ ] **Step 1: Write the failing tests** (append to `tests/test_engine.py`)

```python
import torch
from block_puzzle_ai.batch_env import spawn_distribution as torch_spawn_distribution, tables_for
from block_puzzle_ai.engine import cpp_tables, spawn_sequence
from block_puzzle_ai.game import PieceGenerator
from block_puzzle_ai.pieces import all_pieces, load_piece_table

SCORES = [0, 1, 2999, 3000, 3001, 3002, 3003, 3004, 5999, 6001, 6005, 6006, 6007,
          11999, 12001, 12011, 12012, 12013, 100_000, 16_777_217, 40_000_001]
HISTORIES = [[], [0], [8], [3, 10], [1, 2, 11], [4, 5, 6]]

@pytest.mark.parametrize("score", SCORES)
def test_spawn_sequence_matches_reference(score):
    ids = {p: i for i, p in enumerate(all_pieces())}
    for seed, hist in enumerate(HISTORIES):
        gen = PieceGenerator(random.Random(seed), history=list(hist))
        assert spawn_sequence(seed, score, hist, 500) == [ids[gen.spawn(score)] for _ in range(500)]

@pytest.mark.parametrize("seed", [2**32 + 5, 2**64 - 1])
def test_spawn_sequence_big_seeds(seed):
    ids = {p: i for i, p in enumerate(all_pieces())}
    gen = PieceGenerator(random.Random(seed))
    assert spawn_sequence(seed, 100_000, [], 500) == [ids[gen.spawn(100_000)] for _ in range(500)]

@pytest.mark.parametrize("score", SCORES)
def test_spawn_distribution_matches_batch_env(score):
    t = tables_for("cpu")
    for hist in HISTORIES:
        h = (hist + [-1, -1, -1])[:3]
        expected = torch_spawn_distribution(t, torch.tensor([score]), torch.tensor([h]))[0].tolist()
        got = _engine.spawn_distribution(cpp_tables(), score, h)
        assert max(abs(a - b) for a, b in zip(got, expected)) < 1e-6
        assert abs(sum(got) - 1) < 1e-9

@pytest.mark.parametrize("cells, groups", [
    ([[0]] * 33, list(range(16)) + [15] * 17),        # 33 pieces
    ([[0]] * 17, list(range(17))),                    # 17 groups
    ([list(range(10))], [0]),                         # 10 cells
    ([[100]], [0]),                                   # offset off the board
    ([[0], [0]], [1, 0]),                             # groups not non-decreasing from 0
])
def test_tables_reject_bad_input(cells, groups):
    with pytest.raises(ValueError):
        _engine.Tables(cells, groups, [])
```

- [ ] **Step 2: Run and see it fail**

Run: `uv run pytest tests/test_engine.py -v -k "spawn or tables"`
Expected: FAIL with `ModuleNotFoundError: No module named 'block_puzzle_ai.engine'`.

- [ ] **Step 3: Implement `make_tables`, `spawn` and `spawn_distribution` in `cpp/engine.hpp`, and bind them**

How `spawn` keeps the RNG stream aligned:
- It always makes 3 draws: `r = randrange(1000)`, `group = randrange(allowed)`, then `variant = randrange(group_size[group])`.
- When the cooldown applies, it adds a draw `randrange(8)` between the group and variant draws, which replaces `group`.

`spawn_distribution` does the following:
1. For each of the 1000 values of `r`, take a uniform distribution over the allowed groups, and average the 1000 results.
2. If the cooldown applies, replace this with a uniform distribution over groups 0–7.
3. Split each group's probability evenly over its variants.

- [ ] **Step 4: Implement `engine.py`: `cpp_tables` and `spawn_sequence`**

`cpp_tables` passes `[[r*10 + c for r, c in p.cells] for p in all_pieces(table)]`, `[p.group for p in all_pieces(table)]` and `sorted(HARD_GROUPS)`.

- [ ] **Step 5: Run the tests and see them pass**

Run: `uv run pytest tests/test_engine.py -v`
Expected: all pass, 58 tests in this file.

- [ ] **Step 6: Commit**

```bash
git add cpp/ src/block_puzzle_ai/engine.py tests/test_engine.py
git commit -m "Engine (#11): piece tables, exact spawn and spawn distribution"
```

---

### Task 3: Board operations, legal moves and after-states

**Files:**
- Modify: `cpp/engine.hpp`, `cpp/bindings.cpp`, `src/block_puzzle_ai/engine.py`, `tests/test_engine.py`

**Interfaces:**
- Consumes: `Tables`, `make_tables`, `spawn` and `cpp_tables` from Task 2.
- Produces in C++:
  - `struct State { Board board = 0; int pending[3] = {}; int64_t score = 0, moves = 0, lines = 0; int history[3] = {-1, -1, -1}; };`
  - `struct Placed { Board board; int points; int lines; };`
  - `Board legal_positions(const Tables&, Board board, int piece);`
  - `Placed apply(const Tables&, Board board, int piece, int pos);` does not check legality.
  - `bool has_legal_move(const Tables&, const State&);`
- Produces in Python:
  - `_engine.Game(tables, seed: int, fast_rng: bool = False)`. It holds a copy of `Tables` and one RNG (a `std::variant` of the two classes).
    - Read-only properties: `board_lo`, `board_hi`, `pending`, `score`, `moves`, `lines_cleared`, `history` and `done`.
    - `set_state(board_lo, board_hi, pending: list[int], score: int, history: list[int])` sets moves and lines to 0 and recomputes `done`.
    - `legal_moves() -> list[tuple[int, int, int]]` returns `(slot, row, col)` tuples.
    - `afterstates() -> list[tuple[int, int, int, int, int, int, int]]` returns `(slot, row, col, lo, hi, points, lines)` tuples, in the same order as `legal_moves()`.
    - `spawn_distribution() -> list[float]` uses the state's own score and history.
  - In `engine.py`:
    - `class CppGame(seed: int, table: str = "new", fast_rng: bool = False)`. Its properties are `board: int` (`lo | hi << 64`), `pending: list[Piece]`, `score`, `moves`, `lines_cleared`, `history: list[int]` and `done`.
    - Its methods are `set_state(board: int, pending: Sequence[Piece], score: int, history: Sequence[int])` (history padded to 3 with −1), `legal_moves() -> list[Move]`, `afterstates() -> list[tuple[Move, int, int, int]]` (move, board after, points, lines) and `spawn_distribution()`.

- [ ] **Step 1: Write the failing tests**

```python
from block_puzzle_ai.engine import CppGame
from block_puzzle_ai.game import Game, Move, clear_full_lines, placements, points_for

def test_afterstates_match_reference_on_dense_boards():
    rng, pieces, max_lines = random.Random(7), all_pieces(), 0
    for i in range(1000):
        density = 0.3 + 0.6 * i / 999
        board = sum(1 << c for c in range(100) if rng.random() < density)
        board, _, _ = clear_full_lines(board)  # reachable boards have no full line before a move
        pending = [rng.choice(pieces) for _ in range(3)]
        ref = Game(0)
        ref.board, ref.pending = board, list(pending)
        expected = []
        for m in ref.legal_moves():
            piece = pending[m.slot]
            pl = next(p for p in placements(piece) if (p.row, p.col) == (m.row, m.col))
            after, lines, _ = clear_full_lines(board | pl.mask, pl.lines)
            expected.append((m, after, points_for(piece.size, lines), lines))
            max_lines = max(max_lines, lines)
        cpp = CppGame(0)
        cpp.set_state(board, pending, 0, [])
        assert cpp.legal_moves() == [m for m, *_ in expected]
        assert cpp.afterstates() == expected
        assert cpp.done == (not expected)
    assert max_lines >= 3  # the boards did produce simultaneous row and column clears

def test_set_state_rejects_bad_input():
    g, p = CppGame(0), all_pieces()[:3]
    for args in [(1 << 100, p, 0, []), (0, p, -1, []), (0, p, 0, [16]), (0, p, 0, [-2])]:
        with pytest.raises(ValueError):
            g.set_state(*args)
    raw = _engine.Game(cpp_tables(), 0)
    for pending in ([28, 0, 0], [-1, 0, 0], [0, 0]):
        with pytest.raises(ValueError):
            raw.set_state(0, 0, pending, 0, [-1, -1, -1])
```

- [ ] **Step 2: Run and see it fail**

Run: `uv run pytest tests/test_engine.py -v -k "afterstates or set_state"`
Expected: FAIL with `ImportError: cannot import name 'CppGame'`.

- [ ] **Step 3: Implement the board operations in `cpp/engine.hpp`**

`legal_positions` computes `valid[p] & ~(board >> off_0 | … | board >> off_{n−1})`.

`apply`:
1. Set `b = board | shape[p] << pos`.
2. Row `r` is full when `(b >> 10r) & 0x3FF == 0x3FF`.
3. The full-column bits are the AND of the 10 row fields.
4. Clear the union of the full rows and columns.
5. `points = 10·size + 50·L·(L+1)`, where L counts rows plus columns.

Enumerate moves by walking the set bits of each slot's legal mask with `countr_zero` on the two 64-bit halves.

- [ ] **Step 4: Bind `Game` and implement `CppGame`**

`set_state` rejects each of the following with `std::invalid_argument`:
- a pending list whose length is not 3
- a pending id outside [0, `n_pieces`)
- a history entry outside [−1, `n_groups`)
- a negative score
- `board_hi >> 36 != 0`

- [ ] **Step 5: Run the tests and see them pass**

Run: `uv run pytest tests/test_engine.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add cpp/ src/block_puzzle_ai/engine.py tests/test_engine.py
git commit -m "Engine (#11): bitboard legal moves, placement and line clears"
```

---

### Task 4: Steps, resets and trajectory parity

**Files:**
- Modify: `cpp/engine.hpp`, `cpp/bindings.cpp`, `src/block_puzzle_ai/engine.py`, `tests/test_engine.py`

**Interfaces:**
- Consumes: `State`, `apply`, `legal_positions`, `has_legal_move` and `spawn` (Tasks 2–3), and `CppGame` (Task 3).
- Produces in C++:
  - `template <class Rng> void reset(const Tables&, State&, Rng&);`
  - `template <class Rng> Placed step(const Tables&, State&, int slot, int pos, Rng&);` throws `std::invalid_argument` on an illegal move.
- Produces in Python:
  - `_engine.Game.reset()` and `_engine.Game.step(slot, row, col) -> tuple[int, int, bool]`, returning `(points, lines, done)`. The constructor calls `reset()`. Stepping a finished game raises `ValueError`.
  - `CppGame.reset()` and `CppGame.step(move: Move) -> tuple[int, int, bool]`.

- [ ] **Step 1: Write the failing tests**

```python
def assert_same(ref: Game, cpp: CppGame):
    assert cpp.board == ref.board
    assert cpp.pending == ref.pending
    assert (cpp.score, cpp.moves, cpp.lines_cleared, cpp.done) == (ref.score, ref.moves, ref.lines_cleared, ref.done)
    assert cpp.history == (ref.generator.history[:3] + [-1, -1, -1])[:3]

@pytest.mark.parametrize("block", range(10))
def test_trajectory_parity(block):
    for seed in range(100 * block, 100 * block + 100):
        ref, cpp, pick = Game(seed), CppGame(seed), random.Random(10_000 + seed)
        for _ in range(2):  # the second game checks that reset keeps the spawn history
            assert_same(ref, cpp)
            while not ref.done:
                moves = ref.legal_moves()
                assert cpp.legal_moves() == moves
                move = pick.choice(moves)
                r = ref.step(move)
                assert cpp.step(move) == (r.points, r.lines, r.done)
                assert_same(ref, cpp)
            ref.reset()
            cpp.reset()

DOT, SQUARE3 = load_piece_table()[0][0], load_piece_table()[8][0]  # as in tests/test_game.py
CHECKER = sum(1 << (r * 10 + c) for r in range(10) for c in range(10) if (r + c) % 2 == 0)

def test_step_errors():
    g = CppGame(3)
    g.set_state(1, [DOT] * 3, 0, [])  # cell (0, 0) filled
    with pytest.raises(ValueError):
        g.step(Move(0, 0, 0))  # occupied
    with pytest.raises(ValueError):
        g.step(Move(0, 10, 0))  # off the board
    g.set_state(CHECKER, [SQUARE3] * 3, 0, [])  # no two empty cells are adjacent, and no line is full
    assert g.done and g.legal_moves() == []
    with pytest.raises(ValueError):
        g.step(Move(0, 0, 1))

def test_fast_rng_game_is_deterministic():
    a, b = CppGame(5, fast_rng=True), CppGame(5, fast_rng=True)
    pick = random.Random(0)
    while not a.done:
        move = pick.choice(a.legal_moves())
        assert a.step(move) == b.step(move)
    assert a.board == b.board and a.moves > 0
```

- [ ] **Step 2: Run and see it fail**

Run: `uv run pytest tests/test_engine.py -v -k "trajectory or step_errors or fast_rng"`
Expected: FAIL with `AttributeError: ... 'step'`.

- [ ] **Step 3: Implement `reset` and `step` in C++, then bind them and add them to `CppGame`**

`step` and `reset` follow the order in Global Constraints. `done` is `!has_legal_move(state)` after every reset, step and `set_state`.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: all tests pass. The 1,000 games take only a few seconds, because random games last about 42 moves.

- [ ] **Step 5: Commit**

```bash
git add cpp/ src/block_puzzle_ai/engine.py tests/test_engine.py
git commit -m "Engine (#11): step/reset with trajectory parity to game.py"
```

---

### Task 5: Multithreaded random play

**Files:**
- Modify: `cpp/engine.hpp`, `cpp/bindings.cpp`, `src/block_puzzle_ai/engine.py`, `tests/test_engine.py`

**Interfaces:**
- Consumes: `State`, `reset`, `step`, `legal_positions`, `apply` and `Xoshiro256`.
- Produces in C++:
  - `struct PlayStats { uint64_t moves, games, afterstates; double seconds; };`
  - `PlayStats play_random(const Tables&, int threads, uint64_t moves_per_thread, uint64_t seed, bool afterstates);` throws `std::invalid_argument` if `threads < 1`.
- Produces in Python:
  - `_engine.random_play(tables, threads, moves_per_thread, seed) -> tuple[int, int, float]`, returning `(moves, games, seconds)`.
  - `_engine.afterstate_play(tables, threads, moves_per_thread, seed) -> tuple[int, int, int, float]`, returning `(moves, games, afterstates, seconds)`.
  - Both release the GIL.
  - `engine.py` wraps them as `random_play(threads, moves_per_thread, seed=0, table="new")` and `afterstate_play(...)`, with the same tuple returns.

- [ ] **Step 1: Write the failing tests**

```python
from block_puzzle_ai.engine import afterstate_play, random_play

def test_random_play_counts_and_determinism():
    moves, games, seconds = random_play(2, 10_000, seed=1)
    assert moves == 20_000 and games >= 2 and seconds > 0
    assert random_play(2, 10_000, seed=1)[:2] == (moves, games)
    moves, games, after, _ = afterstate_play(2, 10_000, seed=1)
    assert moves == 20_000 and after > moves

def test_random_play_edge_counts():
    with pytest.raises(ValueError):
        random_play(0, 100)
    assert random_play(3, 0)[:2] == (0, 3)
```

- [ ] **Step 2: Run and see it fail**

Run: `uv run pytest tests/test_engine.py -v -k random_play`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `play_random`**

Thread `i` does the following:
1. Seeds `Xoshiro256(seed_i)`, where `seed_i` comes from `splitmix64` applied to `seed + i`.
2. Calls `reset` on a local `State`. Each reset counts as one game, so the initial game counts too.
3. Makes exactly `moves_per_thread` moves, resetting whenever a game has no legal move.
4. Picks each move uniformly over all legal `(slot, pos)`: draw `k = randrange(popcount of the three masks)`, then select the k-th set bit.
5. With `afterstates = true`, it first calls `apply` on every legal move and keeps the result (sum the points), so the compiler cannot drop the work. It counts the after-states.

Time with `std::chrono::steady_clock` around starting and joining the `std::jthread`s.

- [ ] **Step 4: Run the tests and see them pass**

Run: `uv run pytest tests/test_engine.py -v -k random_play`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add cpp/ src/block_puzzle_ai/engine.py tests/test_engine.py
git commit -m "Engine (#11): multithreaded random and after-state play"
```

---

### Task 6: Benchmark, docs and ticket

**Files:**
- Create: `scripts/bench_engine.py`
- Modify: `docs/RESULTS.md` (new section "C++ engine (ticket #11)" at the end), `README.md` (one line in "How it works", step 2 "Simulator")

**Interfaces:**
- Consumes: `random_play` and `afterstate_play` (Task 5), plus `game.Game`, `batch_env.BatchGame` and `agents.RandomAgent` as reference points.

- [ ] **Step 1: Write `scripts/bench_engine.py [MOVES_PER_THREAD (default 2_000_000)]`**

It prints one table:
- `random_play` and `afterstate_play` at 1, 12 and 24 threads, each with moves/s
- `game.py` random play for 20,000 moves on 1 thread
- `BatchGame` on CUDA with `RandomAgent`: 4,096 games × 500 steps, timed after one warm-up step

- [ ] **Step 2: Run it**

Run: `uv run python scripts/bench_engine.py`
Expected: `random_play` ≥ 1M moves/s on 1 thread and ≥ 10M on 24 threads.

If it falls short, profile with `perf record` on a small C++ driver or on the module. Fix the hot spot, rerun Task 4's tests, and only then rerun the benchmark. If a target still can't be met, record the measured number and the reason instead of the target.

- [ ] **Step 3: Write the results**
- `docs/RESULTS.md`: add the table, the command, the machine (RTX 4070 Laptop, 12 cores / 24 threads, Zen 4), the parity coverage (1,000 trajectories, 1,000 dense boards, the spawn grid), and what the engine is for next (n-tuple training, #10 technique 7).
- `README.md`: one sentence linking to that section.

- [ ] **Step 4: Check the build from a clean checkout**
1. `git clone . /tmp/claude-1000/-home-mulm-Projects-block-puzzle-ai/3f090a50-c133-4850-9a9f-6de069d357c1/scratchpad/clean`
2. In that directory, run `uv sync && uv run pytest -q`.
3. Expected: everything passes.
4. Delete the clone afterwards.

- [ ] **Step 5: Commit**

```bash
git add scripts/bench_engine.py docs/RESULTS.md README.md
git commit -m "Engine (#11): benchmark and results"
```

- [ ] **Step 6: Ticket (ask the user before posting)**

Edit #11:
- tick the scope-A tasks and acceptance items
- add a comment with the benchmark table and the list of dropped items: the batched torch API/DLPack, `--backend cpp`, and "`rl_v1` same scores with the C++ backend"

Close #11 with the final commit message (`Closes #11`) only after the user agrees.
