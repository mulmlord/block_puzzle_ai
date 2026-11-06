# n-tuple TD Learner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** an n-tuple after-state value function, trained by lock-free multithreaded TD(0) self-play in C++ on the #11 engine, that beats `rl_v2` 1-ply on mean score.

**Architecture:**
- `cpp/ntuple.hpp` holds the following, all built on `engine.hpp`:
  - board symmetries
  - `NTuple`: the weights, `value` and `update`
  - `best_move`
  - a persistent multithreaded `Trainer` with restart pools
  - a thread-count-independent `evaluate`
- `cpp/bindings.cpp` exposes them.
- `src/block_puzzle_ai/ntuple.py` owns:
  - tuple selection, meaning shapes reduced to orbit representatives
  - Python symmetry helpers, used by the tests
  - the config and checkpoint I/O
  - the `train` / `evaluate` CLI, which writes `metrics.jsonl` in `train.py`'s format

**Tech Stack:** C++20 (`std::atomic_ref`, `std::jthread`, BMI2 `pext` with a fallback), nanobind with `nb::ndarray`, NumPy, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-11-03-ntuple-learner-design.md`. Read it first. The engine it builds on is `cpp/engine.hpp`; see `docs/superpowers/specs/2026-11-03-cpp-engine-design.md`.

## Global Constraints

- **Value terms:**
  - `V(a)` = Σ over tuples i, Σ over the 8 symmetries σ, of `w[off_i + index_i(σ(board))]`
  - plus Σ over 12 groups g of `w_fit[cooldown][g][bucket(placements of g's variants)]`
  - plus Σ over the 2 remaining pieces p of `w_rem[group(p)][bucket(placements of p)]`
- **Tuple index:** bit j of the index is the board bit at the tuple's j-th cell, with cells in ascending order. Python passes the cells sorted. Tuples have 1–20 cells, each in 0..99, with no duplicates.
- **Weight layout:** the tuple tables in tuple order (table i holds 2^len(tuple_i) entries), then `w_fit` (2×12×8, index `(cooldown·12 + g)·8 + bucket`), then `w_rem` (12×8, index `g·8 + bucket`). All are float32.
- **Buckets:** 0, 1, 2, 3–4, 5–8, 9–16, 17–32, 33+ map to 0..7.
- **Precision:** `value` sums in double and returns double, and `update` takes a double target. At V ≈ 10⁸, float32 sums would carry rounding error of about ±90, which is the size of the differences between moves.
- **Lookups and updates:**
  - `n_lookups = 8·n_tuples + 14`.
  - Update: `δ = target − V(a)`, and every looked-up weight gets `+= α·δ / n_lookups`, once per lookup.
- **The 8 symmetries** are the dihedral group of the 10×10 board. Python's `transform_cell(t, r, c)`:
  1. If `t & 4`, swap r and c (transpose).
  2. If `t & 2`, set r = 9 − r.
  3. If `t & 1`, set c = 9 − c.

  C++ may enumerate the group in any order, because only the set of 8 matters for the sum.
- **The cooldown flag** is `in_cooldown(tables, history)` at decision time, before the step's spawn.
- **Choosing a move:** greedy argmax over legal moves of `points + γ·V(after-state)`. Moves are scanned in engine order (slot-major, position ascending), and the first maximum wins.
- **Training defaults:** γ = 1.0, α = 0.1, `restart_frac` = 0.5, weights start at 0, no ε.
- **Restarts:** snapshot every 50 moves into a ring of 10. Snapshots from the last 500 moves before a game over join the per-thread pool (capacity 4,096, overwritten as a ring).
- **RNG:**
  - Training thread i and evaluation game i each use `Xoshiro256`, seeded with the (i + 1)-th output of a splitmix64 stream started at `seed`.
  - Evaluation results must not depend on the thread count.
- **Shared weights:** threads touch them only through `std::atomic_ref<float>` with `memory_order_relaxed`.
- **Checkpoint `*.ntw`:** the 8-byte magic `BPANTW01`, a little-endian uint32 JSON length, the JSON (`config`, `tuples`, `moves_trained`, `elapsed_min`, `n_weights`), then the raw float32 weights.
- **`metrics.jsonl`:**
  - Train records: `{"type": "train", "moves_per_s", "games", "deaths", "mean_score", "mean_moves", "restarts", "value_mean", "alpha", "elapsed_min", "moves"}`.
  - Eval records: `{"type": "eval", "eval_<key>"` for every key of `evaluate.summarize(...)`, plus `"best_eval_moves_per_death", "elapsed_min", "moves"}`.

## Review Focus

1. **Long games cut every minute:** `Trainer.run` is called in one-minute chunks, and games must carry on across calls. Otherwise the learner never sees the late game of long games. Tested in Task 3, `test_trainer_chunks_equal_one_run`.
2. **Resume losing `best.ntw`:** after `--resume`, `best.ntw` may only be replaced by a run that beats the best `eval_moves_per_death` already in `metrics.jsonl`. Tested in Task 4, `test_resume_keeps_best`.
3. **Ctrl-C during a long run** must save `latest.ntw` before exiting, instead of losing up to 15 minutes of training. Tested in Task 4, `test_interrupt_saves_latest`.
4. **float32 at large values:** with γ = 1, V grows to 10⁷–10⁸, and updates must still move it. At V ≈ 10⁸, a TD step of α·δ must change V by at least half its nominal size. Tested in Task 3, `test_update_precision_at_large_values`.
5. **Tiny caps:** `evaluate_ntuple(max_moves=0)` must return every game as truncated, with 0 moves and 0 points. It must not crash or play a move. Tested in Task 3, `test_evaluate_zero_cap`.

---

### Task 1: Tuple selection and Python symmetries

**Files:**
- Create: `src/block_puzzle_ai/ntuple.py` (only the parts named here)
- Test: `tests/test_ntuple.py`

**Interfaces:**
- Produces (`block_puzzle_ai.ntuple`):
  - `transform_cell(t: int, r: int, c: int) -> tuple[int, int]`
  - `transform_board(t: int, board: int) -> int`: board is a 100-bit int
  - `transform_piece(t: int, piece_id: int, table: str = "new") -> int`: the id of the σ-image, normalized to its top-left
  - `shape_cells(name: str) -> list[tuple[int, int]]`: accepts `"window4"`, `"line"` and `"rect<h>x<w>"`; any other name raises `ValueError`
  - `shapes_to_tuples(names: list[str]) -> list[list[int]]`: orbit representatives as sorted absolute cell lists, one per orbit. Within a shape, the representatives are ordered by cell list, and shapes come in the order given.
  - `fit_bucket(n: int) -> int`

- [ ] **Step 1: Write the failing tests**

```python
import random
import pytest
from block_puzzle_ai.ntuple import fit_bucket, shape_cells, shapes_to_tuples, transform_board, transform_cell, transform_piece
from block_puzzle_ai.pieces import all_pieces

def orbit(cells):
    return {frozenset(r * 10 + c for r, c in (transform_cell(t, i // 10, i % 10) for i in cells)) for t in range(8)}

@pytest.mark.parametrize("name, count", [("window4", 10), ("line", 5)])
def test_orbits(name, count):
    reps = shapes_to_tuples([name])
    assert len(reps) == count and all(r == sorted(r) for r in reps)
    cells = shape_cells(name)
    h, w = 1 + max(r for r, _ in cells), 1 + max(c for _, c in cells)
    for r0 in range(11 - h):
        for c0 in range(11 - w):
            inst = frozenset((r0 + r) * 10 + c0 + c for r, c in cells)
            assert sum(inst in orbit(rep) for rep in reps) == 1
    assert len(shapes_to_tuples(["window4", "line"])) == 15

def test_transforms_form_a_group_and_map_pieces():
    rng = random.Random(0)
    board = rng.getrandbits(100)
    assert len({transform_board(t, board) for t in range(8)}) == 8
    for p in range(len(all_pieces())):
        images = {transform_piece(t, p) for t in range(8)}
        assert {all_pieces()[i].group for i in images} == {all_pieces()[p].group}

def test_fit_bucket_and_bad_shape():
    assert [fit_bucket(n) for n in (0, 1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33, 99)] == [0, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
    with pytest.raises(ValueError):
        shape_cells("blob")
```

- [ ] **Step 2: Run and see them fail**

Run: `uv run pytest tests/test_ntuple.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'block_puzzle_ai.ntuple'`.

- [ ] **Step 3: Implement the five functions in `ntuple.py`.** Representative: list all translations of the shape, then for each unseen one take the lexicographically smallest sorted cell list among its 8 images, and mark the whole orbit as seen.

- [ ] **Step 4: Run and see them pass**

Run: `uv run pytest tests/test_ntuple.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/block_puzzle_ai/ntuple.py tests/test_ntuple.py
git commit -m "n-tuple (#10): tuple orbits and board symmetries"
```

---

### Task 2: The `NTuple` value function in C++

**Files:**
- Create: `cpp/ntuple.hpp`
- Modify: `cpp/bindings.cpp`, `src/block_puzzle_ai/ntuple.py`, `tests/test_ntuple.py`

**Interfaces:**
- Consumes:
  - `bpa::Tables`, `Board`, `legal_positions`, `in_cooldown`, `bit` from `engine.hpp`
  - `shapes_to_tuples`, `transform_board`, `transform_piece` and `fit_bucket` from Task 1
- Produces in C++ (`namespace bpa`):
  - `struct Afterstate { Board board; int remaining[2]; bool cooldown; };`
  - `class NTuple`. Its constructor `NTuple(const Tables&, std::vector<std::vector<int>> tuples)` throws `std::invalid_argument` on bad tuples.
  - Members:
    - `size_t n_weights() const;`
    - `int n_lookups() const;`
    - `float* weights();`
    - `double value(const Afterstate&) const;`
    - `void update(const Afterstate&, double target, float alpha);`
    - `const Tables& tables() const;`
  - Weights live in a `std::vector<float>`, allocated once.
- Produces in Python:
  - `_engine.NTuple(tables, tuples)` with:
    - `.weights`: a writable float32 NumPy view of length `n_weights`, which keeps the model alive
    - `.n_weights`, `.n_lookups`
    - `.value(board_lo, board_hi, remaining: list[int], cooldown: bool) -> float`
    - `.update(board_lo, board_hi, remaining, cooldown, target, alpha) -> None`
  - `remaining` must hold exactly 2 valid piece ids, otherwise `ValueError`.
  - `ntuple.make_model(tuples: list[list[int]], table: str = "new") -> _engine.NTuple`
  - `ntuple.value(model, board: int, remaining: list[int], cooldown: bool) -> float`, which splits the board into lo and hi

- [ ] **Step 1: Write the failing tests** (append)

```python
import numpy as np
from block_puzzle_ai import _engine
from block_puzzle_ai.engine import CppGame, cpp_tables
from block_puzzle_ai.game import placements
from block_puzzle_ai.ntuple import make_model, value

def n_placements(piece, board):
    return sum(not p.mask & board for p in placements(piece))

def brute_value(tuples, w, board, remaining, cooldown):
    pieces, v, off = all_pieces(), 0.0, 0
    for cells in tuples:
        for t in range(8):
            b = transform_board(t, board)
            v += w[off + sum((b >> c & 1) << j for j, c in enumerate(cells))]
        off += 1 << len(cells)
    for g in range(12):
        count = sum(n_placements(p, board) for p in pieces if p.group == g)
        v += w[off + (cooldown * 12 + g) * 8 + fit_bucket(count)]
    for p in remaining:
        v += w[off + 192 + pieces[p].group * 8 + fit_bucket(n_placements(pieces[p], board))]
    return v

def sample_afterstates(n, seed):
    """n/2 after-states from random play, n/2 random boards; (board, remaining ids, cooldown)."""
    rng, out = random.Random(seed), []
    game = CppGame(seed)
    while len(out) < n // 2:
        if game.done:
            game.reset()
        move, after, _, _ = rng.choice(game.afterstates())
        ids = [all_pieces().index(p) for p in game.pending]
        out.append((after, [ids[i] for i in range(3) if i != move.slot], rng.random() < 0.3))
        game.step(move)
    while len(out) < n:
        density = rng.uniform(0.2, 0.7)
        board = sum(1 << c for c in range(100) if rng.random() < density)
        out.append((board, [rng.randrange(28), rng.randrange(28)], rng.random() < 0.3))
    return out

@pytest.fixture
def random_model():
    tuples = shapes_to_tuples(["window4", "line"])
    model = make_model(tuples)
    model.weights[:] = np.random.default_rng(0).normal(size=model.n_weights).astype(np.float32)
    return tuples, model

def test_value_matches_brute_force(random_model):
    tuples, model = random_model
    assert model.n_weights == 10 * 65536 + 5 * 1024 + 288 and model.n_lookups == 8 * 15 + 14
    for board, rem, cd in sample_afterstates(60, 1):
        assert value(model, board, rem, cd) == pytest.approx(brute_value(tuples, model.weights, board, rem, cd), rel=1e-5, abs=1e-3)

def test_value_is_symmetric(random_model):
    _, model = random_model
    for board, rem, cd in sample_afterstates(200, 2):
        v = value(model, board, rem, cd)
        for t in range(8):
            image = value(model, transform_board(t, board), [transform_piece(t, p) for p in rem], cd)
            assert image == pytest.approx(v, rel=1e-5, abs=1e-3)

@pytest.mark.parametrize("tuples", [[list(range(21))], [[100]], [[3, 3]], [[]]])
def test_ntuple_rejects_bad_tuples(tuples):
    with pytest.raises(ValueError):
        _engine.NTuple(cpp_tables(), tuples)

def test_value_rejects_bad_remaining(random_model):
    _, model = random_model
    for rem in ([0], [0, 28], [-1, 0]):
        with pytest.raises(ValueError):
            model.value(0, 0, rem, False)

def test_update_moves_toward_target(random_model):
    _, model = random_model
    for board, rem, cd in sample_afterstates(20, 3):
        before = value(model, board, rem, cd)
        model.update(board & (2**64 - 1), board >> 64, rem, cd, before + 50.0, 0.1)
        after = value(model, board, rem, cd)
        assert before < after < before + 50.0

def test_update_alpha_one_lands_on_target():
    tuples = [[0, 1, 2, 3, 10, 11, 12, 13]]
    model, rng = make_model(tuples), random.Random(4)
    while True:  # a board whose 8 symmetric indices are distinct, with remaining pieces from different groups
        board = rng.getrandbits(100)
        idx = {sum((transform_board(t, board) >> c & 1) << j for j, c in enumerate(tuples[0])) for t in range(8)}
        if len(idx) == 8:
            break
    rem = [0, 27]
    assert all_pieces()[0].group != all_pieces()[27].group
    model.update(board & (2**64 - 1), board >> 64, rem, False, 123.5, 1.0)
    assert value(model, board, rem, False) == pytest.approx(123.5, abs=1e-3)

def test_update_precision_at_large_values():
    model = make_model(shapes_to_tuples(["window4", "line"]))
    model.weights[:] = 1e8 / model.n_lookups
    board, rem = 0, [0, 1]
    before = value(model, board, rem, False)
    model.update(0, 0, rem, False, before + 1000.0, 0.1)
    assert value(model, board, rem, False) - before >= 50.0  # nominal step: 0.1 * 1000 = 100
```

- [ ] **Step 2: Run and see them fail**

Run: `uv run pytest tests/test_ntuple.py -v -k "value or ntuple or update"`
Expected: FAIL with `ImportError: cannot import name 'make_model'`.

- [ ] **Step 3: Implement `cpp/ntuple.hpp`**
- **Symmetries:** compute the 8 transformed boards from three generators, each done row-wise with tables:
  - H: a 1,024-entry table reversing 10 bits, applied per row
  - V: reverse the row order
  - T: a 1,024-entry table of `Board` with a bit at `c·10` for each set bit c of a row, shifted left by r and ORed together
- **Index:** `pext` on the lo and hi halves, joined as `idx = pext(lo, mlo) | pext(hi, mhi) << popcount(mlo)`. Without `__BMI2__`, fall back to a bit loop.
- **Placement counts:** compute the legal-placement popcount of each of the 28 pieces once per `value` call, and reuse it for both the group sums and the remaining pieces.
- **`update`:** recomputes the lookup indices, then adds through `std::atomic_ref<float>` (relaxed). `value` reads through relaxed `atomic_ref` loads.

- [ ] **Step 4: Bind `NTuple`; add `make_model` and `value` to `ntuple.py`.** Return the weights as `nb::ndarray<nb::numpy, float, nb::ndim<1>>` over `weights()`, with `nb::rv_policy::reference_internal`.

- [ ] **Step 5: Run and see them pass**

Run: `uv run pytest tests/test_ntuple.py -v`
Expected: all pass (14).

- [ ] **Step 6: Commit**

```bash
git add cpp/ src/block_puzzle_ai/ntuple.py tests/test_ntuple.py
git commit -m "n-tuple (#10): symmetric n-tuple value function in C++"
```

---

### Task 3: TD update, move choice, trainer and evaluator

**Files:**
- Modify: `cpp/ntuple.hpp`, `cpp/bindings.cpp`, `src/block_puzzle_ai/ntuple.py`, `tests/test_ntuple.py`

**Interfaces:**
- Consumes: `NTuple`, `Afterstate` (Task 2); `State`, `reset`, `step`, `apply`, `legal_positions`, `for_each_bit`, `has_legal_move`, `splitmix64`, `Xoshiro256` (engine).
- Produces in C++:
  - `struct Choice { int slot, pos, points; Afterstate after; double q; };`
  - `bool best_move(const NTuple&, const State&, float gamma, Choice& out);` returns false if the state has no legal move.
  - `class Trainer`, with `Trainer(NTuple&, float gamma, float alpha, double restart_frac, int threads, uint64_t seed);`
    - `TrainStats run(uint64_t moves);`
    - `float alpha;` (public)
    - Per-thread `State`, ring, pool, RNG and current `Afterstate` persist across `run` calls.
  - `std::vector<GameOutcome> evaluate(const NTuple&, int games, uint64_t seed, int64_t max_moves, int threads, float gamma);`
    - `struct GameOutcome { int64_t score, moves, lines; bool truncated; };`
    - `threads < 1` throws.
- Produces in Python:
  - `_engine.Trainer(ntuple, gamma, alpha, restart_frac, threads, seed)` with `.run(moves) -> dict`, keys `moves seconds games deaths fresh_games mean_score mean_moves restarts value_mean`, and a read/write `.alpha`. The GIL is released and the trainer keeps the model alive.
  - `_engine.evaluate_ntuple(ntuple, games, seed, max_moves, threads, gamma=1.0) -> list[tuple[int, int, int, bool]]`, which releases the GIL.
    - The spec's `tables` argument is dropped, because `NTuple` carries its tables.
    - `gamma` is added so that the move choice matches training. It comes from the checkpoint config.
  - `ntuple.evaluate(model, games, seed, max_moves, threads, gamma=1.0) -> list[evaluate.GameResult]`.

`TrainStats` notes:
- `fresh_games` counts the finished games that started from an empty board. `mean_score` and `mean_moves` are taken over those games, or 0 if there were none.
- `deaths == games`, because training has no cap.
- `value_mean` is the mean V of the chosen after-states.
- `run(moves)` gives thread i `moves / threads` moves, plus 1 if `i < moves % threads`.

- [ ] **Step 1: Write the failing tests** (append)

```python
from block_puzzle_ai.ntuple import evaluate as nt_evaluate

def test_evaluate_deterministic_across_threads(random_model):
    _, model = random_model
    model.weights[:] *= 0.01
    one = nt_evaluate(model, games=16, seed=3, max_moves=2000, threads=1)
    assert nt_evaluate(model, games=16, seed=3, max_moves=2000, threads=4) == one
    assert len(one) == 16 and all(r.moves > 0 for r in one)

def test_evaluate_zero_cap(random_model):
    _, model = random_model
    assert all(r.truncated and r.moves == 0 and r.score == 0 for r in nt_evaluate(model, 8, 1, 0, 2))

def test_trainer_chunks_equal_one_run():
    tuples = shapes_to_tuples(["line"])
    a, b = make_model(tuples), make_model(tuples)
    ta, tb = _engine.Trainer(a, 1.0, 0.1, 0.5, 1, 7), _engine.Trainer(b, 1.0, 0.1, 0.5, 1, 7)
    ta.run(3000)
    ta.run(3000)
    tb.run(6000)
    assert np.array_equal(a.weights, b.weights)

def test_training_improves():
    model = make_model(shapes_to_tuples(["window4", "line"]))
    stats = _engine.Trainer(model, 1.0, 0.1, 0.5, 4, 0).run(3_000_000)
    assert stats["moves"] == 3_000_000 and stats["games"] > 0
    results = nt_evaluate(model, games=64, seed=777, max_moves=100_000, threads=4)
    assert sum(r.moves for r in results) / 64 > 150  # random play: 24, tuned heuristic: 279

def test_trainer_and_evaluate_reject_zero_threads(random_model):
    _, model = random_model
    with pytest.raises(ValueError):
        _engine.Trainer(model, 1.0, 0.1, 0.5, 0, 0)
    with pytest.raises(ValueError):
        nt_evaluate(model, 4, 0, 100, 0)
```

- [ ] **Step 2: Run and see them fail**

Run: `uv run pytest tests/test_ntuple.py -v -k "evaluate or trainer or training"`
Expected: FAIL with `ImportError: cannot import name 'evaluate'`.

- [ ] **Step 3: Implement `best_move`, `Trainer` and `evaluate` in `cpp/ntuple.hpp`, following the spec's TD loop.**
- A thread's first `run` begins with `reset`. Later calls carry on from the saved state and after-state.
- On a game over:
  - update toward 0
  - update the stats
  - move the eligible ring snapshots (game id equal, `moves ≥ end − 500`) into the pool
  - restart: with probability `restart_frac` from a random pool state if the pool is non-empty, otherwise with `reset`

  Then choose the first move of the new game.
- `evaluate` hands out game indices through `std::atomic<int>`. It precomputes the seed of each game, and stops a game when `moves ≥ max_moves`, which makes it truncated.

- [ ] **Step 4: Bind both; add `ntuple.evaluate`, which wraps the tuples into `GameResult`s.**

- [ ] **Step 5: Run and see them pass**

Run: `uv run pytest tests/test_ntuple.py -v`
Expected: all pass. Note how long `test_training_improves` takes in the ledger. If it takes over 60 s, rule on it in the ledger.

- [ ] **Step 6: Commit**

```bash
git add cpp/ src/block_puzzle_ai/ntuple.py tests/test_ntuple.py
git commit -m "n-tuple (#10): TD(0) trainer with restarts and 1-ply evaluator"
```

---

### Task 4: Checkpoints and CLI

**Files:**
- Modify: `src/block_puzzle_ai/ntuple.py`, `tests/test_ntuple.py`

**Interfaces:**
- Consumes: `make_model`, `evaluate`, `shapes_to_tuples`, `_engine.Trainer` (Tasks 1–3); `evaluate.summarize`.
- Produces (`block_puzzle_ai.ntuple`):
  - `@dataclass NTupleConfig`, with `shapes: list[str]` (default `["window4", "line"]`), `gamma: float = 1.0`, `alpha: float = 0.1`, `restart_frac: float = 0.5`, `threads: int = 24` and `seed: int = 0`.
  - `save_checkpoint(path, cfg: NTupleConfig, tuples, weights: np.ndarray, moves_trained: int, elapsed_min: float) -> None`. It writes to `path.tmp`, then renames.
  - `load_checkpoint(path) -> tuple[NTupleConfig, list[list[int]], np.ndarray, dict]`. The dict holds `moves_trained` and `elapsed_min`. A bad magic, or a weight count that differs from `n_weights`, raises `ValueError`.
  - `load_model(path) -> tuple[_engine.NTuple, NTupleConfig, dict]`.
  - `main(argv: list[str] | None = None) -> None`, with the subcommands `train` and `evaluate` as in the spec. `train` also takes `--threads`, `--seed`, `--shapes` (comma-separated), `--gamma`, `--restart-frac` and `--chunk-seconds` (default 60).
  - `if __name__ == "__main__": main()`.

Training loop notes:
- Chunk size: start at 50,000 × threads moves. Then size each chunk from the last measured rate × `chunk-seconds`.
- **Stopping:** `--total-moves M` stops when `moves_trained ≥ M`, and `--hours H` stops when `elapsed_min ≥ 60·H`. Both count earlier runs of the same `--out`, so `--hours 8 --resume` finishes an interrupted 8-hour run.
- An eval runs when at least `eval-minutes` have passed since the last one, and always at the end.
- `--eval-minutes 0` means an eval after every chunk.
- After each eval, save `latest.ntw`. Save `best.ntw` when the eval's moves per death beat the best so far. The best so far is the largest `best_eval_moves_per_death` in `metrics.jsonl`, read with `.get`, so records without the key are ignored.
- On `KeyboardInterrupt`, save `latest.ntw`, print a line and return.
- `evaluate --json` prints `summarize(results) | {"agent": "ntuple:<path>", "seconds": s, "results": [asdict(r) ...]}`.

- [ ] **Step 1: Write the failing tests** (append)

```python
import json, signal, subprocess, sys, time
from block_puzzle_ai.ntuple import NTupleConfig, load_checkpoint, load_model, main, save_checkpoint

def test_checkpoint_roundtrip(tmp_path, random_model):
    tuples, model = random_model
    cfg = NTupleConfig(alpha=0.05)
    save_checkpoint(tmp_path / "m.ntw", cfg, tuples, model.weights, 1234, 5.5)
    loaded, cfg2, meta = load_model(tmp_path / "m.ntw")
    assert cfg2 == cfg and meta == {"moves_trained": 1234, "elapsed_min": 5.5}
    assert np.array_equal(loaded.weights, model.weights)
    for board, rem, cd in sample_afterstates(10, 5):
        assert value(loaded, board, rem, cd) == value(model, board, rem, cd)

def test_checkpoint_rejects_corruption(tmp_path, random_model):
    tuples, model = random_model
    path = tmp_path / "m.ntw"
    save_checkpoint(path, NTupleConfig(), tuples, model.weights, 0, 0.0)
    raw = path.read_bytes()
    (tmp_path / "magic.ntw").write_bytes(b"XXXXXXXX" + raw[8:])
    (tmp_path / "short.ntw").write_bytes(raw[:-4])
    for bad in ("magic.ntw", "short.ntw"):
        with pytest.raises(ValueError):
            load_checkpoint(tmp_path / bad)

def train_args(out, moves):
    return ["train", "--out", str(out), "--total-moves", str(moves), "--threads", "2", "--shapes", "line",
            "--eval-minutes", "0", "--eval-games", "8", "--eval-max-moves", "2000", "--chunk-seconds", "1"]

def test_train_resume_and_evaluate_cli(tmp_path, capsys):
    main(train_args(tmp_path, 100_000))
    main(train_args(tmp_path, 200_000) + ["--resume", "--alpha", "0.05"])
    cfg, _, _, meta = load_checkpoint(tmp_path / "latest.ntw")
    assert meta["moves_trained"] >= 200_000 and cfg.alpha == 0.05
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    evals = [r for r in records if r["type"] == "eval"]
    assert evals and {"eval_mean", "eval_median", "eval_max", "eval_moves_per_death", "elapsed_min", "moves"} <= evals[0].keys()
    assert any(r["type"] == "train" and r["moves_per_s"] > 0 for r in records)
    capsys.readouterr()
    main(["evaluate", "--weights", str(tmp_path / "best.ntw"), "--games", "4", "--seed", "1", "--max-moves", "500", "--threads", "2", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert out["games"] == 4 and len(out["results"]) == 4

def test_resume_keeps_best(tmp_path):
    main(train_args(tmp_path, 100_000))
    best = max(json.loads(l).get("best_eval_moves_per_death", 0) for l in (tmp_path / "metrics.jsonl").read_text().splitlines())
    best_bytes = (tmp_path / "best.ntw").read_bytes()
    records = (tmp_path / "metrics.jsonl").read_text()
    (tmp_path / "metrics.jsonl").write_text(records + json.dumps({"type": "eval", "best_eval_moves_per_death": best * 1e6}) + "\n")
    main(train_args(tmp_path, 200_000) + ["--resume"])
    assert (tmp_path / "best.ntw").read_bytes() == best_bytes  # nothing beat the recorded best

def test_interrupt_saves_latest(tmp_path):
    proc = subprocess.Popen([sys.executable, "-m", "block_puzzle_ai.ntuple", "train", "--out", str(tmp_path), "--hours", "1",
                             "--threads", "1", "--shapes", "line", "--eval-minutes", "60", "--chunk-seconds", "1"])
    time.sleep(8)
    proc.send_signal(signal.SIGINT)
    assert proc.wait(timeout=30) == 0
    assert load_checkpoint(tmp_path / "latest.ntw")[3]["moves_trained"] > 0
```

- [ ] **Step 2: Run and see them fail**

Run: `uv run pytest tests/test_ntuple.py -v -k "checkpoint or cli or resume or interrupt"`
Expected: FAIL with `ImportError: cannot import name 'NTupleConfig'`.

- [ ] **Step 3: Implement the config, the checkpoints and `main` in `ntuple.py`.**

- [ ] **Step 4: Run the whole suite**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/block_puzzle_ai/ntuple.py tests/test_ntuple.py
git commit -m "n-tuple (#10): checkpoints, train/evaluate CLI and metrics"
```

---

### Task 5: Throughput and the 1-hour checkpoint run

**Files:**
- Modify: `docs/RESULTS.md`, adding a new section `## n-tuple learner (#10 technique 7)` at the end.

- [ ] **Step 1: Run 1 hour detached**

```bash
setsid nohup uv run python -m block_puzzle_ai.ntuple train --out runs/nt1h --hours 1 > runs/nt1h.log 2>&1 &
```

Expected: `runs/nt1h/metrics.jsonl` gets train records every ~minute and eval records every 15 min.

- [ ] **Step 2: Measure** `moves_per_s` from the train records, using the median of the last 10. The initial target is ≥ 200k on 24 threads. If it falls short, record the measured number and the main cost: use `perf top` on the process for a minute.

- [ ] **Step 3: Evaluate the checkpoint run**

```bash
uv run python -m block_puzzle_ai.ntuple evaluate --weights runs/nt1h/best.ntw --games 64 --seed 777 --max-moves 200000 --json > runs/nt1h/seed777.json
uv run python scripts/bootstrap_mean.py runs/v2_eval/seed777_v2.json runs/nt1h/seed777.json
```

- [ ] **Step 4: Gate.**
  - Write the throughput, the learning curve (one row per eval) and the seed-777 result into the new RESULTS section.
  - Report the 1-hour moves per game over to the user.
  - If they are **below 1,000**, the learner isn't learning: stop and debug with superpowers:systematic-debugging before Task 6.
  - Otherwise continue.

- [ ] **Step 5: Commit**

```bash
git add docs/RESULTS.md
git commit -m "n-tuple (#10): throughput and 1-hour run"
```

---

### Task 6: The long run and acceptance

**Files:**
- Modify: `docs/RESULTS.md` (the same section), `README.md` (one line in the results list)

- [ ] **Step 1: Run 8 hours detached** with the defaults:

```bash
setsid nohup uv run python -m block_puzzle_ai.ntuple train --out runs/nt1 --hours 8 > runs/nt1.log 2>&1 &
```

Record the start time in the ledger. If it is interrupted, resume with `uv run python -m block_puzzle_ai.ntuple train --out runs/nt1 --hours 8 --resume`.

- [ ] **Step 2: Run the acceptance evaluations** on `runs/nt1/best.ntw`, writing JSON files into `runs/nt1_eval/`:
  - `seed777_200k.json`: 64 games, seed 777, `--max-moves 200000`
  - `seed777_10M.json`: the same, with `--max-moves 10000000`
  - Bootstrap: `scripts/bootstrap_mean.py runs/v2_eval/seed777_v2.json runs/nt1_eval/seed777_200k.json`
  - Death rate vs `rl_v2`: `ratio = (deaths_nt / moves_nt) / (deaths_v2 / moves_v2)`, with the Poisson CI as in `scripts/ab_compare.py`

- [ ] **Step 3: Verdict.**
  - **Pass:** the 200k-cap mean beats 6,507,274, and the CI lower bound is > 0.
  - Otherwise report the plateau, with the learning curve and its last-hour slope.
  - Write the tables, the verdict and the next methods, prioritised from the result, into RESULTS. Add one line to README.

- [ ] **Step 4: Commit**

```bash
git add docs/RESULTS.md README.md
git commit -m "n-tuple (#10): 8-hour run and acceptance evaluation"
```

- [ ] **Step 5: #10 comment.** Draft a comment with the results and ask the user before posting it.
