# n-tuple TD learner on the C++ engine (#10 technique 7)

Date: 2026-11-03. Status: approved in chat, awaiting review of this spec.

## Goal

The program goal is a **mean score of 100M**.

- Mean ≈ points per move × moves per game over. At about 117 points per move, 100M needs about **850k moves per game over**.
- That is 15× `rl_v2` 1-ply (57k) and 7× `rl_v2` + search (117k).
- It will take several methods stacked:
  1. a better value function, which is this spec
  2. deep expectimax search with those values in C++, which is the next spec. With 2048 n-tuples, 1-ply → 6-ply went from 413k to 625k.
  3. learner upgrades: temporal-coherence learning rates, optimistic TD, bigger tuples

This spec builds the foundation: an n-tuple value function trained by TD(0) self-play entirely on the CPU, on top of the engine from #11.

**Acceptance bar:** the n-tuple agent, playing 1-ply without search, beats `rl_v2` 1-ply on mean score.

## Background that shapes the design

- **The engine** (`cpp/engine.hpp`) applies every legal move of a position at 7.9M moves/s on 24 threads, and it replays `game.py` exactly.
- **Every piece group is closed under the 8 symmetries of the square:** each group holds all rotations and mirror images of its shape. So a symmetric value function loses nothing, and tables can be shared across symmetries.
- **Cells are binary:** a tuple of k cells has 2ᵏ entries, so 16-cell tuples fit in 64k-entry tables. In 2048, cells take 16 values.
- **The #10 death probe:** game overs come from spawn luck over the last ~3 moves, from positions that look ordinary. What remains to improve is the per-move risk of ordinary positions. An undiscounted value (γ = 1) prices that risk at its full cost, the whole expected future score. γ = 0.997 prices it at about 333 moves of points.
- **The 200k-move cap can't measure 100M:** a capped game tops out near 23M. Results therefore report **moves per game over**, which the cap doesn't saturate, and the mean with a 10M-move cap.

## Scope

In scope:
- the n-tuple value function, the TD(0) trainer and the 1-ply evaluator, in C++
- bindings, plus a Python module with a CLI for training and evaluation
- checkpoints and resume
- tests
- a long run, with results in `docs/RESULTS.md` and a comment on #10

Out of scope, as later specs:
- expectimax search
- temporal-coherence or optimistic TD
- multi-stage tables
- the viewer agent
- a `--agent *.ntw` option in `evaluate.py`

## Files

| File | Purpose |
|---|---|
| `cpp/ntuple.hpp` | board symmetries, `NTuple` (tables, value, TD update), `Trainer`, `evaluate` |
| `cpp/bindings.cpp` | adds `NTuple`, `Trainer` and `evaluate_ntuple` to `_engine` |
| `src/block_puzzle_ai/ntuple.py` | tuple shapes and orbit selection, config, checkpoint I/O, CLI (`train`, `evaluate`) |
| `tests/test_ntuple.py` | tests |

## Value function

`V(a)` is the value of an after-state `a`. An after-state has three parts:
- the board after the placement and its line clears
- the 2 pieces left in the tray
- the cooldown flag, i.e. whether the next spawn is restricted to groups 0–7, computed from the spawn history at decision time

The score phase only matters below 12,000 points, so it is ignored.

V(a) is the sum of three kinds of lookups.

1. **Board tuples.**
   - A *tuple* is a set of up to 20 absolute cells with a float table of 2ᵏ entries. Its index is the tuple's cells in ascending cell order, read as bits. In C++ that is `pext` on both 64-bit halves of the board, then a shift to join them.
   - For each tuple `T` and each of the 8 symmetries σ, add `w_T[index_T(σ(board))]`. That makes 8 lookups per tuple.
   - The symmetries are identity, rot90, rot180, rot270, horizontal flip, vertical flip, transpose and anti-transpose. They are built from three generators (horizontal flip, vertical flip, transpose) using row-wise lookup tables.
   - **Tuple selection happens in Python** (`ntuple.py`):
     - A *shape* is a cell pattern. Take every translation of it that fits on the board.
     - Group these cell sets into orbits under the 8 symmetries.
     - Keep one representative per orbit, the one whose cell list is lexicographically smallest. The representatives are the tuples passed to C++.
   - **Default shapes:**
     - `window4`, the 4×4 square: 49 translations in 10 orbits
     - `line`, the 1×10 row: 20 lines (rows and columns) in 5 orbits
     - That gives 15 tables and 655,360 + 5,120 weights.
2. **Piece fit.**
   - For each of the 12 groups `g`, count the legal placements of all its variants on the after-state board.
   - Put the count in one of 8 buckets: 0, 1, 2, 3–4, 5–8, 9–16, 17–32, 33+.
   - Look up `w_fit[cooldown][g][bucket]`. That is 12 lookups.
   - The count is invariant under the symmetries, because each group is closed under them.
3. **Remaining pieces.** For each of the 2 remaining pieces `p`, look up `w_rem[group(p)][bucket(legal placements of p)]`. That is 2 lookups. The pair (board, p) transformed together gives the same count, so this is invariant too.

All weights live in one float32 array: the tuple tables in order, then `w_fit` (2×12×8), then `w_rem` (12×8). Python sees the array as a NumPy view without copying.

**Invariance:** transforming the board by σ and mapping each remaining piece to its σ-image leaves V unchanged. The transform only permutes the 8 symmetric terms of each tuple sum. This is tested.

## TD(0) after-state training

Each thread runs a loop of greedy self-play, following Szubert & Jaśkowski for 2048:

```
s ← reset(); a, r ← argmax over legal moves of r(m) + V(a(m))
loop:
    s ← step(s, chosen move)              # spawns the next piece
    if s has no legal move:  target ← 0;  restart (pool or empty board)
    else:                    a', r' ← argmax r + V;  target ← r' + γ·V(a')
    update(a, target);  a ← a'
```

- **The update** is `δ = target − V(a)`. Each weight looked up for `a` gets `+= α·δ / n_lookups`, where `n_lookups = 8·n_tuples + 14`.
- **Defaults:**
  - **γ = 1**, so V is the expected remaining score.
  - **α = 0.1**, constant. It can be changed when a run resumes, which allows manual step-downs.
  - The weights start at 0.
  - There is no ε: the spawns provide the exploration.
- **Restarts** (as the `rl_v2` run's `RestartPool`):
  - Each thread snapshots its game state every 50 moves into a ring of 10.
  - When a game ends, the ring's snapshots from the last 500 moves before the game over join that thread's pool (4,096 states).
  - A finished game restarts from a random pool state with probability `restart_frac`, default 0.5. Otherwise it restarts from an empty board.
- **RNG:** each thread uses its own `Xoshiro256`, seeded as in `play_random`: thread i takes the (i + 1)-th output of a splitmix64 stream started at `seed`.
- **Threads:** every thread reads and writes the shared weights without locks, through `std::atomic_ref<float>` with relaxed ordering. On x86 that compiles to plain loads and stores and avoids undefined behaviour.
- **`Trainer.run(moves)`** plays that many moves in total, split across the threads. It releases the GIL and returns statistics:
  - `moves`, `seconds`
  - `games`, the number of finished games
  - `deaths`
  - `mean_score` and `mean_moves` of the finished games that started from an empty board
  - `restarts`
  - `value_mean`, the mean V of the chosen after-states

## Evaluation

`evaluate_ntuple(tables, ntuple, games, seed, max_moves, threads)` plays games 1-ply (argmax r + V) on all threads.
- Game i uses a `Xoshiro256` seeded with the (i + 1)-th output of a splitmix64 stream started at `seed`.
- It returns one `(score, moves, lines, truncated)` per game, in game order. Results are deterministic for a given seed and do not depend on the thread count.
- **Python reuses `evaluate.GameResult` and `evaluate.summarize`,** and `ntuple evaluate --json` prints the same JSON as `evaluate.py`. So `scripts/bootstrap_mean.py` and `scripts/ab_compare.py` compare it directly with the `rl_v2` runs in `runs/v2_eval/`.
- **The seeds are not shared with `evaluate.py`:** "seed 777" means the same protocol size, but different random games from `BatchGame` seed 777. That comparison is statistical either way, because games diverge after their first moves.

## Python module and CLI (`block_puzzle_ai.ntuple`)

- **`shapes_to_tuples(names: list[str]) -> list[list[int]]`** turns shape names into orbit representatives. The shapes are `window4`, `line`, and `rect<h>x<w>` with h·w ≤ 20.
- **`@dataclass NTupleConfig`:** `shapes=["window4", "line"]`, `gamma=1.0`, `alpha=0.1`, `restart_frac=0.5`, `threads=24`, `seed=0`.
- **Checkpoint `*.ntw`:**
  - the 8-byte magic `BPANTW01`
  - a uint32 JSON length
  - the JSON: the config, the tuple cell lists, `moves_trained` and `elapsed_min`
  - the raw float32 weights
- **`train`:** `python -m block_puzzle_ai.ntuple train --out runs/<name> (--hours H | --total-moves M) [--resume] [--alpha A] [--eval-minutes 15] [--eval-games 256] [--eval-max-moves 200000]`
  - Training runs in chunks of 1 minute, measured by moves.
  - After each chunk it appends a `{"type": "train", ...}` record to `runs/<name>/metrics.jsonl`, with `elapsed_min` and `moves`.
  - Every `--eval-minutes` it runs an eval and appends a `{"type": "eval", "eval_mean", "eval_median", "eval_max", "eval_moves_per_death", ...}` record. The keys match `train.py`, so the viewer's training chart shows the run.
  - It saves `latest.ntw` after every eval. It saves `best.ntw` when `eval_moves_per_death` improves, because the cap doesn't saturate that metric.
  - `--resume` continues from `latest.ntw`, with the counters, config and weights. The restart pool starts empty.
- **`evaluate`:** `python -m block_puzzle_ai.ntuple evaluate --weights W --games 64 --seed 777 --max-moves 200000 [--threads 24] [--json]`.

## Tests (`tests/test_ntuple.py`)

1. **Orbits:** `shapes_to_tuples(["window4"])` gives 10 tuples, `["line"]` gives 5, and every translation's cell set is reachable from exactly one representative under the 8 symmetries.
2. **Symmetry invariance:**
   - The weights are seeded randomly.
   - Use 200 random after-states: 100 drawn from random play and 100 random boards.
   - For each, transform the board by each σ and map the remaining pieces to their σ-images. V must be equal within a relative 1e-5.
3. **Brute force:** a pure-Python evaluation over the weights array, the tuple cells, the 8 Python-implemented symmetries and the fit buckets must equal C++ `value()` within a relative 1e-5.
4. **TD update:**
   - `update(a, target)` moves V(a) strictly toward the target.
   - With α = 1, a single tuple whose 8 symmetric indices are distinct, and two remaining pieces from different groups, V(a) lands exactly on the target. All 22 looked-up weights are then distinct.
5. **Training works:** training for 3M moves on 4 threads gives a 1-ply evaluation over 64 games with more than 150 mean moves per game. Random play gets 24 and the tuned heuristic 279.
6. **Determinism:** `evaluate_ntuple` gives identical results for the same seed with 1 and 4 threads.
7. **Checkpoints:** save, then load, gives identical weights and identical `value()` results. `--resume` continues the counters.
8. **Errors:** each of these raises `ValueError`:
   - a tuple with more than 20 cells
   - a cell outside 0–99
   - duplicate cells in a tuple
   - a weights array of the wrong length
   - `threads < 1`
   - a checkpoint with a bad magic

## Acceptance

1. `uv run pytest` passes.
2. **Throughput:** training moves/s on 24 threads are measured and recorded. The initial target is ≥ 200k moves/s; revise it after the first measurement, giving the reason.
3. **A 1-hour run** is a checkpoint: its moves per game over are reported before the long run starts.
4. **The long run** (about 8 h): `best.ntw` is evaluated on 64 games (seed 777) twice, with a 200k cap and with a 10M cap.
   - **Pass:** the 200k-cap mean beats `rl_v2` 1-ply's 6,507,274, and the bootstrap 95% CI of the difference excludes 0.
   - Also report moves per game over and the death-rate ratio vs `rl_v2` (`ab_compare.py` style).
5. **If the bar is not met,** report the plateau and the learning curve honestly. The learner is still delivered, and the next methods (search, temporal-coherence learning, bigger tuples) are prioritised from that result.
6. Results go into `docs/RESULTS.md`, with a comment on #10.
