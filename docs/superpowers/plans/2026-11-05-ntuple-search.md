# n-tuple Expectimax Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the highest mean score that expectimax search over the n-tuple value can reach within an 8-hour, 64-game evaluation, aiming at the program goal of 100M.

**Architecture:**
- `cpp/ntuple.hpp` gets a split value: a per-board `BoardEval` and a cheap rest. It stays bit-identical to today's value.
- A new `cpp/search.hpp` holds:
  - the recursive search, with the next-spawn expectation reusing the known pieces' board evaluations
  - tier selection
  - `evaluate_search`, which streams finished games through a callback
- `cpp/bindings.cpp` exposes the search.
- A new `src/block_puzzle_ai/ntuple_search.py` owns tier parsing, the resumable games file and the summary. `ntuple evaluate` dispatches to it.

**Tech Stack:** C++20, nanobind (`nb::gil_scoped_acquire` in the callback), pytest, uv.

**Spec:** `docs/superpowers/specs/2026-11-05-ntuple-search-design.md`. Read it first, especially "Search", which holds the definitions every task implements.

## Global Constraints

- **Search definitions** are those in the spec's "Search" section:
  - `Q1`, `best1` (0 with no legal move), the child state, `P(x)` from the *parent's* score and history, `E` and `B`
  - top k by Q1 with ties in engine order; `k = 0` means all moves
  - the sum over x in ascending piece id, skipping `P(x) = 0`
- **Child state** of move `m` with spawn `x`:
  - pending `(rem0, rem1, x)`, where `rem0, rem1` are the after-state's `remaining` in order
  - score `parent.score + r(m)`
  - history `[group(x), parent.history[0], parent.history[1]]`
- **Remaining pieces** in a child after-state, as `best_move` computes them: slot 0 → `(p[1], p[2])`, slot 1 → `(p[0], p[2])`, slot 2 → `(p[0], p[1])`.
- **Tiers:**
  - The search uses the *last* tier whose `below` > the root's best Q1. With none, it plays exactly `best_move`.
  - Valid tiers have strictly decreasing `below`, non-decreasing `len(widths)`, non-empty widths, and widths ≥ 0. Anything else raises `std::invalid_argument` (`ValueError` in Python).
- **Determinism:**
  - The search draws no random numbers.
  - Game `g` uses `Xoshiro256(stream_seeds(seed, games)[g])`.
  - Results do not depend on the thread count.
- **Unchanged:** `evaluate`, `best_move`, `Trainer` and `update`. Values stay bit-identical; the build already uses `-ffp-contract=off` and no fast-math.
- **Tests:** each test runs in under 60 s. The full suite is `uv run pytest -q`, and `uv run` rebuilds the extension after `cpp/**` changes.
- **Commits:** messages start `n-tuple search (#10): …` and end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Shared machine:** before any run longer than 10 minutes, check `pgrep -c tfeval`. If tfeval is running, use `--threads 12` and say so in the ledger, or wait. The acceptance run (Task 7) only starts when no tfeval is running.

## Review Focus

1. **The checkpoint's γ (0.999) must reach the search.** If the CLI path dropped it, every value would be on the wrong scale. Test: `test_run_search_eval_uses_checkpoint_gamma` (Task 4).
2. **Games that hit the move cap** must truncate exactly like `evaluate`: a game is truncated when a legal move exists and `moves >= max_moves`. Test: `test_no_tiers_matches_evaluate` uses a cap small enough that some games hit it (Task 3).
3. **A process killed in the middle of writing a line** leaves a partial last line. Resuming must drop it and finish, not crash. Test: `test_resume_drops_partial_last_line` (Task 4).
4. **Resuming with different weights, seed, cap or tiers** must stop with an error, not mix results. Test: `test_resume_rejects_other_settings` (Task 4).
5. **Positions where search is degenerate:** k larger than the number of legal moves, a root with one legal move, near-death states whose children often have no legal move (value 0), and a root with no legal move. Test: `test_search_edge_cases` (Task 2).

---

### Task 1: Split value, bit-identical

**Files:**
- Modify: `cpp/ntuple.hpp` (`NTuple::value`, `NTuple::lookups`)
- Create: `tests/data/ntuple_values_golden.json`, `scripts/make_ntuple_golden.py`
- Test: `tests/test_ntuple_search.py`

**Interfaces:**
- Produces, in `cpp/ntuple.hpp`:
  - `struct BoardEval { double tuples = 0; int count[MAX_PIECES] = {}; int group_count[MAX_GROUPS] = {}; };`
  - `BoardEval NTuple::board_eval(Board b) const`
  - `double NTuple::value(const BoardEval& e, bool cooldown, int rem0, int rem1) const`
  - `double NTuple::value(const Afterstate& a) const`, now `value(board_eval(a.board), a.cooldown, a.remaining[0], a.remaining[1])`

- [ ] **Step 1: Golden values from the current build.** Write `scripts/make_ntuple_golden.py`:
  - It builds the `random_model` of `tests/test_ntuple.py`: tuples `shapes_to_tuples(["window4", "line"])`, weights `np.random.default_rng(0).normal(size=n).astype(np.float32)`.
  - It takes the 300 after-states of `sample_afterstates(300, 7)`, imported from `tests/test_ntuple.py` (the script puts `tests/` on `sys.path`). For each, it records the board and remaining pair as given, with **both** cooldown flags.
  - It writes `[{"board": int, "remaining": [r0, r1], "cooldown": bool, "value": float.hex(v)}]` to `tests/data/ntuple_values_golden.json`.

  Run it **before** touching `ntuple.hpp`: `uv run python scripts/make_ntuple_golden.py`. Expected: 600 entries.
- [ ] **Step 2: Test** `test_split_value_matches_golden`: for every entry, `value(model, board, remaining, cooldown) == float.fromhex(entry["value"])`, compared with `==`, not approx. Run `uv run pytest tests/test_ntuple_search.py -q`. Expected: PASS on the old code.
- [ ] **Step 3: Implement the split** in `cpp/ntuple.hpp`:
  - `board_eval` sums the tuple weights into `tuples`, starting from 0 in today's order (mask-major, then the 8 symmetry boards), and fills `count` and `group_count` as `lookups` does.
  - `value(e, …)` starts from `e.tuples`, adds the 12 fit weights in group order, then `rem0`'s and `rem1`'s weights.
  - `update` keeps using `lookups`.
- [ ] **Step 4: Verify.** `uv run pytest -q`. Expected: all pass, including the golden test and every existing `test_ntuple.py` and `test_engine.py` test.
- [ ] **Step 5: Commit** with the message `n-tuple search (#10): split value into a per-board part`, including the script, the golden file and the test.

---

### Task 2: Search core

**Files:**
- Create: `cpp/search.hpp`
- Modify: `cpp/bindings.cpp` (include `search.hpp`; two new functions)
- Test: `tests/test_ntuple_search.py`

**Interfaces:**
- Consumes: `NTuple::board_eval` and `NTuple::value(const BoardEval&, …)` from Task 1, plus `engine.hpp`'s `legal_positions`, `apply`, `spawn_distribution`, `in_cooldown` and `for_each_bit`.
- Produces, in `cpp/search.hpp`, namespace `bpa`:
  - `struct Tier { double below; std::vector<int> widths; };`
  - `void check_tiers(const std::vector<Tier>&);` throws `std::invalid_argument`.
  - `struct Scored { int slot, pos, points; double q1, score; };`
  - `std::vector<Scored> search_scores(const NTuple&, const State&, float gamma, const std::vector<int>& widths);`
    - Returns the kept root moves (the top `widths[0]` by Q1) in engine order, each with its Q1 and its search score `E(s, m, widths[1:])`.
    - With no legal move it returns an empty vector.
  - `bool search_move(const NTuple&, const State&, float gamma, const std::vector<Tier>&, Choice& out, int& tier, bool& changed);`
    - Returns false with no legal move.
    - `tier` is the index used, or -1 for 1-ply. `changed` is true when the move differs from `best_move`'s.
    - `out.q` is the search score, or Q1 at -1. `out.after` and `out.v` are those of the chosen move.
- Produces, in Python via `_engine`:
  - `search_scores(model, board_lo, board_hi, pending, score, history, gamma, widths) -> list[tuple[slot, pos, points, q1, score]]`
  - `search_move(model, board_lo, board_hi, pending, score, history, gamma, tiers) -> tuple[slot, pos, q, tier, changed] | None`
  - `tiers` is a `list[tuple[float, list[int]]]`, with `float("inf")` allowed. Build the `State` with the existing `read_history` and `check_score` helpers.

- [ ] **Step 1: The Python reference** in `tests/test_ntuple_search.py`. It is a direct transcription of the spec on `CppGame` and `ntuple.value`, with no shortcuts:

```python
HARD = set(HARD_GROUPS)

def cooldown(history):
    return any(h >= 0 and h in HARD for h in history[:3])

def ref_q1(model, gamma, board, pending, score, history):
    """[(slot, pos, points, after_board, remaining, q1)] in engine order."""
    g = CppGame(0)
    g.set_state(board, [all_pieces()[p] for p in pending], score, history)
    out = []
    for move, after, points, _ in g.afterstates():
        rem = [pending[i] for i in range(3) if i != move.slot]
        q1 = points + gamma * value(model, after, rem, cooldown(history))
        out.append((move.slot, move.row * 10 + move.col, points, after, rem, q1))
    return out

def ref_E(model, gamma, parent_score, parent_history, after, rem, points, widths):
    g = CppGame(0)
    g.set_state(0, [all_pieces()[p] for p in (*rem, 0)], parent_score, parent_history)
    total = 0.0
    for x, px in enumerate(g.spawn_distribution()):
        if px == 0:
            continue
        hist = [all_pieces()[x].group, *parent_history[:2]]
        total += px * ref_B(model, gamma, after, [*rem, x], parent_score + points, hist, widths)
    return points + gamma * total

def ref_B(model, gamma, board, pending, score, history, widths):
    q = ref_q1(model, gamma, board, pending, score, history)
    if not q:
        return 0.0
    if not widths:
        return max(m[5] for m in q)
    kept = sorted(q, key=lambda m: -m[5])[: widths[0] or None]  # stable: ties keep engine order
    return max(ref_E(model, gamma, score, history, m[3], m[4], m[2], widths[1:]) for m in kept)
```

  `spawn_distribution` depends only on score and history, so the board and pending pieces passed to `set_state` there are placeholders.
- [ ] **Step 2: Tests** (they fail because `_engine.search_scores` does not exist yet):
  - `test_depth2_matches_reference`:
    - States: 10 mid-game states from random play with `CppGame(seed)`, using seeds 1–10 and 30–300 random moves. At least 2 of them must have a hard group in `history[:3]`. Assert that.
    - The model is the random model of Task 1, with γ 0.999.
    - For k in (1, 4, 0): for each kept move, `search_scores(…, [k])` equals the reference's `ref_E(…, [])` for the same move (`rel=1e-9`). The set of kept moves equals the reference's top k.
    - For k = 0, use only the first 3 states, to stay under 60 s.
  - `test_depth3_matches_reference`: one state from the same sampler, widths `[2, 2]`, against `ref_E(…, [2])` (`rel=1e-9`).
  - `test_search_move_picks_max_and_tiers`:
    - `search_move` with `[(inf, [4])]` plays the kept move with the highest score, first in engine order on ties, and reports tier 0.
    - With `[(inf, [4]), (b, [2, 2])]`, where `b` is above the root's best Q1, it reports tier 1.
    - With `[(b2, [4])]`, where `b2` is below the best Q1, it reports tier -1, and the move equals `best_move`'s, the Q1 argmax from `ref_q1`.
  - `test_search_edge_cases` uses a helper `near_death_states(seed)`. It plays random legal moves with `CppGame(seed)` and keeps the last 3 states before each game over as `(board, pending ids, score, history)`, plus the final state that has no legal move. The test uses seeds 1–20:
    - k = 100 gives the same kept moves and scores as k = 0 on one state.
    - A state with exactly one legal move gives one kept move, whose score equals the reference. Assert that the seeds yield one.
    - On the near-death states, the depth-2 scores (k = 4) equal the reference. Assert that the reference met at least one child with no legal move (`ref_B` returned 0 from an empty `ref_q1`). A state where *every* spawn ends the game cannot exist, because any free cell fits the single-cell piece of group 0.
    - The final state returns `[]` from `search_scores` and `None` from `search_move`.
  - `test_tiers_rejected`: `ValueError` for increasing `below`, decreasing depth, empty widths and a negative width.
- [ ] **Step 3: Run** `uv run pytest tests/test_ntuple_search.py -q`. Expected: FAIL (`AttributeError: … search_scores`).
- [ ] **Step 4: Implement `cpp/search.hpp`.**
  - Use a recursive `expectation(nt, Board after, int rem0, int rem1, const State& parent, int points, float gamma, span widths)`, built as the spec's "Expectation over the spawn" describes:
    - Compute the placements and `board_eval` of `rem0` and `rem1` on `after` once.
    - For each x with `P(x) > 0`, the cooldown flag is `in_cooldown(child history)`. Known-piece placements cost one `value(e, cd, …)` each, and `x`'s placements get a fresh `board_eval`.
    - At the last level take the max Q1. Otherwise keep the top k by Q1 (stable) and recurse.
  - The root computes Q1 exactly like `best_move`, so tier -1 is bit-identical to it.
  - Then add the two bindings.
- [ ] **Step 5: Verify.** `uv run pytest -q`. Expected: all pass, each test under 60 s (check with `--durations=5`).
- [ ] **Step 6: Commit** with the message `n-tuple search (#10): recursive expectimax with tiers`.

---

### Task 3: `evaluate_search` with streamed games

**Files:**
- Modify: `cpp/search.hpp`, `cpp/bindings.cpp`
- Test: `tests/test_ntuple_search.py`

**Interfaces:**
- Consumes: `search_move`, `check_tiers` (Task 2), `stream_seeds`, `parallel`, `reset`, `step`, `GameOutcome` (`ntuple.hpp`).
- Produces, in C++:
  - `struct GameSearch { GameOutcome outcome; std::vector<uint64_t> searched, changed; double seconds = 0; std::vector<double> last_q1; };`
    - `searched` and `changed` have one entry per tier.
    - `last_q1` holds the best Q1 of the last ≤ 20 moves played, oldest first.
  - `std::vector<std::pair<int, GameSearch>> evaluate_search(const NTuple&, int games, uint64_t seed, int64_t max_moves, int threads, float gamma, const std::vector<Tier>&, const std::vector<int>& skip, const std::function<void(int, const GameSearch&)>& on_game);`
    - Returns the games not in `skip`, in game order.
    - `on_game` runs once per finished game under a mutex.
    - An exception in any worker or in `on_game` sets an atomic stop flag. Every worker checks it before each move, and `parallel` raises it.
- Produces, in Python: `_engine.evaluate_ntuple_search(model, games, seed, max_moves, threads, gamma, tiers, skip, on_game)`.
  - The return value is a `list[tuple[game, score, moves, lines, truncated, searched, changed, seconds, last_q1]]`.
  - `on_game` is a Python callable `(game, score, moves, lines, truncated, searched, changed, seconds, last_q1)`, or None. It is called with the GIL acquired. The GIL is released around the C++ call.
  - The game loop is `evaluate`'s, including its truncation rule, with `search_move` instead of `best_move`.

- [ ] **Step 1: Tests:**
  - `test_no_tiers_matches_evaluate`: random model, 24 games, seed 5, `max_moves` 300, 8 threads, tiers `[]`. Assert `(score, moves, lines, truncated)` per game equals `_engine.evaluate_ntuple(…)`, and that at least one game has `truncated` True.
  - `test_search_threads_independent`: tiers `[(inf, [2])]`, 8 games, `max_moves` 200, γ 0.999. The per-game tuples without `seconds` are equal for 1 and 8 threads.
  - `test_skip_and_callback`:
    - Running 6 games with `skip=[1, 4]` returns games `[0, 2, 3, 5]`, identical to the same games of a full run.
    - The callback is called once per returned game, with the same values.
  - `test_callback_exception_stops`: a callback that raises `RuntimeError` on its first call makes `evaluate_ntuple_search` raise `RuntimeError`, even with 8 games of `max_moves` 10⁹. This shows the stop flag ends the other games: the test finishes in seconds, not hours.
  - `test_last_q1`: `last_q1` records the best Q1 of each move *played* (stepped). With `max_moves` 5, every game has `len(last_q1) == 5`. With `max_moves` 300, every game of at least 20 moves has length 20. All values are finite.
- [ ] **Step 2: Run** them. Expected: FAIL (missing `evaluate_ntuple_search`).
- [ ] **Step 3: Implement** `evaluate_search` and the binding. The binding converts `GameSearch` to the tuple above. Its C++ `on_game` lambda takes `nb::gil_scoped_acquire` before calling Python.
- [ ] **Step 4: Verify.** `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** with the message `n-tuple search (#10): evaluate_search with streamed games`.

---

### Task 4: Python module, CLI, resume and weights

**Files:**
- Create: `src/block_puzzle_ai/ntuple_search.py`, `models/ntuple_v1.ntw` (a copy of `runs/nt1/best.ntw`)
- Modify: `src/block_puzzle_ai/ntuple.py` (the `evaluate` subparser options and `evaluate_cmd` dispatch)
- Test: `tests/test_ntuple_search.py`

**Interfaces:**
- Consumes: `_engine.evaluate_ntuple_search` (Task 3), plus `ntuple.load_model`, `evaluate.GameResult` and `evaluate.summarize`.
- Produces, in `ntuple_search.py`:
  - `Tier = tuple[float, list[int]]`
  - `tiers_from_args(search_k: int | None, search_below: float | None, deep_below: float | None, deep_k: list[int] | None) -> list[Tier]`
    - `--search-k K` alone gives `[(inf, [K])]`.
    - `--search-below G` gives `[(G, [K or 8])]`.
    - `--deep-below G2 --deep-k K1,K2` appends `(G2, [K1, K2])`.
    - `--deep-k` without `--deep-below`, or the reverse, raises `SystemExit`, and so does a `deep_k` without exactly 2 values. Invalid orders raise `SystemExit` with the engine's message.
  - `run_search_eval(weights: str | Path, games: int, seed: int, max_moves: int, threads: int, tiers: list[Tier], games_file: str | Path | None) -> dict`
    - γ comes from the checkpoint (`cfg.gamma`).
    - **Games file:**
      - Line 1 is a header: `{"weights": str(weights), "games", "seed", "max_moves", "gamma", "tiers": [[below or null for inf, widths], …]}`.
      - Each later line holds one game: `{"game", "score", "moves", "lines", "truncated", "searched", "changed", "seconds", "last_q1"}`, written by the callback and flushed straight away.
      - **On resume**, a header that differs raises `SystemExit("…games file … other settings …")`. A last line that is not valid JSON is dropped (the file is rewritten without it), and finished games are passed as `skip`.
    - **Return:** `summarize(results)` over all games in game order, plus:
      - `"agent": f"ntuple:{weights}"`
      - `"seconds"`: the wall time of this call, rounded to 0.1
      - `"search": {"tiers", "searched", "changed", "us_per_move"}`, where `us_per_move` = Σ game seconds · 10⁶ / Σ moves
      - `"results": [asdict(GameResult)…]`
- CLI (`ntuple evaluate`): adds `--search-k`, `--search-below`, `--deep-below`, `--deep-k` (comma-separated) and `--games-file`. With any of them, `evaluate_cmd` calls `run_search_eval`. Without them, behaviour is unchanged. `--json` prints the dict; text mode prints the keys as today, plus a `search` line.

- [ ] **Step 1: Tests**, using a checkpoint saved with `save_checkpoint` from the random model with γ 0.999 into `tmp_path`:
  - `test_tiers_from_args`: covers the 4 accepted forms and the 4 rejected ones listed above.
  - `test_run_search_eval_uses_checkpoint_gamma`:
    - Use the random model with weights scaled by 10⁴, so γ changes decisions.
    - With tiers `[(inf, [2])]`, 4 games and a cap of 200, the per-game scores equal `_engine.evaluate_ntuple_search(…, gamma=0.999, …)` and differ from a run at γ 1.0 for at least one game.
    - The header's `gamma` is 0.999.
  - `test_resume_matches_full_run`:
    1. Run 6 games into file A.
    2. Write header plus the first 3 game lines into file B, and run again on B.
    3. B now holds 6 games, and the summary's per-game `results` equal A's.
  - `test_resume_drops_partial_last_line`: as above, but B also gets a cut-off fourth line (`'{"game": 3, "sco'`). The rerun completes with the same results as A.
  - `test_resume_rejects_other_settings`: reusing A with a different seed, cap or tiers, or another weights path, raises `SystemExit`.
  - `test_summary_works_with_scripts`: the `--json` output saved to a file, passed twice to `scripts/bootstrap_mean.py`, exits 0 (run with `subprocess`).
- [ ] **Step 2: Run** them. Expected: FAIL (`ModuleNotFoundError: block_puzzle_ai.ntuple_search`).
- [ ] **Step 3: Implement** `ntuple_search.py` and the CLI wiring.
- [ ] **Step 4: Weights.** Run `cp runs/nt1/best.ntw models/ntuple_v1.ntw`, then `uv run python -m block_puzzle_ai.ntuple evaluate --weights models/ntuple_v1.ntw --games 64 --seed 777 --max-moves 200000`. Expected: mean 6,192,662, exactly as in `runs/nt1_eval/seed777_200k.json`.
- [ ] **Step 5: Verify.** `uv run pytest -q`. Expected: all pass.
- [ ] **Step 6: Commit** with the message `n-tuple search (#10): CLI, resumable games file, models/ntuple_v1.ntw`.

---

### Task 5: Depth-2 experiments

**Files:**
- Create: `runs/nts_d2/*` (not committed), plus a section in `docs/RESULTS.md`

- [ ] **Step 1: Cost probe.** 8 games, seed 2024, cap 20,000, `--search-k` 2, 4 and 8, each with `--games-file runs/nts_d2/probe_k{K}.jsonl --json`. Record `us_per_move` against the 1-ply rate of about 63 µs. If k = 8 costs more than 300× a 1-ply move, profile the expectation before going on: about 120× was expected.
- [ ] **Step 2: Gate calibration.** 8 games, seed 2024, cap 20,000, `--search-below G --search-k 8`, with G in {60000, 70000, 75000, 78000}. Pick the two G whose searched fraction (`searched / moves`) is nearest 0.1% and 1%.
- [ ] **Step 3: Arms.** 64 games, seed 2024, cap 200,000, each with its own `--games-file runs/nts_d2/<arm>.jsonl --json > runs/nts_d2/<arm>.json`:
  - `1ply` (no search options)
  - `k2`, `k4` and `k8` (every move)
  - `gate_lo` and `gate_hi` (k = 8, the two G from step 2)

  Expect up to ~70 min for `k8`. Check `pgrep -c tfeval` first.
- [ ] **Step 4: Compare.**
  - `uv run python scripts/bootstrap_mean.py runs/nts_d2/1ply.json runs/nts_d2/<arm>.json` for each arm.
  - For death rates, write the arms' summaries (without `results`, with `arm` set) into `runs/nts_d2/compare.jsonl`, with `1ply` first, and run `scripts/ab_compare.py`.
  - Answer the spec's question: does every-move search raise points per move?
- [ ] **Step 5: Write up.** In `docs/RESULTS.md`, under a new `## n-tuple search (ticket #10)`, add a "Depth 2" subsection with:
  - the cost table (k → µs per move, × 1-ply)
  - the arms table (mean, median, moves per game over, points per move, searched and changed fraction, death rate with CI)
  - two or three bullets of reading

  Commit with the message `n-tuple search (#10): depth-2 experiments`.

---

### Task 6: Depth-3 tier

**Files:**
- Create: `runs/nts_d3/*` (not committed); `docs/RESULTS.md` (the same section)

- [ ] **Step 1: Danger threshold.** From the `k`-arm games files of Task 5 with the best mean, take each game over's `last_q1` and the best Q1 at each of the last 1, 3 and 10 moves. Use the medians and 10th percentiles at those positions to choose 2–3 thresholds G₂, and report how often the best Q1 falls below each one per move. The searched fraction of the gate arms gives that rate.
- [ ] **Step 2: Cost probe.** 8 games, cap 20,000, best depth-2 k plus `--deep-below G2 --deep-k K1,K2`, with `(K1, K2)` in {(4, 2), (8, 4)}. Record `us_per_move` and the deep-tier rate.
- [ ] **Step 3: Arms.** 64 games, seed 2024, cap 200,000, for each pair (G₂, widths) whose projected wall time fits. **Projection:** µs per move × 1.06M moves × 64 games / 24 threads ≤ 8 h, and the longest game, about 4M moves × µs per move, ≤ 8 h. Run at most 4 arms.
- [ ] **Step 4: Choose the configuration** with the highest mean whose projection fits, compared against the best depth-2 arm with `bootstrap_mean.py`. Write a "Depth 3" subsection with the threshold table, the arms table and the chosen configuration with its projection. Commit with the message `n-tuple search (#10): depth-3 tier and chosen configuration`.

---

### Task 7: Overnight acceptance

**Files:**
- Create: `runs/nts_accept/*` (not committed)
- Modify: `docs/RESULTS.md` (the same section), `README.md` (one line in the results list)

- [ ] **Step 1: Launch** only when `pgrep -c tfeval` prints 0:

```bash
setsid nohup uv run python -m block_puzzle_ai.ntuple evaluate --weights models/ntuple_v1.ntw --games 64 --seed 777 \
  --max-moves 10000000 <chosen search options> --games-file runs/nts_accept/games.jsonl --json \
  > runs/nts_accept/seed777_10M.json 2> runs/nts_accept/err.log &
```

  Record the start time and options in the ledger. If it is interrupted, rerun the same command: finished games are kept.
- [ ] **Step 2: Compare:**
  - `scripts/bootstrap_mean.py runs/nt1_eval/seed777_10M.json runs/nts_accept/seed777_10M.json` against 1-ply at the same cap
  - `scripts/bootstrap_mean.py runs/v2_eval/seed777_v2_search115_top8.json runs/nts_accept/seed777_10M.json`, against `rl_v2` + search. Note that `rl_v2`'s run had a 200k cap with 12 games alive.
  - the death rate with `ab_compare.py`, as in Task 5
- [ ] **Step 3: Write up.** Add an "Acceptance" subsection to RESULTS, following the spec's Acceptance section:
  - mean, median, max, points per move, death rate, µs per move, wall time and the distance to 100M
  - the verdict: the number reached and the bottleneck (survival, points per move or compute)
  - the next methods, prioritised from the result

  Add one README line to the results list, with the row `n-tuple + search` in the second table. Commit with the message `n-tuple search (#10): overnight acceptance`.
- [ ] **Step 4: #10 comment.** Draft it in the scratchpad and ask the user before posting.
