# Expectimax search for the n-tuple agent (#10, next method 1)

Date: 2026-11-05. Status: approved in chat, awaiting review of this spec.

## Goal

The program goal is a **mean score of 100M**. The user chose to aim straight at it: no pass/fail bar, but the highest mean the search can reach within an **8-hour evaluation budget** (64 games on 24 threads).

Starting point, the 8-hour n-tuple run (`runs/nt1/best.ntw`, γ 0.999), on seed `777`, 64 games:

| | Cap | Mean | Moves / game over | Points / move |
|---|---:|---:|---:|---:|
| n-tuple, 1-ply | 10M | 7,111,924 | 75,645 | 94 |
| `rl_v2`, 1-ply | 200k | 6,507,274 | 57,402 | 117 |
| `rl_v2` + search (gated, top 8) | 200k | 11,158,066 (12 games alive at the cap) | 117,387 | 117 |

100M at 94 points per move needs about 1.06M moves per game: 14× today's survival, or less if points per move rise.

## Background

- **Moves and spawns.** A state holds the board, 3 pending pieces, the score and the last 3 spawned groups. A move places one pending piece (slot, position). Then one new piece is appended, drawn from 28 pieces in 12 groups with an exact distribution: `spawn_distribution(score before the move, history)` in `cpp/engine.hpp`.
- **1-ply today.** `best_move` (`cpp/ntuple.hpp`) plays argmax over the ~76 legal moves of `Q1(m) = r(m) + γ·V(a_m)`, where `a_m` is the after-state (board after placement and clears, the 2 remaining pieces, and the cooldown flag of the next spawn).
- **Costs.** A 1-ply move takes about 63 µs per thread: 64 games, 4.65M moves in 12.3 s on 24 threads. About 0.8 µs per after-state value.
- **`rl_v2`'s search** (`agents.LookaheadAgent`) re-scores the top 8 moves one spawn deeper whenever its best 1-ply value is below a gate. That gave +4.65M (1.7×) and halved the death rate. It ran on only 0.12% of moves, so it mostly bought survival.
- **The points-per-move gap** (94 against 117) is the larger one. A 1-ply value misses rewards that one more move would collect, for example a multi-line clear set up for the next piece. Searching every move puts the follow-up reward into the move choice.

## Scope

In scope:
- Splitting `NTuple::value` into a per-board part and a cheap rest, bit-identical to today's value
- A recursive expectimax search with tiers, in C++
- `evaluate_search`, with search statistics and per-game results streamed to a file, so long evaluations can resume
- Bindings and `ntuple evaluate` CLI options
- `models/ntuple_v1.ntw`
- Tests
- Experiments, the overnight acceptance evaluation, a `docs/RESULTS.md` section, and a draft #10 comment

Out of scope:
- Search during training, and any new weights
- The viewer
- Searching one game on several threads
- Sampled spawns, rollouts, MCTS
- Integration with the torch `BatchGame` agents

## Search

Definitions:
- `Q1(m) = r(m) + γ·V(a_m)`, as in `best_move`.
- `best1(s) = max over legal moves m of Q1(m)`, or **0** if `s` has no legal move.
- Child state `s'(m, x)`, for move `m` in `s` and spawned piece `x`:
  - board: `a_m.board`
  - pending pieces: the two pieces not played, in their slot order, then `x`
  - score: `s.score + r(m)`
  - history: `s.history` with `x`'s group pushed on (most recent first)
- `P(x) = spawn_distribution(s.score, s.history)[x]`, the distribution `step` draws from. Pieces with `P(x) = 0` are skipped.

**Recursive score**, with a list of widths `w = [k₁, …, k_d]` (d spawn levels). The root keeps its top `k₁` moves by Q1 and scores each with `E(s, m, [k₂, …, k_d])`:

```
E(s, m, R)        = r(m) + γ · Σ_x P(x) · B(s'(m, x), R)
B(s, [])          = best1(s)
B(s, [k, R'…])    = max over the top k moves m' of s by Q1 of E(s, m', R'),   or 0 if s has no legal move
```

- Depth 2 (one spawn level) is `w = [k]`: each kept move scores `E(s, m, []) = r(m) + γ·Σ_x P(x)·best1(s'(m, x))`, called Q2.
- Depth 3 is `w = [k₁, k₂]`. Each child keeps its top `k₂` moves by Q1 and scores them by Q2.
- Ties in Q1 keep engine order (slot-major, position ascending).
- The root plays the kept move with the highest search score. Ties go to the first in engine order.
- `k = 0` means all legal moves.
- The sum over `x` runs in ascending piece id.

**Tiers.**
- A tier is `(below, widths)`. The search computes the root's `Q1` for all moves, then uses the **last** tier whose `below` is greater than the best Q1. If no tier matches, it plays the 1-ply move.
- Tiers must have strictly decreasing `below` and non-decreasing depth. `below = +∞` means every move.
- Typical tier lists:
  - `[(+∞, [4])]`: depth 2 on every move
  - `[(G, [8])]`: gated, like `rl_v2`
  - `[(+∞, [4]), (G₂, [8, 4])]`: depth 2 everywhere, and depth 3 below G₂
  - `[]`: plain 1-ply
- Thresholds are absolute values on the weights' scale (V ≈ 82k for γ 0.999). They are set by experiment and are not stored in the weights file.

**No randomness.** The search uses exact expectations and draws no random numbers. Game `g` uses the same spawn stream as `evaluate`: `stream_seeds(seed, games)[g]`. So results do not depend on the thread count, and with no tiers the games are exactly today's 1-ply games.

## Implementation

### Split value (`cpp/ntuple.hpp`)

- `struct BoardEval { double tuples; int count[MAX_PIECES]; int group_count[MAX_GROUPS]; }`
  - `tuples` is the sum, starting from 0 and in today's lookup order (tuple-major, then symmetry), of the 8·n_tuples tuple weights.
  - `count[p]` is the number of legal placements of piece `p` on the board, and `group_count[g]` the sum over each group.
- `BoardEval NTuple::board_eval(Board)`.
- `double NTuple::value(const BoardEval&, bool cooldown, int rem0, int rem1)` adds the fit weights in group order, then `rem0`, then `rem1`.
- `value(const Afterstate&)` becomes `value(board_eval(a.board), a.cooldown, a.remaining[0], a.remaining[1])`. The summation order is unchanged, so values are bit-identical to today's, and training and `best_move` behave exactly as before. `update` keeps its own lookups.

### Expectation over the spawn

For a child of a root move, the board `b` and the known pieces `p0`, `p1` do not depend on `x`:
- **Known pieces.** Placements of `p0` and `p1` give the same boards for every `x`. Their `BoardEval` is computed once per child board.
  - Per `x`, only the cheap rest changes: the after-state of a `p0` placement has remaining pieces `(p1, x)`, and a `p1` placement has `(p0, x)`.
  - The cooldown flag is `in_cooldown(history with x pushed)`, which takes at most two values.
- **Placements of `x`** are new boards, with remaining pieces `(p0, p1)`. About 1,100 per child over all `x`.
- Estimated cost: about 1,200 after-state values per kept root move, against ~2,100 done naively. Depth 2 at k = 4–8 is then about 60–120× a 1-ply move.

### Entry points

- `struct Tier { double below; std::vector<int> widths; }`.
- `bool search_move(const NTuple&, const State&, float gamma, const std::vector<Tier>&, Choice& out, SearchStats& st)`. It plays like `best_move` when no tier applies. `SearchStats` counts moves per tier, moves where the search changed the 1-ply choice, and seconds.
- `search_scores(...)`, for tests, returns every kept root move as `(slot, pos, q1, score)`.
- `evaluate_search(nt, games, seed, max_moves, threads, gamma, tiers, skip, on_game)`:
  - Scheduling, seeds and the cap work as in `evaluate`. `skip` lists game indices already finished.
  - `on_game(g, outcome, stats)` is called once per finished game, under a mutex, with the GIL acquired.
  - An exception in a worker or in the callback stops all games at their next move and is raised in Python, using the existing `parallel()`.
- `evaluate` and `best_move` are unchanged.

### Bindings and CLI (`cpp/bindings.cpp`, `block_puzzle_ai.ntuple`)

New options for `ntuple evaluate`:
- `--search-k K`: depth 2 on every move, tier `(+∞, [K])`.
- `--search-below G`: gate that tier at `G` instead. `K` defaults to 8.
- `--deep-below G2 --deep-k K1,K2`: adds the tier `(G2, [K1, K2])`.
- `--games-file PATH`: per-game JSONL file.
  - The first line is a header with the weights path, games, seed, max moves, γ and tiers.
  - Each later line is one game: `game`, `score`, `moves`, `lines`, `truncated`, searched moves per tier, changed moves, and `seconds`.
  - Rerunning with the same arguments skips finished games. A different header stops with an error.
- `--max-moves` stays 200,000 by default. Searched acceptance runs use 10,000,000.

Output:
- The summary and `--json` output keep today's keys and add `search` (tiers, searched and changed moves per tier, µs per move). So `scripts/bootstrap_mean.py` and `scripts/ab_compare.py` work unchanged.
- `results` holds every game in the games file, in game order.
- `seconds` is the wall time of this invocation.
- Stopping: kill the process. Ctrl-C is not handled while C++ runs, as today. Finished games are already on disk.

### Weights

Copy `runs/nt1/best.ntw` to `models/ntuple_v1.ntw` (2.6 MB) and commit it, like `models/rl_v2.pt`.

### Known limit

Games run one per thread, so the longest game sets the wall time. At a mean of ~1M moves, the longest of 64 is about 4M moves, about 5–6 h at ~5 ms per move. That fits in 8 h. Searching one game on several threads waits until the tail is the bottleneck.

## Tests (`tests/test_ntuple_search.py`, each under 60 s)

1. **Split value:** `value(board_eval(b), cooldown, r0, r1) == value(Afterstate)` exactly, on random after-states, both cooldown flags, and all remaining pairs.
2. **Depth 2 against a reference:**
   - A slow Python reference implements the definitions above on `game.py` and `ntuple.value`.
   - On random mid-game states it matches `search_scores` (relative 1e-9) and the chosen move, for k = 1, 4 and 0 (all).
   - One small depth-3 case is checked the same way.
3. **No tiers is 1-ply:** `evaluate_search` with `[]` gives `evaluate`'s score, moves and lines for every game.
4. **Threads:** 1 and 8 threads give identical per-game results.
5. **Resume:** an evaluation stopped after some games, then rerun, gives the same combined results as one run.
6. **Edge cases:** a state with no legal move, children that are game over (value 0), spawn pieces with zero probability, and invalid tier lists (rejected).

## Experiments (seed `2024`)

1. **Depth 2:** 64 games, 200k cap.
   - Arms: 1-ply; every move with k = 2, 4 and 8; gated k = 8 at two thresholds.
   - Record µs per move, mean, median, points per move, moves per game over, and searched and changed moves.
   - Question: does searching every move raise points per move?
2. **Depth-3 tier:**
   - Place the threshold from the distribution of best Q1 over the last moves before game overs.
   - Then sweep 2–3 thresholds and 1–2 width pairs on top of the best depth-2 setting.
3. **Choice:** the highest mean whose projected wall time fits 8 h, from µs per move at ~1M moves per game and the longest-game estimate.

## Acceptance (overnight, when the user's tfeval jobs are not running)

- **Command:** `models/ntuple_v1.ntw`, the chosen tiers, 64 games, seed `777`, 10M-move cap, 24 threads, with a games file, in at most 8 h.
- **Report:**
  - Mean, median and max.
  - Bootstrap 95% CI against n-tuple 1-ply at the same cap (`runs/nt1_eval/seed777_10M.json`, 7,111,924).
  - Comparison with `rl_v2` + search (11,158,066, at a 200k cap with 12 games alive).
  - Points per move, death rate, µs per move, wall time, and the distance to 100M.
- **Verdict:** the number reached, and the bottleneck that sets the next step: survival, points per move, or compute.
- **Write-up:** a `docs/RESULTS.md` section and one README line. Ask the user before posting to #10.

## Risks

- Every-move search may not raise points per move, if V's errors are correlated with the reward the search adds. The experiments show this before any overnight run.
- Gate thresholds depend on the weights' value scale. New weights need new thresholds.
- At high survival, the longest game, not the mean, sets the wall time.
