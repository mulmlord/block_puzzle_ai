# Search-backed targets for the n-tuple value (ticket #10)

## Why

The value sees danger only 1–2 moves ahead (`docs/RESULTS.md`, "Search below 85k"). Every game over ends a fall below Q1 85k, and below 85k search barely helps. So the lever is falling below 85k less often, which needs a value that sees trouble earlier. The temporal-coherence pilot showed that the step size is not the limit: plain TD and TC fine-tunes level off at the same 1-ply survival. This changes the target instead.

## Method

TD(0) today moves V(a), for the after-state a played from state s, towards `max_m' Q1(s', m')`, where s' is a after one sampled spawn (0 with no legal move).

With search-backed targets, the trainer runs `search_scores(s', widths)` at every state s' and uses its results twice. Each kept root move m' (top widths[0] by Q1) has its E(s', m', widths[1:]), computed before any update of this step:

```text
V(a)      ← max_m' E(s', m')               a 2-spawn-deep target: one spawn sampled, the next exact
V(a_m')   ← (E(s', m') − r(m')) / γ        for each kept m', in engine order: the exact expectation over the spawn
V(a)      ← 0                              when s' has no legal move
```

- Every update goes through `NTuple::learn`, so `--tc` applies.
- **Play** is the first maximum of Q1 among the kept moves, which is `best_move`'s move. With `--play-search`, it is the first maximum of E, which is `search_move`'s move. A new game's first move is chosen the same way.
- Restarts, the collapse guard, checkpoints and the periodic 1-ply evaluation are unchanged.

**Cost.** A search with widths [2] costs about 12 times a 1-ply move, and the step makes 3 updates instead of 1.

## Pilot

Both arms fine-tune `models/ntuple_v1.ntw` with `--tc --target-k 2`, side by side for 2 h on 12 threads each, with the TC pilot's settings (cap 0.0208, evaluation every 15 minutes).

| Arm | Play | Question |
|---|---|---|
| S | 1-ply move | Do search-backed targets help when play visits danger often (a game over every ~75k moves, plus restarts)? |
| SP | searched move | Does training on the searched agent's own states help more? Its game overs are about 50× rarer. |

The control is the TC pilot's `tc` arm: the same fine-tune with 1-ply targets.

**Measurement.** The deployed agent searches, so the 1-ply evaluation is only a health check. For each arm's `latest.ntw` and for the control's, the split yardstick runs under the acceptance config (k = 4, [8, 4] below 92k): free play, capture and stage 2, about 45 minutes each on 24 workers. `ntuple_v1` has 14.01M [9.11M, 24.19M] moves per game over there.
- The Q1 thresholds (85k, 92k) stay as they are. γ = 0.999 sets V's scale, so the thresholds keep their meaning across fine-tunes; the report states each arm's mean Q1 on ordinary moves.
  - **Wrong, found in the pilot:** every fine-tune's Q1 is `ntuple_v1`'s plus a near-constant shift (+1.3k to +12.9k), so the fixed tiers sat deeper in danger. The pilot was measured again with every tier shifted by each arm's median shift (`split_yardstick.py --shift`); see `docs/RESULTS.md`, "Search-backed targets pilot".
- Different weights play different games, so comparisons between weights are unpaired, with intervals of about ×/÷ 1.5.

**Bar.** An arm earns a longer run if its death rate relative to the control has a 95% interval below 1. A better point estimate whose interval includes 1 is inconclusive.

## Code

- `cpp/search.hpp`: `search_learn(nt, prev, s, gamma, widths, alpha, play_search, out)`, one step as above.
- `cpp/trainer.hpp`: `Trainer` moves here from `ntuple.hpp`, so that it can use the search. It gains `search_widths` (empty: 1-ply targets, bit-identical to before) and `search_play`.
- `ntuple train --target-k K [--play-search]`; both go into the checkpoint's config.

## Tests

- `search_learn` against a Python transcription built from `search_scores` and `update`: the same weights bit for bit, at depth 2 and 3, the right move under both play rules, and V(prev) ← 0 with no legal move.
- At α = 0, a one-thread trainer with search play plays the same first game as `evaluate_ntuple_search` with tier (+inf, [K]). With 1-ply play, it plays the same first game as `evaluate`.
- With search targets, chunks equal one run, and the CLI writes `target_k` and `play_search` to the config.
