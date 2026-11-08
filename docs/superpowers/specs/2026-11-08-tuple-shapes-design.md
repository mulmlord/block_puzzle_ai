# Larger tuple shapes for the n-tuple value (ticket #10)

## Why

Two value pilots changed how `ntuple_v1`'s tuples are trained: temporal-coherence step sizes, then search-backed targets. Neither moved the searched agent's death rate (`docs/RESULTS.md`, "Search-backed targets pilot"). What is left for the value is what it can represent.

The tuples are `window4` (4×4 squares, 10 tables) and `line` (rows and columns, 5 tables): 655,360 + 5,120 weights. Two things they can't see:
- **Space for big pieces.** A 5-cell line or a 3×3 square needs room that a 4×4 window does not show.
- **Multi-line setups** (the bonus `50·L·(L+1)`). A `line` table sees one row, not two rows that fill together. `rl_v2` makes 117 points per move, the searched n-tuple agent 101.

`shapes_to_tuples` already takes `rect<h>x<w>` (up to 20 cells), so no code changes.

## Pilot

Three arms train from zero weights with `--tc`, side by side for 3 h on 8 threads each, at the default step-size cap for 8 threads (0.0625).

| Arm | Shapes | Tables | Weights | Speed vs W |
|---|---|---:|---:|---:|
| W (control) | `window4,line` | 15 | 0.66M | 1 |
| B | `window4,line,rect2x10`: adds row-pair bands | 20 | 5.90M | 0.92 |
| R | `rect4x5,line`: 4×5 windows instead of 4×4 | 17 | 12.59M | 0.77 |

- 8 threads for 3 h is 1 h of the whole machine. `ntuple_v1`'s 8-hour run found its best weights after 62 minutes on 24 threads.
- W shows whether training from zero with TC at this budget matches `ntuple_v1`, so that B and R are compared with the same training.

**Measurement.** The split yardstick under the acceptance config (k = 4, [8, 4] below 92k), free play seed `31`, capture `32`, stage 2 `41`, on each arm's `best.ntw` (the best periodic 1-ply evaluation).
- **Shifted tiers.** Each arm's tiers are shifted by the median of (its Q1 − `ntuple_v1`'s Q1) on 1,000 of `ntuple_v1`'s 92k entry states. The report gives the spread of that difference, since weights trained from zero need not differ from `ntuple_v1` by a constant.
- Only the 92k tier changes play: [8, 4] runs below 92k and below 85k alike. The 85k tier only splits the measurement.
- The capture stage restarts from at most 5,000 entry states (`--limit`), which bounds its time; e1 still counts all entries.
- Different weights play different games, so comparisons are unpaired.

**Bar.** B or R earns a longer run if its death rate relative to W has a 95% interval below 1, or if its points per move rise with a death-rate interval that includes 1. The report also compares every arm with `ntuple_v1` (14.01M [9.11M, 24.19M] moves per game over, 101.3 points per move) and gives both means.
