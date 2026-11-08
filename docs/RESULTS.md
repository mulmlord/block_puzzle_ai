# Results

## Baselines vs. RL (128 fixed seeds `2024`, 20,000-move cap)

| Agent | Mean | Median | Min | Max | Moves / game | Points / move | Alive at cap |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random legal moves | 1,012 | 900 | 530 | 2,050 | 24 | 42 | 0 |
| Heuristic (greedy, tuned) | 20,879 | 15,515 | 3,170 | 87,560 | 279 | 75 | 0 |
| RL `models/rl_v1.pt` | 1,493,256 | 1,854,600 | 39,150 | 2,206,540 | 13,782 | 108 | 54 |

The heuristic's three weights were grid-searched over 18 combinations (128 games each), then the four best were re-checked on 512 games. The differences between those four were within noise (means 20.5k–22.8k).

## Training run `runs/main` (2 h on an RTX 4070 Laptop GPU)

Config:
- 512 parallel games, ~6–7k moves/s
- γ = 0.99, λ = 0.8, reward × 0.01
- ε from 0.10 to 0.005 over 4M moves
- 32-channel 3-block ResNet + MLP head
- AdamW, lr 3e-4

| Training time | Moves played | Eval mean | Median | Max | Games alive at cap |
|---:|---:|---:|---:|---:|---:|
| 11 min | 4.3M | 223,716 | 171,405 | 1,035,360 | 0/128 |
| 25 min | 8.3M | 1,031,638 | 970,910 | 1,828,080 | 35/128 |
| 37 min | 12.1M | 515,447 | 375,615 | 2,026,340 | 2/128 |
| 49 min | 16.0M | 746,353 | 498,295 | 2,072,770 | 10/128 |
| 62 min | 19.8M | 972,487 | 717,725 | 2,088,050 | 21/128 |
| 75 min | 23.6M | 1,077,314 | 954,765 | 2,107,040 | 24/128 |
| 88 min | 27.4M | 1,088,973 | 1,041,670 | 2,111,870 | 32/128 |
| 102 min | 31.2M | 1,263,239 | 1,358,585 | 2,131,830 | 42/128 |
| 114 min | 35.0M | 952,606 | 742,965 | 2,185,290 | 21/128 |
| 124 min | 37.2M | 1,493,256 | 1,854,600 | 2,206,540 | 54/128 |

Observations:
- The agent passed the heuristic within the first 10 minutes.
- **Points per move rose steadily**, from 75 (heuristic) to about 108, because the agent learned to build stacked rows that complete several lines at once. Under the `50·L·(L+1)` bonus, two lines at once pay 300 instead of 2 × 100.
- **The evaluations are noisy.** Each one plays only 128 games, and survival is heavy-tailed: a single death at move 3,000 instead of at the cap moves the mean by about 15k.
- The 25-minute jump is part of that noise. The trend over the run is clear.

## Long evaluation: games played to the end

64 new seeds (`777`) were run with a 200,000-move cap. **No game reached the cap**: all 64 ended with a real game over.
Command: `uv run python -m block_puzzle_ai.evaluate --agent models/rl_v1.pt --games 64 --seed 777 --max-moves 200000`.

| | Mean | Median | Min | Max | Mean moves | Points / move |
|---|---:|---:|---:|---:|---:|---:|
| RL `rl_v1` | 2,595,820 | 1,831,945 | 31,110 | **11,750,010** | 23,958 | 108 |

**High score: 11,750,010 points in a single game of 108,511 moves**, versus 87,560 for the heuristic's best game. The heuristic number comes from a different run (seed 2024, 128 games, 20,000-move cap), but no heuristic game came close to that cap.

Survival is heavy-tailed:

| Still alive after | 1k moves | 5k | 20k | 50k | 100k |
|---|---:|---:|---:|---:|---:|
| Games (of 64) | 63 | 56 | 28 | 6 | 1 |

Score quantiles:

| p10 | p25 | p50 | p75 | p90 |
|---:|---:|---:|---:|---:|
| 526k | 915k | 1.78M | 3.57M | 5.11M |

## Lookahead search (ticket #10)

`LookaheadAgent` searches one move deeper, but only where the agent senses danger. That is when the best 1-ply Q of a state is below `search_below`. There it re-scores the `top_k` best moves as `r + γ·E[best 1-ply Q after the next spawn]`. The expectation runs over every piece the game can spawn next, with the exact probabilities from `randTiles`. If no move is left, the value is 0. The network is unchanged, and there is no retraining.

Choosing the gate (rl_v1, 256 games, seed `4242`):
- The best Q has a median of 115 and a 2nd percentile of 109.9.
- Q < 100 occurs on only 0.08% of moves, but on 97% of the last moves before a game over. 2 and 3 moves before a game over, it is 66% and 42%.
- So the value net senses a death only 1–3 moves ahead, which is the range a 1-move search can still fix.

### A/B on fresh seeds (256 games, seed `2025`, 20,000-move cap)

Many games reach the cap, so score means are compressed. The comparison therefore uses deaths per move played. CI is a Poisson rate-ratio 95% CI against `base`.

| Arm | Game overs | Moves / game over | Death rate vs base | 95% CI | Mean | Time |
|---|---:|---:|---:|---|---:|---:|
| `base`: 1-ply | 154 | 23,138 | 1.00 | | 1,507,448 | 8 min |
| `fp32`: 1-ply, MLP head in fp32 | 122 | 31,300 | 0.74 | [0.58, 0.94] | 1,649,096 | 9 min |
| search below 100, all moves | 121 | 31,065 | 0.74 | [0.59, 0.95] | 1,587,864 | 9 min |
| **search below 110, top 8** | **70** | **63,305** | **0.37** | **[0.28, 0.48]** | 1,878,605 | 60 min |
| search below 110, top 8, fp32 head | 94 | 44,232 | 0.52 | [0.40, 0.68] | 1,798,531 | 48 min |

Search below 110 runs on about 2% of moves, and on about half of those it changes the move.

The fp32 head was **not adopted**. On 726k identical states, the bf16 and fp32 heads differ in best Q by a mean of −0.001 (std 0.14). Both trigger the search at almost the same rate, so the precision only breaks near-ties. Its two results also point in opposite directions: it helped 1-ply but hurt search.

Command: `uv run python scripts/ab_lookahead.py 256 2025 20000 out.jsonl base,fp32,search-100-all,search-110-8`, then `scripts/ab_compare.py out.jsonl`.

### Long evaluation with search (the ticket's protocol: 64 games, seed `777`, 200,000-move cap)

Command: `uv run python -m block_puzzle_ai.evaluate --agent models/rl_v1.pt --games 64 --seed 777 --max-moves 200000 --search-below 110 --top-k 8`.

| | Mean | Median | Min | Max | Mean moves | Points / move | Time |
|---|---:|---:|---:|---:|---:|---:|---:|
| `rl_v1`, 1-ply (as above) | 2,595,820 | 1,831,945 | 31,110 | 11,750,010 | 23,958 | 108 | 26 min |
| **`rl_v1` + search** | **5,985,303** | **4,408,550** | 60,930 | **21,739,880** | 55,168 | 108 | 94 min |

**The mean rose 2.3×: +3.39M, bootstrap 95% CI [+2.08M, +4.75M]** (`scripts/bootstrap_mean.py`).
- The search does not score faster: points per move stay at 108. All of the gain comes from surviving longer.
- The best game was still alive at the 200,000-move cap with 21,739,880 points. The best game that ended in a real game over scored 17,306,820 points in 159,603 moves.

| Still alive after | 1k moves | 5k | 20k | 50k | 100k | 200k (cap) |
|---|---:|---:|---:|---:|---:|---:|
| 1-ply (of 64) | 63 | 56 | 28 | 6 | 1 | 0 |
| + search (of 64) | 63 | 62 | 45 | 29 | 11 | 1 |

Score quantiles with search:

| p10 | p25 | p50 | p75 | p90 |
|---:|---:|---:|---:|---:|
| 976k | 1.69M | 4.20M | 9.07M | 13.2M |

## Training speed (ticket #9)

`scripts/profile_step.py` measures one training step: 512 envs, mid-game states of `rl_v1`, with a sync between phases. Run with `--eager` for the old code path.

| Phase (ms per step) | Before | After |
|---|---:|---:|
| Net inference (≈38k after-states) | 62.2 | 25.7 |
| `encode` | 9.2 | 5.7 |
| Network update (amortized) | 10.1 | 4.5 |
| `step` + `candidates` | 5.4 | 5.1 |
| **Total** | **86.9 (5.9k moves/s)** | **41.1 (12.5k moves/s): 2.1×** |

What changed:
- **Inference** (`ValueNet.predict`): the net now runs in fixed 2,048-row chunks, replayed as one CUDA graph (`torch.compile(mode="reduce-overhead")`). Neither part pays off alone. At 2,048 rows a layer's activations (13 MB) stay in the 32 MB L2 cache, and the graph removes the kernel launches that dominate passes this small. Eager inference at 2,048 rows is no faster than at 16k rows (64 vs 62 ms).
- **`fit_maps`**: the compare/mask/count epilogue after the matmul is compiled into a single pass.
- **Update** (`Learner`): fused AdamW, forward + backward as a CUDA graph, and one host sync per update instead of one per minibatch.

Tried and dropped:
- **Pruning candidates with the heuristic:** its top 16 moves contain the net's best move only 55% of the time (77% for the top 32).
- **Deduplicating after-states:** only 4.3% are exact duplicates.
- **Wider nets:** 48 and 64 channels cost 1.7× and 2.2× per evaluation, so they are a strength question for #8 rather than a speed-up.

60-minute run from scratch (`runs/speed9`, same config as `runs/main`):

| | Moves / s (median) | Best eval in the first hour | Eval at ~31M moves |
|---|---:|---:|---:|
| `runs/main` (old code) | 6.5k | 1,031,638 (25 min) | 1,263,239 (102 min) |
| `runs/speed9` | 12.9k | 1,224,598 (58 min) | 1,224,598 (58 min) |

Learning per move is unchanged. Each wall-clock hour now covers about twice as many moves.

The CUDA-graphed kernels round bf16 outputs a little differently, which flips some near-tie moves, so individual games diverge. Playing strength is unchanged within noise. Death rate of `rl_v1`, graphed vs eager:
- seed 2024 (128 games): 1.23 [0.90, 1.69]
- seed 2025 (256 games): 0.95 [0.76, 1.20]
- pooled: **1.04 [0.87, 1.25]**

## Longer training: `rl_v2` (ticket #8)

`models/rl_v2.pt` is a fine-tune of `rl_v1`. It plays the same way (greedy 1-ply, with optional search), but it survives about 2.4× longer and scores 8% more per move.

### What changed in training

- **fp32 value head.** Under bf16 autocast the network's output V ≈ 110 is rounded to steps of 0.5, which with reward × 0.01 is 50 points. The median gap between the two best moves is only about 24 points, so the rounding changed the greedy move on **23.6%** of `rl_v1`'s states, and on 53% at reward × 0.003. The MLP head now runs in fp32 (`ValueNet.set_fp32_head`). That cuts the mean error of V from 12.8 to 3.2 points, for about 5% more inference time. Checkpoints saved without it, such as `rl_v1`, still run as they were trained. This is the likely reason why the fp32-head arms in the lookahead A/B above pointed in opposite directions.
- **Longer horizon:** γ = 0.997 instead of 0.99. The reward scale is set to 1 − γ = 0.003, so V stays on the same scale, near the points per move. The search gate still had to move; see below.
- **Restarts near deaths** (#10, technique 3). Every 50 steps all games are snapshotted. When a game ends in a game over, its snapshots from 50–500 moves before the end join a 65k-state pool, and half of all finished games restart from a random pool state.
- **Schedule:** the learning rate warms up over 1M moves, then follows a cosine from 3e-4 to 3e-5. ε is fixed at 0.001.
- **`best.pt` is selected on moves per game over**, not on the mean score, because the 20k-move cap saturates the mean.
- **Resumable runs:** `--resume runs/<name>` continues from `resume.pt`, which is saved every 10 min. A snapshot goes to `ckpt/` at every eval.

### Pilots (30M moves from `rl_v1` each; 256 games, seed `2025`, 20,000-move cap)

| Arm | Game overs | Moves / game over | Death rate vs γ 0.99 | Points / move | Mean |
|---|---:|---:|---|---:|---:|
| `rl_v1` (no fine-tune) | 146 | 24,255 | | 108 | 1,497,820 |
| A: γ 0.99 | 160 | 20,127 | 1.00 | 121 | 1,526,140 |
| B: γ 0.997 | 139 | 25,825 | 0.78 [0.62, 0.98] | 114 | 1,595,842 |
| **C: γ 0.997 + restarts** | **130** | **28,700** | **0.70 [0.56, 0.88]** | 113 | **1,643,864** |
| D: γ 0.999 + restarts | 162 | 20,216 | 1.00 [0.80, 1.25] | 100 | 1,284,951 |

- Fine-tuning at γ = 0.99 trades survival for points: the death rate rises and points per move rise. With a death worth only about 100 moves of reward, the agent takes more risk for bigger clears.
- γ = 0.997 recovers survival. Restarts add a bit more, though C vs B alone is 0.90 [0.71, 1.14].
- γ = 0.999 is worse on both counts. At reward × 0.001, immediate rewards are small next to the value error, and 30M moves are too few to learn a horizon of 1,000 moves.
- The first attempt at B, with the bf16 head, got *worse* during training (training games fell from 7.4k to 5.7k moves). It was stopped, and the pilots were rerun with the fp32 head.

### Long run `runs/v2` (arm C, 400M moves, 9.8 h, 12.5k moves/s)

`uv run python -m block_puzzle_ai.train --init models/rl_v1.pt --out runs/v2 --total-moves 400000000 --gamma 0.997 --reward-scale 0.003 --restart-frac 0.5 --eps-start 0.001 --eval-minutes 45`

Evals: 256 games, seed `2024`, 20,000-move cap.

| Time | Moves | Game overs | Moves / game over | Points / move | Mean | Alive at cap |
|---:|---:|---:|---:|---:|---:|---:|
| 0.8 h | 33M | 125 | 30,099 | 100 | 1,477,000 | 131/256 |
| 1.7 h | 66M | 154 | 21,811 | 106 | 1,388,755 | 102/256 |
| 2.5 h | 98M | 191 | 14,301 | 111 | 1,188,191 | 65/256 |
| 3.3 h | 131M | 113 | 34,021 | 111 | 1,673,502 | 143/256 |
| 4.1 h | 165M | 89 | 46,684 | 116 | 1,883,620 | 167/256 |
| 5.0 h | 198M | 148 | 23,713 | 117 | 1,610,730 | 108/256 |
| 5.8 h | 232M | 72 | **59,799** | 117 | 1,969,683 | 184/256 |
| 6.6 h | 267M | 156 | 21,541 | 120 | 1,579,271 | 100/256 |
| 7.5 h | 302M | 121 | 32,099 | 121 | 1,838,843 | 135/256 |
| 8.3 h | 337M | 118 | 32,806 | 123 | 1,855,616 | 138/256 |
| 9.1 h | 373M | 127 | 29,794 | 123 | 1,818,452 | 129/256 |
| 9.8 h | 400M | 104 | 38,544 | 123 | 1,923,349 | 152/256 |

- Evals swing a lot while the learning rate is high: from 14k to 60k moves per game over, and neighbouring evals differ by up to 2.8×.
- Points per move climb throughout, from 100 to 123. Survival peaked at 232M moves, and the low-lr end of the run settled at 30–39k moves per game over. Late training drifts towards greedier play, as pilot A did.
- `best.pt` is the 232M snapshot. On fresh seeds (`2025`) it held its lead over the final weights, so it was released as `rl_v2`:

| Seed `2025`, 256 games, 20k cap | Game overs | Moves / game over | Death rate vs `rl_v1` | Points / move | Mean |
|---|---:|---:|---|---:|---:|
| `rl_v1` | 146 | 24,255 | 1.00 | 108 | 1,497,820 |
| **`rl_v2` = `best.pt` (232M)** | **80** | **52,740** | **0.46 [0.35, 0.60]** | 117 | 1,930,164 |
| final weights (400M) | 103 | 39,284 | 0.62 [0.48, 0.79] | 123 | 1,939,407 |

### Long evaluation (the ticket's protocol: 64 games, seed `777`, 200,000-move cap)

Commands:
- `uv run python -m block_puzzle_ai.evaluate --agent models/rl_v2.pt --games 64 --seed 777 --max-moves 200000`
- add `--search` for the search row. It uses the gate stored in the checkpoint; see below.

| | Mean | Median | Min | Max | Moves / game over | Points / move | Alive at cap | Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `rl_v1` | 2,595,820 | 1,831,945 | 31,110 | 11,750,010 | 23,958 | 108 | 0 | 26 min |
| `rl_v1` + search (gate 110) | 5,985,303 | 4,408,550 | 60,930 | 21,739,880 | 56,043 | 108 | 1 | 94 min |
| **`rl_v2`** | **6,507,274** | **4,665,275** | 12,110 | 23,400,040 | **57,402** | **117** | 2 | 10 min |
| `rl_v2` + search (gate 110) | 8,396,430 | 6,506,605 | 865,650 | 23,512,410 | 76,560 | 117 | 4 | 11 min |
| **`rl_v2` + search (gate 115)** | **11,158,066** | **7,589,755** | 13,380 | 23,464,090 | **117,387** | 117 | **12** | 12 min |

All search rows re-score the 8 best moves (`--top-k 8`, the default).

The ticket's acceptance test is `rl_v2` vs `rl_v1`, both 1-ply: **mean +3.91M (2.5×), bootstrap 95% CI [+2.39M, +5.54M]**.
- Without any search, `rl_v2` already matches `rl_v1` with search, and it plays about 9× faster. The faster evaluations are also helped by the #9 speed-ups.
- With search at gate 115:
  - +4.65M [+2.13M, +7.12M] over `rl_v2` alone; death rate 0.49 [0.34, 0.71]
  - +5.17M [+2.81M, +7.49M] over `rl_v1` with search
  - 4.3× `rl_v1`
  - 12 of the 64 games were still alive at the cap, so this mean is a lower bound.
  - It searched on 0.12% of moves, and the search changed the move on half of those.
- The best game that ended in a real game over scored 22,892,280 points.
- Re-scoring *all* moves instead of the top 8 (`--top-k 0`) scored 12,332,520 with 16 games at the cap, but it took about 26% more time per move. The difference is not significant: top 8 minus all moves is −1.17M [−4.05M, +1.64M], death rate 1.20 [0.81, 1.77].

| Still alive after | 1k moves | 5k | 20k | 50k | 100k | 200k (cap) |
|---|---:|---:|---:|---:|---:|---:|
| `rl_v1` (of 64) | 63 | 56 | 28 | 6 | 1 | 0 |
| `rl_v2` (of 64) | 63 | 58 | 48 | 28 | 10 | 2 |
| `rl_v2` + search, gate 115 (of 64) | 63 | 63 | 56 | 41 | 25 | 12 |

### 256 games played to the end (seed `2026`)

The ticket asks for 256 games as well: the high score is the luckiest of n games, so it grows with n.

Commands:
- `uv run python -m block_puzzle_ai.evaluate --agent models/rl_v2.pt --games 256 --seed 2026 --max-moves 2000000`
- `rl_v1` was run with `--max-moves 200000`, which none of its games reached.

| | Mean | Median | Min | Max | Moves / game over | Points / move | Time |
|---|---:|---:|---:|---:|---:|---:|---:|
| `rl_v1` | 2,562,693 | 1,848,425 | 700 | 15,268,300 | 23,669 | 108 | 10 min |
| **`rl_v2`** | **6,824,363** | **4,587,690** | 25,090 | **38,818,140** | **58,348** | **117** | 43 min* |

\*The GPU was shared with another evaluation for most of this run.

- **The mean rose 2.66×: +4.26M, bootstrap 95% CI [+3.36M, +5.19M].** All 256 games of both agents ended in a real game over.
- **High score: 38,818,140 points** in 331,683 moves of plain 1-ply play.
  - The "luckiest of n" estimate predicts about 42M: mean length × H₂₅₆ ≈ 58.3k × 6.1 ≈ 357k moves, at 117 points per move.

| Still alive after | 1k moves | 5k | 20k | 50k | 100k | 200k | 300k |
|---|---:|---:|---:|---:|---:|---:|---:|
| `rl_v1` (of 256) | 246 | 211 | 109 | 33 | 4 | 0 | 0 |
| `rl_v2` (of 256) | 248 | 231 | 176 | 110 | 53 | 11 | 1 |

`rl_v2` score quantiles:

| p10 | p25 | p50 | p75 | p90 |
|---:|---:|---:|---:|---:|
| 620k | 1.70M | 4.60M | 9.46M | 14.99M |

### Lookahead search with `rl_v2`: the gate has to move

`rl_v2`'s value function is much sharper than `rl_v1`'s. Over 8.6M moves (256 games, seed `4242`, `scripts/danger_trace.py`):

| | Median best Q | 2nd percentile | Moves with Q < 110 | Moves with Q < 115 |
|---|---:|---:|---:|---:|
| `rl_v1` | 115 | 109.9 | 2% | |
| `rl_v2` | 118.0 | 116.1 | 0.03% | 0.18% |

Share of game overs flagged by each gate, by how many moves before the game over:

| Gate | 1 move | 2 moves | 3 moves | 5 moves |
|---|---:|---:|---:|---:|
| `rl_v1`, Q < 110 | 97% | 66% | 42% | |
| `rl_v2`, Q < 110 | 96% | 65% | 40% | 19% |
| `rl_v2`, Q < 115 | 100% | 89% | 70% | 45% |

So for `rl_v2`, the old gate of 110 still catches deaths 1–3 moves ahead about as often as before, but it fires 70× less often. A/B on fresh seeds (`2027`, 128 games, 200,000-move cap; `scripts/ab_lookahead.py 128 2027 200000 out.jsonl base,search-110-8,search-115-8 models/rl_v2.pt`):

| Arm | Game overs | Moves / game over | Death rate vs 1-ply | Mean | Median | Searched moves |
|---|---:|---:|---|---:|---:|---:|
| `rl_v2`, 1-ply | 127 | 49,499 | 1.00 | 5,748,927 | 3,379,135 | |
| search below 110, top 8 | 122 | 62,646 | 0.79 [0.62, 1.01] | 6,985,762 | 5,327,580 | 0.03% |
| **search below 115, top 8** | **109** | **106,338** | **0.47 [0.36, 0.60]** | **10,594,480** | **8,695,410** | 0.12% |

Gate 115 vs gate 110: 0.59 [0.46, 0.76]. The gate now lives in the checkpoint (`search_below`, 115 for `rl_v2`). Checkpoints without it, such as `rl_v1`, fall back to 110. `evaluate --search` and the viewer's "RL + lookahead" use the checkpoint's gate.

### Where `rl_v2` + search still dies (ticket #10)

The question was whether the remaining game overs could be prevented by more search, or only by a better value function.

- **Method:**
  - Play 64 games with `rl_v2` and gate-115 search (seed `4243`, 200,000-move cap), which gave 50 game overs.
  - Keep the state 1, 3, 10 and 30 moves before each game over.
  - Replay each state 16 times with freshly drawn pieces (the three pending pieces are kept), and count how often the agent is still alive 100 moves later.
  - Compare three policies: plain 1-ply, gate 115, and search on every move for the first 40 moves.
- The script was a one-off and is not in the repo.

| Moves before the game over | 1-ply | Gate 115 | Search on every move | Gate fired (Q < 115) | Median best Q |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.66 | 0.67 | 0.69 | 98% | 80.7 |
| 3 | 0.90 | 0.93 | 0.94 | 48% | 115.1 |
| 10 | 0.996 | 0.999 | 0.999 | 2% | 117.3 |
| 30 | 0.999 | 0.999 | 0.999 | 0% | 118.3 |

- **No position was lost.** Even 1 move before a game over, two thirds of fresh draws survive the next 100 moves, and none of the 200 states died on all 16 replays. These games ended because of the particular pieces that came next.
- **Ten moves out, the positions are ordinary.** Their survival matches that of a typical position (about 0.999 per 100 moves at 117k moves per game over). Their median Q is 117.3, against 118 overall. The danger builds up only over the last ~3 moves, where the gate already fires.
- **More search has little left to gain.** Searching every move instead of gating:
  - 1 move before: +0.02 [−0.01, +0.06] survival (bootstrap 95% CI over states)
  - 3 moves before: +0.01 [−0.01, +0.04]
  - It cost 44× the compute.
- **Consequences:**
  - Distilling the search into training is dropped. At gate 115 the search costs only about 20% more time, and it acts only in the last few moves, where the net already senses danger.
  - What remains is the per-move risk of ordinary positions, which is a training problem, not a search problem.

## C++ engine (ticket #11)

`block_puzzle_ai._engine` (sources in `cpp/`, Python wrapper `block_puzzle_ai.engine`) is a single-game bitboard twin of `game.py`, meant for CPU-heavy work: the n-tuple value functions of #10 technique 7, and search.
- The board is an `unsigned __int128`.
- Each piece's legal positions come from one mask expression over its ≤ 9 cells.
- Full rows and columns come from the 10 row fields.
- The spawn RNG is an exact port of `randTiles`, and a bit-exact port of CPython's `random.Random` lets C++ games replay `game.Game(seed)` games.
- `uv sync` builds it with scikit-build-core and nanobind.

**Parity** (`tests/test_engine.py`):
- 1,000 seeded games, two each (so the spawn history carries over the reset), stepped in lock-step with `game.py`. The legal moves, board, pieces, score, points, lines and game over were identical at every move.
- Every legal move on 1,000 random boards filled from 30% to 90%, including clears of 3+ lines at once.
- Spawns and the exact spawn distribution at scores around each threshold, and at scores where float32 rounding matters.
- The RNG for seeds up to 2⁶⁴ − 1.

**Throughput** (`uv run python scripts/bench_engine.py`, AMD Ryzen 7 7840HX with 12 cores / 24 threads, RTX 4070 Laptop GPU):

| Simulator | Moves/s | Note |
|---|---:|---|
| C++ `random_play`, 1 thread | 9.8M | 24.2 moves per game, like the random baseline |
| C++ `random_play`, 12 threads | 84.6M | |
| C++ `random_play`, 24 threads | **107.3M** | |
| C++ `afterstate_play`, 1 thread | 0.75M | 57M after-states/s |
| C++ `afterstate_play`, 24 threads | 7.9M | 603M after-states/s |
| `game.py`, 1 thread | 20.9k | |
| `BatchGame` on CUDA, 4,096 games, random agent | 135k | |

- **Random play:** the targets were ≥ 1M moves/s on one thread and ≥ 10M on all threads. Both are beaten about 10×.
- **After-state play** applies every legal move before each move (about 76 per move), which is the per-move work of a 1-ply learner before any value lookups. This sets the ceiling for CPU n-tuple training on this machine: about 8M moves/s, against the GPU trainer's 12.5k moves/s.
- 12 → 24 threads adds only 27%: the second hyper-thread of each core shares its execution units.

## n-tuple learner (ticket #10, technique 7)

`block_puzzle_ai.ntuple` learns a symmetric n-tuple value function by TD(0) on after-states, on the C++ engine. Training is lock-free on 24 threads, and the agent plays greedy 1-ply. The design is in `docs/superpowers/specs/2026-11-03-ntuple-learner-design.md`.
- Features: tuple shapes `window4` and `line`, plus fit counts for each piece group and for the 2 remaining pieces.
- Settings: γ = 1, restart fraction 0.5.

### Getting a stable 1-hour run

The first two launches failed:
- **Divergence after 12M moves:** with α 0.1 on 24 threads, the mean value went from 1.3M to 4.5e22, then to NaN.
  - Stale concurrent updates on shared hot weights make the effective step grow with α × threads. 1 and 4 threads at α 0.1 were stable; 12 were not.
  - The α cap now defaults to min(0.1, 0.5 / threads), which is 0.021 at 24 threads. Non-finite values raise an error instead of crashing the process.
- **Collapse after 480M moves (43 min):** the run had reached 25k moves per game over. Within 2 minutes it fell to 29 moves per game and never recovered.
  - Cause: with γ = 1, each game over moves its weights by about α × V. Once V is about 2.4M, a chance cluster of game overs reorders the policy, which causes more game overs.
  - It could be provoked on demand from the 32-minute checkpoint (1% random moves for 3M moves), and a quarter of the step size survived it. The restart pool and lock-free races were ruled out. The collapsed state is permanent: it can only be left by rescaling the weights.
  - Fix: after every 10-second chunk, α = min(cap, 12,500 / mean V).
  - A rollback guard backs this up. If a chunk has ≥ 50 game overs at more than 4× the recent median rate, or mean V falls below 0.8× its recent maximum, the guard restores the weights from up to 3 chunks earlier and halves the cap.

### 1-hour run `runs/nt1h`

Command: `uv run python -m block_puzzle_ai.ntuple train --out runs/nt1h --hours 1`

- **627M moves in 60.7 min.** That is 1.6× the moves of the whole 9.8-hour `rl_v2` run.
- **Throughput:** median 183.5k moves/s over the last 10 chunks (225k in the first chunk, while games are short). The target was ≥ 200k. It has not been profiled.
- **Stability:** the guard never fired. α fell smoothly from 0.021 to 0.0052 as mean V rose from 38k to 2.39M.

Periodic evaluations (256 games, seed `2024`, 200,000-move cap):

| Minutes | Moves trained | Mean | Median | Moves / game over | Points / move |
|---:|---:|---:|---:|---:|---:|
| 16 | 161M | 2,221,902 | 1,603,245 | 27,761 | 80 |
| 31 | 321M | 2,283,489 | 1,537,010 | 28,487 | 80 |
| **47 (`best.ntw`)** | **481M** | **2,826,588** | **1,985,890** | **35,526** | 80 |
| 61 | 627M | 2,730,577 | 1,988,640 | 34,101 | 80 |

Long evaluation (the ticket's protocol: 64 games, seed `777`, 200,000-move cap; `uv run python -m block_puzzle_ai.ntuple evaluate --weights runs/nt1h/best.ntw --games 64 --seed 777 --max-moves 200000`):

| | Mean | Median | Min | Max | Moves / game over | Points / move | Alive at cap | Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `rl_v1` | 2,595,820 | 1,831,945 | 31,110 | 11,750,010 | 23,958 | 108 | 0 | 26 min |
| **`rl_v2`** | **6,507,274** | **4,665,275** | 12,110 | 23,400,040 | **57,402** | **117** | 2 | 10 min |
| n-tuple, 1 h | 2,285,927 | 1,658,510 | 121,290 | 9,750,180 | 28,622 | 80 | 0 | 31 s |

- **vs `rl_v2`:** mean −4.22M, bootstrap 95% CI [−5.80M, −2.73M]. Death rate 2.01 [1.41, 2.84].
- **Points per move are the larger gap.** They have stayed at 80 since the first evaluation, against 117 for `rl_v2` and 108 for `rl_v1`. Even with `rl_v2`'s survival, 80 points per move would give a mean of only about 4.6M.
- **Survival is still rising,** and it already beats `rl_v1`'s 24.0k moves per game over. The best evaluation came at 47 minutes, and the last one was within noise of it.
- The plan's 1-hour gate (≥ 1,000 moves per game over) is passed, so the 8-hour run comes next.

### γ pilot

With γ = 1, points per move stayed at 80. The suspected cause was that a V of millions drowns the multi-line bonus `50·L·(L+1)`. So before the 8-hour run, three short runs were compared at 161M moves each, on 12 threads (the other 12 were busy), with the same evaluation as the periodic one (256 games, seed `2024`, 200,000-move cap):

| γ | Threads | Mean | Moves / game over | Points / move | α at 161M |
|---:|---:|---:|---:|---:|---:|
| 1 (1-hour run) | 24 | 2,221,902 | 27,761 | 80.4 | 0.0084 |
| 1 (control) | 12 | 1,566,358 | 19,461 | 80.5 | 0.0082 |
| **0.999** | 12 | **3,356,660** | **40,006** | 84.6 | 0.0417 |
| 0.9995 | 12 | 3,201,683 | 39,370 | 82.0 | 0.0417 |

- **The points-per-move explanation was wrong.** γ < 1 adds only 2–4 points per move.
- **Survival doubled,** and the thread count does not explain it. But γ comes bundled with the step size: with γ 0.999, V stays near 81k, so the schedule `min(cap, 12,500 / V)` never lowers α below its cap.
- Each row is a single evaluation, and snapshots of one run vary more than that: at 110M moves, γ 0.999 and the γ = 1 control were level (25.7k and 24.9k moves per game over).
- γ 0.999 was chosen for the 8-hour run.

### 8-hour run `runs/nt1`

Command: `uv run python -m block_puzzle_ai.ntuple train --out runs/nt1 --hours 8 --gamma 0.999`

- **9.16 billion moves in 480.7 min,** at about 300k moves/s on 24 threads. That is 23× the moves of the whole `rl_v2` run.
- **The best weights came after 62 minutes.** From 47 to 78 minutes the run held about 70k moves per game over and 93–94 points per move.
- **A collapse at about 80 minutes.** Training moves per game over fell from 73k to 3.6k within 10 minutes, at α 0.0013. So γ < 1 does not prevent collapses.
  - The guard fired 8 times in all: 4 times between 32 and 49 minutes, and 4 more within 2 minutes at 81–83 minutes. Each one halved the α cap, from 0.021 to **8.1·10⁻⁵**, 1/256 of the start.
  - The rollbacks did not stop the collapse. They restore weights at most 3 chunks (30 s) back, which was already inside it.
- **After the collapse, the run cycled for 6.5 hours without beating its 62-minute best.** α stayed at 8.1·10⁻⁵. Training moves per game over swung between about 12k and 60k, in cycles of about 90 minutes, and V rose and fell with them (78k–83k). The last hour sat on the top of one cycle: 57.5k, 59.4k, 58.0k and 56.9k moves per game over in the periodic evaluations.
- Points per move follow survival: 93–94 at the best, 81–83 in the troughs.

Periodic evaluations (256 games, seed `2024`, 200,000-move cap), selected:

| Minutes | Moves trained | Mean | Moves / game over | Points / move |
|---:|---:|---:|---:|---:|
| 16 | 0.25B | 3,636,322 | 42,751 | 86 |
| 31 | 0.52B | 485,191 | 6,097 | 80 |
| 47 | 0.77B | 6,205,611 | 70,341 | 93 |
| **62 (`best.ntw`)** | **1.04B** | **6,319,104** | **70,785** | **94** |
| 78 | 1.31B | 6,029,913 | 68,235 | 94 |
| 93 | 1.70B | 209,775 | 2,682 | 78 |
| 124 | 2.36B | 4,258,370 | 53,036 | 82 |
| 325 | 6.20B | 902,329 | 11,130 | 81 |
| 371 | 7.08B | 4,377,927 | 53,168 | 85 |
| 417 | 7.99B | 1,004,416 | 12,363 | 81 |
| 464 | 8.88B | 4,901,886 | 59,379 | 85 |
| 481 | 9.16B | 4,667,025 | 56,906 | 85 |

All 32 evaluations are in `runs/nt1/metrics.jsonl`.

### Acceptance

Long evaluation (64 games, seed `777`; `uv run python -m block_puzzle_ai.ntuple evaluate --weights runs/nt1/best.ntw --games 64 --seed 777 --max-moves 200000 --json`). Death rates are relative to `rl_v2`, with Poisson 95% CIs (`scripts/ab_compare.py`):

| | Cap | Mean | Median | Min | Max | Moves / game over | Points / move | Alive at cap | Death rate | Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`rl_v2`** | 200k | **6,507,274** | 4,665,275 | 12,110 | 23,400,040 | 57,402 | **117** | 2 | 1 | 10 min |
| n-tuple, 1 h | 200k | 2,285,927 | 1,658,510 | 121,290 | 9,750,180 | 28,622 | 80 | 0 | 2.01 [1.41, 2.84] | 31 s |
| **n-tuple, 8 h (`best.ntw`)** | 200k | 6,192,662 | 4,224,940 | 19,940 | 18,863,510 | **72,672** | 94 | 6 | 0.79 [0.55, 1.13] | 12 s |
| n-tuple, 8 h (`latest.ntw`) | 200k | 5,788,879 | **5,220,285** | 117,420 | 17,087,440 | 71,167 | 85 | 3 | 0.81 [0.57, 1.15] | 13 s |
| n-tuple, 8 h (`best.ntw`) | 10M | 7,111,924 | 4,224,940 | 19,940 | **42,324,610** | 75,645 | 94 | 0 | | 17 s |

- **Verdict: not passed.** The mean is −0.31M against `rl_v2`, bootstrap 95% CI [−2.35M, +1.76M], so the two are statistically tied. The plan's bar was a lower bound above 0.
- **Survival is better than `rl_v2`'s for the first time,** at 72.7k against 57.4k moves per game over, but the death rate (0.79 [0.55, 1.13]) is not significantly lower.
- **Points per move are the whole gap:** 94 against 117. With `rl_v2`'s points per move, the same moves per game (65.9k at this cap) would give a mean of about 7.7M.
- **vs the 1-hour run:** +3.91M [+2.42M, +5.46M].
- **High score: 42,324,610 points** in one 451,088-move game without search, with the 10M cap. That beats `rl_v2`'s 38,818,140 (on the 256 seeds `2026`), though on different seeds.
- **Evaluation is about 48× faster than `rl_v2`'s**: 12 s against 593 s for the same 64 games.

### Next methods, in order

1. **Lookahead search on the n-tuple agent.** Search added +4.65M (1.7×) to `rl_v2` and halved its death rate. Evaluation here is 48× cheaper, so deeper or ungated search is affordable. It needs no retraining.
2. **Training stability.** Replace the guard's halving, which froze α at 1/256 after 83 minutes, with per-weight adaptive steps (temporal-coherence learning, the usual choice for 2048 n-tuple networks). Also look at why survival cycles every ~90 minutes even at tiny α. One guess is that danger states are forgotten when game overs become rare. Rollbacks need a longer window to be of any use.
3. **Points per move.** γ was ruled out, so the features are next: tuple shapes that can represent setups for clearing several lines at once.

## n-tuple search (ticket #10)

Expectimax search over the n-tuple value (`cpp/search.hpp`, design in `docs/superpowers/specs/2026-11-05-ntuple-search-design.md`):
- Each kept move is scored by `r + γ · Σ_x P(x) · best1(child)`, with the exact spawn distribution and no sampling.
- "k" is the number of root moves kept, by 1-ply value Q1.
- A gate searches only when the root's best Q1 is below G. Q1 is about 92.5k on ordinary moves.

Weights: `models/ntuple_v1.ntw`, a copy of `runs/nt1/best.ntw` (γ 0.999).

### Depth 2

Command: `uv run python -m block_puzzle_ai.ntuple evaluate --weights models/ntuple_v1.ntw --games 64 --seed 2024 --max-moves 200000 --search-k K` (or `--search-below G --search-k 8`), plus `--games-file … --json`.

Cost, in µs per move per thread:

| k | 8 games on 8 threads | 64 games on 24 threads | × 1-ply (24 threads) |
|---:|---:|---:|---:|
| 1-ply | 28 | 49 | 1 |
| 2 | 511 | 1,002 | 20 |
| 4 | 1,017 | 1,949 | 40 |
| 8 | 1,999 | 4,214 | 86 |

Arms (64 games, seed `2024`, 200,000-move cap, 24 threads). Death rates are relative to 1-ply, with Poisson 95% CIs (`scripts/ab_compare.py`):

| Arm | Searched | Changed | Mean | Median | Alive at cap | Moves / game over | Points / move | Death rate | Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1-ply | 0 | 0 | 5,069,704 | 3,766,075 | 1 | 54,787 | 94.0 | 1 | 11 s |
| gate G = 90k | 0.085% | 0.041% | 12,138,866 | 12,813,315 | 21 | 192,133 | 94.0 | 0.29 [0.19, 0.42] | 22 s |
| gate G = 92k | 0.84% | 0.40% | 16,716,380 | 18,817,385 | 46 | 631,186 | 94.2 | 0.087 [0.051, 0.147] | 37 s |
| k = 2 | 100% | 19.9% | 18,432,223 | 19,689,075 | 58 | 1,997,181 | 98.4 | 0.027 [0.012, 0.063] | 9.4 min |
| k = 4 | 100% | 28.7% | 19,951,879 | 20,268,075 | 63 | 12,600,958 | 101.3 | 0.004 [0.001, 0.031] | 19 min |
| **k = 8** | 100% | 32.0% | **20,365,126** | **20,619,130** | 63 | 12,640,489 | **103.1** | 0.004 [0.001, 0.031] | 41 min |

- **Search buys survival, by an order of magnitude or more.**
  - Searching only 0.84% of moves (G = 92k) cuts the death rate 11× for 1.4× the cost of 1-ply.
  - Searching every move with k = 2 cuts it 36× (6 game overs in 12M moves).
  - k = 4 and k = 8 had one game over each.
- **Every-move search raises points per move:** 94 → 98.4, 101.3 and 103.1 for k = 2, 4 and 8. Gated search does not (94.0–94.2). That answers the spec's question, but the gain is 5–10%. The mean moves through survival.
- **The 200k cap now binds:** 58–63 of 64 games are alive at it. Means near 20M are about 200k × points per move, so they no longer rank the every-move arms. k = 8 against k = 2 is +1.93M [+0.85M, +3.11M].
- **The plan's projection is obsolete.** It assumed about 1.06M moves per game at the 10M cap. k = 2 already gives about 2M moves per game over, so the acceptance is now bound by compute (long games), not by survival.

### Depth 3

**Where to place the depth-3 tier.** Game overs give only 2–4 moves of warning. Best Q1 before the game overs (`runs/nts_d3/danger.py`), against about 92.5k on ordinary moves:

| Arm | Game overs | 10 moves before | 3 moves before | Last move |
|---|---:|---:|---:|---:|
| k = 2 | 6 | 92.2k (p10 89.4k) | 87.4k (p10 63.4k) | 74.1k (p10 52.9k) |
| gate G = 92k | 18 | 92.5k (p10 91.8k) | 91.5k (p10 75.9k) | 70.7k (p10 52.6k) |

With k = 2 on every move, the best Q1 falls below G₂ = 90k on 0.005% of moves, below 91.5k on 0.044–0.047% and below 92k on 0.17–0.19%. One depth-3 search costs 42 ms on one thread with widths [4, 2], and 159 ms with [8, 4].

Arms: k = 2 on every move plus the tier (G₂, widths); 64 games, seed `2024`, 200,000-move cap, 24 threads. Death rates are relative to k = 2:

| Arm | Deep tier rate | µs / move | Mean | Game overs | Moves / game over | Points / move | Death rate | vs k = 2 (mean) | Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| k = 2 | 0 | 1,002 | 18,432,223 | 6 | 1,997,181 | 98.4 | 1 | | 9.4 min |
| G₂ 91.5k, [4, 2] | 0.047% | 1,133 | 18,478,363 | 7 | 1,715,593 | 98.5 | 1.16 [0.39, 3.46] | +0.05M [−1.39M, +1.48M] | 11 min |
| **G₂ 92k, [4, 2]** | 0.19% | 1,164 | **19,492,663** | 2 | 6,335,280 | 98.5 | 0.32 [0.06, 1.56] | **+1.06M [+0.07M, +2.21M]** | 11 min |
| G₂ 91.5k, [8, 4] | 0.044% | 1,060 | 19,287,005 | 2 | 6,267,769 | 98.5 | 0.32 [0.06, 1.58] | +0.85M [−0.30M, +2.09M] | 11 min |
| G₂ 92k, [8, 4] | 0.17% | 1,347 | 19,429,431 | 2 | 6,314,220 | 98.5 | 0.32 [0.06, 1.57] | +1.00M [−0.08M, +2.19M] | 13 min |

- The tier helps when it covers enough positions (92k) or searches wide enough ([8, 4]). [4, 2] at 91.5k did not help.
- The three 2-game-over arms cannot be told apart. Together they have 6 game overs in 38M moves, against k = 2's 6 in 12M: a death rate of 0.32 [0.10, 0.98].
- Points per move do not change. The tier fires too rarely to matter for them.

**Chosen configuration:** `--search-k 2 --deep-below 92000 --deep-k 4,2`, the highest mean.

**Projection to the acceptance (seed `777`, 10M cap).** Survival is modelled as a constant death rate per move, so expected moves per game = m · (1 − e^(−cap / m)), where m is moves per game over:

| m | Moves / game | Mean (at 98.5 points / move) | Work on 24 threads |
|---|---:|---:|---:|
| 6.33M (point estimate) | 5.03M | 495M | 4.3 h |
| 1.75M (95% low) | 1.75M | 172M | 1.5 h |
| 52M (95% high) | 9.10M | 897M | 7.8 h |

- A 10M-move game takes about 3.2 h at the 24-thread cost.
- All 64 games at the cap would take 8.6 h, which is over the 8-hour budget only in the case where almost no game dies.
- Every move with k = 4 or k = 8 does not fit. k = 4 is 1,949 µs per move with about 12.6M moves per game over (1 game over), which projects to about 10 h.

### Acceptance

Command (2026-11-06, 04:35–08:06; 24 threads, no `tfeval` running):

```bash
uv run python -m block_puzzle_ai.ntuple evaluate --weights models/ntuple_v1.ntw --games 64 --seed 777 --max-moves 10000000 \
  --search-k 2 --deep-below 92000 --deep-k 4,2 --games-file runs/nts_accept/games.jsonl --json
```

Seed `777`, 64 games. Death rates are relative to n-tuple 1-ply, with Poisson 95% CIs (`scripts/ab_compare.py`):

| | Cap | Mean | Median | Min | Max | Moves / game over | Points / move | Alive at cap | Death rate | Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| n-tuple, 1-ply | 10M | 7,111,924 | 4,224,940 | 19,940 | 42,324,610 | 75,645 | 94.0 | 0 | 1 | 17 s |
| `rl_v2` + search | 200k | 11,158,066 | 7,589,755 | | 23,464,090 | 117,387 | **117.0** | 12 | 0.64 [0.45, 0.93] | 12 min |
| **n-tuple + search** | 10M | **365,777,118** | **320,815,295** | 5,749,560 | **984,183,190** | **3,774,622** | 98.4 | 1 | **0.020 [0.014, 0.028]** | 3.51 h |

- **The 100M goal is reached:** the mean is 365.8M, 3.7× the goal. 51 of the 64 games scored over 100M.
- **vs n-tuple 1-ply at the same cap:** +358.7M, bootstrap 95% CI [+292.7M, +426.8M]. That is 51×, almost all of it from survival: the death rate is 50× lower.
- **vs `rl_v2` + search:** +354.6M [+288.6M, +422.9M]. Its run had a 200k cap with 12 games alive, so its mean is a lower bound.
- **High score: 984,183,190 points** in game 0. It was still alive at the 10M-move cap.
- **Cost:** 1,059 µs per move per thread. The depth-3 tier ran on 449k moves (0.19%) and changed the move on 49% of them; depth 2 changed 20% of all moves.
  - The longest game (10M moves) took 2.94 h. The whole evaluation took 3.51 h of the 8-hour budget.
- **The projection was optimistic but inside its range.** The seed-`2024` arm had 2 game overs and gave 6.33M moves per game over (95% range 1.75M–52M); seed `777` gives 3.77M from 63 game overs. The mean is 366M against a projected 495M [172M, 897M].

**Verdict:** 365.8M, with 3.5 of the 8 hours used. Survival still sets the mean, and the value function is the bottleneck:
- In 28 of the 63 game overs, the best Q1 was still ≥ 92k three moves before the end: the value saw the danger only 1–2 moves ahead.
  - *Correction:* this said that the deep tier never fired for those. It did: the best Q1 was below 85.2k on the last move of all 63 game overs, so the tier ran on the last move of every one, and on the last two in 22 of the 28 ([Yardstick](#yardstick-ticket-10-goal-mean-1b)).
- Points per move (98.4) are fixed by k = 2 and stay below `rl_v2`'s 117.

### Next methods, in order

1. **Spend the unused compute on wider search.** At seed `2024`, k = 4 on every move had 1 game over in 12.6M moves and 101 points per move, against k = 2's 6 game overs and 98.4.
   - At the 10M cap, k = 4 projects to about 10 h. The 8-hour budget is set by the longest games, which run on one thread each.
   - So this needs either k = 3, or searching one game on several threads once fewer games than threads are left.
2. **A value function that sees danger.** Almost half the game overs give no warning in Q1 three moves ahead, so wider search alone cannot catch them.
   - Options: training on search-backed targets, and the temporal-coherence step sizes from the n-tuple learner's list.
3. **Points per move:** tuple features that represent multi-line setups, as before. `rl_v2` gets 117 per move.
4. **The 10M cap now binds.** At this survival, about 7% of games reach it, worth about 985M points each. Comparing means at higher survival will need a higher cap, and more compute.

## Yardstick (ticket #10, goal: mean 1B)

The next goal is a mean of 1B. At the acceptance's survival, a game over in free play costs about 4M moves, about 4 minutes of the whole machine. The precision depends only on the number of game overs D: the 95% interval on moves per game over is about ×/÷ e^(2/√D). The acceptance's 63 game overs took 3.5 hours.

**Both means.** Every mean is reported twice:
- with the simulator's full scores;
- with each game clamped at 2³¹ − 1 = 2,147,483,647, because the game stores the score as a 32-bit int (`docs/RULES.md`, offset 0x4dc).

Clamping the score is the same as capping the game at INT32_MAX / points per move: 21.8M moves at 98.4 points per move. At 98.4 points per move, a mean of 1B needs about 10.2M moves per game over with full scores, and about 12.2M with clamped scores.

The engine computes the spawn level in 64 bits, so it keeps dealing hard pieces after the score passes 2³¹. In the game, the score would wrap negative there, and a negative level allows only groups 0–8 (inferred from `randTiles`, not observed).

### Short capped games (`block_puzzle_ai.yardstick`)

**Method.** With a constant risk per move, game lengths are exponential with mean m, the moves per game over:
- The mean is points per move × m without a cap.
- It is points per move × m · (1 − e^(−cap / m)) with games stopped at a cap.

Many short games stopped at a cap estimate m as well as a few long games with the same number of game overs, and no single game sets the wall time.

**Validation** (2026-11-06, 15:41–17:04, `runs/yard/run.sh accept`): the acceptance config on 96 new games, stopped at 1M moves.

```bash
uv run python -m block_puzzle_ai.ntuple evaluate --weights models/ntuple_v1.ntw --games 96 --seed 2027 --max-moves 1000000 \
  --threads 24 --search-k 2 --deep-below 92000 --deep-k 4,2 --games-file runs/yard/accept.jsonl --json > runs/yard/accept.json
uv run python -m block_puzzle_ai.yardstick runs/yard/accept.json runs/nts_accept/seed777_10M.json --cap 10000000
```

| Run | Game overs | Moves | Moves / game over | Mean at the 10M cap, projected (full / clamped) |
|---|---:|---:|---:|---|
| 96 games, 1M cap, seed `2027` | 11 | 90.3M | 8.21M [4.59M, 16.44M] | 569.0M [400.5M, 737.5M] / 569.0M |
| Acceptance: 64 games, 10M cap, seed `777` | 63 | 237.8M | 3.77M [2.95M, 4.91M] | 345.3M [280.6M, 420.4M] / 345.3M (observed 365.8M / 365.8M) |

- **The validation failed.** The acceptance's death rate is 2.17× [1.15, 4.13] the 1M-cap run's (p ≈ 0.02).
- **Two explanations:**
  - Chance.
  - A risk that grows with game age. Within the acceptance, games up to 1M moves old had 4.80M [2.75M, 9.29M] moves per game over, and older games 3.53M [2.69M, 4.75M]; the late / early risk ratio is 1.33 [0.81, 2.20].
- **Spawns cannot explain a rising risk:** past a score of 12M (about 120k moves), every draw of r ≥ 1 allows all 12 groups.
- **Chance is the likelier explanation.** The split yardstick below, run on games 150k–550k moves old, gives the acceptance's value.
- **Not used to choose arms.** Even without a bias, 11 game overs give ×/÷ 1.8, too coarse to rank k = 3 or 4 against k = 2. The tool remains for reporting: both means and the risk by game age.

### Split at danger (`scripts/split_yardstick.py`, `scripts/split_estimate.py`)

**Every game over ends a visit to danger.** In all 74 game overs of the two runs, the best Q1 was below 85.2k on the last move, so the deep tier (Q1 < 92k) fired on every one. Hence

```
death rate = e1 · q1 · p2
  e1  entries into Q1 < 92k per move                                     free play
  q1  P(Q1 falls below 85k before it is back at ≥ 92k | entry)           restarts from the entry states
  p2  P(game over before Q1 is back at ≥ 92k | first move below 85k)     restarts from the states where Q1 fell below 85k
```

- **Restarts** start at saved states with new spawns. The state (board, tray, score and history) is all that the future depends on.
- **Comparisons are paired.** Spawns depend only on the RNG and the groups already spawned, not on the moves, and the spawn level is fixed past a score of 12M. So restarts with the same seed face the same pieces under every arm.

**Runs** (acceptance config, 24 worker processes, 2026-11-06, `runs/yard/split1`):

| Step | What | Result | Time |
|---|---|---|---:|
| Free play, seed 31 | 24 games: 150k moves to warm up, then 400k counted moves | 8,879 entries in 9.6M moves: e1 = 9.25e-4 [8.98e-4, 9.52e-4] per move. Deep search on 0.19% of moves, as in the acceptance | 16 min |
| One stage, seed 31 | 40 restarts per entry, until Q1 ≥ 92k or a game over | 92 game overs in 355,160 restarts: 2.59e-4 [1.92e-4, 3.35e-4] per entry; 2.0 moves per restart | 32 min |
| Capture, seed 32 | 20 restarts per entry, stopped at the first move below 85k | 987 captured states: q1 = 0.556% [0.462%, 0.657%]. No game over without falling below 85k | 16.5 min |
| Stage 2, seed 41 | 10 restarts per captured state | 361 game overs: p2 = 3.66% [2.93%, 4.44%]; 4.2 moves per restart | 1.6 min |

**Validation:**

| Estimate | Moves / game over | Mean, 10M cap (full / clamped) | Mean, no cap (full / clamped) |
|---|---:|---|---|
| Free play, acceptance (63 game overs) | 3.77M [2.95M, 4.91M] | 365.8M / 365.8M (observed) | |
| Free play, both runs (74 game overs) | 4.43M [3.53M, 5.65M] | | |
| Split, one stage | 4.17M [3.22M, 5.69M] | 373.4M [302.4M, 463.4M] / 373.4M | 410.8M [316.6M, 559.9M] / 408.6M [316.2M, 547.8M] |
| Split, two stages | 5.32M [4.23M, 6.91M] | 443.6M [377.2M, 520.0M] / 443.6M | 523.5M [416.3M, 679.9M] / 514.8M [414.0M, 651.0M] |

- **Both split estimates agree with free play.**
- **The two-stage estimate is about 6× cheaper.** Its interval matches about 64 free-play game overs, the acceptance's precision, and it took 34 minutes against 3.5 hours.
- **All the risk is below 85k.**
  - 353,222 of the one-stage restarts never fell below 85k, and none of them ended the game.
  - All 92 game overs came from the 1,938 restarts (0.55%) that did.
  - The capture run agrees: 0 game overs in 177,580 restarts without first falling below 85k.

### Search below 85k

Stage 2 for four arms, which differ only below 85k. All four use the same 987 captured states and seeds (`runs/yard/stage2.sh`, `runs/yard/split1/roll_41_s2_*.json`):

| Arm | Widths below 85k | One search (1 thread) | p2 | Death rate vs `accept` (paired) | Moves / game over | Mean, 10M cap (full / clamped) | Extra cost per move |
|---|---|---:|---:|---:|---:|---|---:|
| `accept` | [4, 2] | 12 ms | 3.66% | 1 | 5.32M | 443.6M / 443.6M | |
| `w8` | [8, 4] | 31 ms | 3.56% | 0.972 [0.934, 1.010] | 5.47M | 451.9M / 451.9M | +1 µs |
| `d4` | [4, 2, 2] | 0.65 s | 3.56% | 0.972 [0.946, 0.996] | 5.47M | 451.9M / 451.9M | +19 µs |
| `w8d4` | [8, 4, 2] | 1.84 s | 3.45% | 0.945 [0.905, 0.983] | 5.63M | 460.4M / 460.4M | +47 µs |

- **Below 85k, search barely matters.** About 3.5% of the visits that fall below 85k end the game whatever the search does: 150× more search there removes only 5.5% of the game overs. Those game overs are mostly forced by the pieces that come.
- **The lever is falling below 85k less often.** That happens at e1 · q1 = 5.1e-6 per move, once per 195k moves.
  - A clamped 1B mean needs 12.2M moves per game over: 2.3× fewer such falls at the same p2 if the two-stage 5.32M is right, and 3.2× fewer if the acceptance's 3.77M is.
  - What can lower the rate: the search on every move, which changes e1; the search between 85k and 92k, which changes q1 (all four arms use [4, 2] there); and the value function.
- **The paired design resolves effects of about 5%.** With free play, that would take thousands of game overs.

### Search on every move

The split yardstick for k = 3 and k = 4 on every move, each with [4, 2] below 92k (2026-11-06, 19:58–21:09; `runs/yard/karms.sh`, `runs/yard/split_k3`, `runs/yard/split_k4`).
- Free play uses the same worker seeds as k = 2, so all three arms see the same pieces. The games still diverge, so these comparisons are unpaired (independent bootstraps, `runs/yard/compare_dirs.py`).
- The cost is per move at tier 0, with 24 worker processes.

| k | Points / move | e1 per move | q1 | p2 | Moves / game over | Death rate vs k = 2 | Mean, 10M cap (full / clamped) | Mean, no cap (full / clamped) | Cost / move |
|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| 2 | 98.43 | 9.25e-4 | 0.556% | 3.66% | 5.32M [4.23M, 6.91M] | 1 | 443.6M / 443.6M | 523.5M / 514.8M | 1.5 ms |
| 3 | 100.32 | 4.70e-4 | 0.762% | 4.19% | 6.67M [4.29M, 11.32M] | 0.80 [0.47, 1.32] | 519.6M / 519.6M | 668.8M / 641.8M | 2.0 ms |
| **4** | **101.34** | 3.29e-4 | 0.673% | 3.97% | **11.36M [8.08M, 17.23M]** | **0.47 [0.30, 0.71]** | 673.9M / 673.9M | **1,151.3M [818.4M, 1,746.5M] / 973.0M [759.1M, 1,235.8M]** | 2.9 ms |

- **k = 4 halves the death rate.** It enters danger 64% less often. The visits it does make are a little more severe: q1 is 21% higher and p2 8% higher.
- **k = 3 also enters danger half as often**, but its visits are more severe, and its net effect is not significant.
- **k = 4 projects to 1.15B with full scores, and 973M [759M, 1,236M] clamped.** A clamped 1B at 101.3 points per move needs 11.8M moves per game over, inside k = 4's interval but above its point estimate.
- **Check against free play.** The free-play phases had 2, 1 and 3 game overs, where the split predicts 1.8, 1.4 and 0.84. For k = 4's 3, P(≥ 3) = 0.054.

**Cost of a 1B acceptance with k = 4** (64 games, at the seed-`2024` cost of 1,949 µs per move per thread on 24 threads):

| Cap | Moves / game | Moves, 64 games | Mean (full / clamped) | 24 threads | Longest game, 1 thread |
|---:|---:|---:|---|---:|---:|
| 22M | 9.72M | 622M | 985M / 973M | 14.0 h | 11.9 h |
| 25M | 10.10M | 647M | 1,024M / 973M | 14.6 h | 13.5 h |
| 50M | 11.22M | 718M | 1,137M / 973M | 16.2 h | 27.1 h |

- **A cap of about 22M moves costs the clamped mean nothing.** Every point past 21.2M moves is clamped anyway.
- **Fitting this in 8 hours needs two things:**
  - a k = 4 search about 1.8× faster;
  - splitting one game's search across idle threads, about 5 threads for the last game.

### Search below 92k on top of k = 4

Two arms, each restarted from k = 4's 3,163 entry states with the same seeds and repetitions (seed `32`, 20 each) as the k = 4 capture run (2026-11-06, 21:12–22:53; `runs/yard/danger_arms.sh`, `runs/yard/split_k4`). Each arm uses its own widths both between 85k and 92k and below 85k.
- **q1 is paired**: the same restart sees the same pieces in both arms.
- **p2 is not paired**, because each arm falls below 85k in different states.
- Extra cost is per move, in thread time at 24 workers, against k = 4's 1,949 µs.

| Below 92k | q1 | q1 vs k = 4 (paired) | p2 | Moves / game over | Death rate vs k = 4 | Mean, no cap (full / clamped) | Extra cost / move |
|---|---:|---:|---:|---:|---:|---|---:|
| [4, 2] (k = 4) | 0.673% | 1 | 3.97% | 11.36M [8.08M, 17.23M] | 1 | 1,151.3M / 973.0M [759.1M, 1,235.8M] | 0 |
| **[8, 4]** | 0.571% | **0.847 [0.753, 0.938]** | 3.80% | **14.01M [9.11M, 24.19M]** | 0.81 [0.60, 1.02] | **1,420.2M / 1,107.1M [833.0M, 1,430.5M]** | **+129 µs (+6.6%)** |
| [16, 8] | 0.575% | 0.854 [0.731, 0.986] | 3.43% | 15.36M [8.91M, 30.72M] | 0.74 [0.48, 1.02] | 1,556.6M / 1,164.8M [819.4M, 1,551.4M] | +454 µs (+23%) |

- **[8, 4] between 85k and 92k falls below 85k 15% less often.** Of the restarts that fell below 85k in either arm, 137 did only under [4, 2] and 72 only under [8, 4].
- **[16, 8] adds nothing on q1 over [8, 4]** (1.008 [0.895, 1.144], paired), at 3.5× the cost. Its lower p2 is within the noise of unpaired states.
- **With [8, 4], the clamped mean projects to 1.11B** (no cap), with an interval that still reaches down to 833M.
- **The acceptance config becomes** `--search-k 4 --deep-below 92000 --deep-k 8,4`.

**Cost of a 1B acceptance with k = 4 and [8, 4] below 92k** (64 games, at 2,077 µs per move per thread on 24 threads):

| Cap | Moves / game | Moves, 64 games | Mean (full / clamped) | 24 threads | Longest game, 1 thread |
|---:|---:|---:|---|---:|---:|
| 22M | 11.10M | 710M | 1,124M / 1,107M | 17.1 h | 12.7 h |
| 25M | 11.66M | 746M | 1,181M / 1,107M | 17.9 h | 14.4 h |

- **One night needs a search about 2.1× faster**, plus about 2 threads for each of the last few games.

### Speed

Profile of the k = 4 search before the changes (one thread, `perf`):
- **Legal-placement counts** (`legal_positions` for all 28 pieces of every evaluated board): about 33%.
- **Tuple lookups**: about 35%.
- **The 8 board symmetries**: about 16%.

95% of the evaluated boards are a parent board plus one piece, with no line cleared. `NTuple::child_eval` evaluates them from the parent's lookups:
- The piece's images under the symmetries are translations of precomputed images.
- Only the lookups that read the piece's cells are redone, about 34 of 120.
- Each piece's legal positions lose one precomputed mask, shifted.

The sum keeps `board_eval`'s order, so values are bit for bit the same: the golden values pass, and all benchmark scores are unchanged.

Benchmark: `uv run python -m block_puzzle_ai.ntuple evaluate --weights models/ntuple_v1.ntw --games G --seed 2024 --max-moves 4000 --threads T --search-k 4 --deep-below 92000 --deep-k 8,4 --json`. Times are per move per thread, one game per thread, ±3% between runs.

| Version | 1 thread | 12 threads | 24 threads | Moves per second on 24 threads |
|---|---:|---:|---:|---:|
| Before | 1,004 µs | 1,041 µs | 2,098 µs | 1 |
| **Children from the parent's lookups** | 785 µs | 829 µs | **1,276 µs** | **1.65×** |

- **Before, SMT added nothing**: 24 threads ran at half the per-thread speed of 12. Now the longest chain is the in-order sum of the 120 weights, and the second hyperthread fills its gaps.
- **Idle threads join the remaining games** (`Helpers` in `cpp/search.hpp`). A searched move runs as tasks:
  - first the part each kept root move shares;
  - then one task per (kept move, spawned piece).

  The results are the same on any number of threads. One game takes 865 µs per move on 1 thread, 258 µs on 4, and 168 µs on 12 or 24. Four games on 24 threads take 263 µs per move each.
- **Projected acceptance time** (64 games, cap 22M, m = 14.0M; a simulation with these rates): **10.9 h [9.8, 12.1]**, against 12.6 h with one thread per game and 19.3 h before.

Tried and dropped:
- **A sum per symmetry**: 8 accumulators instead of one chain. It is +9% on 24 threads (1.30× on one), but it changes values in the last bits and breaks the golden values.
- **Vectorized counts, and weights kept as doubles**: no gain beyond the noise.
- **Pruning placements by an upper bound** (the largest weight over all supersets of each lookup's pattern): it prunes 0.6% of placements, so the bound is far too loose.
- **Searching only root moves within δ of the best Q1.** The moves k = 4 picks are at most 109 below the best Q1 on 99% of moves (k = 4 changes the 1-ply move on 29% of moves). With δ = 200, the search keeps 3.87 of 4 moves on average.
- **Incremental evaluation of line clears**: 4.6% of children clear a line, worth at most about 6%. Not done.

### 1B acceptance

Command (2026-11-06 23:25 – 2026-11-07 12:55; 24 threads, no `tfeval` running at launch):

```bash
uv run python -m block_puzzle_ai.ntuple evaluate --weights models/ntuple_v1.ntw --games 64 --seed 777 --max-moves 22000000 \
  --threads 24 --search-k 4 --deep-below 92000 --deep-k 8,4 --games-file runs/accept1b/games.jsonl --json
uv run python -m block_puzzle_ai.yardstick runs/accept1b/games.jsonl
```

Seed `777`, 64 games. "Clamped" counts each game at most 2³¹ − 1 = 2,147,483,647:

| Search | Cap | Mean | Mean, clamped | Median | Min | Max | Moves / game over | Points / move | Alive at cap | Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| k = 2, [4, 2] below 92k (100M acceptance) | 10M | 365,777,118 | 365,777,118 | 320,815,295 | 5,749,560 | 984,183,190 | 3,774,622 | 98.4 | 1 | 3.51 h |
| **k = 4, [8, 4] below 92k** | 22M | **1,164,852,934** | **1,149,503,618** | **1,070,481,810** | 46,680,120 | **2,229,751,750** | **14,148,484** | **101.3** | 12 | 13.5 h |

- **The 1B goal is reached with both means.**
  - Full scores: 1,164.9M, bootstrap 95% CI [985.7M, 1,345.2M].
  - Clamped: 1,149.5M [975.4M, 1,324.2M].
  - Both intervals reach just below 1B: 3.5% of the bootstrap resamples have a full mean below 1B, and 4.6% a clamped mean below 1B.
- **35 of the 64 games scored over 1B, and 61 over 100M.**
  - The 12 games alive at the 22M cap had all passed 2³¹, so the cap costs the clamped mean nothing.
  - The high score is 2,229,751,750 (game 43, alive at the cap). The game itself would show at most 2,147,483,647.
- **The death rate is 0.27 [0.18, 0.39] of the 100M acceptance's**: 52 game overs in 735.7M moves, against 63 in 237.8M.
- **The split yardstick's projection held.**
  - It projected 14.01M [9.11M, 24.19M] moves per game over; free play gave 14.15M [10.8M, 18.9M].
  - At the 22M cap it projected 1,124M / 1,107M (full / clamped); the observed means are 1,165M / 1,150M.
  - The risk does not grow with game age: the late half of the moves has 1.17 [0.68, 2.01] times the early half's death rate.
- **Search:** k = 4 changed the 1-ply move on 28.7% of moves. The deep tier ran on 441,666 moves (0.060%) and changed the move on 58.6% of them.
- **Time: 13.5 h, against the projected 10.9 h [9.8, 12.1].**
  - The run used 1,586 µs of thread time per move, against 1,276 µs in the speed benchmark.
  - The benchmark runs for seconds. Over hours, the laptop held all cores at about 3.2 GHz and 87 °C.
  - The longest game took 9.9 h.

### Next methods, in order

The 1B goal is met. If the goal moves higher:
1. **The value function.** Danger shows only 1–2 moves ahead, and below 85k about 3.5% of visits are lost whatever the search does. The 8-hour training run found its best weights after 62 minutes and did not improve over the remaining 6.5 hours, so stable training (temporal-coherence step sizes) comes first, then search-backed targets.
2. **Points per move.** `rl_v2` gets 117 per move, against 101.3 here.
3. **The clamp sets a ceiling.** The clamped mean cannot exceed 2,147.5M, and a game past about 21.2M moves counts as 2,147.5M however long it goes on. A clamped 1.5B would need about 28M moves per game over, twice today's.

## Value function (ticket #10)

### Temporal-coherence pilot

**Method.** Temporal coherence (Beal & Smith 1999; Jaśkowski 2018 for 2048 n-tuple networks) gives every weight its own step size:
- E sums the steps the weight was asked to take, and A their sizes.
- Each step is scaled by |E| / A: 1 while the steps agree in sign, and towards 0 while they cancel.

`ntuple train --tc` keeps (E, A) as doubles per weight and saves them in checkpoints. `--init` starts a new run from a checkpoint's weights.

**Runs** (2026-11-07, 15:26–16:58, `runs/tc_pilot/run.sh`). Both arms fine-tuned `models/ntuple_v1.ntw` for 1.5 h, side by side on 12 threads each. Both used the 24-thread runs' step-size cap of 0.0208 and the collapse guard. The evaluation is the periodic one (256 games, seed `2024`, 200,000-move cap); on it, `ntuple_v1` gave 70,785 moves per game over and 94.0 points per move.

```bash
common="--init models/ntuple_v1.ntw --hours 1.5 --threads 12 --alpha 0.0208 --eval-minutes 15"
uv run python -m block_puzzle_ai.ntuple train --out runs/tc_pilot/plain $common
uv run python -m block_puzzle_ai.ntuple train --out runs/tc_pilot/tc $common --tc
```

| Arm | Rollbacks | Cap at the end | Moves | Moves / game over at 17 / 34 / 52 / 68 / 85 / 92 min | Mean of the 6 | Points / move |
|---|---:|---:|---:|---|---:|---:|
| Plain TD | 3 (at 38M, 69M and 95M moves) | 0.0026 | 647M | 84.4k / 80.4k / 88.6k / 73.0k / 75.9k / 86.8k | 81.5k | 91.8–94.6 |
| **TC** | **0** | 0.0208 | 633M | 82.7k / 78.3k / 83.8k / 79.2k / 73.4k / 77.3k | 79.1k | 93.6–94.5 |

- **TC keeps training stable at the full step size.**
  - Plain TD collapsed three times in its first 16 minutes. Each time the guard restored older weights and halved the cap, from 0.0208 to 0.0026.
  - TC ran all 90 minutes at 0.0208 without a rollback.
- **It trains no better.** The two arms are level, and both are about 10% above `ntuple_v1`. One evaluation's interval is ×/÷ 1.14, and neither arm's points per move moved.
- **Fine-tuning has reached what these tuples can represent,** so step sizes are not the limit. The next lever is the target: train the value towards what the search finds.

### Search-backed targets pilot

**Method** (`docs/superpowers/specs/2026-11-07-search-targets-design.md`). At every state it reaches, the trainer runs the depth-2 search on the top 2 moves (`--target-k 2`) and trains towards its scores:
- the previous after-state towards the best search score, which looks one spawn further ahead than a 1-ply target;
- each searched move's after-state towards its exact expectation over the next spawn.

Arm S plays the 1-ply move. Arm SP plays the searched move (`--play-search`), as the deployed agent does. Both fine-tuned `models/ntuple_v1.ntw` with TC and the TC pilot's settings, side by side for 2 h on 12 threads each (2026-11-07, 17:27–19:28, `runs/st_pilot/run.sh`). The control is the TC pilot's `tc` arm: the same fine-tune with 1-ply targets.

```bash
common="--init models/ntuple_v1.ntw --hours 2 --threads 12 --alpha 0.0208 --eval-minutes 15 --tc --target-k 2"
uv run python -m block_puzzle_ai.ntuple train --out runs/st_pilot/S $common
uv run python -m block_puzzle_ai.ntuple train --out runs/st_pilot/SP $common --play-search
```

**Training.** The deployed agent searches, so the periodic 1-ply evaluation is only a health check.

| Arm | Moves | Moves / s | Game overs in training | Rollbacks | 1-ply moves / game over, every ~15 min | Mean | Points / move |
|---|---:|---:|---:|---:|---|---:|---:|
| Control (1.5 h) | 633M | 132k | 8,103 | 0 | 82.7k / 78.3k / 83.8k / 79.2k / 73.4k / 77.3k | 79.1k | 93.6–94.5 |
| S | 124M | 18k | 5,074 | 0 | 50.9k / 40.8k / 28.7k / 11.1k / 52.1k / 58.2k / 44.3k / 36.8k | 40.4k | 81.9–94.7 |
| SP | 120M | 17k | 406 | 0 | 26.7k / 29.0k / 26.3k / 24.9k / 21.3k / 21.6k / 21.1k / 21.6k | 24.1k | 94.7–99.8 |

- Search targets train at about 1/7 of the moves per second.
- **Both arms got worse at 1-ply.** S lost half of the control's moves per game over, SP 70%, and SP kept falling.
- SP's training had 12× fewer game overs than S's in about as many moves, so it also had 12× fewer restarts near game overs.

**The fine-tunes shift Q1.** On the same states, each fine-tune's best Q1 is `ntuple_v1`'s plus a nearly constant amount (`runs/st_pilot/q1_shift.py`: 1,000 states from `ntuple_v1`'s 1-ply play, and 1,000 of its 92k entry states from the k = 4 free play).

| Weights | Shift on ordinary states: median [p10, p90] | Shift on 92k entry states: median [p10, p90] |
|---|---|---|
| Control | +1,236 [+1,207, +1,272] | +1,315 [+1,245, +1,403] |
| S | +4,732 [+4,674, +4,804] | +4,880 [+4,368, +5,048] |
| SP | +12,952 [+12,885, +13,030] | +12,947 [+12,724, +13,858] |

- A constant barely changes the moves: all moves' Q1 and search scores move together, and only children with no legal move, which stay at 0, look worse by it.
- **It moves the danger tiers.** `ntuple_v1`'s median Q1 on ordinary states is 92,581, only 581 above the 92k tier. Shifted by 1.3k–12.9k, the fine-tunes entered Q1 < 92k 15–90 times less often, so the deep search ran only in deeper danger.
- So the spec's assumption that the thresholds keep their meaning across fine-tunes was wrong. `split_yardstick.py --shift D` adds D to every tier threshold. Each arm was measured again with D = its median shift on the entry states (2026-11-08, 00:23–03:09, `runs/st_pilot/calib.sh`).

**Yardstick** (k = 4, [8, 4] below 92k; free play seed `31`, capture `32`, stage 2 `41`; `runs/st_pilot/yard_*` with fixed tiers, `yardc_*` shifted).
- Different weights play different games, so the ratios are unpaired: independent bootstraps (`runs/st_pilot/st_compare.py`).
- The `ntuple_v1` row is the acceptance config's split estimate from "Search below 92k on top of k = 4".
- q1 here is the chance of falling below 85k. With fixed tiers, S and SP also had game overs without falling below 85k (0.79% and 5.3% of restarts), which the moves per game over include.

| Weights | Tiers | Points / move | e1 per move | q1 | p2 | Moves / game over | Death rate vs control | Mean, no cap (full / clamped) |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `ntuple_v1` | fixed | 101.3 | 3.29e-4 | 0.571% | 3.80% | 14.01M [9.11M, 24.19M] | | 1,420.2M / 1,107.1M |
| Control | fixed | 102.0 | 2.20e-5 | 8.13% | 12.1% | 4.64M [2.76M, 9.79M] | 1 | 473.2M / 468.1M |
| S | fixed | 100.9 | 8.54e-6 | 17.9% | 7.61% | 5.44M [3.74M, 8.93M] | 0.85 [0.42, 1.99] | 548.7M / 537.7M |
| SP | fixed | 108.3 | 3.75e-6 | 55.4% | 23.6% | 1.45M [0.99M, 2.38M] | 3.20 [1.55, 7.50] | 157.2M / 157.2M |
| Control | +1,315 | 102.1 | 3.96e-4 | 0.430% | 4.01% | 14.66M [8.26M, 29.96M] | 1 | 1,495.7M / 1,139.8M |
| S | +4,880 | 100.9 | 1.47e-3 | 0.234% | 3.39% | 8.53M [6.58M, 11.53M] | **1.72 [0.92, 3.64]** | 861.1M / 790.0M |
| SP | +12,947 | 108.3 | 5.53e-4 | 0.940% | 8.84% | 2.00M [1.62M, 2.54M] | **7.34 [4.05, 15.38]** | 216.4M / 216.4M |

- **Search-backed targets don't help the searched agent.** With shifted tiers, S's death rate is 1.72 [0.92, 3.64] times the control's and SP's 7.34 [4.05, 15.38] times. Neither meets the bar, and SP is clearly worse.
  - S's shifted 92k tier fired 3.7 times as often as the control's, so S got more deep search and still died more often.
- **SP trades safety for points.** It learns the value of its own searched play: 108.3 points per move against the control's 102.1 (+6%), at 7 times the death rate. A likely cause is its training, which rarely reached game overs.
- **1-ply gains don't carry over to the search.** The control, about 10% better than `ntuple_v1` at 1-ply, is level with it under the deployed search: death rate 0.96 [0.41, 2.05].
- **Comparing weights needs shifted tiers.** With fixed tiers the control looked 3.02 [1.24, 6.72] times worse than `ntuple_v1`. Comparisons between weights now use `--shift` with the median Q1 shift on `ntuple_v1`'s 92k entry states.
- Check against free play: with shifted tiers, the three arms had 2, 1 and 6 game overs, where the split predicts 0.66, 1.12 and 4.80.
- **Neither the step size (TC pilot) nor the target moved the searched agent's death rate.** Fine-tuning these tuples has stopped paying off. What remains for the value is the representation: more or larger tuples.
