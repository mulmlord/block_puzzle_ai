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
