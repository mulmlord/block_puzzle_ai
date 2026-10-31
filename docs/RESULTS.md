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
