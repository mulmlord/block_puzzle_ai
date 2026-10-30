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
