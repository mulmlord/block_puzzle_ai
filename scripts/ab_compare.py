"""Compare A/B arms on death rate (deaths per move), with a Poisson rate-ratio 95% CI against the first arm."""

import json
import math
import sys

arms = [json.loads(line) for line in open(sys.argv[1])]
base = arms[0]
base_moves = base["moves_per_death"] * base["deaths"]
print(f"{'arm':<16}{'deaths':>7}{'moves/death':>13}{'mean':>10}{'median':>10}{'ratio':>8}{'95% CI':>16}{'secs':>7}")
for arm in arms:
    moves = arm["moves_per_death"] * arm["deaths"]
    ratio = (arm["deaths"] / moves) / (base["deaths"] / base_moves)
    se = math.sqrt(1 / arm["deaths"] + 1 / base["deaths"])
    lo, hi = ratio * math.exp(-1.96 * se), ratio * math.exp(1.96 * se)
    print(
        f"{arm['arm']:<16}{arm['deaths']:>7}{arm['moves_per_death']:>13.0f}{arm['mean']:>10.0f}{arm['median']:>10.0f}"
        f"{ratio:>8.2f}{f'[{lo:.2f}, {hi:.2f}]':>16}{arm['seconds']:>7}"
    )
