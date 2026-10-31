"""Bootstrap 95% CI of the mean-score difference between two `evaluate --json` runs (B minus A)."""

import json
import random
import statistics
import sys

a, b = ([r["score"] for r in json.load(open(path))["results"]] for path in sys.argv[1:3])
rng = random.Random(0)
diffs = sorted(
    statistics.fmean(rng.choices(b, k=len(b))) - statistics.fmean(rng.choices(a, k=len(a))) for _ in range(20_000)
)
print(f"A mean {statistics.fmean(a):,.0f}  B mean {statistics.fmean(b):,.0f}")
print(f"B - A {statistics.fmean(b) - statistics.fmean(a):,.0f}  95% CI [{diffs[500]:,.0f}, {diffs[19_499]:,.0f}]")
