import sys
import numpy as np

d = np.load(sys.argv[1])
thresholds = [float(t) for t in sys.argv[2].split(",")] if len(sys.argv) > 2 else [40, 60, 80, 100]
died = d["died"]
games = [d[f"g{i}"] for i in range(len(died))]
allq = np.concatenate([g[:, 0] for g in games])
fill = np.concatenate([g[:, 2] for g in games])
print(f"moves {len(allq):,}  deaths {died.sum()}  best-Q pct: " + " ".join(f"p{p}={np.percentile(allq, p):.1f}" for p in (0.1, 1, 2, 5, 10, 50)))
# Q at k moves before the final (terminal) state, for games that died
for k in (1, 2, 3, 5, 10, 20, 50):
    qk = np.array([g[-k, 0] for g, dd in zip(games, died) if dd and len(g) >= k])
    print(f"{k:3d} moves before death: median Q {np.median(qk):6.1f}  " + "  ".join(
        f"<{thr}: {np.mean(qk < thr):5.1%}" for thr in thresholds))
print("share of all moves below threshold:", "  ".join(f"<{thr}: {np.mean(allq < thr):6.2%}" for thr in thresholds))
# minimum Q over the last 20 moves of dying games: did the trajectory pass through a flagged state?
for w in (5, 10, 20, 50):
    mins = np.array([g[-w:, 0].min() for g, dd in zip(games, died) if dd])
    print(f"min Q over last {w:2d} moves: " + "  ".join(f"<{thr}: {np.mean(mins < thr):5.1%}" for thr in thresholds))
print("filled cells overall p50/p90/p99:", np.percentile(fill, [50, 90, 99]))
