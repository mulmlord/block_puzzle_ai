"""Moves per game over and projected means from split_yardstick.py runs (method in its docstring).

    uv run python scripts/split_estimate.py DIR ROLL                    # one stage: ROLL ran without --capture
    uv run python scripts/split_estimate.py DIR CAPTURE S2_A [S2_B ...] # two stages: CAPTURE ran with --capture,
                                                                        # each S2 restarts from its captured states
        [--arm CAPTURE_X S2_X ...]

S2 arms must differ only below 85k, so they share e1 and q1 and their death-rate ratios are p2 ratios, paired by state
and seed; the first S2 file is the baseline. An --arm differs below 92k too: its own capture run from the same entry
states (same --seed and --reps as CAPTURE, so q1 is paired) and its own stage 2; e1 is shared. Every mean is reported with full scores and with each game
clamped at INT32_MAX (the game's 32-bit score). Intervals: a bootstrap over free-play workers for e1, times a cluster
bootstrap over the 92k entry states for q1 and p2 jointly.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from block_puzzle_ai.yardstick import INT32_MAX, expected_capped

B = 4000
ACCEPT_PPM = 98.43  # points per move of the acceptance config, for free-play files from before points were recorded


def fmt(x: float) -> str:
    return f"{x / 1e6:,.2f}M"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, help="directory with free.json")
    ap.add_argument("roll", type=Path, help="restarts from the free-play entries")
    ap.add_argument("stage2", type=Path, nargs="*", help="restarts from the states captured in ROLL")
    ap.add_argument("--cap", type=float, default=1e7, help="also project the mean at this move cap")
    ap.add_argument("--arm", nargs=2, type=Path, action="append", default=[], metavar=("CAPTURE", "S2"),
                    help="another arm from the same free play: its capture run and its stage 2")
    a = ap.parse_args(argv)
    rng = np.random.default_rng(0)

    free = json.loads((a.dir / "free.json").read_text())
    moves_w = np.array([r["counted"] for r in free], float)
    entries_w = np.array([len(r["entries"]) for r in free], float)
    e1 = entries_w.sum() / moves_w.sum()
    wb = rng.integers(0, len(free), (B, len(free)))
    e1_b = entries_w[wb].sum(1) / moves_w[wb].sum(1)
    ppm = sum(r["points"] for r in free) / moves_w.sum() if all("points" in r for r in free) else ACCEPT_PPM
    print(f"free play: {int(moves_w.sum()):,} moves, {ppm:.2f} points per move, e1 = {e1:.4e} per move "
          f"[{np.quantile(e1_b, .025):.4e}, {np.quantile(e1_b, .975):.4e}]")

    rows = json.loads(a.roll.read_text())
    n1 = len(rows)
    reps = np.array([len(r) for r in rows], float)
    dead = np.array([sum(x[0] == 1 for x in r) for r in rows], float)
    hits = np.array([sum(x[0] == 2 for x in r) for r in rows], float)
    sb = rng.integers(0, n1, (B, n1))
    w_b = np.stack([np.bincount(s, minlength=n1) for s in sb])  # draws of each entry state per bootstrap

    def report(name: str, lam: float, lam_b: np.ndarray) -> None:
        m = 1 / lam
        m_lo, m_hi = np.quantile(1 / lam_b, [0.025, 0.975])
        print(f"{name}: moves per game over {fmt(m)} [{fmt(m_lo)}, {fmt(m_hi)}]")
        for cap in (None, a.cap):
            c = np.inf if cap is None else cap
            full = [ppm * expected_capped(x, c) for x in (m, m_lo, m_hi)]
            clamp = [ppm * expected_capped(x, min(c, INT32_MAX / ppm)) for x in (m, m_lo, m_hi)]
            label = "no cap" if cap is None else f"{cap:,.0f}-move cap"
            print(f"    mean, {label}: {fmt(full[0])} [{fmt(full[1])}, {fmt(full[2])}], "
                  f"clamped at 2^31: {fmt(clamp[0])} [{fmt(clamp[1])}, {fmt(clamp[2])}]")

    if not a.stage2:
        if hits.any():
            sys.exit(f"{a.roll} ran with --capture: pass the stage-2 files too")
        p = dead.sum() / reps.sum()
        p_b = (w_b * dead).sum(1) / (w_b * reps).sum(1)
        print(f"one stage: P(game over | entry) = {p:.3e} [{np.quantile(p_b, .025):.3e}, {np.quantile(p_b, .975):.3e}]"
              f", {int(dead.sum())} game overs in {int(reps.sum()):,} restarts from {n1:,} entry states "
              f"({int((dead > 0).sum())} with a game over)")
        report("one stage", e1 * p, e1_b * p_b)
        return

    def stage1(rows: list) -> tuple:
        """q1 and the game overs before 85k from a capture run, with bootstraps on the shared draws w_b."""
        if len(rows) != n1:
            sys.exit(f"capture runs differ: {len(rows)} and {n1} entry states")
        reps = np.array([len(r) for r in rows], float)
        hits = np.array([sum(x[0] == 2 for x in r) for r in rows], float)
        dead = np.array([sum(x[0] == 1 for x in r) for r in rows], float)
        cluster = np.array([i for i, r in enumerate(rows) for x in r if x[0] == 2])  # entry state of each capture
        secs = sum(x[3] for r in rows for x in r) / reps.sum()
        return (hits.sum() / reps.sum(), (w_b * hits).sum(1) / (w_b * reps).sum(1), dead.sum() / reps.sum(),
                (w_b * dead).sum(1) / (w_b * reps).sum(1), cluster, secs, int(hits.sum()), int(dead.sum()), reps.sum())

    def stage2(path: Path, cluster: np.ndarray) -> tuple:
        s2 = json.loads(path.read_text())
        if len(s2) != len(cluster):
            sys.exit(f"{path}: {len(s2)} states, its capture run has {len(cluster)}")
        frac = np.array([sum(x[0] == 1 for x in r) / len(r) for r in s2])
        secs = sum(x[3] for r in s2 for x in r) / sum(len(r) for r in s2)
        wc_b = w_b[:, cluster]
        return frac.mean(), (wc_b * frac).sum(1) / wc_b.sum(1), secs

    def show(name: str, s1: tuple, s2: tuple) -> tuple:
        q1, q1_b, d0, d0_b, _, _, nh, nd, nr = s1
        p2, p2_b, secs = s2
        print(f"{name}: q1 = {q1:.4e} [{np.quantile(q1_b, .025):.4e}, {np.quantile(q1_b, .975):.4e}] ({nh} captures, "
              f"{nd} game overs in {int(nr):,} restarts), p2 = {p2:.4f} [{np.quantile(p2_b, .025):.4f}, "
              f"{np.quantile(p2_b, .975):.4f}], {secs:.2f} s per stage-2 restart")
        lam, lam_b = e1 * (d0 + q1 * p2), e1_b * (d0_b + q1_b * p2_b)
        report(f"    {name}", lam, lam_b)
        return lam, lam_b

    base1 = stage1(rows)
    base = None
    for path in a.stage2:
        s2 = stage2(path, base1[4])
        lam, lam_b = show(path.stem, base1, s2)
        if base is None:
            base = (path.stem, lam, lam_b, s2[2])
            continue
        r_b = lam_b / base[2]
        print(f"    death rate vs {base[0]}: {lam / base[1]:.3f} [{np.quantile(r_b, .025):.3f}, "
              f"{np.quantile(r_b, .975):.3f}] (paired), extra cost {e1 * base1[0] * (s2[2] - base[3]) * 1e6:+.2f} µs per move")
    for cap_path, s2_path in a.arm:
        s1 = stage1(json.loads(cap_path.read_text()))
        s2 = stage2(s2_path, s1[4])
        lam, lam_b = show(s2_path.stem, s1, s2)
        r_b = lam_b / base[2]
        extra = e1 * ((s1[5] - base1[5]) + s1[0] * s2[2] - base1[0] * base[3]) * 1e6
        q_b = s1[1] / base1[1]
        print(f"    q1 vs {base[0]}: {s1[0] / base1[0]:.3f} [{np.quantile(q_b, .025):.3f}, {np.quantile(q_b, .975):.3f}] "
              f"(paired)")
        print(f"    death rate vs {base[0]}: {lam / base[1]:.3f} [{np.quantile(r_b, .025):.3f}, "
              f"{np.quantile(r_b, .975):.3f}] (q1 paired), extra cost {extra:+.2f} µs per move")

if __name__ == "__main__":
    main()
