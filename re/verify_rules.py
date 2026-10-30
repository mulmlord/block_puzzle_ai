"""Emulate JewelPuzzleGameLayer placement / line-detection / scoring code inside Unicorn and
compare against a Python reference model of the rules spec.

Run: uv run python re/verify_rules.py <libMyGame.so> [src/block_puzzle_ai/data/pieces.json]
"""
import json
import random
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm64_emu import Emulator  # noqa: E402
from unicorn.arm64_const import UC_ARM64_REG_X0, UC_ARM64_REG_X19, UC_ARM64_REG_SP  # noqa: E402

SO = sys.argv[1]
PIECES = sys.argv[2] if len(sys.argv) > 2 else str(
    Path(__file__).resolve().parents[1] / "src/block_puzzle_ai/data/pieces.json"
)
N = 10

MATCH = "_ZN20JewelPuzzleGameLayer20matchTileByRowAndColEP22JewelPuzzlePendingNodeii"
NOMATCH = "_ZN20JewelPuzzleGameLayer12noMatchTilesEP22JewelPuzzlePendingNode"
CHECK = "_ZN20JewelPuzzleGameLayer12checkToCrushEv"
CRUSH = "_ZN20JewelPuzzleGameLayer17crushPendingTilesEv"

crush_calls = []


def on_crush(e):
    lst_b, lst_e = e.u64(layer + 0x4B8), e.u64(layer + 0x4C0)
    ptrs = [e.u64(p) for p in range(lst_b, lst_e, 8)]
    lines = struct.unpack("<i", e.read(layer + 0x4E4, 4))[0]
    crush_calls.append((ptrs, lines))
    return None


dummy = lambda e: 0x5000_0000  # noqa: E731  (non-null fake cocos object)
emu = Emulator(
    SO,
    overrides={
        "_ZN22JewelPuzzleGameElement10setMyColorEi": lambda e: None,
        "_ZN22JewelPuzzleGameElement15setChildOpacityEf": lambda e: None,
        CRUSH: on_crush,
        "_ZN20JewelPuzzleGameLayer9animScoreERN7cocos2d4Vec2Eibb": lambda e: None,
        "_ZN7cocos2d9DelayTime6createEf": lambda e: 0,
        "_ZN7cocos2d8CallFunc6createERKNSt6__ndk18functionIFvvEEE": lambda e: 0,
        "_ZN7cocos2d8Sequence6createEPNS_16FiniteTimeActionEz": lambda e: 0,
    },
)
uc = emu.uc

# fake layer with a vtable whose every slot is a no-op stub (runAction etc.)
virt = emu._new_stub("layer::virtual")
emu.overrides["layer::virtual"] = lambda e: None
vtable = emu.malloc(8 * 256)
for i in range(256):
    uc.mem_write(vtable + 8 * i, struct.pack("<Q", virt))
layer = emu.malloc(0x900)
uc.mem_write(layer, struct.pack("<Q", vtable))
cells = [[emu.malloc(0x600) for _ in range(N)] for _ in range(N)]
for r in range(N):
    for c in range(N):
        uc.mem_write(layer + 0x510 + r * 0x50 + c * 8, struct.pack("<Q", cells[r][c]))
node = emu.malloc(0x400)
tile = emu.malloc(0x6C)
uc.mem_write(node + 0x318, struct.pack("<Q", tile))
uc.mem_write(layer + 0x458, struct.pack("<Q", node))


def set_board(board):
    for r in range(N):
        for c in range(N):
            uc.mem_write(cells[r][c] + 0x580, bytes([board[r][c], 0]))


def set_piece(matrix):
    flat = [v for row in matrix for v in row]
    uc.mem_write(tile, struct.pack("<25i", *flat))
    i0, j0 = next((i, j) for i in range(5) for j in range(5) if matrix[i][j] >= 1)
    n = sum(1 for v in flat if v >= 1)
    uc.mem_write(node + 0x330, struct.pack("<ii", i0, j0))
    uc.mem_write(node + 0x2F8, struct.pack("<i", n))
    return i0, j0


# ---------------- reference model ----------------
def ref_fits(board, matrix, R, C):
    for i in range(5):
        for j in range(5):
            if matrix[i][j] >= 1:
                r, c = R + i, C + j
                if not (0 <= r < N and 0 <= c < N) or board[r][c]:
                    return False
    return True


def ref_any_fit(board, matrix):
    return any(ref_fits(board, matrix, R, C) for R in range(-4, N) for C in range(-4, N))


def ref_full_lines(board):
    rows = [r for r in range(N) if all(board[r])]
    cols = [c for c in range(N) if all(board[r][c] for r in range(N))]
    cleared = {(r, c) for r in rows for c in range(N)} | {(r, c) for c in cols for r in range(N)}
    return len(rows) + len(cols), cleared


def ref_points(n_cells, lines):
    return 10 * n_cells + 50 * lines * (lines + 1)


tables = json.load(open(PIECES))["variants"]
all_mats = [t["matrix"] for tab in tables.values() for g in tab for t in g]
rng = random.Random(1234)

# ---- 1. matchTileByRowAndCol vs reference, all anchors, random boards ----
mism = checked = 0
for trial in range(25):
    dens = rng.choice([0.0, 0.2, 0.4, 0.6, 0.8])
    board = [[1 if rng.random() < dens else 0 for _ in range(N)] for _ in range(N)]
    set_board(board)
    for m in all_mats:
        i0, j0 = set_piece(m)
        for r in range(N):
            for c in range(N):
                for i in range(5):  # clear tip flags in the footprint (what resetTipsTiles does)
                    for j in range(5):
                        rr, cc = r - i0 + i, c - j0 + j
                        if 0 <= rr < N and 0 <= cc < N:
                            uc.mem_write(cells[rr][cc] + 0x581, b"\0")
                got = emu.call(MATCH, layer, node, r, c) & 1
                exp = ref_fits(board, m, r - i0, c - j0)
                checked += 1
                if got != exp:
                    mism += 1
print(f"[1] matchTileByRowAndCol: {checked} cases, {mism} mismatches")

# ---- 2. noMatchTiles vs reference ----
mism = checked = 0
for trial in range(300):
    dens = rng.choice([0.5, 0.7, 0.8, 0.9])
    board = [[1 if rng.random() < dens else 0 for _ in range(N)] for _ in range(N)]
    set_board(board)
    for m in all_mats:
        set_piece(m)
        got = emu.call(NOMATCH, layer, node) & 1
        exp = not ref_any_fit(board, m)
        checked += 1
        mism += got != exp
print(f"[2] noMatchTiles: {checked} cases, {mism} mismatches")

# ---- 3. checkToCrush line detection + no-clear scoring ----
mism = checked = 0
ptr2rc = {cells[r][c]: (r, c) for r in range(N) for c in range(N)}
for trial in range(400):
    board = [[1 if rng.random() < 0.5 else 0 for _ in range(N)] for _ in range(N)]
    for r in rng.sample(range(N), rng.randint(0, 3)):
        board[r] = [1] * N
    for c in rng.sample(range(N), rng.randint(0, 3)):
        for r in range(N):
            board[r][c] = 1
    set_board(board)
    set_piece(rng.choice(all_mats))
    score0 = rng.randint(0, 5000)
    uc.mem_write(layer + 0x4DC, struct.pack("<i", score0))
    uc.mem_write(layer + 0x4B8, b"\0" * 24)  # empty crush list vector
    crush_calls.clear()
    emu.call(CHECK, layer)
    exp_lines, exp_cleared = ref_full_lines(board)
    n = struct.unpack("<i", emu.read(node + 0x2F8, 4))[0]
    checked += 1
    if exp_lines == 0:
        score1 = struct.unpack("<i", emu.read(layer + 0x4DC, 4))[0]
        ok = not crush_calls and score1 == score0 + ref_points(n, 0)
    else:
        ptrs, lines = crush_calls[0]
        got_cleared = [ptr2rc[p] for p in ptrs]
        ok = lines == exp_lines and len(got_cleared) == len(set(got_cleared)) and set(got_cleared) == exp_cleared
    mism += not ok
print(f"[3] checkToCrush (line count, dedupe, no-clear score): {checked} cases, {mism} mismatches")

# ---- 4. crushPendingTiles score arithmetic (0x42801c..0x42804c) ----
mism = checked = 0
for lines in range(0, 7):
    for n in range(1, 10):
        score0 = 777
        uc.mem_write(layer + 0x4E4, struct.pack("<i", lines))
        uc.mem_write(layer + 0x4DC, struct.pack("<i", score0))
        uc.reg_write(UC_ARM64_REG_X0, n)
        uc.reg_write(UC_ARM64_REG_X19, layer)
        uc.reg_write(UC_ARM64_REG_SP, 0x7F0F_F000)
        uc.emu_start(0x42801C, 0x42804C)
        got = struct.unpack("<i", emu.read(layer + 0x4DC, 4))[0] - score0
        checked += 1
        mism += got != ref_points(n, lines)
        if n == 1:
            print(f"    L={lines}: clear points for a 1-cell piece = {got}")
print(f"[4] crush score formula 10*cells + 50*L*(L+1): {checked} cases, {mism} mismatches")
