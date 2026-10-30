# Reverse engineering notes

Target: `game.puzzle.blockpuzzle` v6.1 (base APK + `split_config.arm64_v8a.apk`).
The original APKs and anything extracted from them are **not** part of this repository.

## 1. Artifact inventory (REA)

| Artifact | SHA-256 | Notes |
|---|---|---|
| `game.puzzle.blockpuzzle_6.1.apk` | `9710be0c…a05c` | Java/Kotlin shell (`classes.dex`, ads/Play services), Cocos Studio UI (`assets/res/**.ExportJson`), sprites, sounds |
| `split_config.arm64_v8a.apk` | `0dda4c64…f90f` | Single native lib `lib/arm64-v8a/libMyGame.so` (9.2 MB) |
| `libMyGame.so` | `45cc3c3f…b69d` | AArch64, NDK r28c, stripped, but C++ dynamic symbols exported |

REA `open_binary` + `inspect_artifact` + `extract_artifact` produced the inventories; the engine is **Cocos2d-x** (C++), so the
game rules live in `libMyGame.so`, not in the dex. Relevant classes (exported symbols):

- `JewelPuzzleGameLayer` – classic mode (board, placement, clears, scoring, game over)
- `JewelPuzzleBombGameLayer` – bomb mode
- `JewelPuzzlePendingNode` / `JewelPuzzlePendingElement` – the pieces offered below the board
- `PendingTileManager` / `JewelPuzzlePendingTile` – piece table and piece RNG
- `JewelPuzzleGameManage` – persistent data (max score, saved board)

Tooling notes: no REA decompiler backend (Ghidra/IDA/Hopper/Kuna) or JADX is configured on this machine, and the host GNU objdump
has no AArch64 target. Disassembly was done with `llvm-objdump`; functions that build data procedurally were **emulated**
with Unicorn (`re/arm64_emu.py`), i.e. the shipped AArch64 code computed the results, with libc/libc++/cocos imports stubbed.

## 2. Piece table — `PendingTileManager::init()` @ `0x42fc08`

`JewelPuzzlePendingTile` is 0x6c bytes: `int matrix[5][5]`, `int color` (+0x64), `int variant` (+0x68).
`PendingTileManager` holds `std::vector<std::vector<JewelPuzzlePendingTile>>` (12 groups; the inner vector holds the
rotations of one shape — the player cannot rotate, a random rotation is dealt) and a `std::vector<int>` history at +0x18.

`init()` reads `UserDefault::getBoolForKey("isOld", false)` and builds one of two tables. Reproduce with:

```bash
uv run python re/extract_pieces.py path/to/libMyGame.so src/block_puzzle_ai/data/pieces.json
```

Group (index = color) for the default table (`isOld = false`, fresh installs):

| # | shape | variants |
|---|---|---|
| 0 | 1×1 dot | 1 |
| 1 | domino | 2 |
| 2 | I3 | 2 |
| 3 | I4 | 2 |
| 4 | I5 | 2 |
| 5 | small L (3 cells) | 4 |
| 6 | 2×2 square | 1 |
| 7 | big L (5 cells, 3×3 corner) | 4 |
| 8 | 3×3 square | 1 |
| 9 | T (4 cells) | 4 |
| 10 | U (5 cells) | 4 |
| 11 | plus (5 cells) | 1 |

With `isOld = true`, groups 9–11 are instead J/L tetromino (4), T tetromino (4) and S/Z (4).

## 3. Piece RNG — `PendingTileManager::randTiles(int p)` @ `0x42f798`

```
history.resize(min(len, 3))                 # keep 3 most recent group ids
r = rand() % 1000
v = (int)(r / 1000.0f * p)
if   v <= 2999:          g = rand() % (12 - 3)   # groups 0..8
elif v / 16 <= 374:      g = rand() % (12 - 2)   # v < 6000  -> groups 0..9
elif v / 32 <= 374:      g = rand() % (12 - 1)   # v < 12000 -> groups 0..10
else:                    g = rand() % 12         # all groups
if any(h in {8, 10, 11} for h in history):       # bitmask 0xd00
    g = rand() % 8                               # no hard piece right after a hard piece
history.insert(0, g)
return groups[g][rand() % len(groups[g])]
```

Evidence: `cmp w21,#0xbb7` (2999), `lsr #4 / lsr #5` vs `#0x176` (374), mask `#0xd00`, `and #7` (rand % 8 with C sign semantics).

## 4. Piece score — `JewelPuzzlePendingNode::configData()` @ `0x42ef2c`, `getScore()` @ `0x42f650`

`configData` walks the 5×5 matrix and increments field `+0x2f8` for every cell `>= 1`; `getScore()` returns it,
i.e. a piece is worth its number of cells.

## 5. Board, placement, clears, scoring, game over

See [`RULES.md`](RULES.md).
