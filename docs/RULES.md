# Block Puzzle (game.puzzle.blockpuzzle v6.1): classic-mode rules spec

Source: `libMyGame.so` (arm64-v8a), `JewelPuzzleGameLayer` (classic mode). Static analysis, plus Unicorn emulation of the pure-logic functions.
Confidence labels:
- **[EMU]**: emulated in Unicorn and matched a Python reference model with 0 mismatches.
- **[OBS]**: read directly from the disassembly.
- **[INF]**: inferred.
- **[UNK]**: unknown.

Verification script: [`re/verify_rules.py`](../re/verify_rules.py) (output reproduced at the end).

Layer field map used below (offsets from `JewelPuzzleGameLayer*`):

| off | meaning |
|---|---|
| 0x418/0x41c | board origin x / y (pixel) |
| 0x458 | currently dragged `JewelPuzzlePendingNode*` |
| 0x478..0x488 | `vector<JewelPuzzlePendingNode*>` pending pieces (display order left→right) |
| 0x498 | game-over flag |
| 0x4b8..0x4c8 | crush list `vector<JewelPuzzleGameElement*>` |
| 0x4dc | **score** (int) |
| 0x4e0 | displayed score (UI count-up) |
| 0x4e4 | lines completed by this move |
| 0x510 | `JewelPuzzleGameElement* board[10][10]` (row stride 0x50) |

Element fields: +0x580 `filled` (bool), +0x581 `tip` (temporary placement-preview flag), +0x584 color.
Pending node fields: +0x2f8 cell count (`getScore()`), +0x318 `JewelPuzzlePendingTile*`, +0x330/+0x334 anchor (i0,j0), +0x338/+0x33c drop (row,col).

---

## 1. Board

- **10 × 10** grid. **[OBS]**
  - `restart` @0x425a94–0x425b3c creates elements with nested `i<10`, `j<10` loops and stores them at `0x510 + i*0x50 + j*8`.
  - `checkToCrush`, `matchTiles`, `noMatchTiles`, `resetTipsTiles` and `saveMatrixData` all loop 10×10. `matchTileByRowAndCol` bounds-checks `0..9`.
- **Coordinates. [OBS]**
  - `board[row][col]`; row 0 is the **top** row and col 0 is the left column.
  - `getPointByRowCol(row,col)` @0x423c74: `x = originX + col*66`, `y = originY - row*66` (cell = 66 px).
  - `getRowAndColByPoint` @0x423cb4 is the inverse: `ceil((originY-y-33)/66)`, `ceil((x-originX-33)/66)`.
- **Cell state. [OBS]** A cell is occupied iff `element+0x580 != 0`. Color (+0x584) is cosmetic.
- **Persisted save**, in `JewelPuzzleGameManage::playerMatrixData` (0x9400e0); written by `saveMatrixData` @0x427410 after every successful move. **[OBS]**
  - `int color[10][10]` (row-major, stride 40 B), with -1 = empty.
  - `+0x190` score (written only if ≥1).
  - `+0x194..+0x1a8` three `(group, variant)` pairs for the pending pieces (`tile+0x64/+0x68`), saved only when exactly 3 are pending.
  - `resetMatrixData` @0x42ad38 sets all cells to -1 and zeroes the rest.
  - `validMatrixData` @0x42ac78 is true iff score>0 or any cell ≠ -1. `init` @0x424854 then either resumes via `loadFromArchieve` (followed by `checkGameOver`) or calls `restart`.

## 2. Pending pieces and spawning

- **Always 3 pieces are offered. [OBS]**
  - `restart` @0x425b88/0x425d4c/0x425f04 creates three `PendingNode`s, each via `randTiles(score)` (score = 0 there), and pushes them in order left, middle, right (rects 0x438, 0x428, 0x448).
  - `resetAllPendingNode` @0x427a90 lays out vector index i at x-offset `{-3.6, 0, +3.6}[i]*66` (float table @0x2cea68).
- **Refill is one piece at a time, immediately after every successful placement. [OBS]** `onTouchEnded` @0x4269fc–0x426ab8 runs this sequence:
  1. remove the placed node from the vector (memmove; the order of the others is kept),
  2. `moveToMatch` (animation only),
  3. **`bornPendingNode`**, which creates one node with `randTiles(this->score@0x4dc)` (@0x4276ec) and **appends it at the end** (rightmost slot),
  4. `resetAllPendingNode`,
  5. `checkToCrush` (scoring and clears),
  6. `saveMatrixData`.

  So slots behave like a queue: placing slot k shifts the slots right of k one place left, and the new piece always lands in slot 2.
- **`randTiles` argument = current game score *before* this move's points.** `bornPendingNode` runs before `checkToCrush` adds the points. **[OBS]**
- **No fit guarantee.** `randTiles` never reads the board; nothing re-rolls an unplaceable piece. **[OBS]**
- **The player cannot rotate pieces.** The variant is fixed at spawn: `JewelPuzzlePendingNode` has no rotate method, and the touch handlers only move or scale. **[OBS]**
- **Spawn algorithm** (`PendingTileManager::randTiles` @0x42f798). `history` is the vector at mgr+0x18 and persists across games; `restart` does not clear it. The tables have G = 12 groups. **[OBS]**

```python
def rand_tiles(score, history, groups):          # groups: list of variant lists, len 12
    if len(history) > 3: del history[3:]          # keep 3 most recent (front = newest)
    r = rand() % 1000
    v = int(float32(r) / float32(1000.0) * float32(score))   # trunc toward 0
    G = len(groups)
    if   v <= 2999: g = rand() % (G - 3)   # groups 0..8
    elif v <= 5999: g = rand() % (G - 2)   # 0..9
    elif v <= 11999: g = rand() % (G - 1)  # 0..10
    else:           g = rand() % G         # 0..11
    if any(h in (8, 10, 11) for h in history):   # checks the <=3 previous groups
        g = rand() % 8                            # force a small group 0..7
    history.insert(0, g)
    variant = rand() % len(groups[g])
    return groups[g][variant]
```

- The group is chosen uniformly among the allowed groups and then the variant uniformly. Pieces are therefore not weighted by variant count. **[OBS]**
- **Score thresholds** (v ≈ uniform on [0, 0.999·score]): **[INF]**
  - Groups 9..11 are impossible until score ≥ ~3004.
  - For score s ≥ 12000 the four ranges are hit with probabilities ≈ 3000/s, 3000/s, 6000/s and 1−12000/s.
- **"Big piece" cooldown. [OBS]** A spawn of group 8, 10 or 11 stays in the 3-entry history. The next 3 spawns are therefore forced into groups 0..7, and those forced spawns never extend the cooldown.
- **RNG.** Bionic `rand()`, seeded once with `srand(time(NULL))` @0x417654. The exact sequence is irrelevant for simulation. **[OBS]**
- **Piece table selection** (resolves "isOld"). **[OBS]**
  - `PendingTileManager::init` @0x42fc40 calls `UserDefault::getIntegerForKey("isOld", 0)` (vtable slot +0x8 = `getIntegerForKey(const char*, int)`, resolved from `vtable for cocos2d::UserDefault` @0x90da70). It takes the "old" table only if the value **== 1** (`cmp w20,#1` @0x42fc74).
  - `AppDelegate` @0x41816c–0x4181e8 sets "isOld" once (slot +0x38 = `setIntegerForKey`). It writes 1 if prior progress exists (best score, a saved game, or two other player fields), else 2.
  - **A fresh install therefore uses the `isOld=false` table** (`pieces.json` variant `"new"`): 28 tiles in 12 groups. Group 8 = 3×3 square, 9 = T (4 rotations), 10 = U (4 rotations), 11 = plus.

## 3. Placement

- **Anchor.** `configData` @0x42ef2c iterates the 5×5 matrix row-major (`i` = row, `j` = column; row 0 = top, same orientation as the board). The first filled cell becomes the anchor: `+0x330 = i0`, `+0x334 = j0`, and `+0x360` = its sprite. **[OBS]**
- **Drop position.** `canPlace` @0x426b54 (and `matchTiles` @0x42668c for the drag preview) converts the anchor sprite's world position to the nearest board cell `(r, c)`. It rejects positions outside 0..9 and then calls `matchTileByRowAndCol(node, r, c)`. **[OBS]**
- **`matchTileByRowAndCol` @0x429020 [EMU]: 147,500 cases (25 random boards × 59 tiles × 100 anchors), 0 mismatches.**
  - Matrix cell (i,j) maps to board cell `(r + i - i0, c + j - j0)`. Equivalently, the 5×5 top-left sits at `(R, C) = (r - i0, c - j0)`.
  - The placement is valid iff **every** filled matrix cell maps inside 0..9 × 0..9 onto an **empty** board cell.
  - All-or-nothing; there is no other constraint (no adjacency rule, no colour rule).
  - The `tip` flag (+0x581) is only scratch state and is reset each time by `resetTipsTiles`.
- **Commit** (`onTouchEnded` @0x426960–0x4269f4). **[OBS]**
  - If valid, every flagged cell gets `filled = 1` plus the piece colour.
  - If invalid, the "Move-Fail" sound plays and the piece returns to its slot with no state change.
- **Simulator form.** Action = (slot ∈ {0,1,2}, R, C). It is legal iff `fits(board, piece, R, C)`. For any piece there are at most 10×10 distinct legal placements, one per anchor cell. **[INF, follows from EMU]**

## 4. Line clearing

- **`checkToCrush` @0x426d64 [EMU]: 400 random boards, 0 mismatches.**
  1. Sets `L = 0` (+0x4e4).
  2. Scans all 10 rows (0x426dcc loop). Each completely filled row has its 10 elements pushed with `pushToCrushList`, and `L++`.
  3. Scans all 10 columns (0x426f40 loop) the same way.
  4. Both scans use the board **after** the piece was placed and **before** any removal, so rows and columns are detected **simultaneously**.
- **`pushToCrushList` @0x427bec** appends an element only if it is not already in the list. A cell shared by a full row and a full column is cleared **once**, but **L counts lines** (row + column = 2). **[EMU]**
- **`crushPendingTiles` @0x427d50** removes the cells synchronously in the same call: for each listed element `setMyColor(-1)` and `filled = 0` (@0x427dd4). The cleared cells do not shift; there is no gravity. **[OBS]**
- If the crush list is empty, nothing is removed. **[OBS]**

## 5. Scoring

**Formula** (one placement of a piece with `n` cells that completes `L` lines; L = rows + columns):

```
points = 10*n + 50*L*(L+1)
score += points
```

- **L = 0** (@0x4270f8–0x42711c): `score += n*10` (`madd w8, w0, #10, score`). **[EMU]**
- **L ≥ 1** (@0x42801c–0x428048): `w9 = 5L+5`, then `w8 = w9*L + n`, then `score = w8*10 + score`. **[EMU]**: emulated over L = 0..6 and n = 1..9, 0 mismatches.
- The line bonus by L, for a 1-cell piece including its 10 placement points:

| L | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| total | 110 | 310 | 610 | 1010 | 1510 | 2110 |
| line bonus alone | 100 | 300 | 600 | 1000 | 1500 | 2100 |

- `n` is `JewelPuzzlePendingNode::getScore()` (+0x2f8), which `configData` sets to the number of matrix entries ≥ 1. The node at +0x458 is still the placed piece while `checkToCrush` runs, because it is only cleared at @0x4268e4. **[OBS]**
- **No combo or streak across moves**, **no all-clear bonus**, **no per-cleared-cell bonus**. Shared cells do not matter for scoring. **[OBS]** Evidence:
  - `checkToCrush` writes 0x4d0 and 0x4d8 (@0x4270b8), but nothing in the classic layer ever reads them.
  - The 0x4a0 vector feeding 0x4d0 is never filled.
  - The formula uses only n and L.
- **Score storage and best score. [OBS]**
  - The score is updated immediately in `checkToCrush` / `crushPendingTiles`. `update()` @0x4275e4 only animates the displayed value by +3 per frame.
  - Best score is updated only at game over: `showGameOver` @0x428d70 copies the score to `JewelPuzzleGameManage::TotalScore` and calls `setGameMaxScore(score)` @0x42ac38, which sets `playerData+4 = max(best, score)` and saves.
- `animScore` @0x428214 is display only. It shows the same `points` value.

## 6. Game over

- **Condition. [EMU]: `noMatchTiles` 17,700 cases, 0 mismatches.** `checkGameOver` @0x4254c0 sets `gameOver = 1`. For each pending piece, `noMatchTiles` @0x428a4c tries `matchTileByRowAndCol` at all 100 anchors. If any piece fits anywhere, `gameOver = 0` (unplaceable pieces are greyed out to opacity 100).
- **Game over iff none of the 3 pending pieces has a legal placement anywhere on the board.**
- **When it is checked. [OBS]**
  - After every successful placement, via a delayed CallFunc. The lambda operator() resolves to `checkGameOver`: 0.4 s if no clear (vtable 0x8df2b8 → 0x429428), 0.8 s after a clear (0x8df738 → 0x429974).
  - It sees the post-clear board and the 3 pending pieces, including the newly spawned one.
  - It also runs once after resuming a saved game.
- **On game over. [OBS]** Delay 1 s, then `showGameOver`:
  - `resetMatrixData` (clears the save), records best score, shows a full-screen ad (lambda @0x42951c), and opens `JewelOverLayer`.
  - `JewelOverLayer` buttons: restart, leave to menu, share, leaderboard (`reportScore`), rate.
- **No revive or continue exists** (no such method or button). **[OBS]**

## 7. Other mechanics

- **`enterBtnRefresh`** @0x424944 is wired to the widget named **"btnprev"** (@0x42446c via GOT 0x921850). It does **not** refresh pieces. It replaces the scene with `JewelPuzzleMainMenuLayer`. The game is already saved, so re-entering resumes it. **[OBS]**
- **`enterBtnPrev`** @0x424aac is wired to **"btnreplay"** (GOT 0x921858). It opens `JewelPauseLayer`, which has close (resume), restart (plus a full-screen ad) and leave. **There is no undo**: no previous state is stored anywhere. **[OBS]**
- **`showTipsTiles` / `resetTipsTiles`** are only the translucent drag-preview "ghost" (cells with the tip flag drawn at opacity 160). They are not a hint system and are not called on any gameplay path. **[OBS]**
- **No other mechanics. [OBS]** There are no timers, no lives, no power-ups and no in-game purchases affecting play in classic mode. `update()` only animates the score text.

## 8. Bomb mode (brief)

- **A separate mode.** **[OBS]**
  - Entered from `JewelPuzzleMainMenuLayer::enterBtnPlayBombGame`, implemented by `JewelPuzzleBombGameLayer`.
  - It has its own save slot (`playerBombMatrixData`) and its own best score (`setBombGameMaxScore`).
  - Board, placement, line clearing and scoring are the same: identical formula at @0x41fb08 and @0x420b84–0x420b9c.
  - `randBombTiles` is instruction-for-instruction identical to `randTiles`.
- **Bomb mechanic.** **[OBS]** **[INF]** where marked.
  - `layer+0x4e8` counts **non-clearing** placements (@0x41fe2c).
  - When `count % 5 == 0`, `randBomb` @0x4213f8 converts a uniformly random **filled** cell into a bomb (`changeToBomb`: +0x550 = 1, counter +0x598 = 9) and resets the count to 0. If the board is empty, it sets the count to 4 so the next non-clearing move retries.
  - On every non-clearing placement, every bomb ticks down via `subBombCount` (inlined loop @0x41fc20; `notifyBomb` itself is never called).
  - A bomb reaching ≤ 0 calls `shakeScreen` @0x421ebc. That sets the game-over flag (@0x421f0c) and its CallFunc (@0x422c50) calls `showGameOver`: **game over**.
  - Moves that clear lines do not tick bombs.
  - Clearing a bomb's row or column removes it, because `setMyColor(-1)` clears +0x550 @0x4230f8. **[INF]** that this fully defuses it.
  - Normal game over (no piece fits) also applies.

---

## Simulator reference (classic mode)

```python
N = 10
def fits(board, cells, R, C):            # cells: [(i,j)] filled entries of the 5x5 matrix
    return all(0 <= R+i < N and 0 <= C+j < N and not board[R+i][C+j] for i, j in cells)

def step(state, slot, R, C):
    piece = state.pending[slot]
    if not fits(state.board, piece.cells, R, C):
        return INVALID                     # game: piece snaps back, nothing changes
    for i, j in piece.cells: state.board[R+i][C+j] = 1
    del state.pending[slot]                                  # others keep their order
    state.pending.append(rand_tiles(state.score, state.history, TABLE))  # score BEFORE this move
    rows = [r for r in range(N) if all(state.board[r])]
    cols = [c for c in range(N) if all(state.board[r][c] for r in range(N))]
    for r in rows:
        for c in range(N): state.board[r][c] = 0
    for c in cols:
        for r in range(N): state.board[r][c] = 0
    L = len(rows) + len(cols)
    reward = 10*len(piece.cells) + 50*L*(L+1)
    state.score += reward
    done = not any(fits(state.board, p.cells, R2, C2)
                   for p in state.pending for R2 in range(-4, N) for C2 in range(-4, N))
    return reward, done

def reset(state):   # restart(): empty board, score 0, 3 x rand_tiles(0); history NOT cleared
    state.board = [[0]*N for _ in range(N)]; state.score = 0
    state.pending = [rand_tiles(0, state.history, TABLE) for _ in range(3)]
```

Observation-relevant hidden state: the spawn distribution depends on `score` and on the last 3 spawned groups (`history`). Include both if the agent should be able to anticipate spawns.

## Remaining unknowns / caveats

- **UI timing.** `checkGameOver` runs 0.4/0.8 s after a move and the player can keep playing during that delay. The outcome is identical to checking after each move: if nothing fits, no move is possible. **[INF]**
- **Float precision.** `v` is computed in float32 (`scvtf`/`fdiv`/`fmul`/`fcvtzs`). Use float32 if bit-exact threshold behaviour at boundaries (e.g. score ≈ 3003) matters. **[OBS]**
- **"isOld" decision fields.** The meaning of `[GOT 0x921638]+0x4b0` and `playerData+8` is not identified. They only matter for upgraded installs (which table they get), not for fresh installs. **[UNK]**
- **Bomb-mode details** (whether the count reset to 4 means "4 more moves", and the full defuse behaviour) were not emulated. **[INF]**
- **Display and save side effects** are not part of the rules and were not reproduced: the score is saved only if ≥1, and the pending pieces are saved only when exactly 3 exist.

## Verification output (`uv run python re/verify_rules.py path/to/libMyGame.so`)

```
[1] matchTileByRowAndCol: 147500 cases, 0 mismatches
[2] noMatchTiles: 17700 cases, 0 mismatches
[3] checkToCrush (line count, dedupe, no-clear score): 400 cases, 0 mismatches
    L=0: clear points for a 1-cell piece = 10
    L=1: clear points for a 1-cell piece = 110
    L=2: clear points for a 1-cell piece = 310
    L=3: clear points for a 1-cell piece = 610
    L=4: clear points for a 1-cell piece = 1010
    L=5: clear points for a 1-cell piece = 1510
    L=6: clear points for a 1-cell piece = 2110
[4] crush score formula 10*cells + 50*L*(L+1): 63 cases, 0 mismatches
```

The 59 tiles tested are both tables: 28 tiles (`"new"` table) plus 31 tiles (`"old"` table). In the emulation, cocos and UI calls (`setMyColor`, `setChildOpacity`, `animScore`, `DelayTime`, `CallFunc`, `Sequence`, layer virtuals) were stubbed. `crushPendingTiles` was intercepted to capture the crush list and L, and its arithmetic block was run separately.
