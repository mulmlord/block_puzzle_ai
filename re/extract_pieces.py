"""Extract the piece (PendingTile) table by emulating PendingTileManager::init() from libMyGame.so.

Usage: uv run python re/extract_pieces.py <path/to/libMyGame.so> [out.json]

PendingTileManager holds std::vector<std::vector<JewelPuzzlePendingTile>>. Each tile is 0x6c bytes:
int matrix[5][5] followed by int color (0x64) and one trailing int (0x68). init() reads the
UserDefault bool "isOld" and builds the table in one of two variants; both are dumped.
"""

from __future__ import annotations

import json
import sys

from arm64_emu import Emulator

TILE_SIZE = 0x6C
USER_DEFAULT_GET_INSTANCE = "_ZN7cocos2d11UserDefault11getInstanceEv"


def run_init(so_path: str, is_old: bool) -> list[list[dict]]:
    state: dict[str, int] = {}

    def get_instance(emu: Emulator) -> int:
        if "obj" not in state:
            vtable = emu.malloc(8 * 128)
            vfunc = emu._new_stub("UserDefault::virtual")
            for i in range(128):
                emu.uc.mem_write(vtable + 8 * i, vfunc.to_bytes(8, "little"))
            obj = emu.malloc(16)
            emu.uc.mem_write(obj, vtable.to_bytes(8, "little"))
            state["obj"] = obj
        return state["obj"]

    emu = Emulator(
        so_path,
        overrides={
            USER_DEFAULT_GET_INSTANCE: get_instance,
            "UserDefault::virtual": lambda e: int(is_old),
        },
    )
    this = emu.malloc(0x100)
    emu.call("_ZN18PendingTileManager4initEv", this)

    groups = []
    begin, end = emu.u64(this), emu.u64(this + 8)
    for g in range(begin, end, 24):
        t_begin, t_end = emu.u64(g), emu.u64(g + 8)
        tiles = []
        for t in range(t_begin, t_end, TILE_SIZE):
            vals = emu.i32s(t, TILE_SIZE // 4)
            matrix = [vals[r * 5 : r * 5 + 5] for r in range(5)]
            tiles.append({"matrix": matrix, "color": vals[25], "extra": vals[26]})
        groups.append(tiles)
    return groups


def render(matrix: list[list[int]]) -> list[str]:
    return ["".join("#" if v else "." for v in row) for row in matrix]


def normalize(groups: list[list[dict]]) -> list[list[dict]]:
    """Cells relative to the shape's top-left corner, keeping the original matrix for reference."""
    result = []
    for tiles in groups:
        group = []
        for t in tiles:
            cells = [(r, c) for r in range(5) for c in range(5) if t["matrix"][r][c]]
            r0 = min(r for r, _ in cells)
            c0 = min(c for _, c in cells)
            group.append(
                {"cells": [[r - r0, c - c0] for r, c in cells], "color": t["color"], "matrix": t["matrix"]}
            )
        result.append(group)
    return result


def main() -> None:
    so_path = sys.argv[1]
    tables = {"new": run_init(so_path, False), "old": run_init(so_path, True)}
    for variant, groups in tables.items():
        print(f"===== isOld={variant == 'old'}: {len(groups)} groups")
        for gi, tiles in enumerate(groups):
            print(f"-- group {gi}: {len(tiles)} tiles")
            for t in tiles:
                print(f"   color={t['color']} variant={t['extra']}  " + " | ".join(render(t["matrix"])))
    if len(sys.argv) > 2:
        out = {
            "source": "PendingTileManager::init() @0x42fc08 in libMyGame.so (emulated); "
            "tile = int[5][5] + color + variant",
            "variants": {name: normalize(groups) for name, groups in tables.items()},
        }
        with open(sys.argv[2], "w") as fh:
            json.dump(out, fh, separators=(",", ":"))


if __name__ == "__main__":
    main()
