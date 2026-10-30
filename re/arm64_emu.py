"""Minimal AArch64 ELF emulator for running isolated game-logic functions from libMyGame.so.

The shared library is mapped as data into Unicorn; nothing from the APK runs on the host CPU.
Imported functions (libc, libc++) and selected engine functions (cocos2d::UserDefault) are
replaced with Python stubs so pure game-logic routines can execute in isolation.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Callable

from elftools.elf.elffile import ELFFile
from elftools.elf.relocation import RelocationSection
from unicorn import UC_ARCH_ARM64, UC_HOOK_CODE, UC_MODE_ARM, Uc, UcError
from unicorn.arm64_const import (
    UC_ARM64_REG_LR,
    UC_ARM64_REG_PC,
    UC_ARM64_REG_SP,
    UC_ARM64_REG_TPIDR_EL0,
    UC_ARM64_REG_X0,
    UC_ARM64_REG_X1,
    UC_ARM64_REG_X2,
)

PAGE = 0x1000
STUB_BASE = 0x7000_0000
HEAP_BASE = 0x6000_0000
HEAP_SIZE = 0x0100_0000
STACK_BASE = 0x7F00_0000
STACK_SIZE = 0x0010_0000
TLS_BASE = 0x7E00_0000
RET_INSN = struct.pack("<I", 0xD65F03C0)  # ret

R_AARCH64_GLOB_DAT = 1025
R_AARCH64_JUMP_SLOT = 1026
R_AARCH64_RELATIVE = 1027


def _align_down(x: int) -> int:
    return x & ~(PAGE - 1)


def _align_up(x: int) -> int:
    return (x + PAGE - 1) & ~(PAGE - 1)


@dataclass
class Emulator:
    path: str
    overrides: dict[str, Callable[["Emulator"], int | None]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.uc = Uc(UC_ARCH_ARM64, UC_MODE_ARM)
        self.symbols: dict[str, int] = {}
        self.stubs: dict[int, str] = {}
        self._heap_next = HEAP_BASE
        self._next_stub = STUB_BASE
        self.calls: list[str] = []
        with open(self.path, "rb") as fh:
            elf = ELFFile(fh)
            self._map_segments(elf)
            self._load_symbols(elf)
            self._apply_relocations(elf)
        self.uc.mem_map(HEAP_BASE, HEAP_SIZE)
        self.uc.mem_map(STACK_BASE, STACK_SIZE)
        self.uc.mem_map(TLS_BASE, PAGE)
        self.uc.reg_write(UC_ARM64_REG_TPIDR_EL0, TLS_BASE)
        self.exit_addr = self._new_stub("__emu_exit__")
        self.uc.hook_add(UC_HOOK_CODE, self._on_stub, begin=STUB_BASE, end=STUB_BASE + 0x10_0000)

    # ---- loading -------------------------------------------------------------------------
    def _map_segments(self, elf: ELFFile) -> None:
        segs = [s for s in elf.iter_segments() if s["p_type"] == "PT_LOAD"]
        lo = _align_down(min(s["p_vaddr"] for s in segs))
        hi = _align_up(max(s["p_vaddr"] + s["p_memsz"] for s in segs))
        self.uc.mem_map(lo, hi - lo)
        for s in segs:
            self.uc.mem_write(s["p_vaddr"], s.data())
        self.uc.mem_map(STUB_BASE, 0x10_0000)

    def _load_symbols(self, elf: ELFFile) -> None:
        dynsym = elf.get_section_by_name(".dynsym")
        for sym in dynsym.iter_symbols():
            if sym["st_shndx"] != "SHN_UNDEF" and sym["st_value"]:
                self.symbols.setdefault(sym.name, sym["st_value"])
        self._dynsym = dynsym

    def _new_stub(self, name: str) -> int:
        addr = self._next_stub
        self._next_stub += 4
        self.uc.mem_write(addr, RET_INSN)
        self.stubs[addr] = name
        return addr

    def _apply_relocations(self, elf: ELFFile) -> None:
        stub_for: dict[str, int] = {}
        for sec in elf.iter_sections():
            if not isinstance(sec, RelocationSection):
                continue
            for rel in sec.iter_relocations():
                rtype = rel["r_info_type"]
                off = rel["r_offset"]
                if rtype == R_AARCH64_RELATIVE:
                    self.uc.mem_write(off, struct.pack("<Q", rel["r_addend"]))
                elif rtype in (R_AARCH64_GLOB_DAT, R_AARCH64_JUMP_SLOT):
                    name = self._dynsym.get_symbol(rel["r_info_sym"]).name
                    if name in self.overrides or name not in self.symbols:
                        if name not in stub_for:
                            stub_for[name] = self._new_stub(name)
                        target = stub_for[name]
                    else:
                        target = self.symbols[name]
                    self.uc.mem_write(off, struct.pack("<Q", target + rel["r_addend"]))

    # ---- runtime helpers -----------------------------------------------------------------
    def malloc(self, size: int) -> int:
        addr = self._heap_next
        self._heap_next += (size + 15) & ~15
        if self._heap_next > HEAP_BASE + HEAP_SIZE:
            raise MemoryError("emulated heap exhausted")
        self.uc.mem_write(addr, b"\0" * size)
        return addr

    def reg(self, r: int) -> int:
        return self.uc.reg_read(r)

    def read(self, addr: int, size: int) -> bytes:
        return bytes(self.uc.mem_read(addr, size))

    def u64(self, addr: int) -> int:
        return struct.unpack("<Q", self.read(addr, 8))[0]

    def i32s(self, addr: int, n: int) -> list[int]:
        return list(struct.unpack(f"<{n}i", self.read(addr, 4 * n)))

    def _on_stub(self, uc: Uc, address: int, size: int, user_data: object) -> None:
        name = self.stubs.get(address)
        if name is None:
            return
        if name == "__emu_exit__":
            uc.emu_stop()
            return
        self.calls.append(name)
        handler = self.overrides.get(name) or _DEFAULT_STUBS.get(name)
        if handler is None:
            uc.emu_stop()
            raise RuntimeError(f"unhandled import called: {name}")
        result = handler(self)
        if result is not None:
            uc.reg_write(UC_ARM64_REG_X0, result & 0xFFFF_FFFF_FFFF_FFFF)

    def call(self, func: int | str, *args: int, max_insns: int = 50_000_000) -> int:
        addr = self.symbols[func] if isinstance(func, str) else func
        regs = [UC_ARM64_REG_X0 + i for i in range(8)]
        for r, v in zip(regs, args):
            self.uc.reg_write(r, v)
        self.uc.reg_write(UC_ARM64_REG_SP, STACK_BASE + STACK_SIZE - 0x1000)
        self.uc.reg_write(UC_ARM64_REG_LR, self.exit_addr)
        try:
            self.uc.emu_start(addr, self.exit_addr, count=max_insns)
        except UcError as exc:
            pc = self.reg(UC_ARM64_REG_PC)
            raise RuntimeError(f"emulation fault at pc={pc:#x}: {exc}") from exc
        return self.reg(UC_ARM64_REG_X0)


def _memcpy(emu: Emulator) -> int:
    dst, src, n = emu.reg(UC_ARM64_REG_X0), emu.reg(UC_ARM64_REG_X1), emu.reg(UC_ARM64_REG_X2)
    if n:
        emu.uc.mem_write(dst, emu.read(src, n))
    return dst


def _memset(emu: Emulator) -> int:
    dst, val, n = emu.reg(UC_ARM64_REG_X0), emu.reg(UC_ARM64_REG_X1), emu.reg(UC_ARM64_REG_X2)
    if n:
        emu.uc.mem_write(dst, bytes([val & 0xFF]) * n)
    return dst


def _fail(name: str) -> Callable[[Emulator], int | None]:
    def handler(emu: Emulator) -> int | None:
        raise RuntimeError(f"target called {name}")

    return handler


_DEFAULT_STUBS: dict[str, Callable[[Emulator], int | None]] = {
    "_Znwm": lambda e: e.malloc(e.reg(UC_ARM64_REG_X0)),
    "_Znam": lambda e: e.malloc(e.reg(UC_ARM64_REG_X0)),
    "malloc": lambda e: e.malloc(e.reg(UC_ARM64_REG_X0)),
    "_ZdlPv": lambda e: None,
    "_ZdaPv": lambda e: None,
    "free": lambda e: None,
    "memcpy": _memcpy,
    "memmove": _memcpy,
    "memset": _memset,
    "__stack_chk_fail": _fail("__stack_chk_fail"),
    "abort": _fail("abort"),
    "__cxa_throw": _fail("__cxa_throw"),
}
