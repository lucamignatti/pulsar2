#!/usr/bin/env python3
"""Disassemble Rocket League UFunctions from an RLSDK-Generator dump + live process.

Replaces the manual Script-blob walk in research/reports/RL_UFUNCTION_RE.md.

    python tools/rl_ufunction_disasm.py --sdk path/to/RLSDK --function TAGame.Car_TA.IsBumperHit
    python tools/rl_ufunction_disasm.py --sdk path/to/RLSDK --function IsBumperHit --raw
    python tools/rl_ufunction_disasm.py --sdk path/to/RLSDK --list Car_TA.Is

Requires a same-session ObjectDump.txt (inject RLSDK-Generator, then run this
while RocketLeague.exe is still that process). Heap UFunction pointers do not
survive a restart.
"""

from __future__ import annotations

import argparse
import ctypes
import re
import struct
import sys
from ctypes import wintypes
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# UFunction / UStruct offsets (RLSDK-Generator Core layout, 64-bit).
# Script.Data/Count sit in UStruct padding; verified live 2026-08-26.
# ---------------------------------------------------------------------------
OFF_UFIELD_NEXT = 0x60
OFF_UOBJECT_CLASS = 0x50
OFF_CHILDREN = 0x88
OFF_PROPERTY_SIZE = 0x90
OFF_SCRIPT_DATA = 0x98
OFF_SCRIPT_COUNT = 0xA0
OFF_FUNCTION_FLAGS = 0x130
OFF_INATIVE = 0x138
OFF_FRIENDLY_NAME = 0x13C
OFF_NUM_PARMS = 0x145
OFF_PARMS_SIZE = 0x146
OFF_FUNC = 0x158
OFF_UPROPERTY_OFFSET = 0x98

FUNC_FINAL = 0x00000001
FUNC_DEFINED = 0x00000002
FUNC_NATIVE = 0x00000400
FUNC_PUBLIC = 0x00020000
FUNC_PROTECTED = 0x00080000
FUNC_HAS_DEFAULTS = 0x00800000

PROCESS_VM_READ = 0x0010
PROCESS_QUERY_INFORMATION = 0x0400
TH32CS_SNAPPROCESS = 0x00000002
TH32CS_SNAPMODULE = 0x00000008
TH32CS_SNAPMODULE32 = 0x00000010
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
MAX_PATH = 260

TOKEN_NAMES = {
    0x00: "Local",
    0x01: "Inst",
    0x02: "Default",
    0x03: "StateVar",
    0x04: "Return",
    0x05: "Switch",
    0x06: "Jump",
    0x07: "JumpIfNot",
    0x08: "Stop",
    0x09: "Assert",
    0x0A: "Case",
    0x0B: "Nothing",
    0x0C: "LabelTable",
    0x0D: "GotoLabel",
    0x0E: "EatReturnValue",
    0x0F: "Let",
    0x10: "DynArrayElem",  # vanilla; unused here (Psyonix 0x57 0x00)
    0x11: "New",
    0x12: "ClassContext",
    0x13: "MetaCast",
    0x14: "LetBool",
    0x15: "EndParmValue",
    0x16: "EndParms",
    0x17: "Self",
    0x18: "Skip",
    0x19: "Context",
    0x1A: "ArrayElem",
    0x1B: "VirtualCall",
    0x1C: "FinalCall",
    0x1D: "Int",
    0x1E: "Float",
    0x1F: "String",
    0x20: "ObjectConst",
    0x21: "NameConst",
    0x22: "RotatorConst",
    0x23: "VectorConst",
    0x24: "Byte",
    0x25: "0",
    0x26: "1",
    0x27: "True",
    0x28: "False",
    0x29: "NativeParm",
    0x2A: "None",
    0x2B: "Local",  # Psyonix: vanilla hole; locals/parms
    0x2C: "IntByte",
    0x2D: "BoolVar",
    0x2E: "DynCast",
    0x2F: "Iterator",
    0x30: "IteratorPop",
    0x31: "IteratorNext",
    0x32: "StructEq",
    0x33: "StructNe",
    0x34: "UnicodeString",
    0x35: "StructMember",
    0x36: "DynArrayLen",  # vanilla; unused here (Psyonix 0x57 0x01)
    0x37: "GlobalCall",
    0x38: "Cast",
    0x3A: "ReturnNothing",
    0x3F: "EmptyDelegate",
    0x41: "DebugInfo",
    0x43: "DelegateProp",
    0x46: "LocalOut",  # observed as out-struct instance on StructMember
    0x48: "EmptyParm",
    0x4A: "EmptyParm",
    0x4B: "EmptyDelegate",  # live: zero a delegate; vanilla InstanceDelegate is dead
    0x4C: "EndOfScript",  # vanilla 0x53; logs "Execution beyond end of script"
    0x53: "Template",  # WORD skip + UObject* + BYTE; not EOS
    0x54: "StructInit",
    0x57: "DynArray",  # Psyonix: sub-op then operands (vanilla InsertItem)
    0x58: "DynArrayResult",
    0x5B: "Seq2",  # two exprs; not an array token
    0x5E: "SkipMark",  # observed in front of && / native ops
}

# 0x57 sub-ops. Live table qword_142388790; 0x10-0x1F unused.
# 0x20+ take a delegate (FindFunction on capture object) — compiler lambdas.
DYNARRAY_SUB = {
    0x00: "Element",
    0x01: "Length",
    0x02: "Insert",
    0x03: "Remove",
    0x04: "Find",
    0x05: "FindStruct",
    0x06: "Add",
    0x07: "AddItem",
    0x08: "RemoveItem",
    0x09: "InsertItem",
    0x0A: "Iterator",
    0x0B: "AddItems",
    0x0C: "Append",
    0x0D: "AddUnique",
    0x0E: "Sort",
    0x0F: "SortStable",
    0x20: "Sorted",
    0x21: "Filter",
    0x22: "Map",
    0x23: "Reduce",
    0x24: "First",
    0x25: "Every",
    0x26: "Any",
    0x27: "Concat",
}

# iNative fallback if Core_classes.cpp is missing
INATIVE_FALLBACK = {
    129: "Not_PreBool",
    130: "AndAnd_BoolBool",
    131: "XorXor_BoolBool",
    132: "OrOr_BoolBool",
    150: "Less_IntInt",
    151: "Greater_IntInt",
    152: "LessEqual_IntInt",
    171: "Multiply_FloatFloat",
    174: "Add_FloatFloat",
    175: "Subtract_FloatFloat",
    182: "MultiplyEqual_FloatFloat",
    183: "DivideEqual_FloatFloat",
    211: "Subtract_PreVector",
    212: "Multiply_VectorFloat",
    213: "Multiply_FloatVector",
    215: "Add_VectorVector",
    216: "Subtract_VectorVector",
    217: "EqualEqual_VectorVector",
    219: "Dot_VectorVector",
    223: "AddEqual_VectorVector",
    224: "SubtractEqual_VectorVector",
    225: "VSize",
    226: "Normal",
}


class PROCESSENTRY32(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", ctypes.c_char * MAX_PATH),
    ]


class MODULEENTRY32(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("th32ModuleID", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("GlblcntUsage", wintypes.DWORD),
        ("ProccntUsage", wintypes.DWORD),
        ("modBaseAddr", ctypes.c_void_p),
        ("modBaseSize", wintypes.DWORD),
        ("hModule", wintypes.HMODULE),
        ("szModule", ctypes.c_char * 256),
        ("szExePath", ctypes.c_char * MAX_PATH),
    ]


kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
kernel32.Process32First.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32)]
kernel32.Process32First.restype = wintypes.BOOL
kernel32.Process32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32)]
kernel32.Process32Next.restype = wintypes.BOOL
kernel32.Module32First.argtypes = [wintypes.HANDLE, ctypes.POINTER(MODULEENTRY32)]
kernel32.Module32First.restype = wintypes.BOOL
kernel32.Module32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(MODULEENTRY32)]
kernel32.Module32Next.restype = wintypes.BOOL
kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.ReadProcessMemory.argtypes = [
    wintypes.HANDLE,
    wintypes.LPCVOID,
    wintypes.LPVOID,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t),
]
kernel32.ReadProcessMemory.restype = wintypes.BOOL
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32.CloseHandle.restype = wintypes.BOOL


def find_pid(exe: str = "RocketLeague.exe") -> int | None:
    snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snap == INVALID_HANDLE_VALUE:
        return None
    try:
        pe = PROCESSENTRY32()
        pe.dwSize = ctypes.sizeof(PROCESSENTRY32)
        if not kernel32.Process32First(snap, ctypes.byref(pe)):
            return None
        while True:
            name = pe.szExeFile.decode("utf-8", "replace")
            if name.lower() == exe.lower():
                return int(pe.th32ProcessID)
            if not kernel32.Process32Next(snap, ctypes.byref(pe)):
                return None
    finally:
        kernel32.CloseHandle(snap)


def module_base(pid: int, exe: str = "RocketLeague.exe") -> int | None:
    snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, pid)
    if snap == INVALID_HANDLE_VALUE:
        return None
    try:
        me = MODULEENTRY32()
        me.dwSize = ctypes.sizeof(MODULEENTRY32)
        if not kernel32.Module32First(snap, ctypes.byref(me)):
            return None
        while True:
            name = me.szModule.decode("utf-8", "replace")
            if name.lower() == exe.lower():
                return int(me.modBaseAddr or 0)
            if not kernel32.Module32Next(snap, ctypes.byref(me)):
                return None
    finally:
        kernel32.CloseHandle(snap)


class ProcessReader:
    def __init__(self, pid: int):
        self.pid = pid
        self.handle = kernel32.OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, False, pid)
        if not self.handle:
            raise OSError(f"OpenProcess({pid}) failed: {ctypes.get_last_error()}")

    def close(self) -> None:
        if self.handle:
            kernel32.CloseHandle(self.handle)
            self.handle = None

    def read(self, addr: int, size: int) -> bytes:
        buf = (ctypes.c_ubyte * size)()
        n = ctypes.c_size_t()
        ok = kernel32.ReadProcessMemory(self.handle, ctypes.c_void_p(addr), buf, size, ctypes.byref(n))
        if not ok or n.value != size:
            raise OSError(f"RPM 0x{addr:X} size={size} err={ctypes.get_last_error()}")
        return bytes(buf)

    def u64(self, addr: int) -> int:
        return struct.unpack_from("<Q", self.read(addr, 8))[0]

    def u32(self, addr: int) -> int:
        return struct.unpack_from("<I", self.read(addr, 4))[0]

    def u16(self, addr: int) -> int:
        return struct.unpack_from("<H", self.read(addr, 2))[0]


@dataclass
class DumpIndex:
    dump_base: int = 0
    gobjects: int = 0
    gobjects_offset: int = 0
    by_addr: dict[int, str] = field(default_factory=dict)
    by_name: dict[str, int] = field(default_factory=dict)
    names: dict[int, str] = field(default_factory=dict)  # FName index
    inative: dict[int, str] = field(default_factory=dict)

    def short(self, addr: int) -> str:
        full = self.by_addr.get(addr)
        if not full:
            return f"0x{addr:X}"
        # "Function TAGame.Car_TA.IsBumperHit" → last component
        parts = full.split()
        qual = parts[-1] if parts else full
        if "." in qual:
            return qual.rsplit(".", 1)[-1]
        return qual

    def full(self, addr: int) -> str:
        return self.by_addr.get(addr, f"0x{addr:X}")


def parse_object_dump(path: Path) -> DumpIndex:
    idx = DumpIndex()
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"Base:\s*(0x[0-9A-Fa-f]+)", text)
    if m:
        idx.dump_base = int(m.group(1), 16)
    m = re.search(r"GObjects:\s*(0x[0-9A-Fa-f]+)", text)
    if m:
        idx.gobjects = int(m.group(1), 16)
    m = re.search(r"Offset:\s*(0x[0-9A-Fa-f]+)", text)
    if m:
        idx.gobjects_offset = int(m.group(1), 16)
    line_re = re.compile(r"UObject\[\d+\]\s+(.+?)\s+(0x[0-9A-Fa-f]+)\s*$")
    for line in text.splitlines():
        m = line_re.match(line.strip())
        if not m:
            continue
        name = m.group(1).rstrip()
        addr = int(m.group(2), 16)
        idx.by_addr[addr] = name
        idx.by_name[name] = addr
        if name.startswith("Function "):
            idx.by_name[name[len("Function ") :]] = addr
    return idx


def parse_name_dump(path: Path, idx: DumpIndex) -> None:
    name_re = re.compile(r"Name\[(\d+)\]\s+(\S+)")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = name_re.match(line.strip())
        if m:
            idx.names[int(m.group(1))] = m.group(2)


def resolve_fname(idx: DumpIndex, index: int, number: int) -> str | None:
    base = idx.names.get(index)
    if not base:
        return None
    if number:
        return f"{base}_{number - 1}"
    return base


def fill_inative_from_live(proc: ProcessReader, idx: DumpIndex) -> int:
    """Overwrite iNative > 0x80 with UFunction.FriendlyName (FName at +0x13C)."""
    n = 0
    funcs = [
        (addr, name)
        for addr, name in idx.by_addr.items()
        if name.startswith("Function ")
    ]
    # Core.Object last so operator iNatives win over same-index aliases.
    funcs.sort(key=lambda x: x[1].startswith("Function Core.Object."))
    for addr, _name in funcs:
        try:
            inn = proc.u16(addr + OFF_INATIVE)
        except OSError:
            continue
        if inn <= 0x80:
            continue
        try:
            fi = proc.u32(addr + OFF_FRIENDLY_NAME)
            fn = proc.u32(addr + OFF_FRIENDLY_NAME + 4)
        except OSError:
            continue
        friendly = resolve_fname(idx, fi, fn)
        if not friendly or friendly == "None":
            continue
        idx.inative[inn] = friendly
        n += 1
    return n


def parse_inative_cpp(sdk: Path, idx: DumpIndex) -> None:
    cpp = sdk / "SDK_HEADERS" / "Core_classes.cpp"
    if not cpp.is_file():
        idx.inative = dict(INATIVE_FALLBACK)
        return
    func = None
    for line in cpp.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"// Function (\S+)", line)
        if m:
            func = m.group(1)
        m = re.search(r"iNative\[(\d+)\]", line)
        if m and func:
            idx.inative[int(m.group(1))] = func.rsplit(".", 1)[-1]
    for k, v in INATIVE_FALLBACK.items():
        idx.inative.setdefault(k, v)


def find_function(idx: DumpIndex, query: str) -> tuple[int, str]:
    q = query.strip()
    if q.lower().startswith("0x"):
        addr = int(q, 16)
        return addr, idx.full(addr)
    if q in idx.by_name:
        addr = idx.by_name[q]
        return addr, idx.by_addr.get(addr, q)
    key = "Function " + q
    if key in idx.by_name:
        return idx.by_name[key], key
    uniq: dict[int, str] = {}
    for addr, name in idx.by_addr.items():
        if not name.startswith("Function "):
            continue
        qual = name[len("Function ") :]
        if qual == q or qual.endswith("." + q) or name.endswith(q):
            uniq[addr] = name
    if len(uniq) == 1:
        addr, name = next(iter(uniq.items()))
        return addr, name
    if not uniq:
        raise SystemExit(f"no Function matching {query!r}")
    shown = "\n".join(f"  {n}  0x{a:X}" for a, n in list(uniq.items())[:20])
    raise SystemExit(f"{len(uniq)} matches for {query!r}:\n{shown}")


def flag_names(flags: int) -> str:
    bits = []
    if flags & FUNC_FINAL:
        bits.append("Final")
    if flags & FUNC_DEFINED:
        bits.append("Defined")
    if flags & FUNC_NATIVE:
        bits.append("Native")
    if flags & FUNC_PUBLIC:
        bits.append("Public")
    if flags & FUNC_PROTECTED:
        bits.append("Protected")
    if flags & FUNC_HAS_DEFAULTS:
        bits.append("HasDefaults")
    return "|".join(bits) if bits else hex(flags)


def field_leaf(name: str) -> str:
    if "." in name:
        return name.rsplit(".", 1)[-1]
    parts = name.split()
    return parts[-1] if parts else name


class Decoder:
    def __init__(self, data: bytes, idx: DumpIndex, raw: bool = False):
        self.data = data
        self.idx = idx
        self.i = 0
        self.raw = raw
        self._guard = 0

    def remaining(self) -> int:
        return len(self.data) - self.i

    def peek(self) -> int:
        if self.i >= len(self.data):
            return 0x4C
        return self.data[self.i]

    def _need(self, n: int) -> bool:
        return self.remaining() >= n

    def _dump_ptr_at(self, off: int = 0) -> int | None:
        if self.remaining() < off + 8:
            return None
        p = struct.unpack_from("<Q", self.data, self.i + off)[0]
        return p if p in self.idx.by_addr else None

    def u8(self) -> int:
        if not self._need(1):
            return 0x4C
        b = self.data[self.i]
        self.i += 1
        return b

    def u16(self) -> int:
        if not self._need(2):
            self.i = len(self.data)
            return 0
        v = struct.unpack_from("<H", self.data, self.i)[0]
        self.i += 2
        return v

    def u32(self) -> int:
        if not self._need(4):
            self.i = len(self.data)
            return 0
        v = struct.unpack_from("<I", self.data, self.i)[0]
        self.i += 4
        return v

    def f32(self) -> float:
        if not self._need(4):
            self.i = len(self.data)
            return 0.0
        v = struct.unpack_from("<f", self.data, self.i)[0]
        self.i += 4
        return v

    def ptr(self) -> int:
        if not self._need(8):
            self.i = len(self.data)
            return 0
        v = struct.unpack_from("<Q", self.data, self.i)[0]
        self.i += 8
        return v

    def cstring(self) -> str:
        start = self.i
        while self.i < len(self.data) and self.data[self.i] != 0:
            self.i += 1
        s = self.data[start : self.i].decode("ascii", "replace")
        if self.i < len(self.data):
            self.i += 1
        return s

    def obj(self, addr: int) -> str:
        return self.idx.short(addr)

    def fname(self) -> str:
        index = self.u32()
        number = self.u32()
        base = self.idx.names.get(index, f"Name[{index}]")
        if number:
            return f"{base}_{number - 1}"
        return base

    def expr(self) -> str:
        self._guard += 1
        if self._guard > 8000:
            return "<depth>"
        try:
            return self._expr()
        finally:
            self._guard -= 1

    def _expr(self) -> str:
        if self.i >= len(self.data):
            return "<eof>"
        op = self.i
        t = self.u8()
        if self.raw:
            prefix = f"[{op:03X} {t:02X}] "
        else:
            prefix = ""

        # Psyonix: locals/parms are 0x2B (and 0x46 out). Vanilla 0x00 Local is
        # unused unless the next 8 bytes resolve in ObjectDump.
        if t == 0x00:
            p = self._dump_ptr_at(0)
            if p is not None:
                self.i += 8
                return prefix + self.obj(p)
            return prefix + "self"
        if t in (0x01, 0x02, 0x03, 0x2B, 0x46):
            return prefix + self.obj(self.ptr())
        if t == 0x48:
            # Not LocalOut (that's 0x46). 0x48 is empty/default parm unless a dump ptr follows.
            p = self._dump_ptr_at(0)
            if p is not None:
                self.i += 8
                return prefix + self.obj(p)
            return prefix + ""

        if t == 0x04:
            inner = self.expr()
            if inner in ("", "Nothing", "<eof>"):
                return prefix + "return"
            if inner.startswith("return"):
                return prefix + inner
            return prefix + f"return {inner}"
        if t == 0x06:
            return prefix + f"goto {self.u16():03X}"
        if t == 0x07:
            tgt = self.u16()
            cond = self.expr()
            return prefix + f"if (!{self._wrap(cond, self._PREC_UNARY)}) goto {tgt:03X}"
        if t == 0x09:
            line = self.u16()
            cond = self.expr()
            return prefix + f"assert({cond}) /*{line}*/"
        if t == 0x0A:
            n = self.u16()
            if n == 0xFFFF:
                return prefix + "default:"
            val = self.expr()
            return prefix + f"case {val}:"
        if t == 0x0B:
            return prefix + "Nothing"
        if t == 0x11:
            # EX_New: outer, name, flags, class; optional trailing Nothing.
            outer, nm, fl, cls = self.expr(), self.expr(), self.expr(), self.expr()
            if self.peek() == 0x0B:
                self.u8()
            args = [x for x in (outer, nm, fl) if x and x != "Nothing"]
            if args:
                return prefix + f"new({', '.join(args)}) {cls}"
            return prefix + f"new {cls}"
        if t == 0x0E:
            # EatReturnValue: optional property then the discarded call
            if self.remaining() >= 8:
                p = struct.unpack_from("<Q", self.data, self.i)[0]
                if p in self.idx.by_addr:
                    self.i += 8
                    return prefix + f"/*eat {self.obj(p)}*/ {self.expr()}"
            return prefix + self.expr()
        if t == 0x0F:
            lhs, rhs = self.expr(), self.expr()
            return prefix + f"{lhs} = {rhs}"
        if t == 0x14:
            lhs, rhs = self.expr(), self.expr()
            return prefix + f"{lhs} = {rhs}"
        if t == 0x16:
            return prefix + ")"
        if t == 0x17:
            return prefix + "self"
        if t == 0x18:
            self.u16()
            return prefix + self.expr()
        if t == 0x5B:
            # Optional DebugInfo, then two exprs (seen in front of array Insert/Add).
            self._eat_debuginfo()
            a, b = self.expr(), self.expr()
            if a and b and a != b:
                return prefix + f"{a}; {b}"
            return prefix + (b or a)
        if t == 0x5E:
            return prefix + self.expr()
        if t == 0x19 or t == 0x12:
            # 64-bit Psyonix: object, WORD skip, UField* rvalue, optional 0x00, member.
            if self.peek() == 0x00 and self._dump_ptr_at(1) is None:
                self.u8()
            obj = self.expr()
            skip = self.u16()
            rval = None
            if self.remaining() >= 8:
                p = struct.unpack_from("<Q", self.data, self.i)[0]
                if p == 0 or p in self.idx.by_addr:
                    self.i += 8
                    if p:
                        rval = self.obj(p)
            if self.peek() == 0x00 and self._dump_ptr_at(1) is None:
                self.u8()
            member = self.expr()
            _ = skip
            if not member:
                return prefix + obj
            if rval and member != rval and not member.startswith(rval + ".") and "(" not in member:
                return prefix + f"{obj}.{rval}.{member}"
            return prefix + f"{obj}.{member}"
        if t == 0x1B or t == 0x37:
            name = self.fname()
            args = self._args()
            return prefix + f"{name}({args})"
        if t == 0x1C:
            fn = self.obj(self.ptr())
            args = self._args()
            return prefix + f"{fn}({args})"
        if t == 0x1D:
            return prefix + str(ctypes.c_int32(self.u32()).value)
        if t == 0x1E:
            return prefix + f"{self.f32():g}"
        if t == 0x1F:
            return prefix + repr(self.cstring())
        if t == 0x20:
            return prefix + self.obj(self.ptr())
        if t == 0x21:
            return prefix + self.fname()
        if t == 0x22:
            p, y, r = self.u32(), self.u32(), self.u32()
            return prefix + f"rot({p},{y},{r})"
        if t == 0x23:
            x, y, z = self.f32(), self.f32(), self.f32()
            return prefix + f"vec({x:g},{y:g},{z:g})"
        if t == 0x24 or t == 0x2C:
            return prefix + str(self.u8())
        if t == 0x25:
            return prefix + "0"
        if t == 0x26:
            return prefix + "1"
        if t == 0x27:
            return prefix + "true"
        if t == 0x28:
            return prefix + "false"
        if t == 0x29:
            return prefix + self.obj(self.ptr())
        if t == 0x2A:
            return prefix + "None"
        if t == 0x2D:
            return prefix + self.expr()
        if t == 0x2E or t == 0x13:
            cls = self.obj(self.ptr())
            return prefix + f"({cls})({self.expr()})"
        if t == 0x32 or t == 0x33:
            st = self.obj(self.ptr())
            a, b = self.expr(), self.expr()
            op = "==" if t == 0x32 else "!="
            return prefix + f"({a} {op} {b} /*{st}*/)"
        if t == 0x1A:
            # Static UArrayProperty: index, then array (vanilla EX_ArrayElement).
            idx = self.expr()
            arr = self.expr()
            return prefix + f"{arr}[{idx}]"
        if t == 0x35:
            member = self.obj(self.ptr())
            _struct = self.ptr()
            if self.remaining() >= 2:
                self.i += 2  # flags
            inst = self.expr()
            return prefix + f"{inst}.{member}"
        if t == 0x36:
            # Vanilla EX_DynArrayLength; this build leaves GNatives[0x36] empty.
            return prefix + f"{self.expr()}.Length"
        if t == 0x41:
            self._eat_debuginfo_payload()
            return prefix + self.expr()
        if t == 0x57:
            return prefix + self._dynarray()
        if t == 0x58:
            arr = self.expr()
            val = self.expr()
            return prefix + f"{arr}.Result = {val}"
        if t == 0x38:
            cst = self.u8()
            inner = self.expr()
            # IntToFloat / IntToBool / similar: keep the value, drop the cast token.
            if cst in (0x3D, 0x3E, 0x3F, 0x41, 0x42):
                return prefix + inner
            return prefix + f"cast({cst:02X}, {inner})"
        if t == 0x3A:
            if self.remaining() >= 8:
                p = struct.unpack_from("<Q", self.data, self.i)[0]
                if p in self.idx.by_addr:
                    self.i += 8
            return prefix + ""
        if t == 0x3F or t == 0x4B:
            return prefix + "None"
        if t == 0x43:
            # Delegate: FName of the callback (compiler lambda or named fn).
            # Native ReadObject is 8 bytes; compiler also emits Object=None.
            p = self._dump_ptr_at(0)
            if p is not None:
                self.i += 8
                return prefix + self.obj(p)
            name = self.fname()
            if self.remaining() >= 8 and struct.unpack_from("<Q", self.data, self.i)[0] == 0:
                self.i += 8
            return prefix + name
        if t == 0x4A:
            return prefix + "/*optional*/"
        if t == 0x4C:
            return prefix + ""
        if t == 0x53:
            # Psyonix template token: skip WORD (abs Script offset) + UObject*
            # (relocated, unused at exec) + BYTE vs a global int. Fall through
            # on match; mismatch jumps to ScriptStart+skip. Not EndOfScript.
            if self.remaining() < 11:
                self.i = len(self.data)
                return prefix + ""
            skip = self.u16()
            typ = self.obj(self.ptr())
            expected = self.u8()
            return prefix + f"/*template {typ} == {expected} skip {skip:03X}*/"
        if t == 0x54:
            # Skip field-init exprs until EndParms, optional DebugInfo, then the instance.
            inits = self._args_until_endparms()
            self._eat_array_tail()
            inst = self.expr()
            body = "; ".join(x for x in inits if x)
            if inst and body:
                return prefix + f"{inst}{{{body}}}"
            return prefix + (inst or f"{{{body}}}")
        if 0x60 <= t <= 0x6F:
            inn = ((t - 0x60) << 8) | self.u8()
            args = self._native_args(inn)
            name = self.idx.inative.get(inn, f"native[{inn}]")
            return prefix + self._fmt_native(name, inn, args)
        if t >= 0x70:
            args = self._native_args(t)
            name = self.idx.inative.get(t, f"native[{t}]")
            return prefix + self._fmt_native(name, t, args)

        # Unknown: dump a few bytes so the walk can be fixed
        rest = self.data[op : min(op + 16, len(self.data))].hex(" ")
        self.i = len(self.data)
        return prefix + f"<tok {t:02X} @ {op:03X}: {rest}>"

    def _eat_endparms(self) -> None:
        if self.peek() == 0x16:
            self.u8()

    def _eat_debuginfo_payload(self) -> None:
        """Bytes after the 0x41 token. Magic 100 → 3 ints + 1 byte (13)."""
        if self.remaining() < 4:
            return
        magic = struct.unpack_from("<I", self.data, self.i)[0]
        if magic == 100 and self.remaining() >= 13:
            self.i += 13

    def _eat_debuginfo(self) -> None:
        if self.peek() != 0x41:
            return
        self.u8()
        self._eat_debuginfo_payload()

    def _eat_array_tail(self) -> None:
        # Method-style array ops skip one terminator (usually EndParms) then
        # optional EX_DebugInfo. Matches the native `Code++; if (*Code==0x41)`.
        if self.remaining():
            self.u8()
        self._eat_debuginfo()

    def _skip16(self) -> None:
        self.u16()

    def _args_until_endparms(self) -> list[str]:
        parts: list[str] = []
        while self.i < len(self.data) and self.peek() not in (0x16, 0x41, 0x4C, 0x30, 0x31):
            start = self.i
            part = self.expr()
            if part and not part.startswith("<tok") and part not in ("<eof>", "<depth>", "Nothing", ")"):
                parts.append(part)
            if self.i == start or part.startswith("<tok") or part in ("<eof>", "<depth>"):
                break
        return parts

    def _callish(self, arr: str, meth: str, args: list[str]) -> str:
        args = [a for a in args if a]
        return f"{arr}.{meth}({', '.join(args)})"

    def _lambda_operands(
        self,
        *,
        prop: bool = False,
        seed: bool = False,
        result_expr: bool = False,
    ) -> tuple[str, str | None, str | None]:
        """After the array expr: WORD skip, delegate, optional dest UProperty*,
        optional seed expr, EndParms+DebugInfo, optional RESULT copy expr.

        Filter/Map/Reduce/Sorted ReadObject a UProperty (not the UFunction).
        The lambda is FindFunction'd by name on the capture object.
        """
        self._skip16()
        fn = self.expr()
        dest_prop = self.obj(self.ptr()) if prop else None
        seed_s = self.expr() if seed else None
        self._eat_array_tail()
        if result_expr:
            self.expr()
        return fn, dest_prop, seed_s

    def _dynarray(self) -> str:
        """Psyonix EX_DynArray: 0x57, sub-op, operands. Vanilla Length/Insert/Remove are dead."""
        sub = self.u8()
        name = DYNARRAY_SUB.get(sub, f"Op_{sub:02X}")

        if sub == 0x00:
            self.u8()  # bounds-check flag
            idx = self.expr()
            arr = self.expr()
            return f"{arr}[{idx}]"
        if sub == 0x01:
            return f"{self.expr()}.Length"

        arr = self.expr()

        if sub == 0x02:
            idx, count = self.expr(), self.expr()
            self._eat_array_tail()
            return self._callish(arr, "Insert", [idx, count])
        if sub == 0x03:
            idx, count = self.expr(), self.expr()
            extra = self.expr()
            self._eat_array_tail()
            args = [idx, count]
            if extra and extra not in (")", "Nothing"):
                args.append(extra)
            return self._callish(arr, "Remove", args)
        if sub == 0x04:
            _flag = self.u8()
            self._skip16()
            item = self.expr()
            self._eat_array_tail()
            meth = "Contains" if _flag else "Find"
            return self._callish(arr, meth, [item])
        if sub == 0x05:
            _flag = self.u8()
            self._skip16()
            prop, item = self.expr(), self.expr()
            self._eat_array_tail()
            meth = "Contains" if _flag else "Find"
            return self._callish(arr, meth, [prop, item])
        if sub == 0x06:
            count = self.expr()
            self._eat_array_tail()
            return self._callish(arr, "Add", [count])
        if sub == 0x07:
            self._skip16()
            item = self.expr()
            self._eat_array_tail()
            return self._callish(arr, "AddItem", [item])
        if sub == 0x08:
            self._skip16()
            item = self.expr()
            self._eat_array_tail()
            return self._callish(arr, "RemoveItem", [item])
        if sub == 0x09:
            self._skip16()
            idx, item = self.expr(), self.expr()
            self._eat_array_tail()
            return self._callish(arr, "InsertItem", [idx, item])
        if sub == 0x0A:
            idx = self.expr()
            val = self.expr()
            self._skip16()
            body: list[str] = []
            while self.i < len(self.data) and self.peek() not in (0x30, 0x31, 0x4C):
                start = self.i
                s = self.expr()
                if s and s not in ("Nothing", ")"):
                    body.append(s)
                if self.i == start:
                    break
            while self.peek() in (0x30, 0x31):
                self.u8()
            inner = "; ".join(body)
            return f"foreach ({idx}, {val} in {arr}) {{ {inner} }}"
        if sub == 0x0B:
            self._skip16()
            items = self._args_until_endparms()
            self._eat_array_tail()
            return self._callish(arr, "AddItem", items)
        if sub in (0x20, 0x21, 0x22):
            fn, _, _ = self._lambda_operands(prop=True, result_expr=True)
            return self._callish(arr, name, [fn])
        if sub == 0x23:
            fn, _, seed = self._lambda_operands(prop=True, seed=True, result_expr=True)
            return self._callish(arr, "Reduce", [fn, seed])
        if sub in (0x24, 0x25, 0x26):
            fn, _, _ = self._lambda_operands()
            return self._callish(arr, name, [fn])
        if sub == 0x27:
            self._skip16()
            other = self.expr()
            self._skip16()
            self.obj(self.ptr())
            self._eat_array_tail()
            self.expr()
            return self._callish(arr, "Concat", [other])

        # Sort (0x0E/0x0F) and remaining Psyonix extras: skip WORD, args, tail.
        self._skip16()
        args = self._args_until_endparms()
        self._eat_array_tail()
        return self._callish(arr, name, args)

    def _native_args(self, inn: int) -> list[str]:
        # AndAnd/OrOr: left, optional SkipMark, right, EndParms.
        # Do not eat a raw WORD — Psyonix uses 0x5E, not a bare skip size.
        parts: list[str] = []
        while self.i < len(self.data) and len(parts) < 16:
            if self.peek() == 0x16:
                self.u8()
                break
            if self.peek() == 0x4C:
                break
            start = self.i
            part = self.expr()
            if part and not part.startswith("<tok") and part not in ("<eof>", "<depth>"):
                parts.append(part)
            elif part.startswith("<tok") or part in ("<eof>", "<depth>") or self.i == start:
                if part.startswith("<tok"):
                    parts.append(part)
                break
        return parts

    # Higher = tighter. Atoms (idents, calls, already-parenthesized) win.
    _PREC_ATOM = 100
    _PREC_UNARY = 8
    _PREC_MUL = 6
    _PREC_ADD = 5
    _PREC_CMP = 4
    _PREC_EQ = 3
    _PREC_AND = 2
    _PREC_OR = 1
    _OP_PREC = (
        (">>>", 6),
        (">>", 6),
        ("<<", 6),
        ("**", 7),
        (">=", 4),
        ("<=", 4),
        ("==", 3),
        ("!=", 3),
        ("&&", 2),
        ("||", 1),
        ("^^", 2),
        ("+", 5),
        ("-", 5),
        ("*", 6),
        ("/", 6),
        (">", 4),
        ("<", 4),
    )
    _ASSOC = {"+", "*", "&&", "||"}

    @classmethod
    def _fully_wrapped(cls, s: str) -> bool:
        if len(s) < 2 or s[0] != "(" or s[-1] != ")":
            return False
        depth = 0
        for i, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return i == len(s) - 1
        return False

    @classmethod
    def _prec(cls, s: str) -> int:
        s = s.strip()
        if not s or cls._fully_wrapped(s):
            return cls._PREC_ATOM
        depth = 0
        min_p = cls._PREC_ATOM
        i = 0
        while i < len(s):
            ch = s[i]
            if ch == "(":
                depth += 1
                i += 1
                continue
            if ch == ")":
                depth -= 1
                i += 1
                continue
            if depth != 0:
                i += 1
                continue
            if ch == " ":
                rest = s[i + 1 :]
                matched = False
                for op, prec in cls._OP_PREC:
                    if rest.startswith(op) and (len(rest) == len(op) or rest[len(op)] == " "):
                        min_p = min(min_p, prec)
                        i += 1 + len(op)
                        matched = True
                        break
                if matched:
                    continue
            i += 1
        return min_p

    def _wrap(self, s: str, parent_prec: int, *, right: bool = False, op: str | None = None) -> str:
        s = s.strip()
        if not s:
            return s
        child = self._prec(s)
        if child > parent_prec:
            return s
        if child == parent_prec:
            if right and op not in self._ASSOC:
                return f"({s})"
            return s
        return f"({s})"

    def _binop(self, a: str, op: str, b: str, prec: int, assign: bool) -> str:
        if assign:
            inner = "*" if op in ("*=", "/=") else "+" if op in ("+=", "-=") else None
            return f"{a} {op} {self._wrap(b, prec, right=True, op=inner)}"
        return f"{self._wrap(a, prec, right=False, op=op)} {op} {self._wrap(b, prec, right=True, op=op)}"

    def _fmt_native(self, name: str, inn: int, args: list[str]) -> str:
        args = [a for a in args if a]
        # UFunction.FriendlyName for iNative > 0x80 is "*" / ">" / "Dot" / "VSize".
        if name in ("!", "-") and len(args) == 1:
            return f"{name}{self._wrap(args[0], self._PREC_UNARY)}"
        if name.startswith("Subtract_Pre") and len(args) == 1:
            return f"-{self._wrap(args[0], self._PREC_UNARY)}"
        if name.startswith("Not_Pre") and len(args) == 1:
            return f"!{self._wrap(args[0], self._PREC_UNARY)}"
        if name in ("Dot", "Cross") and len(args) == 2:
            return f"{name}({args[0]}, {args[1]})"
        friendly_bin = {
            "*": (self._PREC_MUL, False),
            "/": (self._PREC_MUL, False),
            "+": (self._PREC_ADD, False),
            "-": (self._PREC_ADD, False),
            "**": (7, False),
            ">>>": (self._PREC_MUL, False),
            ">>": (self._PREC_MUL, False),
            "<<": (self._PREC_MUL, False),
            ">=": (self._PREC_CMP, False),
            "<=": (self._PREC_CMP, False),
            "==": (self._PREC_EQ, False),
            "!=": (self._PREC_EQ, False),
            "&&": (self._PREC_AND, False),
            "||": (self._PREC_OR, False),
            "^^": (self._PREC_AND, False),
            ">": (self._PREC_CMP, False),
            "<": (self._PREC_CMP, False),
            "*=": (self._PREC_MUL, True),
            "/=": (self._PREC_MUL, True),
            "+=": (self._PREC_ADD, True),
            "-=": (self._PREC_ADD, True),
        }
        if name in friendly_bin and len(args) == 2:
            prec, assign = friendly_bin[name]
            return self._binop(args[0], name, args[1], prec, assign)
        if len(args) == 2:
            a, b = args[0], args[1]
            for prefix, op, assign, rhs_prec in (
                ("MultiplyEqual_", "*=", True, self._PREC_MUL),
                ("DivideEqual_", "/=", True, self._PREC_MUL),
                ("AddEqual_", "+=", True, self._PREC_ADD),
                ("SubtractEqual_", "-=", True, self._PREC_ADD),
                ("MultiplyMultiply_", "**", False, 7),
                ("GreaterGreaterGreater_", ">>>", False, self._PREC_MUL),
                ("GreaterGreater_", ">>", False, self._PREC_MUL),
                ("LessLess_", "<<", False, self._PREC_MUL),
                ("GreaterEqual_", ">=", False, self._PREC_CMP),
                ("LessEqual_", "<=", False, self._PREC_CMP),
                ("EqualEqual_", "==", False, self._PREC_EQ),
                ("NotEqual_", "!=", False, self._PREC_EQ),
                ("AndAnd_", "&&", False, self._PREC_AND),
                ("OrOr_", "||", False, self._PREC_OR),
                ("XorXor_", "^^", False, self._PREC_AND),
                ("Multiply_", "*", False, self._PREC_MUL),
                ("Divide_", "/", False, self._PREC_MUL),
                ("Add_", "+", False, self._PREC_ADD),
                ("Subtract_", "-", False, self._PREC_ADD),
                ("Greater_", ">", False, self._PREC_CMP),
                ("Less_", "<", False, self._PREC_CMP),
            ):
                if name.startswith(prefix):
                    return self._binop(a, op, b, rhs_prec, assign)
            if name.startswith("Dot_"):
                return f"Dot({a}, {b})"
            if name.startswith("Cross_"):
                return f"Cross({a}, {b})"
        if inn == 129 and len(args) == 1:
            return f"!{self._wrap(args[0], self._PREC_UNARY)}"
        return f"{name}({', '.join(args)})"

    def _args(self) -> str:
        return ", ".join(self._native_args(-1))

    def statements(self) -> list[str]:
        out: list[str] = []
        while self.i < len(self.data):
            if self.peek() == 0x4C:
                break
            start = self.i
            s = self.expr()
            if not s:
                if self.i == start:
                    self.i += 1
                continue
            out.append(s)
            if self.i == start:
                break
        return out


def walk_children(proc: ProcessReader, idx: DumpIndex, fn_addr: int) -> list[str]:
    out: list[str] = []
    p = proc.u64(fn_addr + OFF_CHILDREN)
    seen = 0
    while p and seen < 64:
        seen += 1
        name = idx.full(p)
        off = proc.u32(p + OFF_UPROPERTY_OFFSET)
        out.append(f"  +0x{off:02X}  {name}")
        nxt = proc.u64(p + OFF_UFIELD_NEXT)
        if nxt == p:
            break
        p = nxt
    return out


def extract_vtable_disp(code: bytes) -> int | None:
    """Best-effort: last RIP-relative-free `call [reg+disp]` / `jmp [reg+disp]` in the exec thunk."""
    try:
        import capstone  # type: ignore
    except ImportError:
        return None
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    md.detail = True
    last = None
    for insn in md.disasm(code, 0):
        if insn.mnemonic in ("call", "jmp") and insn.op_str.startswith("qword ptr ["):
            m = re.search(r"\+ 0x([0-9a-f]+)", insn.op_str, re.I)
            if m:
                last = int(m.group(1), 16)
            m = re.search(r"\+ ([0-9]+)]", insn.op_str)
            if m and last is None:
                last = int(m.group(1))
    return last


def disassemble_function(
    proc: ProcessReader,
    idx: DumpIndex,
    addr: int,
    name: str,
    raw: bool,
    live_base: int | None,
    hexdump: bool = False,
) -> str:
    flags = proc.u64(addr + OFF_FUNCTION_FLAGS)
    inative = proc.u16(addr + OFF_INATIVE)
    nparms = proc.read(addr + OFF_NUM_PARMS, 1)[0]
    psize = proc.u16(addr + OFF_PARMS_SIZE)
    func = proc.u64(addr + OFF_FUNC)
    script_data = proc.u64(addr + OFF_SCRIPT_DATA)
    script_count = proc.u32(addr + OFF_SCRIPT_COUNT)
    lines: list[str] = []
    lines.append(f"// {name}")
    lines.append(f"// flags=[{flags:#010x}] {flag_names(flags)}  iNative={inative}  parms={nparms} size=0x{psize:X}")
    if live_base and func:
        rva = func - live_base
        lines.append(f"// Func=0x{func:X}  RVA=0x{rva:X}")
    else:
        lines.append(f"// Func=0x{func:X}")
    kids = walk_children(proc, idx, addr)
    if kids:
        lines.append("// children:")
        lines.extend(kids)
    native = bool(flags & FUNC_NATIVE)
    defined = bool(flags & FUNC_DEFINED)
    if native:
        lines.append("// NATIVE — Func is the exec thunk; real body is a vtable slot.")
        try:
            thunk = proc.read(func, 96)
            disp = extract_vtable_disp(thunk)
            if disp is not None:
                lines.append(f"// vtable+0x{disp:X} ({disp})  (capstone, from thunk)")
            else:
                lines.append("// (pip install capstone to recover vtable displacement from the thunk)")
        except OSError as e:
            lines.append(f"// could not read thunk: {e}")
    if defined and script_count > 0 and script_data:
        lines.append(f"// Script 0x{script_data:X}  count=0x{script_count:X} ({script_count})")
        blob = proc.read(script_data, script_count)
        if hexdump:
            for off in range(0, min(len(blob), 256), 16):
                chunk = blob[off : off + 16]
                lines.append(f"// {off:03X}  {chunk.hex(' ')}")
            if len(blob) > 256:
                lines.append(f"// ... {len(blob) - 256} more bytes")
        dec = Decoder(blob, idx, raw=raw)
        lines.append("")
        try:
            stmts = dec.statements()
            for s in stmts:
                lines.append(s)
        except (struct.error, IndexError, OSError) as e:
            lines.append(f"// decode error at {dec.i:03X}: {e}")
        if dec.i < len(blob) - 1:
            leftover = blob[dec.i :].hex(" ")
            lines.append(f"// leftover @{dec.i:03X}: {leftover[:160]}")
    elif not native:
        lines.append("// no Script blob (empty Defined or dump/process mismatch)")
    return "\n".join(lines)


def list_functions(idx: DumpIndex, substr: str) -> None:
    s = substr.lower()
    rows = sorted(
        ((n, a) for a, n in idx.by_addr.items() if n.startswith("Function ") and s in n.lower()),
        key=lambda x: x[0],
    )
    for n, a in rows:
        print(f"0x{a:X}  {n}")
    print(f"// {len(rows)} matches", file=sys.stderr)


def resolve_sdk(arg: str | None) -> Path:
    if arg:
        p = Path(arg)
        if (p / "ObjectDump.txt").is_file():
            return p
        inner = p / "RLSDK"
        if (inner / "ObjectDump.txt").is_file():
            return inner
        raise SystemExit(f"no ObjectDump.txt under {p}")
    env = Path(__file__).resolve().parents[1]
    candidates = [
        env.parent / "RLSDK-Generator" / "generated",
        Path.cwd() / "RLSDK",
    ]
    for c in candidates:
        if not c.exists():
            continue
        dumps = sorted(c.glob("**/ObjectDump.txt"), key=lambda x: x.stat().st_mtime, reverse=True)
        if dumps:
            return dumps[0].parent
    raise SystemExit("pass --sdk path/to/RLSDK (folder that contains ObjectDump.txt)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sdk", help="RLSDK folder (ObjectDump.txt + SDK_HEADERS)")
    ap.add_argument("--function", "-f", action="append", default=[], help="Function name or unique suffix (repeatable)")
    ap.add_argument("--list", help="list Function names containing this substring")
    ap.add_argument("--raw", action="store_true", help="prefix each expr with [off tok]")
    ap.add_argument("--hex", action="store_true", help="hexdump the first 256 Script bytes")
    ap.add_argument("--pid", type=int, help="RocketLeague.exe pid (default: find by name)")
    ap.add_argument("--exe", default="RocketLeague.exe")
    args = ap.parse_args()

    sdk = resolve_sdk(args.sdk)
    dump_path = sdk / "ObjectDump.txt"
    idx = parse_object_dump(dump_path)
    nd = sdk / "NameDump.txt"
    if nd.is_file():
        parse_name_dump(nd, idx)
    parse_inative_cpp(sdk, idx)
    print(f"// sdk {sdk}", file=sys.stderr)
    print(f"// dump base 0x{idx.dump_base:X}  {len(idx.by_addr)} objects", file=sys.stderr)

    if args.list:
        list_functions(idx, args.list)
        return 0
    if not args.function:
        ap.error("pass --function NAME or --list SUBSTR")

    pid = args.pid or find_pid(args.exe)
    if not pid:
        raise SystemExit(f"{args.exe} not running — start the game (same session as ObjectDump)")
    live_base = module_base(pid, args.exe)
    print(f"// pid {pid}  live base 0x{live_base:X}" if live_base else f"// pid {pid}", file=sys.stderr)
    if live_base and idx.dump_base and live_base != idx.dump_base:
        print(
            f"// WARNING dump Base 0x{idx.dump_base:X} != live 0x{live_base:X} — regenerate ObjectDump",
            file=sys.stderr,
        )

    proc = ProcessReader(pid)
    try:
        n = fill_inative_from_live(proc, idx)
        print(f"// iNative>0x80 FriendlyName {n}", file=sys.stderr)
        for q in args.function:
            addr, name = find_function(idx, q)
            print(disassemble_function(proc, idx, addr, name, args.raw, live_base, args.hex))
            print()
    finally:
        proc.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
