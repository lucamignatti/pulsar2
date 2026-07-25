#!/usr/bin/env python3
"""Read the trainer's own stdout: the per-iteration report block, and the
wrapper's crash/restart lines.

Shared by `tools/remote/dashboard.py` (the Training card) and `trainerctl
metrics`, so the web and SSH paths read one parser. Everything here is
READ-ONLY — nothing in this file can touch a live run.

The trainer prints a curated block once per iteration, delimited by a rule of
'=' and made of "Key: value" rows with thousands separators:

    ========================================
    Average Step Reward: 0.4125
    ...
    Total Iterations: 1,407

Deliberately SCHEMA-FREE: keys are whatever the C++ `Report` emits, in the order
it emits them. Every hardcoded metric list in this stack has gone stale at least
once — the curated console block carried 27 rows for subsystems that had been
deleted, right up until 2026-07-25. A parser that follows the trainer cannot
drift from it.

Completeness is decided without naming any key: the newest chunk is accepted
only if its key set covers the previous chunk's. A block still being written has
strictly fewer keys, so it loses and we fall back one iteration (~11 s stale,
which is invisible next to a 15 s poll).
"""

import os
import re
import time

# The delimiter the trainer prints BEFORE each block (Report::Display).
DELIM_RE = re.compile(r"^={10,}\s*$")
# "Key: value" — the key may contain '/', spaces and '-'; the value is free text
# (a number with thousands separators, or scientific notation). A leading " - "
# marks a sub-row: the trainer nests the timing breakdown under Collection Time
# and Consumption Time, which is exactly what you want when diagnosing a
# throughput drop from a phone, so the nesting is preserved rather than flattened.
ROW_RE = re.compile(r"^(\s*-\s+)?([A-Za-z][\w/ .\-+()]*?):\s*(.+?)\s*$")

# Wrapper (tools/run_trainer.sh) lines, which land in the same log via stderr.
LAUNCH_RE = re.compile(r"^\[([\d\- :]+)\] launch #(\d+):")
CRASH_RE = re.compile(r"^\[([\d\- :]+)\] CRASH: (.+?)$")
PLANNED_RE = re.compile(r"^\[([\d\- :]+)\] planned restart \(exit 99\)")
BACKOFF_RE = re.compile(r"^\[([\d\- :]+)\] (\d+) consecutive crashes")

TAIL_BYTES = 64 * 1024  # comfortably more than two report blocks


def _to_number(text):
    """'417,600,512' -> 417600512.0; '1.216597e-04' -> 0.0001216597; else None."""
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def parse_blocks(text):
    """Split report text into blocks. Each block is a list of
    {"k", "v", "sub"} rows in emission order."""
    chunks, cur = [], None
    for line in text.splitlines():
        if DELIM_RE.match(line):
            if cur is not None:
                chunks.append(cur)
            cur = []
            continue
        if cur is None:
            continue  # pre-first-delimiter remnant of a tail read
        m = ROW_RE.match(line)
        if m:
            cur.append({"k": m.group(2), "v": m.group(3), "sub": bool(m.group(1))})
    if cur is not None:
        chunks.append(cur)
    return [c for c in chunks if c]


def latest_report(path, tail_bytes=TAIL_BYTES):
    """Newest COMPLETE report block from a log file.

    Returns {"rows": [{"k","v","sub"}], "numbers": {k: float},
             "age_secs": int} or None."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - tail_bytes))
            text = f.read().decode("utf-8", "replace")
        age = int(time.time() - os.stat(path).st_mtime)
    except Exception:
        return None

    blocks = parse_blocks(text)
    if not blocks:
        return None
    block = blocks[-1]
    # See the module docstring: accept the newest chunk only if it covers the
    # previous one's keys, else it is still being printed.
    if len(blocks) >= 2:
        prev = blocks[-2]
        if not {r["k"] for r in prev}.issubset({r["k"] for r in block}):
            block = prev

    numbers = {}
    for r in block:
        n = _to_number(r["v"])
        if n is not None:
            numbers[r["k"]] = n
    return {"rows": block, "numbers": numbers, "age_secs": age}


class RestartTracker:
    """Incremental scan of a log for the wrapper's launch/crash lines.

    Logs reach hundreds of MB; a 15 s poll cannot re-read them. State is keyed on
    (path, offset) and only new bytes are scanned. A rotated or truncated log
    resets the accumulator."""

    def __init__(self):
        self.path = None
        self.offset = 0
        self.launches = 0
        self.crash_total = 0    # every crash ever seen in this log
        self.crashes = []       # newest last, ring-buffered for display
        self.planned = 0
        self.backoff = None     # last crash-loop backoff line

    def update(self, path):
        try:
            size = os.stat(path).st_size
        except Exception:
            return self.snapshot()
        if path != self.path or size < self.offset:
            self.__init__()
            self.path = path
        try:
            # BINARY mode: the offset is a byte count, and TextIOWrapper.seek()
            # only accepts opaque cookies from its own tell(). Decoding per line
            # also keeps the offset honest when the log holds invalid UTF-8.
            with open(path, "rb") as f:
                f.seek(self.offset)
                for raw in f:
                    if not raw.endswith(b"\n"):
                        break  # partial trailing line; re-read it next poll
                    self.offset += len(raw)
                    self._consume(raw.decode("utf-8", "replace").rstrip("\n"))
        except Exception:
            pass
        return self.snapshot()

    def _consume(self, line):
        if not line.startswith("["):
            return  # cheap reject: every wrapper line is timestamp-prefixed
        m = LAUNCH_RE.match(line)
        if m:
            self.launches = max(self.launches, int(m.group(2)))
            return
        m = CRASH_RE.match(line)
        if m:
            self.crash_total += 1
            self.crashes.append({"time": m.group(1), "what": m.group(2)})
            del self.crashes[:-10]
            return
        if PLANNED_RE.match(line):
            self.planned += 1
            return
        m = BACKOFF_RE.match(line)
        if m:
            self.backoff = {"time": m.group(1), "count": int(m.group(2))}

    def snapshot(self):
        return {
            "launches": self.launches,
            "planned_restarts": self.planned,
            "crash_count": self.crash_total,
            "crashes": self.crashes[-5:],
            "backoff": self.backoff,
        }


def format_report(report):
    """Human-readable rendering for `trainerctl metrics`."""
    if not report:
        return "(no report block in the log yet)"
    width = max(len(r["k"]) + (2 if r["sub"] else 0) for r in report["rows"])
    out = [f"newest report block ({report['age_secs']}s since last log write)", ""]
    for r in report["rows"]:
        label = ("- " if r["sub"] else "") + r["k"]
        out.append(f"  {label.ljust(width)}  {r['v']}")
    return "\n".join(out)


if __name__ == "__main__":
    import sys
    print(format_report(latest_report(sys.argv[1])))
