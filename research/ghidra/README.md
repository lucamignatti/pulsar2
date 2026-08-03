# Headless Ghidra against RocketLeague.exe

The MCP bridge needs the Ghidra GUI running with the plugin loaded (HTTP on :8089).
It is usually NOT running. Headless works without it and needs no GUI:

```bash
/home/luca/Documents/ghidra_12.1.2_PUBLIC/support/analyzeHeadless \
  /home/luca/Projects/rocketsim-fixing rocketsim-fixing \
  -process RocketLeague.exe -noanalysis \
  -scriptPath <dir containing the .java> \
  -postScript FindDodge.java
```

Gotchas, each cost a round trip:
- The project LOCATION is the directory holding the `.gpr`
  (`/home/luca/Projects/rocketsim-fixing`), not its parent.
- `-noanalysis` is essential: the program is already analysed, and re-analysing
  RocketLeague.exe takes hours.
- Ghidra 12 has no `DefinedDataIterator.definedStrings(Program)`.
- UE3 property names are **FName table entries, not defined string Data** -- a scan of
  all 536142 defined data items finds nothing. Search raw bytes with `find(addr, bytes)`.

## Located so far (property-name table, contiguous)

| range | names |
|---|---|
| `0x141e23116`-`0x141e23262` | `Dodge*`, `DodgeImpulse*` |
| `0x141e2327e`-`0x141e232de` | `DoubleJump*` |
| `0x141e23316`-`0x141e23456` | `AirControl*` |

These are the three components `speed_flip` combines, and speed_flip is the largest
confirmed sim-to-real defect (583 uu, 94.6 deg orientation, ~0 noise floor -- see
SIM2REAL_AUDIT.md S22). `dodge_then_cancel` alone is only 5.3 uu, so the defect is in the
INTERACTION of dodge cancel + air roll + boost, not the cancel itself.

NOT YET DONE: read the CDO default values for these properties and compare them against
RocketSim. Nothing has been diagnosed or patched from this.

## Rule

Do not patch physics off aggregates or plausibility. Two "fixes" (COASTING_BRAKE_FACTOR
0.15->0.11 and dividing dodge torque by the inverse inertia tensor) were both regressions
that looked right and were caught only by the scripted maneuver test. Every patch goes
through research/maneuvers/ and a real-game capture before it is believed.

## Runtime UObject dump — the way past the RTTI wall (planned)

S33 established the static tree walk is blocked: no MSVC RTTI, and most CarComponent natives
sit behind SHARED exec thunks that dispatch on a vtable. The way through is UE3's own
reflection, read from the LIVE process:

- `GObjects` (TArray<UObject*>) -> every UClass / UFunction / UProperty as a live object
- **`UFunction::Func`** = the native function pointer. Vtable resolution WITHOUT RTTI.
- **`UClass::Defaults`** = the CDO in memory, holding real property VALUES. The .upk
  constants without touching packages or decryption.

One technique clears both blockers at once (S31 constants, S33 vtables).

### Do NOT bother scanning the exe for GNames statically

Tried it. `GNames` is populated at RUNTIME, so it is not statically initialised and the scan
only finds lookalikes. The specific decoy: an Oodle compressor-name table at
`0x14233a320`, whose entries are "None", "SuperFast", "Optimal1", "Kraken", "Mermaid",
"LZNA", "LZH", "LZNIB". Its `"None"` is Oodle's null compressor, NOT FName index 0 -- the
tell is that the entry-header int32 decodes to ASCII garbage (1701869896 = "HHHf") rather
than a small index.

### The actual route

Find `GObjects`/`GNames` by their ACCESSORS, not their contents: locate a function that
dereferences them (FName construction, `UObject::StaticFindObject`, class registration) and
read the global it references. Then attach read-only to a running instance and walk the
arrays. Read-only, offline, single-player.

Cross-check available for free: `research/maneuvers` already pins behaviour to a real capture
with a median 0.00 uu noise floor, so any constant recovered this way can be validated
against it immediately rather than trusted on sight.
