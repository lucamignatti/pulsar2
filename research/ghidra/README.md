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
