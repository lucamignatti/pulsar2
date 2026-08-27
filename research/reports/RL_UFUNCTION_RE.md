# Walking Rocket League UFunctions (SDK → bytecode / native)

**Status**: REFERENCE (method). Author: user, 2026-08-26. Verification appendix added
the same day against the live Aug-10 build.

Rocket League is UE3/UDK. Gameplay types live in `TAGame`, shared types in
`ProjectX` / `Engine` / `Core`. Almost every interesting mechanic is a UFunction on
`Car_TA`, `Ball_TA`, `RBActor_TA`, or a `CarComponent_*_TA`.

The whole pipeline: **name in the SDK → classify Native vs UScript → either IDA the
vtable native or decode the Script blob.** Heap addresses and PIDs change every launch;
offsets, flags, and function names do not.

Two execution paths:

| Kind | FunctionFlags bit | Body lives in | How you read it |
|---|---|---|---|
| UScript | `FUNC_Defined` (0x2), no `FUNC_Native` | `UStruct.Script` TArray (bytecode) | Dump bytes, walk `EExprTokens` |
| Native | `FUNC_Native` (0x400) | x64 in `RocketLeague.exe` | `UFunction.Func` is an exec thunk; the real code is a vtable slot |

The SDK generator emits layouts and flags. It does **not** emit UScript bodies or CDO
defaultproperty values. Those you read live.

## 1. Tool stack

**SDK (names, offsets, flags)**
- RLSDK-Generator — CodeRed-based, RL-specific. Build `RLSDKGenerator.dll`, inject into
  a running game, it walks GObjects/GNames and writes:
  - `SDK_HEADERS/TAGame_classes.hpp` (member offsets)
  - `TAGame_structs.hpp` (`FCarBallInteractionSettings`, `FTimeOfImpactData`, …)
  - `ObjectDump.txt` (`UObject[n] Type Full.Name 0xHEAP`)
  - `*.cpp` ProcessEvent stubs only — **not** the UnrealScript source
- Upstream: CodeRed-Generator, UE3SDKGenerator
- Token enum (vanilla UE3): `UnStack.h` `EExprToken`

**Live memory**
- Debugger/reader that can attach to `RocketLeague.exe` (Cheat Engine, x64dbg, …).
  Need: read heap + module, AOB scan, hardware execute breakpoints.
- **Do not INT3 `PhysicsScene_Step`.** The physics tick is extremely hot; a software BP
  there hitch-kills the process. Use hardware execute BPs on callees (`AddForce`,
  `GetTimeOfImpact` entry, a `movss` store, …).

**Native decompile**
- IDA / Ghidra / Binary Ninja on `RocketLeague.exe`. Image base in IDA is typically
  `0x140000000`. Convert live rip with `IDA_ea = 0x140000000 + (live - module_base)`.

**Optional, not required**
- EliotVU/Unreal-Library decompiles .upk bytecode. RL packages are encrypted
  (RLUPKTool). Live `Script` dump skips that.
- BakkesMod: hook by string name `Function TAGame.CarComponent_Dodge_TA.ApplyDodgeImpulse`.
  Same names as ObjectDump.

## 2. Generate an SDK

1. Clone RLSDK-Generator, set `GConfig::m_outputPathParentDir` in
   `src/Engine/RocketLeague/Configuration.cpp`.
2. Build VS-Release (`RLSDKGenerator.dll`).
3. Launch Rocket League (freeplay is enough).
4. Inject the DLL. Wait until `ObjectDump.txt` and `SDK_HEADERS/` appear.

Offsets can move after a patch; regenerate rather than trusting an old SDK.

You **get**: class layouts (`ABall_TA::CarInteraction` at +0x800 size 0x38,
`ACarComponent_Dodge_TA::MinDodgeTorqueTime` at +0x350), function flags in comments
(`[0x00820103]` vs `[0x00080400]`), ObjectDump lines with per-process heap pointers.

You **do not get**: UScript source/bytecode, CDO defaultproperty values
(`PushZScale = 0.35`, dodge impulse 90000, …), native x64 bodies.

## 3. Finding a target function

Start from a **name**, not an address.

**A. You know the Unreal name** (Bakkes log, SDK, leak, "code soul" `.uc`):
```
Function TAGame.Car_TA.ApplyBallImpactForces
Function TAGame.RBActor_TA.GetTimeOfImpact
Function TAGame.CarComponent_Dodge_TA.ApplyDodgeImpulse
Function TAGame.Car_TA.ShouldDemolish
Function TAGame.Car_TA.IsBumperHit
```
Grep `ObjectDump.txt`. The heap address is valid until the UObject GC moves/destroys it.
UFunction objects usually survive a session; actor instances do not (`Ball_TA` dies on
reset — always re-find via class vtable / Class pointer, never cache a ball `this` from
yesterday).

**B. You know the type, not the function** — grep the generated header for
`class ABall_TA` / `ACar_TA` / `ARBActor_TA` / `ACarComponent_Dodge_TA`, read methods and
fields, then grep ObjectDump for `Function TAGame.<Class>.<Method>`.

**C. You only know a string in-game** — IDA: search for
`ACarComponent_Dodge_TAexecApplyDodgeImpulse` (native registration). That xref is the
*name*, not the body. The body is the vtable slot (§6).

**D. Children chain (locals and parms)** — `UFunction.Children` (+0x88) is a linked
UField list: parms, return, locals. ObjectDump lists them:
```
Function TAGame.Car_TA.IsBumperHit
  StructProperty  ...IsBumperHit.Impact
  BoolProperty    ...IsBumperHit.ReturnValue
  ObjectProperty  ...IsBumperHit.OtherCar
```
A local typed `TimeOfImpactData` means the function *has* TOI data. It does **not** mean
it calls native `GetTimeOfImpact`. `IsBumperHit` fills that struct via
`InitTimeOfImpactFromOldRBState` (Fraction = 0). `ShouldDemolish` calls the sweep first.

## 4. Stable memory layouts

Offsets in the UObject, not ASLR.

**UFunction (size 0x160)**

| Off | Field | Use |
|---|---|---|
| +0x80 | SuperField | |
| +0x88 | Children | first parm/local |
| +0x90 | PropertySize | parm blob size |
| +0x98 | Script.Data | bytecode pointer (Defined) |
| +0xA0 | Script.Count | bytecode size |
| +0x130 | FunctionFlags | Native vs Defined |
| +0x138 | iNative | operator index if native opcode |
| +0x158 | Func | exec thunk (always a module ptr) |

**UProperty**: +0x70 ArrayDim, +0x74 ElementSize, +0x78 PropertyFlags, +0x98 Offset in
owner, +0xC8 `UStructProperty.Struct` (if struct).

**UObject instance (actors)**: +0x00 vtable (module address if live), +0x50 `UClass*`.

**Flag recipes**
```
0x00000400  FUNC_Native
0x00000002  FUNC_Defined          // has Script
0x00000001  FUNC_Final
0x00020000  FUNC_Public
0x00080000  FUNC_Protected
0x00800000  FUNC_HasDefaults      // struct locals with construction
```
- `ApplyBallImpactForces [0x00820103]` → Final|Defined|Public|HasDefaults → **UScript**
- `ApplyDodgeImpulse [0x00080400]` → Native|Protected → **native**
- `GetDodgeImpulse` / `ApplyForces` / `PrePhysicsStep` `[0x00000400]` → **native**

`Func` is never the interesting native for RL dodge. It is always the exec thunk
(ProcessEvent helper that reads bytecode parms, then `call [vtable+slot]`).

## 5. Native path (dodge, AddForce, GetTimeOfImpact)

```
ObjectDump name
  → UFunction*
  → flags have FUNC_Native
  → Func at +0x158  = exec thunk (IDA it)
  → thunk ends in  (*this->vtable[slot])(this, ...)
  → slot byte offset / 8 = vtable index
  → read vtable[slot] from a live CDO/archetype of that class
  → that pointer is the real native
  → IDA / decompile
```

Worked example, `CarComponent_Dodge_TA`:

| UFunction | Func (exec) | vtable off | Meaning |
|---|---|---|---|
| PrePhysicsStep | execPrePhysicsStep | +0x6A0 (1696) | deactivate when ActiveTime > DodgeTorqueTime |
| ApplyForces | execApplyForces | +0x6A8 (1704) | `if (ActiveTime==0) ApplyDodgeImpulse(); ApplyTorqueForces()` |
| GetDodgeImpulse | execGetDodgeImpulse | +0x6B0 (1712) | dir → world impulse |
| ApplyDodgeImpulse | execApplyDodgeImpulse | +0x6B8 (1720) | GetDodgeImpulse + AddForce mode 1 at COM |
| ApplyTorqueForces | execApplyTorqueForces | calls native directly | pitch-cancel after MinDodgeTorqueTime |

**CDO vs archetype for native constants**: `TAGame.Default__CarComponent_Dodge_TA` often
has zeros. Defaults live on `Archetypes.CarComponents.CarComponent_Dodge` (+0x328 …
+0x364). Same for `Ball_TA.CarInteraction`: class CDO is zeros; `Archetypes.Ball.Ball_Default`
and spawned balls have the curve.

**Native AddForce**: `RBActor_TA.AddForce` is native `[0x00024401]`. Optional ForceMode at
parm +0x0C. Instant Δv vs accumulate vs ÷mass is in the PhysX wrapper the thunk calls —
confirm with a live BP on mode in r8/rdx, do not guess from the UScript name "Force".

## 6. UScript path (bytecode)

```
ObjectDump name → UFunction* → FUNC_Defined, no FUNC_Native
  → Func is ProcessInternal (shared)
  → Script.Data / Script.Count at +0x98 / +0xA0
  → dump Count bytes
  → walk EExprToken, resolve every 8-byte pointer via ObjectDump
```

Vanilla tokens (`UnStack.h`) hold for 0x00–0x2A and 0x2C+, with two Psyonix/64-bit nits:
- `0x2B` + `UProperty*` = local / parm (vanilla left 0x2B unused). Confirmed on
  `IsBumperHit` parms.
- `0x5E` appears as a skippable-operand marker in front of `&&` / native ops
  (vanilla `EX_Skip` is 0x18).
- `0x46` + `UProperty*` showed up as the instance of `EX_StructMember` for out structs.
  Treat as "property ref" if ObjectDump resolves it.

Natives: byte ≥ 0x70 is `iNative`. Map via generated `Core_classes.cpp` comments
(`iNative[151]` → `Greater_IntInt`, 226 → `Normal`, 225 → `VSize`, 219 → `Dot`,
216 → `Subtract_VectorVector`, 182 → `MultiplyEqual_FloatFloat`, 130 → `AndAnd_BoolBool`).

- `EX_FinalFunction (0x1C)`: 8-byte `UFunction*`, then args until `EX_EndFunctionParms (0x16)`
- `EX_StructMember (0x35)`: `UProperty*` member, `UScriptStruct*` type, 2 flag bytes, then instance
- `EX_Let (0x0F)`: lvalue expr, rvalue expr
- `EX_JumpIfNot (0x07)`: WORD skip (byte offset from after the skip), then condition

Pointer resolution: any `0x00000191........` / `0x00000190........` in the blob is a
`UObject*`. Grep ObjectDump. A linear scan printing token / resolved name / iNative
already reconstructs the `.uc`.

**Do not use `Children` as `Script`.** The first child of `IsBumperHit` is `OtherCar` at a
nearby heap address; easy to misread as bytecode.

## 7. Defaultproperties (CDO / archetype / instance)

Order of preference:
1. Spawned instance in the map (`Ball_TA` whose Class is `TAGame.Ball_TA`, vtable in-module)
2. Archetype `Archetypes.Ball.Ball_Default`, `Archetypes.CarComponents.CarComponent_Dodge`
3. Class CDO `TAGame.Default__Ball_TA` — often all zeros for NeedCtorLink structs

Finding instances after a map reset: AOB the class vtable (first qword of a known-good
CDO), filter hits whose +0x50 is `Class TAGame.Ball_TA`. Stale ObjectDump actor pointers
look like PhysX heap, not a module vtable.

`FInterpCurveFloat`: TArray of `FInterpCurvePointFloat` (size 0x1C: InVal, OutVal,
tangents, weights, InterpMode). Count at +0x08 of the curve. Walk `Count * 0x1C` at Data.

## 8. Functions walked in the original pass

**Ball extra impulse — UScript + native AddForce**

| Object | Kind | Result |
|---|---|---|
| `Car_TA.ApplyBallImpactForces` | UScript `[0x00820103]`, parm blob 0x84, local `ImpulseOffset` at +0x78 | Script extra; not AddImpulse |
| `Ball_TA.CarInteraction` +0x800 | struct 0x38 | live: PushZScale=0.35, PushForwardScale=0.65, MaxRelativeSpeed=4600, curve 5 linear knots, bSkipScriptForces=0, Restitution=0 / Friction=2 unused by this script |
| `RBActor_TA.AddForce` | Native | mode 2 on the ball extra path (instant Δv, ÷mass, COM). Debug string `"VehicleHitBall"` in the UScript call |

Script reads PushFactorCurve, PushZScale, PushForwardScale, MaxRelativeSpeed. It does
**not** reference Restitution / Friction / bSkipScriptForces. Empty curve → JumpIfNot
skips the whole body. RocketSim already matches the scales and the 4-knot curve; the extra
(1400, 0.6) knot is collinear with 500→2300.

**Demo vs bump TOI**

| Object | Kind | Result |
|---|---|---|
| `RBActor_TA.GetTimeOfImpact` | Native convex sweep | only xref from UScript `ShouldDemolish` (after demo-speed gate). `FTimeOfImpactData` size 0x4C. Fraction < 1 stored on real demos |
| `Car_TA.InitTimeOfImpactFromOldRBState` | UScript | `Impact.Fraction = 0`; copies both cars' poses + hit loc/normal. Fallback when sweep misses |
| `Car_TA.ShouldDemolish` | UScript | sweep; if Fraction >= 1 then Init; then demo cones |
| `Car_TA.IsBumperHit` | UScript, local `Impact: TimeOfImpactData` | starts with Init, **never** GetTimeOfImpact. Then IsCarHitAngleWithinForwardAngle / IsHitLocationWithinForwardAngle / curve / IsCarWithinForwardEllipticalCone / IsValidImpactNormalHit on `PhysicsConfig.CarInteractionSettings` (car bump cones, not `Ball_TA.CarInteraction`) |

`GetTimeOfImpact` is **not** physics CCD. Ordinary `BumpCar` / `ApplyBallImpactForces` do
not call it. Do not reuse a demo-sweep breakpoint to "debug bumps."

**Dodge — all native.** `ApplyDodgeImpulse`, `GetDodgeImpulse`, `ApplyForces`,
`ApplyTorqueForces`, `PrePhysicsStep` — exec thunks at `Func`, real code on the vtable.
Archetype floats match sim: deadzone 0.5, impulses 90000/96000 (= 500 and 500×16/15 UU/s
at mass 180), speed scales 1.9/1.0/2.5, torques 260/224, DodgeTorqueTime=0.65,
MinDodgeTorqueTime=0.041, Z damp 0.35 / 0.15 / 0.06.

## 9. Live confirmation

Bytecode tells you who calls whom. A BP tells you whether it runs on this tick.

1. Resolve `UFunction*` from ObjectDump (or Bakkes FindFunction).
2. Native: BP the **vtable native**, not the exec thunk (thunks run for every
   ProcessEvent, including unused optional-parm skips).
3. UScript: BP a callee native (`AddForce`, `GetTimeOfImpact`) rather than ProcessInternal.
4. **Hardware execute only** on hot paths.
5. Record `this` class (`*(this+0x50)`), force mode, and whether `UFunction*` in the stack
   matches the name you think.

Example: car–ball extra never hit AddImpulse; AddForce `this` was `Ball_TA`, mode always 2.
That closed a wrong IDA guess.

## 10. How to pick the next function

1. Name it in the SDK or ObjectDump.
2. Read FunctionFlags. Native → thunk → vtable → IDA. Defined → dump Script.
3. Read Children. Locals tell you the data model.
4. Grep Script for callee `UFunction*`. One AOB of that heap pointer answers "does A call
   B?" without a full decoder.
5. Defaults: instance / archetype, not `Default__`.
6. Do not score the wrong event. Bakkes `ImpulseType::BallImpact` / `CarImpact` are hooks
   on `ApplyBallImpactForces` / `ApplyCarImpactForces` (car Δv), not
   `ArenaEvent::CarHitBall` / `CarHitCar`. Demo TOI ≠ Bullet CCD.

## 11. Pitfalls

- ObjectDump actor pointers go stale after reset. UFunction / UClass / CDO usually stay.
- `UFunction.Func` on natives is a dead exec thunk. "parse byte 0x41, call vtable" means
  you are in the thunk. Keep going.
- Same struct name, different object. `Ball_TA.CarInteraction`
  (`FCarBallInteractionSettings`) ≠ `Car_TA` / `PhysicsConfig.CarInteractionSettings`
  (`FCarInteractionConfig` bumper/demo cones).
- 120 Hz tapes only if you are fitting sim2real. Layout RE does not care; scoring does.
- Regenerating the SDK after a patch is cheaper than relocating every RVA.

## 12. Minimal recipe

```
1. Inject RLSDK-Generator → ObjectDump + headers
2. Grep ObjectDump for Function TAGame.<Class>.<Name>
3. Read UFunction+0x130 flags
4. Native:
     read +0x158 Func → IDA exec thunk → vtable displacement
     read that slot from class CDO vtable → IDA real native
     read constants from Archetypes.* not Default__*
5. UScript:
     read +0x98 Data, +0xA0 Count → dump
     0x1C = call (next 8 bytes = UFunction*)
     0x2B = local (next 8 bytes = UProperty*)
     byte >= 0x70 = iNative (Core_classes.cpp)
     grep callee pointers vs ObjectDump
6. Confirm with a hardware BP on the native leaf, not PhysicsScene_Step
```

---

# Appendix: static-only variant, verified 2026-08-26 (Aug-10 build)

The vtable step can be done **without injecting anything**, because vtables are static
`.rdata` in the PE. This was used to resolve the dodge activation gate on the live build.

**Anchor problem**: our older Ghidra project holds a *different* RocketLeague.exe, and its
RVAs do not transfer (a known-good `ApplyTorqueForces` address from it found 0 hits).
Build the anchor from the current binary instead.

**Chain that worked** (all static, no debugger):

1. Build the native name→VA map by scanning `.rdata`/`.text` for the UE3
   `FNativeFunctionLookup` pairs (`{char* "<Class>exec<Func>", void* thunk}`). 4,135
   natives resolve on this build. (`research/tools` scratch script; regenerate per patch.)
2. Decompile the exec thunk to get the vtable displacement, exactly as §5:
   - `ACarComponent_TAexecCanActivate` @ `0x1406936d0` → `(**(code **)(*this + 0x698))`
   - `ACarComponent_Dodge_TAexecApplyDodgeImpulse` @ `0x140e88810` → `+0x6b8`
   - `ACarComponent_Dodge_TAexecGetDodgeImpulse` @ `0x140e88750` → `+0x6b0`
   - `ACarComponent_Dodge_TAexecApplyTorqueForces` @ `0x140e88860` → calls
     `FUN_140eb6060` **directly** (no vtable), matching §5's "calls native directly"
3. Get a real (non-thunk) native of the class from step 2 — here `FUN_140eb6060`.
4. Find its caller: `FUN_140eb3490` calls it → that is `Dodge::ApplyForces`, which §5 says
   lives at vtable **+0x6a8**.
5. Byte-search `.rdata` for the qword `0x140eb3490`. Each hit is a vtable slot; the base is
   `hit - 0x6a8`. Two hits → `CarComponent_Dodge_TA` and `CarComponent_Dodge_KO_TA`.
   (Do **not** try to find the vtable base by walking back to the start of a run of
   `.text` pointers — vtables are packed contiguously, the walk spans 17k slots.)
   Also beware: a Ghidra `[DATA]` xref may land in `.pdata` (RVA triplets), not a vtable.

**Resolved vtables (Aug-10 build)**

| slot | Dodge_TA @ `0x141e4c358` | Dodge_KO_TA @ `0x141e4ca18` |
|---|---|---|
| CanActivate +0x698 | `0x140eca050` | `0x140eca050` |
| PrePhysicsStep +0x6a0 | `0x140f1add0` | `0x140f1add0` |
| ApplyForces +0x6a8 | `0x140eb3490` | `0x140eb3490` |
| GetDodgeImpulse +0x6b0 | `0x140eee600` | `0x140eedd90` |
| ApplyDodgeImpulse +0x6b8 | `0x140eb1a20` | `0x140eb1370` |

Shared activation/step/forces, differing impulses — the expected shape, which is itself a
consistency check on the whole chain.

**`CarComponent_Dodge_TA::CanActivate` @ `0x140eca050`**

```c
car = this->Car;                                   // component +0x288
if (car->flags_0x7F8 & 1) {                        // (A) CACHED car-level bit
    t = (this->flags_0x278 & 0x10) ? 0.0f : this->f_0x298;
    if (this->f_0x29c <= t) {                      // (B) timer threshold
        ... delegate via car->vtable+0x218 sets local_10 ...   // (C)
        if (local_10 && (this->i_0x2ec < 1 || this->i_0x2e8 < this->i_0x2ec))
            return 1;                              // (D) use count
    }
}
return 0;
```

**The load-bearing observation: `CanActivate` never queries wheel contact.** There is no
`IsOnGround`, no `GetNumWheelContacts`, no raycast — the ground term is a **cached bit**
(`car+0x7F8` bit 0; the same bitfield holds `bSuperSonic` at bit 0x20, confirmed
separately). Our engine gates the dodge press on a **live** `is_on_ground` recomputed each
tick, which is a different thing: a cache written at a different point in the tick is
exactly a one-tick offset.

That is a candidate *mechanism* for the empirically-fitted 1-tick level-takeoff contact
hold in `GGL_DODGE_GROUND_HOLD` (see `WAVEDASH_GATE.md`) — but it is **not yet proof**.
Open, and the next step if this is ever picked up:

- What is `car+0x7F8` bit 0, and **when in the tick is it written?** 26 `test byte
  [reg+0x7F8], imm8` read sites exist; there are no immediate-mode `or`/`and` writers, so
  it is written via a computed value and needs either careful analysis of the read sites or
  a live hardware BP (§9) correlating the bit against wheel state.
- If the bit turns out to be a ground cache latched at a specific tick phase, the fitted
  hold should be **replaced by a port of that latch** — which would likely also address the
  residual `tilt_on_side` contact error (32% of its ticks), since a cached flag does not
  care about per-wheel tilt the way our live 3-wheel query does.
