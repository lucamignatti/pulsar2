# Scripted sim-vs-real maneuver test

Match-play comparison can only measure what the policy happened to do, and can never
repeat a situation. This runs a **fixed action sequence from a fixed state** in both
venues, so any divergence is pure physics.

33 segments, weighted toward the reported problem areas: landings at various attitudes and
speeds, and flips that interact with the ground (wavedash, dodge-at-contact, dodge-just-
before-landing, landing mid-rotation). Also covers free-air dodges, double jump, ground
driving, air control and wall driving.

Each segment begins with a **state set**, so segments are independent — one maneuver's
error cannot leak into the next, and each is scored on its own.

## Running it

**1. Sim side**

```
cd research/maneuvers/sim_runner
cargo run --release -- ../../../build/collision_meshes ../maneuvers.txt ../sim_maneuvers.tsv
```

**2. Real side** — with Rocket League running:

```
cd rlbot-run
GGL_SCRIPT=../research/maneuvers/maneuvers.txt \
GGL_SCRIPT_OUT=$PWD/real_maneuvers.tsv \
./play.sh match_vs_human 1
```

`GGL_SCRIPT` bypasses the policy entirely — the bot replays the script instead of
inferring. It logs `GGL_SCRIPT: segment N/33 "<name>" done` as it goes and
`ALL SEGMENTS COMPLETE` at the end. Requires `enable_state_setting = true` in the match
config (already set in `match_vs_human.toml` / `match_vs_nexto.toml`).

**3. Compare**

```
python3 research/maneuvers/compare.py \
    research/maneuvers/sim_maneuvers.tsv rlbot-run/real_maneuvers.tsv
```

## Reading the result

Per segment: position error (p50/p90/max), velocity error, forward- and up-vector angle
error, and ground-flag mismatches. Segments are ranked by median position error at the end.

Expect sub-uu medians on driving and free-air segments — that is the established
per-window floor. A landing or flip segment that stands out by 10x+ is a real finding.

## Two things that matter for interpretation

- **Decision-rate cap.** The real game delivers packets at its RENDER rate. With
  `UncappedFramerate=False` (the default) that is 60 Hz, so real rows land on every OTHER
  sim tick and the comparator intersects them. Set `UncappedFramerate=True` in
  `TASystemSettings.ini` and sustain >=120 fps for full per-tick resolution.
- **The ball is parked** at (-3500, 4800, 93) by both runners. The default kickoff ball
  sits at (0,0,93) with radius 91 — its top is z~184, and the drop segments were landing
  on the ball instead of the floor until this was added. If you edit segment positions,
  keep them clear of the park spot.
