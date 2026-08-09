"""Headless RLBotSim match with a real scoreline.

run_match.sh hardcodes match.toml and reports no score, so this runs a chosen config
and polls the packet for team scores. The relay message loop MUST run in a background
thread or MatchManager.packet is never populated.
"""
import os, sys, time
from pathlib import Path
from rlbot.managers import MatchManager
from rlbot.config import load_match_config

HERE = Path(__file__).parent
cfgName, duration = sys.argv[1], int(sys.argv[2])

cfg = load_match_config(HERE / cfgName)
mm = MatchManager()
# rlbot's RLBOT_SERVER_PORT module constant is hardcoded to 23234 and ignores the
# environment - and 23234 is the VIZ's own VizRLBotServer. Without this the client
# silently connects to the viewer and the match sits at 0-0 forever.
# GGL_RLBOT_PORT: run against an RLBotServer on a non-default port (lets a sim match
# run while the real game holds the default 23233/23234 pair hostage).
mm.rlbot_server_port = int(os.environ.get("GGL_RLBOT_PORT", "23234"))
mm.start_match(cfg, wait_for_start=False, ensure_server_started=False)
print(f"[score] match started: {cfgName}, {duration}s", flush=True)
for _ in range(60):
    if mm.packet is not None and mm.packet.players: break
    time.sleep(1.0)
else:
    print("[score] WARNING: no packet with players after 60s", flush=True)

p = mm.packet
names = [c.name for c in p.players] if p and p.players else []
print(f"[score] cars: {names}", flush=True)
if len(names) < 2:
    print("[score] ABORT: fewer than 2 cars spawned", flush=True)

t0 = time.time(); last = (0, 0); stalls = 0
while time.time() - t0 < duration:
    time.sleep(1.0)
    p = mm.packet
    if p is None or not p.teams:
        stalls += 1
        continue
    s = tuple(t.score for t in p.teams)
    if s != last:
        print(f"[score] {s[0]}-{s[1]} at t={time.time()-t0:.0f}s", flush=True)
        last = s

tot = last[0] + last[1]
print(f"[score] FINAL blue(Pulsar)={last[0]} orange(Nexto)={last[1]}  stalls={stalls}", flush=True)
print(f"[score] PULSAR SHARE = {100*last[0]/tot:.1f}%  ({tot} goals)" if tot else "[score] NO GOALS", flush=True)
try:
    mm.stop_match(); mm.disconnect()
except Exception:
    pass
