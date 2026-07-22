"""Build a match of a given team size from a base toml, then hold the connection.

Usage: run_match_holder.py <match_config.toml> <team_size>

The base toml lists ONE participant per team (the "primary"), e.g. Human vs Pulsar2,
or Pulsar2 vs Element. We expand each team up to <team_size> cars:
  - the primary is kept as-is (a Human stays human),
  - remaining slots are filled with a bot: the first bot config on that team, or
    - if the team's primary is a Human (no bot to clone) - the default fill bot
      (pulsar-bot/bot.toml). So "fill my team with bots too" = your teammates are Pulsar2.
"""

import os
import signal
import sys
import time
import tomllib
import traceback
from pathlib import Path

from rlbot.managers import MatchManager
from rlbot.config import load_match_config, load_player_config, get_human

CFG_PATH = Path(sys.argv[1]).resolve()
TEAM_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 1
ROOT = CFG_PATH.parent
DEFAULT_FILL_BOT = ROOT / "pulsar-bot" / "bot.toml"  # fills human teammates / empty slots


def resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (ROOT / q)


# Keep the base config's match settings (map, mutators, instant_start, etc.);
# we only replace the participant list.
cfg = load_match_config(CFG_PATH)

with open(CFG_PATH, "rb") as f:
    raw = tomllib.load(f)
cars = raw.get("cars", [])

players = []
for team in (0, 1):
    team_cars = [c for c in cars if int(c.get("team", 0)) == team]

    # Filler bot for this team: first bot config listed on the team, else the default.
    filler = None
    for c in team_cars:
        if c.get("config_file"):
            filler = resolve(c["config_file"])
            break
    if filler is None:
        filler = DEFAULT_FILL_BOT

    built = []
    for c in team_cars[:TEAM_SIZE]:
        if (c.get("type") or "rlbot").lower() == "human":
            built.append(get_human(team))
        else:
            built.append(load_player_config(resolve(c["config_file"]), team))
    while len(built) < TEAM_SIZE:
        built.append(load_player_config(filler, team))

    players.extend(built)

cfg.player_configurations = players

# Eval/replay mode (REPLAY=1): tell Rocket League to auto-save a replay of the match.
# NOTE: the toml parser ignores an `auto_save_replay` key, so this must be set on the
# config object here, not in the toml.
REPLAY = os.environ.get("REPLAY") == "1"
if REPLAY:
    cfg.auto_save_replay = True
    cfg.skip_replays = False
    print("[holder] EVAL mode: auto_save_replay=True, skip_replays=False", flush=True)

# play.sh tears the holder down with a plain SIGTERM (kill_all), but Python's default
# SIGTERM handler exits WITHOUT running the finally block below - which would skip the
# replay save entirely. Re-raise SIGTERM as KeyboardInterrupt so both Ctrl-C and the
# harness teardown reach the same graceful shutdown path.
_shutting_down = False

def _begin_shutdown(signum, frame):
    # First signal starts the graceful path; ignore any further signals at the OS
    # level so a second one (Ctrl-C sends SIGINT to the group AND play.sh sends
    # SIGTERM) can't interrupt the replay-flush sleep and truncate the save.
    global _shutting_down
    if _shutting_down:
        return
    _shutting_down = True
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    raise KeyboardInterrupt

signal.signal(signal.SIGTERM, _begin_shutdown)
signal.signal(signal.SIGINT, _begin_shutdown)

# Seconds to let Rocket League flush the .replay to disk after StopMatch, before we
# disconnect. play.sh's teardown waits longer than this so the save completes while
# RLBotServer is still alive.
REPLAY_FLUSH_SECS = 5

man = MatchManager()
try:
    man.start_match(cfg, wait_for_start=False, ensure_server_started=False)
    print(f"[holder] match config sent: team_size={TEAM_SIZE}, {len(players)} cars", flush=True)
    while True:
        time.sleep(0.5)
except KeyboardInterrupt:
    pass
except Exception:
    traceback.print_exc()
finally:
    # Matches are UNLIMITED-length (a hard requirement), so they never end on their
    # own - and RLBot only writes a replay when it receives a StopMatch ("RLBot does
    # not actually detect the ending of matches", MatchManager.connect docstring).
    # So in EVAL mode we explicitly StopMatch (shutdown_server stays False - the RL
    # server keeps running) to trigger auto_save_replay, then give RL a moment to
    # flush the file before dropping the socket. This is how we get BOTH: unlimited
    # play, and a replay of the whole thing saved when you end the session.
    if REPLAY:
        try:
            print("[holder] EVAL end: StopMatch -> triggering replay save...", flush=True)
            man.stop_match()  # StopCommand(shutdown_server=False)
            time.sleep(REPLAY_FLUSH_SECS)
            print("[holder] replay flush window elapsed.", flush=True)
        except Exception:
            traceback.print_exc()
    print("[holder] exiting (disconnect only; NOT shutting down server)", flush=True)
    try:
        man.disconnect()
    except Exception:
        pass
