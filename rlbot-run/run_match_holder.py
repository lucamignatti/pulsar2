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

# QUALIFIER RACE MODE (QUAL_FOR/QUAL_AGAINST env, armed by play.sh for eval-vs-Nexto):
# the session is a sequence of ATTEMPTS at "score QUAL_FOR before the opponent scores
# QUAL_AGAINST", inside one holder process. Our team reaching QUAL_FOR ends the match
# (StopMatch in the finally block -> auto_save_replay writes the .replay) and stamps
# rlbot-run/QUALIFIED; the opponent reaching QUAL_AGAINST restarts the match (fresh
# 0-0 - existing_match_behavior="Restart") and the grind continues. Unset -> the old
# hold-forever behavior. The opponent is at FULL STRENGTH here: play.sh only arms this
# in plain eval mode, where no HANDICAPS markers exist - a qualifier replay must show
# a real Nexto or it shows nothing.
QUAL_FOR = int(os.environ.get("QUAL_FOR", "0"))
QUAL_AGAINST = int(os.environ.get("QUAL_AGAINST", "0"))
QUAL_TEAM = int(os.environ.get("QUAL_TEAM", "0"))  # our team index in the toml (0 = blue/Pulsar)
QUAL_ON = QUAL_FOR > 0 and QUAL_AGAINST > 0
QUAL_MARKER = Path(__file__).parent / "QUALIFIED"
if QUAL_ON:
    QUAL_MARKER.unlink(missing_ok=True)
    print(f"[holder] QUALIFIER mode: first to {QUAL_FOR} before opponent reaches "
          f"{QUAL_AGAINST}; failed attempts auto-restart", flush=True)

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

# A FAILED qualifier attempt must leave NO replay behind: the whole point of the grind
# is that the .replay in the Demos folder IS the evidence of a passed attempt, so a
# folder littered with failures makes the successful one unidentifiable.
# The old code restarted with start_match() and never sent StopMatch, on the assumption
# that "no StopMatch => no replay". That was never verified, and auto_save_replay is set
# on the config, so RL may well flush one when the match is replaced. So: end each failed
# attempt EXPLICITLY, then delete whatever replay it produced.
REPLAY_DIR = Path(os.environ.get("REPLAY_DIR", "")) if os.environ.get("REPLAY_DIR") else None
REPLAY_SETTLE_SECS = 3  # let RL finish writing before we look for new files


def _replay_snapshot() -> set:
    if not REPLAY_DIR or not REPLAY_DIR.is_dir():
        return set()
    try:
        return set(REPLAY_DIR.glob("*.replay"))
    except OSError:
        return set()


def _discard_new_replays(before: set, why: str) -> None:
    """Delete any .replay that appeared since `before` was taken."""
    if not REPLAY_DIR:
        return
    time.sleep(REPLAY_SETTLE_SECS)
    for f in sorted(_replay_snapshot() - before):
        try:
            f.unlink()
            print(f"[holder] discarded replay of {why}: {f.name}", flush=True)
        except OSError as e:
            print(f"[holder] WARNING could not delete {f.name}: {e}", flush=True)


# Rocket League does NOT silently auto-save here: on StopMatch it pops the "name your
# replay" dialog and waits for KEYBOARD INPUT. The old fixed 5s flush window expired
# with that dialog still open, play.sh tore the session down, and the replay of a
# passing 42-7 run was lost. So after StopMatch, POLL for the new .replay instead of
# sleeping a fixed time - the human needs time to type a name and confirm.
REPLAY_WAIT_SECS = int(os.environ.get("REPLAY_WAIT_SECS", "300"))


def _wait_for_replay(before: set, timeout: int) -> bool:
    """Block until a new .replay appears (RL's save dialog needs a human). True if saved."""
    if not REPLAY_DIR:
        time.sleep(REPLAY_FLUSH_SECS)
        return False
    print("\n" + "#" * 62, flush=True)
    print("#  ROCKET LEAGUE IS ASKING YOU TO NAME THE REPLAY.", flush=True)
    print("#  Switch to the game, type a name, and confirm the save.", flush=True)
    print(f"#  Waiting up to {timeout}s - nothing is torn down until you do.", flush=True)
    print("#" * 62 + "\n", flush=True)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        new = _replay_snapshot() - before
        if new:
            time.sleep(REPLAY_SETTLE_SECS)  # let the write finish
            for f in sorted(new):
                print(f"[holder] REPLAY SAVED: {f}", flush=True)
            return True
        time.sleep(2)
    print(f"[holder] WARNING: no replay after {timeout}s - was the dialog confirmed?",
          flush=True)
    return False


session_replays = _replay_snapshot()   # baseline for the end-of-session replay wait
man = MatchManager()
try:
    man.start_match(cfg, wait_for_start=False, ensure_server_started=False)
    print(f"[holder] match config sent: team_size={TEAM_SIZE}, {len(players)} cars", flush=True)
    attempt = 1
    attempt_replays = _replay_snapshot()   # baseline for "did THIS attempt write one?"
    last = (-1, -1)
    # After a restart the packet can lag with the DEAD match's score still >= the
    # threshold, which would re-trigger instantly; stay disarmed until 0-0 is seen.
    armed = True
    while True:
        time.sleep(0.5)
        if not QUAL_ON:
            continue
        p = man.packet
        if p is None or not p.teams or len(p.teams) < 2:
            continue
        us = p.teams[QUAL_TEAM].score
        them = p.teams[1 - QUAL_TEAM].score
        if not armed:
            if us == 0 and them == 0:
                armed = True
                print(f"[holder] attempt {attempt} live at 0-0", flush=True)
            continue
        if (us, them) != last:
            print(f"[holder] attempt {attempt}: {us}-{them} "
                  f"(need {QUAL_FOR} before {QUAL_AGAINST})", flush=True)
            last = (us, them)
        if us >= QUAL_FOR:
            print(f"[holder] *** QUALIFIED {us}-{them} on attempt {attempt} *** "
                  f"stopping match to save the replay", flush=True)
            QUAL_MARKER.write_text(f"{us}-{them} attempt={attempt}\n")
            break  # -> finally: StopMatch + flush -> auto_save_replay writes the file
        if them >= QUAL_AGAINST:
            print(f"[holder] attempt {attempt} FAILED at {us}-{them}; "
                  f"ending match, discarding its replay, starting attempt "
                  f"{attempt + 1}", flush=True)
            # STOP the failed match explicitly (shutdown_server=False keeps RLBotServer
            # up), bin any replay it wrote, THEN start a brand new one.
            try:
                man.stop_match()
            except Exception:
                traceback.print_exc()
            _discard_new_replays(attempt_replays, f"failed attempt {attempt}")
            attempt += 1
            last = (-1, -1)
            armed = False
            attempt_replays = _replay_snapshot()
            man.start_match(cfg, wait_for_start=False, ensure_server_started=False)
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
            # RL pops a name-the-replay dialog; wait for the FILE, not a fixed timeout.
            _wait_for_replay(session_replays, REPLAY_WAIT_SECS)
        except Exception:
            traceback.print_exc()
    print("[holder] exiting (disconnect only; NOT shutting down server)", flush=True)
    try:
        man.disconnect()
    except Exception:
        pass
