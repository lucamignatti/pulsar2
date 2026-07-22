"""Start a headless RLBotSim match from match.toml.

RLBotServer (talking to RLBotSim as its game backend) must already be running.
Usage: .venv/bin/python start_match.py [seconds]
"""
import sys
import time
from pathlib import Path

from rlbot.managers import MatchManager
from rlbot.config import load_match_config

HERE = Path(__file__).parent
MATCH_TOML = HERE / "match.toml"

if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 25

    cfg = load_match_config(MATCH_TOML)
    print(f"[start_match] loaded config: {len(cfg.player_configurations)} players, "
          f"auto_start_agents={cfg.auto_start_agents}", flush=True)

    mm = MatchManager()
    # ensure_server_started=False => connect to the already-running server instead of
    # trying to launch a bundled RLBotServer.
    mm.start_match(cfg, wait_for_start=False, ensure_server_started=False)
    print(f"[start_match] match started; running {duration}s...", flush=True)

    time.sleep(duration)

    mm.stop_match()
    mm.disconnect()
    print("[start_match] done", flush=True)
