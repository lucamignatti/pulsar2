#!/bin/bash
# Headless scored match: RLBotServer + RLBotSim backend + score_match.py.
# Mirrors run_match.sh's startup dance but runs a chosen config and reports goals.
# Everything niced: the trainer owns this box.
cd "$(dirname "$0")"
CFG="${1:-match_vs_nexto.toml}"
DUR="${2:-300}"

# This script PRODUCES A NUMBER, so the trainer guard matters more here than anywhere.
# "Everything niced" above is not enough: the trainer holds all 24 cores, Nexto is
# single-threaded Python doing a per-packet attention forward while our bot is a C++
# MLP, so under load the score measures scheduler priority rather than skill (this is
# what produced the bogus "62-38 vs Nexto" on 2026-07-31). CLAUDE.md: never look for
# the trainer with `pgrep -f` - it matches its own command line - resolve /proc/<pid>/exe.
for e in /proc/[0-9]*/exe; do
	t=$(readlink "$e" 2>/dev/null) || continue
	# Rebuilding build/ under a live trainer is the documented-safe workflow, and it
	# replaces the inode - the running process's exe link then reads
	# ".../GigaLearnBot (deleted)". Strip that or the guard misses the common case.
	t=${t% (deleted)}
	case "$t" in *build/GigaLearnBot)
		if [ "${ALLOW_TRAINER:-0}" != "1" ]; then
			echo "[driver] REFUSING: trainer live (pid $(basename "$(dirname "$e")"))."
			echo "[driver] Run 'tools/trainerctl stop' first - a score taken under the"
			echo "[driver] trainer is not a measurement. Override: ALLOW_TRAINER=1"
			exit 1
		fi
		echo "[driver] WARNING: trainer live and ALLOW_TRAINER=1 - score is NOT valid."
		;;
	esac
done

: > core.log; : > rlbotsim.log; : > scored.log

# The real game STEALS the backend slot. RocketLeague launched with `-rlbot` connects
# out to port 23233 the moment a server opens it - before rlbot_sim can - and the
# "Connected to Rocket League" readiness check cannot tell the two apart. The match
# then spawns into a game sitting at a menu: zero cars forever (diagnosed 2026-07-31
# from exactly those logs). It is also physics contamination for the sim-vs-real
# isolation experiment this harness exists for. Refuse until it is closed.
for c in /proc/[0-9]*/cmdline; do
	line=$(cat "$c" 2>/dev/null | tr '\0' ' ') || continue
	case "$line" in *RocketLeague.exe*)
		echo "[driver] REFUSING: real Rocket League is running (pid $(basename "$(dirname "$c")"))."
		echo "[driver] It will steal the sim's backend port (23233). Quit the game first."
		exit 1 ;;
	esac
done
# A leftover RLBotServer from play.sh holds the ports; ours would fail to bind.
pkill -f RLBotServer 2>/dev/null; sleep 1

nice -n 19 ./RLBotServer > core.log 2>&1 &
CORE=$!
for i in $(seq 1 40); do ss -ltn 2>/dev/null | grep -q ':23234' && break; sleep 0.5; done

for attempt in $(seq 1 10); do
	nice -n 19 ./RLBotSim/target/release/rlbot_sim --headless --rlbot-port 23233 --meshes ./collision_meshes >> rlbotsim.log 2>&1 &
	SIM=$!
	for j in $(seq 1 10); do grep -q "Connected to Rocket League" core.log 2>/dev/null && break 2; sleep 0.3; done
	kill "$SIM" 2>/dev/null
done
grep -q "Connected to Rocket League" core.log 2>/dev/null && echo "[driver] backend connected" || echo "[driver] BACKEND FAILED"

RLBOT_SERVER_PORT=23334 nice -n 19 .venv/bin/python score_match.py "$CFG" "$DUR" 2>&1 | tee scored.log

pkill -f GigaLearnRLBot 2>/dev/null
pkill -f score_match.py 2>/dev/null
pkill -f "nexto/bot.py" 2>/dev/null
kill "$SIM" 2>/dev/null; kill "$CORE" 2>/dev/null
sleep 1
echo "=== bot logs ==="
for f in pulsar-bot/bot.*.log; do [ -f "$f" ] && { echo "--- $f"; tail -6 "$f"; }; done
echo "=== errors ==="
grep -iE "error|panic|exception|traceback|abort" core.log rlbotsim.log scored.log 2>/dev/null | head -10
