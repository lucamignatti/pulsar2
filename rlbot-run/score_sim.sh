#!/bin/bash
# Scored SIM match (rlbot_sim backend) on ALTERNATE ports, so it can run while the real
# game is open. `RLBotServer <port>` moves the client port and shifts the backend off
# 23233 (probed 2026-08-09: `RLBotServer 23434` reports "expecting Rocket League on
# port 23235"), and the game's -rlbot flag pins it to 23233 -- so it cannot steal the
# backend slot here, which is what run_scored.sh's game-refusal guard exists for.
# Usage: ./score_sim.sh [match_config] [duration_s]
set -u
cd "$(dirname "$0")"
CFG="${1:-match_vs_nexto.toml}"
DUR="${2:-600}"
SRV_PORT=23434
BACKEND_PORT=23235   # what RLBotServer $SRV_PORT reports it expects RL on

for e in /proc/[0-9]*/exe; do
	t=$(readlink "$e" 2>/dev/null) || continue
	t=${t% (deleted)}
	case "$t" in *build/GigaLearnBot)
		echo "[sim] REFUSING: trainer live (pid $(basename "$(dirname "$e")"))."; exit 1 ;;
	esac
done

: > core_sim.log; : > rlbotsim.log; : > scored_sim.log; rm -f pulsar-bot/bot.*.log 2>/dev/null
rm -f nexto/HANDICAPS pulsar-bot/HANDICAPS 2>/dev/null

./RLBotServer $SRV_PORT > core_sim.log 2>&1 &
CORE=$!
trap 'kill $CORE $SIM 2>/dev/null; pkill -f GigaLearnRLBot 2>/dev/null; pkill -f "nexto/bot.py" 2>/dev/null' EXIT
for i in $(seq 1 40); do ss -ltn 2>/dev/null | grep -q ":$SRV_PORT" && break; sleep 0.5; done

for attempt in $(seq 1 10); do
	./RLBotSim/target/release/rlbot_sim --headless --rlbot-port $BACKEND_PORT --meshes ./collision_meshes >> rlbotsim.log 2>&1 &
	SIM=$!
	for j in $(seq 1 10); do grep -q "Connected to Rocket League" core_sim.log 2>/dev/null && break 2; sleep 0.3; done
	kill "$SIM" 2>/dev/null
done
grep -q "Connected to Rocket League" core_sim.log 2>/dev/null && echo "[sim] backend connected" || { echo "[sim] BACKEND FAILED"; exit 1; }

GGL_RLBOT_PORT=$SRV_PORT .venv/bin/python -u score_match.py "$CFG" "$DUR" 2>&1 | tee scored_sim.log
