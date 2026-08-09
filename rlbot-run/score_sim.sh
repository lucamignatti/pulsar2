#!/bin/bash
# Scored SIM match (rlbot_sim backend) on ALTERNATE ports, so it can run while the real
# game is open. RLBotServer picks its BACKEND port dynamically (first free from 23233),
# and the game's -rlbot flag pins the game to 23233 -- so with no other server running,
# an alternate-port server still opens 23233 and the game STEALS the backend slot (this
# happened: a "sim" match at 18:48 actually played in the real game while rlbot_sim
# died retrying 23235). Defense: hold 23233 ourselves with a dummy listener BEFORE the
# server starts, forcing its backend elsewhere, then read the actual port from its log.
# Usage: ./score_sim.sh [match_config] [duration_s]
set -u
cd "$(dirname "$0")"
CFG="${1:-match_vs_nexto.toml}"
DUR="${2:-600}"
SRV_PORT=23434

for e in /proc/[0-9]*/exe; do
	t=$(readlink "$e" 2>/dev/null) || continue
	t=${t% (deleted)}
	case "$t" in *build/GigaLearnBot)
		echo "[sim] REFUSING: trainer live (pid $(basename "$(dirname "$e")"))."; exit 1 ;;
	esac
done

: > core_sim.log; : > rlbotsim.log; : > scored_sim.log; rm -f pulsar-bot/bot.*.log 2>/dev/null
rm -f nexto/HANDICAPS pulsar-bot/HANDICAPS 2>/dev/null

# Dummy listeners occupying 23233 AND 23234: 23233 so the game (pinned there by
# -rlbot) can never reach this server's backend slot, and 23234 because the server's
# dynamic game-port scan otherwise lands on its own default CLIENT port -- it logs
# "Connected to Rocket League" for a backend there but never forwards the match config
# to it (observed 2026-08-09: zero messages reached rlbot_sim on 23234).
pkill -f "ggl_port_holder" 2>/dev/null; sleep 0.3
python3 -c "# ggl_port_holder
import socket,time
held=[]
for p in (23233, 23234):
    try:
        s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('127.0.0.1',p)); s.listen(1); held.append(p)
    except OSError:
        print(f'[holder] port {p} already taken', flush=True)
print(f'[holder] holding {held}', flush=True)
time.sleep(1e9)" &
DUMMY=$!
sleep 0.5

RLBOT_LOG_LEVEL=debug ./RLBotServer $SRV_PORT > core_sim.log 2>&1 &
CORE=$!
trap 'kill $CORE $SIM $DUMMY 2>/dev/null; pkill -f GigaLearnRLBot 2>/dev/null; pkill -f "nexto/bot.py" 2>/dev/null' EXIT
for i in $(seq 1 40); do ss -ltn 2>/dev/null | grep -q ":$SRV_PORT" && break; sleep 0.5; done
BACKEND_PORT=""
for i in $(seq 1 20); do
	BACKEND_PORT=$(grep -oE "expecting Rocket League on port [0-9]+" core_sim.log | grep -oE "[0-9]+$" | head -1)
	[ -n "$BACKEND_PORT" ] && break; sleep 0.5
done
[ -n "$BACKEND_PORT" ] || { echo "[sim] could not determine backend port from core_sim.log"; exit 1; }
[ "$BACKEND_PORT" = "23233" ] && { echo "[sim] backend landed on 23233 despite dummy - the game could steal it; aborting"; exit 1; }
echo "[sim] server port $SRV_PORT, backend port $BACKEND_PORT"

for attempt in $(seq 1 10); do
	RUST_BACKTRACE=1 ./RLBotSim/target/release/rlbot_sim --headless --rlbot-port "$BACKEND_PORT" --meshes ./collision_meshes >> rlbotsim.log 2>&1 &
	SIM=$!
	for j in $(seq 1 20); do grep -q "Connected to Rocket League" core_sim.log 2>/dev/null && break 2; sleep 0.3; done
	kill "$SIM" 2>/dev/null
done
grep -q "Connected to Rocket League" core_sim.log 2>/dev/null && echo "[sim] backend connected" || { echo "[sim] BACKEND FAILED"; exit 1; }

GGL_RLBOT_PORT=$SRV_PORT .venv/bin/python -u score_match.py "$CFG" "$DUR" 2>&1 | tee scored_sim.log
