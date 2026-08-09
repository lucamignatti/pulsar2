#!/bin/bash
# Scored match against the REAL game: RLBotServer + running RocketLeague (-rlbot) as
# backend + score_match.py. The sim twin is run_scored.sh (rlbot_sim backend); this is
# the other half of the same experiment, so the flow mirrors it deliberately.
# Usage: ./score_real.sh [match_config] [duration_s]
set -u
cd "$(dirname "$0")"
CFG="${1:-match_vs_nexto.toml}"
DUR="${2:-600}"

# Same trainer guard as run_scored.sh: a score under the trainer is not a measurement.
for e in /proc/[0-9]*/exe; do
	t=$(readlink "$e" 2>/dev/null) || continue
	t=${t% (deleted)}
	case "$t" in *build/GigaLearnBot)
		echo "[real] REFUSING: trainer live (pid $(basename "$(dirname "$e")"))."; exit 1 ;;
	esac
done
# This one REQUIRES the real game (run_scored.sh refuses it; we need it as backend).
GAME=0
for c in /proc/[0-9]*/cmdline; do
	line=$(cat "$c" 2>/dev/null | tr '\0' ' ') || continue
	case "$line" in *RocketLeague.exe*) GAME=1 ;; esac
done
[ "$GAME" = 1 ] || { echo "[real] REFUSING: RocketLeague.exe not running."; exit 1; }
# EAC guard (precise form from play.sh: ignore the installer).
for c in /proc/[0-9]*/cmdline; do
	line=$(cat "$c" 2>/dev/null | tr '\0' ' ') || continue
	case "$line" in *[Ss]etup*) continue ;; esac
	case "$line" in *RocketLeague_EAC.exe*|*EasyAntiCheat_EOS.exe*)
		echo "[real] REFUSING: EAC running."; exit 1 ;;
	esac
done

rm -f nexto/HANDICAPS pulsar-bot/HANDICAPS 2>/dev/null
: > core_play.log; : > scored.log; rm -f pulsar-bot/bot.*.log 2>/dev/null
pkill -f RLBotServer 2>/dev/null; sleep 1

RLBOT_LOG_LEVEL=info ./RLBotServer > core_play.log 2>&1 &
CORE=$!
trap 'kill $CORE 2>/dev/null; pkill -f GigaLearnRLBot 2>/dev/null; pkill -f RLBotServer 2>/dev/null' EXIT
for i in $(seq 1 40); do ss -ltn 2>/dev/null | grep -q ':23234' && break; sleep 0.5; done
for i in $(seq 1 60); do grep -q "Connected to Rocket League" core_play.log && break; sleep 0.5; done
grep -q "Connected to Rocket League" core_play.log || { echo "[real] game never connected"; exit 1; }
echo "[real] game connected; starting scored match: $CFG ${DUR}s"

.venv/bin/python -u score_match.py "$CFG" "$DUR" 2>&1 | tee scored.log
