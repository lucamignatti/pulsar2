#!/bin/bash
# End-to-end headless match driver: start RLBotServer, start RLBotSim as its game
# backend, start the match (which should auto-start both cars), let them play, then
# tear everything down and dump logs.
cd "$(dirname "$0")"
DUR="${1:-30}"

: > core.log
: > rlbotsim.log
: > start_match.log

# 1) core
./RLBotServer > core.log 2>&1 &
CORE=$!

# 2) wait for the bot-facing port (23234)
for i in $(seq 1 40); do ss -ltn 2>/dev/null | grep -q ':23234' && break; sleep 0.5; done

# 3) game backend (RLBotSim), connecting out to core on 23233 by default. rlbot_sim's
# own connect attempt is flaky (it gives up after 4 tries and panics) - retry by
# relaunching the whole process rather than just waiting longer, until core confirms.
for attempt in $(seq 1 10); do
	./RLBotSim/target/release/rlbot_sim --headless --rlbot-port 23233 --meshes ./collision_meshes >> rlbotsim.log 2>&1 &
	SIM=$!
	for j in $(seq 1 10); do grep -q "Connected to Rocket League" core.log 2>/dev/null && break 2; sleep 0.3; done
	kill "$SIM" 2>/dev/null
done
grep -q "Connected to Rocket League" core.log 2>/dev/null && echo "[driver] game backend connected"

# 4) send the match config (auto_start_agents=true should launch pulsar-bot/run.sh)
timeout $((DUR + 20)) .venv/bin/python start_match.py "$DUR" >> start_match.log 2>&1 &
MATCH=$!

# 5) play
sleep "$DUR"
sleep 4

# 6) tear down
pkill -f GigaLearnRLBot 2>/dev/null
pkill -f start_match.py 2>/dev/null
kill "$SIM" 2>/dev/null
kill "$CORE" 2>/dev/null

echo "=========== core.log ==========="
cat core.log
echo "=========== rlbotsim.log (tail) ==========="
tail -60 rlbotsim.log
echo "=========== start_match.log ==========="
cat start_match.log
echo "=========== bots created ==========="
grep -c "Created RLBot bot" core.log rlbotsim.log 2>/dev/null
echo "=========== errors ==========="
grep -iE "error|panic|exception|traceback" core.log rlbotsim.log start_match.log 2>/dev/null | head -20
echo "=========== DONE ==========="
