#!/bin/bash
# Launch wrapper for the GigaLearn RLBot v5 bot.
# The RLBotServer/RLBotSim runs this (run_command_linux) with cwd = this file's dir
# and sets RLBOT_SERVER_IP / RLBOT_SERVER_PORT / RLBOT_AGENT_ID in the environment.
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="/home/luca/Projects/pulsar2-3.0"
# GigaLearnRLBot is dynamically linked against libGigaLearnCPP, libRLBotCPP and libtorch.
export LD_LIBRARY_PATH="$REPO/build:$REPO/GigaLearnCPP/libtorch/lib:$LD_LIBRARY_PATH"
# Inference device. The bot defaulted to CPU (RLBotMain reads GGL_USE_GPU, which nothing
# set), which at tickSkip 1 spends 2.78 ms of the 8.33 ms per-tick budget on the forward
# pass alone - a third of the budget, on a box that is usually also running the viz
# viewer. The GPU is idle during a match (the trainer lives on the cluster), so use it.
export GGL_USE_GPU="${GGL_USE_GPU:-1}"
# Per-decision debug JSONL (exact obs/mask/action + raw packet jump fields + air-state
# transitions) -> debug.<pid>.jsonl next to this script. NOT "a few MB": at ts1/120 Hz it
# serialises a 230-float obs every tick and reached 736 MB in one session, so it is real
# per-tick CPU. Keep it ON for measurement runs; set GGL_DEBUG_JSONL=0 to play clean.
export GGL_DEBUG_JSONL="${GGL_DEBUG_JSONL:-1}"
# Gap-verification handicaps, written by play.sh's mode tokens (see its comments for
# why a marker file instead of env prefixes: the env chain silently dropped them once).
if [ -f "$HERE/HANDICAPS" ]; then
	set -a; . "$HERE/HANDICAPS"; set +a
fi
# Capture each car's stdout+stderr + exit code to bot.<pid>.log (one file per Pulsar2 car,
# so 2v2/3v3 instances don't clobber each other). Segfault leaves exit 139; an uncaught
# C++ exception leaves its what() text here.
# Decision rate travels with the staged checkpoint (written by play.sh's sync_checkpoint).
# Without this the bot silently runs at its 15 Hz default on a 120 Hz (ts1) policy.
if [ -z "$GGL_TICK_SKIP" ] && [ -f "$HERE/checkpoint/TICKSKIP" ]; then
	export GGL_TICK_SKIP="$(cat "$HERE/checkpoint/TICKSKIP")"
fi

LOG="$HERE/bot.$$.log"
"$REPO/build/GigaLearnRLBot" "$HERE/checkpoint" > "$LOG" 2>&1
echo "=== GigaLearnRLBot exited with code $? ===" >> "$LOG"
